"""
C-index fairness analysis across models and demographic subgroups.

Computes per-fold concordance index using lifelines, then runs pairwise
Wilcoxon signed-rank tests on fold-level deltas with Holm-Bonferroni correction.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).resolve().parents[3] / "results" / "training_csvs"
OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "resnet_baseline": DATA_DIR / "resnet_baseline_compiled.csv",
    "ecg_fm": DATA_DIR / "ecg_fm_compiled.csv",
    "ecg_fm_e2e": DATA_DIR / "ecg_fm_e2e_compiled.csv",
}

# Folds present in ecg_fm_e2e (4 and 6 are absent)
# ECG_FM_E2E_FOLDS = {1, 2, 3, 5, 7, 8, 9, 10}
ALL_FOLDS = set(range(1, 11))

RACE_SUBGROUPS = ["White", "Black", "Asian", "Other", "Unknown"]
SEX_SUBGROUPS = ["Male", "Female"]


def normalize_sex(series: pd.Series) -> pd.Series:
    return series.str.strip().str.title()


def load_model_data() -> dict[str, pd.DataFrame]:
    dfs = {}
    for model, path in MODELS.items():
        df = pd.read_csv(path)
        df["Sex"] = normalize_sex(df["Sex"])
        dfs[model] = df
    return dfs


def compute_cindex_per_fold(
    df: pd.DataFrame, folds: set[int]
) -> dict[int, float | None]:
    """ Return {fold: cindex} for the given subset dataframe and fold set. """
    results = {}
    for fold in sorted(folds):
        fold_df = df[df["fold_num"] == fold]
        if fold_df["event"].sum() < 2:
            results[fold] = None
            continue
        try:
            ci = concordance_index(
                fold_df["duration"],
                -fold_df["log_risk"], # -ve log_risk
                fold_df["event"],
            )
            results[fold] = ci
        except Exception:
            results[fold] = None
    return results


def get_subgroup_mask(df: pd.DataFrame, subgroup: str) -> pd.Series:
    if subgroup == "Overall":
        return pd.Series(True, index=df.index)
    if subgroup in SEX_SUBGROUPS:
        return df["Sex"] == subgroup
    if subgroup in RACE_SUBGROUPS:
        return df["PatientRace"] == subgroup
    raise ValueError(f"Unknown subgroup: {subgroup}")


# Used when hpc folds were incomplete
def folds_for_model(model: str) -> set[int]:
    # if model == "ecg_fm_e2e":
    #     return ECG_FM_E2E_FOLDS
    # else:
    #     return ALL_FOLDS
    return ALL_FOLDS

# Used when hpc folds were incomplete
def paired_folds(model_a: str, model_b: str) -> set[int]:
    """ Folds valid for a paired comparison between two models. """
    return folds_for_model(model_a) & folds_for_model(model_b)



# Main
def compute_all_cindices(
    dfs: dict[str, pd.DataFrame],
) -> dict[tuple[str, str], dict[int, float | None]]:
    """
    Returns {(model, subgroup): {fold: cindex_or_None}}.
    """
    subgroups = ["Overall"] + SEX_SUBGROUPS + RACE_SUBGROUPS
    cache: dict[tuple[str, str], dict[int, float | None]] = {}

    for model, df in dfs.items():
        folds = folds_for_model(model)
        for subgroup in subgroups:
            mask = get_subgroup_mask(df, subgroup)
            subset = df[mask]
            cache[(model, subgroup)] = compute_cindex_per_fold(subset, folds)

    return cache


def build_results_table(
    cache: dict[tuple[str, str], dict[int, float | None]],
) -> pd.DataFrame:
    rows = []
    subgroups = ["Overall"] + SEX_SUBGROUPS + RACE_SUBGROUPS

    for model in MODELS:
        for subgroup in subgroups:
            fold_map = cache[(model, subgroup)]
            valid = {f: v for f, v in fold_map.items() if v is not None}
            skipped = sorted(f for f, v in fold_map.items() if v is None)

            cindices = list(valid.values())
            mean_ci = np.mean(cindices) if cindices else float("nan")
            std_ci = np.std(cindices, ddof=1) if len(cindices) > 1 else float("nan")
            n_folds = len(cindices)
            skipped_str = ";".join(str(f) for f in skipped) if skipped else ""

            rows.append(
                {
                    "model": model,
                    "subgroup": subgroup,
                    "mean_cindex": round(mean_ci, 6),
                    "std_cindex": round(std_ci, 6),
                    "n_folds": n_folds,
                    "skipped_folds": skipped_str,
                }
            )

    return pd.DataFrame(rows)


def build_pairwise_table(
    cache: dict[tuple[str, str], dict[int, float | None]],
) -> pd.DataFrame:
    subgroups = ["Overall"] + SEX_SUBGROUPS + RACE_SUBGROUPS
    model_list = list(MODELS.keys())
    pairs = [
        (model_list[i], model_list[j])
        for i in range(len(model_list))
        for j in range(i + 1, len(model_list))
    ]

    raw_rows = []

    for subgroup in subgroups:
        for model_a, model_b in pairs:
            folds = sorted(paired_folds(model_a, model_b))

            # Collect paired fold-level deltas (only where both are valid)
            deltas = []
            paired_folds_used = []
            for fold in folds:
                ci_a = cache[(model_a, subgroup)].get(fold)
                ci_b = cache[(model_b, subgroup)].get(fold)
                if ci_a is not None and ci_b is not None:
                    deltas.append(ci_b - ci_a)
                    paired_folds_used.append(fold)

            if len(deltas) < 2:
                # Not enough paired observations to run test
                raw_rows.append(
                    {
                        "subgroup": subgroup,
                        "model_a": model_a,
                        "model_b": model_b,
                        "delta": float("nan"),
                        "p_value": float("nan"),
                        "p_value_corrected": float("nan"),
                        "significant": False,
                    }
                )
                continue

            delta_mean = float(np.mean(deltas))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_val = wilcoxon(deltas)

            raw_rows.append(
                {
                    "subgroup": subgroup,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta": round(delta_mean, 6),
                    "p_value": p_val,
                    "p_value_corrected": float("nan"),  # filled after correction
                    "significant": False,
                }
            )

    df = pd.DataFrame(raw_rows)

    # Holm-Bonferroni correction across all tests that have a valid p-value
    valid_mask = df["p_value"].notna()
    if valid_mask.sum() > 0:
        _, corrected, _, _ = multipletests(
            df.loc[valid_mask, "p_value"], method="holm"
        )
        df.loc[valid_mask, "p_value_corrected"] = corrected
        df.loc[valid_mask, "significant"] = corrected < 0.05

    df["p_value"] = df["p_value"].round(6)
    df["p_value_corrected"] = df["p_value_corrected"].round(6)

    return df

def main():
    print("Loading data...")
    dfs = load_model_data()

    print("Computing per-fold C-indices...")
    cache = compute_all_cindices(dfs)

    print("Building results table...")
    results_df = build_results_table(cache)
    results_path = OUT_DIR / "cindex_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    print("Building pairwise comparison table...")
    pairwise_df = build_pairwise_table(cache)
    pairwise_path = OUT_DIR / "cindex_pairwise.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"  Saved: {pairwise_path}")

    print("\nResults preview:")
    print(results_df.to_string(index=False))
    print("\nPairwise preview (significant rows):")
    sig = pairwise_df[pairwise_df["significant"]]
    print(sig.to_string(index=False) if not sig.empty else "  (none significant)")


if __name__ == "__main__":
    main()
