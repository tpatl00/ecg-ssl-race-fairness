"""
Compile fold-level eval prediction CSVs into one analysis-ready results CSV per model.

For each model, loads all available fold eval_preds CSVs, averages log_risk per patient,
joins demographic metadata, consolidates PatientRace, and writes to results/training_csvs/.
"""

from pathlib import Path

import pandas as pd

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDS_ROOT = REPO_ROOT / "training_outputs" / "ecg-fm-risk"
METADATA_PATH = (
    REPO_ROOT
    / "harvard-emory-dataset"
    / "hf_risk_i0001_and_i0006_prevalent_mi_hyp_additional_factors.csv"
)
OUT_DIR = REPO_ROOT / "results" / "training_csvs"

ALL_FOLDS = list(range(1, 11))

MODELS = {
    "resnet_baseline": {
        "prefix": "resnet_baseline_fold_",
        "folds": ALL_FOLDS,
    },
    "ecg_fm": {
        "prefix": "ecg_fm_fold_",
        "folds": ALL_FOLDS,
    },
    "ecg_fm_e2e": {
        "prefix": "ecg_fm_e2e_fold_",
        "folds": ALL_FOLDS,
    },
}

# Race consolidation
def consolidate_race(race_str):
    if pd.isna(race_str):
        return "Unknown"
    r = str(race_str).upper()
    if r in ("WHITE", "CAUCASIAN"):
        return "White"
    elif "BLACK" in r or "AFRICAN" in r:
        return "Black"
    elif r == "ASIAN":
        return "Asian"
    else:
        return "Other"


# Load metadata
def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(
        METADATA_PATH,
        usecols=["BDSPPatientID", "Sex", "PatientRace", "EventStatus", "FollowUpTimeMonths"],
    )
    # BDSPPatientID is stored as float (e.g. 111189075.0); convert to int string to match prediction patient_id
    meta["BDSPPatientID"] = meta["BDSPPatientID"].astype("Int64").astype(str)
    return meta


# Fold-level row count validation
def validate_fold_lengths() -> None:
    print("\nFold-level row count validation:")

    # Only validate folds that exist in all three models (e2e is the limiting model)
    shared_folds = MODELS["ecg_fm_e2e"]["folds"]

    rows = []
    any_mismatch = False

    for fold in shared_folds:
        counts = {}
        for model_name, cfg in MODELS.items():
            fold_dir = PREDS_ROOT / f"{cfg['prefix']}{fold}"
            csv_path = fold_dir / "eval_results" / f"eval_preds_fold_{fold}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                counts[model_name] = len(df)
            else:
                counts[model_name] = None

        present = [v for v in counts.values() if v is not None]
        mismatch = len(set(present)) > 1
        if mismatch:
            any_mismatch = True

        rows.append(
            {
                "fold": fold,
                **counts,
                "status": "MISMATCH" if mismatch else "ok",
            }
        )

    val_df = pd.DataFrame(rows).set_index("fold")
    print(val_df.to_string())

    if any_mismatch:
        print("\nWARNING: Row count mismatches detected across models for one or more folds.")
        print("Proceeding with compilation, but review the flagged folds above.")
    else:
        print("\nAll shared folds have matching row counts across models.")


# Compile one model
def compile_model(model_name: str, cfg: dict, meta: pd.DataFrame) -> pd.DataFrame:
    prefix = cfg["prefix"]
    folds = cfg["folds"]

    fold_dfs = []
    loaded_folds = []

    for fold in folds:
        fold_dir = PREDS_ROOT / f"{prefix}{fold}"
        csv_path = fold_dir / "eval_results" / f"eval_preds_fold_{fold}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found — skipping fold {fold}")
            continue
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        df["patient_id"] = df["patient_id"].str.strip()
        fold_dfs.append(df)
        loaded_folds.append(fold)

    if not fold_dfs:
        raise RuntimeError(f"No eval CSVs found for model '{model_name}'")

    print(f"  Loaded folds: {loaded_folds}")

    for fold, df in zip(loaded_folds, fold_dfs):
        df["_fold"] = fold
    all_preds = pd.concat(fold_dfs, ignore_index=True)

    # Each patient appears in exactly one eval fold; keep log_risk and fold_num directly
    compiled = all_preds.rename(columns={"_fold": "fold_num"})[
        ["patient_id", "log_risk", "fold_num", "event", "duration"]
    ].copy()

    # Join metadata
    compiled = compiled.merge(
        meta,
        left_on="patient_id",
        right_on="BDSPPatientID",
        how="left",
    ).drop(columns=["BDSPPatientID"])

    # Consolidate race
    compiled["PatientRace"] = compiled["PatientRace"].apply(consolidate_race)

    # Final column order
    compiled = compiled[
        ["patient_id", "log_risk", "fold_num", "event", "duration", "Sex", "PatientRace", "EventStatus", "FollowUpTimeMonths"]
    ]

    return compiled


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fold-length validation (shared folds across all models)")
    print("=" * 60)
    validate_fold_lengths()

    meta = load_metadata()
    print(f"\nMetadata loaded: {len(meta):,} rows")

    print("\n" + "=" * 60)
    print("Compiling models")
    print("=" * 60)

    for model_name, cfg in MODELS.items():
        print(f"\n[{model_name}]")
        compiled = compile_model(model_name, cfg, meta)

        out_path = OUT_DIR / f"{model_name}_compiled.csv"
        compiled.to_csv(out_path, index=False)

        race_dist = compiled["PatientRace"].value_counts().to_dict()
        meta_missing = compiled["Sex"].isna().sum()

        print(f"  Patients in output : {len(compiled):,}")
        print(f"  Metadata join misses: {meta_missing}")
        print(f"  PatientRace dist   : {race_dist}")
        print(f"  Written to         : {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
