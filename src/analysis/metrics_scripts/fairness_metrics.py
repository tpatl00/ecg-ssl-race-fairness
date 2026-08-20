import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path(__file__).parents[3] / "results" / "metrics" / "cindex_results.csv"
OUTPUT_CSV = Path(__file__).parents[3] / "results" / "metrics" / "fairness_metrics.csv"

AXES = {
    "Sex": ["Male", "Female"],
    # Race axis restricted to White, Black, Asian only.
    # 'Other' and 'Unknown' are catch-all categories with ambiguous racial identity —
    # including them inflates PG/CV artificially and obscures meaningful between-group
    # disparity. Formal fairness metrics are only computed over groups with clinically
    # and demographically meaningful identities.
    "Race": ["White", "Black", "Asian"],
}


def compute_fairness(df: pd.DataFrame, model: str, axis: str, subgroups: list[str]) -> dict:
    rows = df[(df["model"] == model) & (df["subgroup"].isin(subgroups))].copy()
    rows = rows.set_index("subgroup").reindex(subgroups).dropna(subset=["mean_cindex"])

    values = rows["mean_cindex"]
    best = values.idxmax()
    worst = values.idxmin()

    return {
        "model": model,
        "axis": axis,
        "PG": values.max() - values.min(),
        "CV": np.std(values, ddof=0) / np.mean(values),
        "WGP": values.min(),
        "best_subgroup": best,
        "worst_subgroup": worst,
    }


def main():
    df = pd.read_csv(INPUT_CSV)
    records = []
    for model in df["model"].unique():
        for axis, subgroups in AXES.items():
            records.append(compute_fairness(df, model, axis, subgroups))

    out = pd.DataFrame(records, columns=["model", "axis", "PG", "CV", "WGP", "best_subgroup", "worst_subgroup"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(out.to_string(index=False))
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
