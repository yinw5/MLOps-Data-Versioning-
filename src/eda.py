"""EDA on whichever version is checked out in data/ right now.

    python ../src/eda.py --name v1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import LIFTS, TARGET, TEST_PATH, TRAIN_PATH


def main():
    """main"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="label for the output folder, e.g. v1")
    args = parser.parse_args()
    out = Path("reports") / f"eda_{args.name}"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    data = pd.concat([train, test])

    lines = [
        f"EDA for dataset {args.name}",
        (f"rows: {len(data):,} (train {len(train):,} / test {len(test):,}), "
         f"columns: {data.shape[1]}"),
        "",
        "Missing values (% of rows):",
        (data.isna().mean() * 100).round(1).sort_values(ascending=False).to_string(),
        "",
        "Numeric summary:",
        data.describe().T.round(2).to_string(),
    ]
    if "gender" in data:
        lines += ["", "Gender counts:", data["gender"].value_counts(dropna=False).to_string()]
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary)
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data[TARGET], bins=60, ax=axes[0])
    axes[0].set_title(f"{args.name}: total_lift distribution")
    sns.boxplot(data=data[LIFTS], ax=axes[1])
    axes[1].set_title(f"{args.name}: individual lifts")
    # log-ish scale or the v1 outliers squash everything
    axes[1].set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(out / "target_and_lifts.png", dpi=120)

    numeric = data.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{args.name}: correlation of numeric columns")
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=120)

    if "gender" in data:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=data, x="gender", y=TARGET, ax=ax)
        ax.set_yscale("symlog")
        ax.set_title(f"{args.name}: total_lift by gender")
        fig.tight_layout()
        fig.savefig(out / "total_lift_by_gender.png", dpi=120)

    print(f"\nsaved summary and plots to {out}/")


if __name__ == "__main__":
    main()
