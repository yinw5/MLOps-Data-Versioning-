"""Prints v1 vs v2 and non-DP vs DP tables from whatever is in metrics/."""

import json
from pathlib import Path

import pandas as pd

METRICS = Path("metrics")


def load(name):
    """load metrics/<name>.json, None if it's not there yet"""
    path = METRICS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    """main"""
    cols = ["rmse", "mae", "median_ae", "r2", "n_test"]
    rows = {name: load(name) for name in ["v1", "v2"]}
    table = pd.DataFrame({k: v for k, v in rows.items() if v}).T
    if not table.empty:
        print("Baseline LinearRegression, v1 vs v2:")
        print(table[cols + ["n_train", "n_features"]].to_string(), "\n")

    dp = load("v2_dp")
    if dp:
        dp_rows = {"non-DP": dp["non_dp"]}
        for run in dp["dp"]:
            dp_rows[f"DP (noise={run['noise_multiplier']})"] = run
        dp_table = pd.DataFrame(dp_rows).T
        extra = [c for c in ["noise_multiplier", "epsilon"] if c in dp_table]
        print(f"Neural net on v2, non-DP vs DP (delta={dp['delta']}, epochs={dp['epochs']}, "
              f"batch={dp['batch_size']}, clip={dp['l2_norm_clip']}):")
        print(dp_table[cols[:-1] + extra].to_string())


if __name__ == "__main__":
    main()
