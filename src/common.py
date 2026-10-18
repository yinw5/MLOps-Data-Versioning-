"""Stuff shared by all the scripts (columns, split settings, preprocessing, metrics)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "total_lift"
LIFTS = ["deadlift", "candj", "snatch", "backsq"]
# total_lift is just the sum of the lifts, so using them as features would be cheating.
# ids/names aren't useful either
EXCLUDED = [TARGET, *LIFTS, "athlete_id", "name"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_split(data_dir=DATA_DIR):
    """read data/train.csv + data/test.csv, return X_train, X_test, y_train, y_test"""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test.csv", low_memory=False)
    features = [c for c in train.columns if c not in EXCLUDED]
    return train[features], test[features], train[TARGET], test[TARGET]


def build_preprocessor(x):
    """numeric: fill median + scale, categorical: fill most common + one-hot.

    picks the columns from the data itself so v1 and v2 both work without changes
    """
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,  # team/affiliate have tons of values, cap it
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )


def regression_metrics(y_true, y_pred):
    """rmse, mae, median ae, r2"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": len(y_true),
    }


def save_json(obj, path):
    """dump to json (makes the folder if needed)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
