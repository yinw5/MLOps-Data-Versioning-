"""Train a baseline linear regression on the dataset version currently in data/.

This file never changes between v1 and v2 -- only the data checked out by the
versioning tool changes.

Usage:
    python ../src/train.py --name v1
"""

import argparse

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from common import build_preprocessor, load_split, regression_metrics, save_json


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="label for the metrics file, e.g. v1")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_split()
    model = Pipeline(
        [("prep", build_preprocessor(x_train)), ("model", LinearRegression())]
    )
    model.fit(x_train, y_train)
    metrics = regression_metrics(y_test, model.predict(x_test))
    metrics.update(model="LinearRegression", dataset=args.name, n_train=len(x_train),
                   n_features=x_train.shape[1])

    save_json(metrics, f"metrics/{args.name}.json")
    for key, value in metrics.items():
        print(f"{key:>10}: {value}")


if __name__ == "__main__":
    main()
