"""Task 14-15: regular NN vs DP-SGD NN (tensorflow privacy) on v2.

Same network/optimizer/epochs for both, the DP one just adds clipping + noise.
epsilon comes from compute_dp_sgd_privacy.

run from dvc/ or gitlfs/:
    python ../src/dp_train.py --name v2
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
    compute_dp_sgd_privacy_statement,
)
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasSGDOptimizer,
)

from common import build_preprocessor, load_split, regression_metrics, save_json

BATCH_SIZE = 250
EPOCHS = 15
LEARNING_RATE = 0.05
L2_NORM_CLIP = 1.0
DELTA = 1e-5  # < 1/n, n is ~24k


def make_model(n_inputs):
    """small MLP, used for both the DP and non-DP runs"""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_inputs,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def fit_and_score(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x_train, y_train_scaled, x_test, y_test, y_scaler, noise_multiplier=None):
    """train once (DP if noise_multiplier is set) and score on test"""
    tf.keras.utils.set_random_seed(42)
    model = make_model(x_train.shape[1])
    if noise_multiplier is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        loss = tf.keras.losses.MeanSquaredError()
    else:
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=L2_NORM_CLIP,
            noise_multiplier=noise_multiplier,
            num_microbatches=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )
        # DP optimizer needs the loss per example (no reduction) so it can clip each one
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE)
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x_train, y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = y_scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()
    return regression_metrics(y_test, y_pred)


def main():
    """main"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="dataset label, e.g. v2")
    parser.add_argument(
        "--noise", default="1.1", help="comma-separated noise multipliers, e.g. 0.5,1.1,2.0"
    )
    args = parser.parse_args()

    x_train_df, x_test_df, y_train, y_test = load_split()
    prep = build_preprocessor(x_train_df)
    x_train = prep.fit_transform(x_train_df).astype(np.float32)
    x_test = prep.transform(x_test_df).astype(np.float32)

    # microbatches have to be full batches, so drop the last partial batch
    n = (len(x_train) // BATCH_SIZE) * BATCH_SIZE
    x_train, y_train = x_train[:n], y_train.to_numpy()[:n]
    # scale y too, otherwise the loss is huge and clipping wrecks the gradients
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)

    results = {"dataset": args.name, "n_train": n, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
               "learning_rate": LEARNING_RATE, "l2_norm_clip": L2_NORM_CLIP, "delta": DELTA}

    print("Training non-DP model ...")
    results["non_dp"] = fit_and_score(x_train, y_train_scaled, x_test, y_test, y_scaler)
    print("  ", results["non_dp"])

    results["dp"] = []
    for noise in (float(v) for v in args.noise.split(",")):
        print(f"Training DP model (noise_multiplier={noise}) ...")
        metrics = fit_and_score(x_train, y_train_scaled, x_test, y_test, y_scaler, noise)
        epsilon, _ = compute_dp_sgd_privacy(
            n=n, batch_size=BATCH_SIZE, noise_multiplier=noise, epochs=EPOCHS, delta=DELTA
        )
        metrics.update(noise_multiplier=noise, epsilon=float(epsilon))
        results["dp"].append(metrics)
        print("  ", metrics)
        print(compute_dp_sgd_privacy_statement(
            number_of_examples=n, batch_size=BATCH_SIZE, num_epochs=EPOCHS,
            noise_multiplier=noise, delta=DELTA,
        ))

    save_json(results, f"metrics/{args.name}_dp.json")
    print(f"saved metrics/{args.name}_dp.json")


if __name__ == "__main__":
    main()
