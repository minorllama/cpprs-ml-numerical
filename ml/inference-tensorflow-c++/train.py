"""Fit a calibration curve and export a TensorFlow SavedModel for C++."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def make_data():
    rng = np.random.default_rng(42)
    raw = rng.uniform(-10, 50, size=(1800, 1)).astype(np.float32)
    ideal = raw + 0.8 + 0.004 * (raw - 20) ** 2
    reference = ideal + rng.normal(0, 0.15, size=raw.shape).astype(np.float32)
    order = rng.permutation(len(raw))
    return [(raw[i], reference[i]) for i in np.split(order, [1200, 1500])]


def build_model(raw_train):
    normalize = tf.keras.layers.Normalization(axis=-1, name="normalize")
    normalize.adapt(raw_train)
    raw = tf.keras.Input(shape=(1,), dtype=tf.float32)
    hidden = tf.keras.layers.Dense(16, activation="tanh", name="hidden_1")(normalize(raw))
    hidden = tf.keras.layers.Dense(16, activation="tanh", name="hidden_2")(hidden)
    correction = tf.keras.layers.Dense(1, name="correction")(hidden)
    calibrated = tf.keras.layers.Add()([raw, correction])
    model = tf.keras.Model(raw, calibrated)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="mse",
        metrics=["mae"],
    )
    return model


class Calibrator(tf.Module):
    def __init__(self, model):
        super().__init__()
        normalize = model.get_layer("normalize")
        self.mean = tf.constant(normalize.mean.numpy())
        self.scale = tf.constant(
            np.maximum(np.sqrt(normalize.variance.numpy()), tf.keras.backend.epsilon())
        )
        self.parameters = tuple(
            tuple(tf.constant(value) for value in model.get_layer(name).get_weights())
            for name in ("hidden_1", "hidden_2", "correction")
        )

    @tf.function(input_signature=[tf.TensorSpec([None, 1], tf.float32, name="raw_c")])
    def predict(self, raw_c):
        hidden = (raw_c - self.mean) / self.scale
        for weights, bias in self.parameters[:-1]:
            hidden = tf.math.tanh(tf.matmul(hidden, weights) + bias)
        weights, bias = self.parameters[-1]
        return {"calibrated_c": raw_c + (tf.matmul(hidden, weights) + bias)}


def export_model(model, directory, raw_test):
    calibrator = Calibrator(model)
    np.testing.assert_allclose(
        calibrator.predict(raw_test)["calibrated_c"],
        model(raw_test, training=False),
        rtol=0,
        atol=1e-4,
    )
    tf.saved_model.save(
        calibrator, str(directory), signatures={"serving_default": calibrator.predict}
    )

    # Read the actual graph endpoints from the exported signature.
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        meta = tf.compat.v1.saved_model.load(session, ["serve"], str(directory))
    signature = meta.signature_def["serving_default"]
    endpoints = [signature.inputs["raw_c"].name, signature.outputs["calibrated_c"].name]
    (directory / "endpoints.txt").write_text("\n".join(endpoints) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("saved_model"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.exists():
        parser.error("--output already exists. Choose a fresh directory.")

    tf.keras.utils.set_random_seed(42)
    train, val, test = make_data()
    raw_train, ref_train = train
    raw_test, ref_test = test
    model = build_model(raw_train)
    model.fit(
        *train,
        validation_data=val,
        epochs=args.epochs,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )],
        verbose=2,
    )

    def mae(prediction):
        return float(np.mean(np.abs(prediction - ref_test)))

    print(f"Uncorrected test MAE: {mae(raw_test):.3f} °C")
    for degree, name in [(1, "Linear"), (2, "Quadratic")]:
        coefficients = np.polynomial.polynomial.polyfit(raw_train[:, 0], ref_train[:, 0], degree)
        prediction = np.polynomial.polynomial.polyval(raw_test, coefficients)
        print(f"{name} test MAE: {mae(prediction):.3f} °C")
    print(f"Neural network test MAE: {mae(model(raw_test, training=False).numpy()):.3f} °C")
    export_model(model, args.output, raw_test)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
