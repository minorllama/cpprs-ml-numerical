"""Compare the C++ executable with Python on a batch of five readings."""
import argparse
import subprocess

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="saved_model")
    parser.add_argument("--binary", default="./predict")
    args = parser.parse_args()

    values = [-10.0, 2.5, 20.0, 32.5, 50.0]
    signature = tf.saved_model.load(args.model).signatures["serving_default"]
    expected = signature(raw_c=tf.constant([[v] for v in values], dtype=tf.float32))
    expected = expected["calibrated_c"].numpy().ravel()
    stdout = subprocess.check_output(
        [args.binary, args.model, *map(str, values)], text=True
    )
    actual = np.asarray([float(line) for line in stdout.splitlines()])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
    print(f"Python and C++ agree on all {len(values)} readings within 0.0001 °C.")


if __name__ == "__main__":
    main()
