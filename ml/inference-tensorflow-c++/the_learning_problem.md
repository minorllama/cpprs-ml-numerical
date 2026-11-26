# The learning problem

We have the following toy problem data: a sensor reports a raw reading and a reference thermometer supplies a calibrated value. Collect paired readings and learn a function that predicts the reference value from the raw reading. Each example has a measured target and the output is a continuous number, so we have a **supervised regression** learning problem. 

This project generates the pairs with the following calibration curve:

```text
reference = raw + 0.8 + 0.004 * (raw - 20)^2 + measurement noise
```

where the raw readings range from −10 to 50 °C. At 20 °C the reference is about 20.8 °C. At 0 °C it is about 2.4 °C. The varying error makes an offset alone insufficient. The synthetic generator makes it easy to see what the model should learn and experiment without hardware.

| Data | Count | Used for |
| --- | ---: | --- |
| Training | 1,200 | Fit normalization, baselines, and network weights |
| Validation | 300 | Monitor validation loss and select the best network weights |
| Test | 300 | Compare models on readings withheld from fitting and selection |


Using `train.py`, we can generate some data with `make_data()`, then build the model,`build_model()`, and call fil, `model.fit()`. The model is then exported, see  `saved_model`, for inference. 

# Using TensorFlow C++ 

Get tensorflow with `curl fL https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-cpu-linux-x86_64.tar.gz -o libtensorflow.tar.gz` and build the inference module: 

```bash
TF_ROOT="$PWD/tensorflow"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic predict.cpp \
  -I"$TF_ROOT/include" -L"$TF_ROOT/lib" -ltensorflow -o predict
```

and the run inference, with tensorflow library on `LD_LIBRARY_PATH`: 
```bash
LD_LIBRARY_PATH="$TF_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./predict saved_model 0 20 42
```

Finally verify with:
```bash
$ LD_LIBRARY_PATH="$TF_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="" python verify.py
```


# `predict.cpp` walk through

To do: walk through the tensorflow boiler plate in `predict.cpp`