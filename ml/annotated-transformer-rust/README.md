# The Annotated Transformer in Rust

A Rust implementation of the encoder–decoder model from [Harvard NLP’s The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). The model is built from tensor operations and primitive layers in [tch-rs](https://docs.rs/crate/tch/0.27.0). Rust runs the model, autograd, training loop, and decoding. The Python wheel in the setup supplies LibTorch.

With `pytorch` version `2.27` in python environment, build with  

```bash
unset LIBTORCH_BYPASS_VERSION_CHECK
export LIBTORCH_USE_PYTORCH=1
torch_lib="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo run --locked -- copy
```

For NVIDIA GPU, use CUDA-enabled **PyTorch 2.15.0** wheel in place of the CPU wheel and set `CUDA_VISIBLE_DEVICES=""` to force CPU execution with the CUDA wheel.
