# Numerical algorithms

Implementation of various numerical algorithms and some neural nets in `rust`, `c++` (using `cuda`, `cuda-oxide`, and `burn` for `rust`). 

* The `cuda-rust/*` directories contain select examples, some from [Oxford CUDA course](https://people.maths.ox.ac.uk/gilesm/cuda/), implemented using NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide). 

* The `cuda-cpp/*` directories contain some examples from [Oxford CUDA course](https://people.maths.ox.ac.uk/gilesm/cuda/) in CUDA: 

    - [cuda-cpp/monte-carlo](https://github.com/minorllama/numerical_algorithms/tree/main/cuda-cpp/monte-carlo) is an an example of using `cuRAND`.

    - [cuda-cpp/tensor-cublas](https://github.com/minorllama/numerical_algorithms/tree/main/cuda-cpp/tensor-cublas) shows using tensor cores and `cuBLAS`.

* The path `ml/two_class_nn` contains implementation of a basic neural net classifier for a two class problem in `rust` and `cpp`, meant as getting-started reference examples for machine learning in C++ and rust. 

* The `recipes` directory contains various numerical recipes, some from [Numerical recipes in C](https://numerical.recipes/book.html).    
