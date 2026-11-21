# Numerical algorithms

Implementation of various numerical algorithms and some neural nets in `rust`, `c++` (using `cuda`, `cuda-oxide`, and `burn` for `rust`). 

* The `cuda-rust/*` directories contain select examples, some from [Oxford CUDA course](https://people.maths.ox.ac.uk/gilesm/cuda/), implemented with NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide). 

* The `cuda-cpp/*` directories contain select examples from [Oxford CUDA course](https://people.maths.ox.ac.uk/gilesm/cuda/) in CUDA: 

    - [cuda-cpp/monte-carlo](numerical_algorithms/cuda-cpp/monte-carlo) is an an example of using `cuRAND`.

    - [cuda-cpp/tensor-cublas](numerical_algorithms/cuda-cpp/tensor-cublas) shows using tensor cores and `cuBLAS`.

* The `ml/two_class_nn` contains implementation of a basic neural net classifier for a two class problem in `rust` and `cpp`; these are meant as getting-started reference examples. 

* The `recipes` directory contains various numerical recipes, some from [Numerical recipes in C](https://numerical.recipes/book.html).    
