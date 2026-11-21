NVCC ?= nvcc
ARCH ?= sm_89
CXXFLAGS = -O3 -std=c++17 -lineinfo -arch=$(ARCH)
LDLIBS = -lcublas

tensor_cublas_example: main.cu common.hpp resources.hpp kernels.cuh
	$(NVCC) $(CXXFLAGS) main.cu $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -f tensor_cublas_example
