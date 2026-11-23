NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo
HEADERS = cuda_support.hpp options.hpp workload.hpp kernels.cuh worker_demo.hpp pipeline.hpp laplace_demo.hpp
DEMOS = stream multithread pipeline laplace
BINARIES = $(addsuffix _legacy,$(DEMOS)) $(addsuffix _per_thread,$(DEMOS))

.PHONY: all clean check-cpu
all: $(BINARIES)

%_legacy: main.cu $(HEADERS) Makefile
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) --default-stream legacy -Xcompiler=-fopenmp 	  -DDEMO_NAME='"$*"' -DSTREAM_MODE='"legacy"' main.cu -o $@

%_per_thread: main.cu $(HEADERS) Makefile
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) --default-stream per-thread -Xcompiler=-fopenmp 	  -DDEMO_NAME='"$*"' -DSTREAM_MODE='"per-thread"' main.cu -o $@

cpu_checks: cpu_checks.cpp options.hpp workload.hpp
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Wpedantic cpu_checks.cpp -o $@

check-cpu: cpu_checks
	./cpu_checks

clean:
	rm -f $(BINARIES) cpu_checks
