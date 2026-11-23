#pragma once
#include "cuda_support.hpp"
#include "options.hpp"
#include "workload.hpp"

/// Run one deliberately small block, with optional whole-block start/finish messages.
__global__ void worker_kernel(float* out, std::size_t count, int rounds, int job, bool trace) {
    if (trace && threadIdx.x == 0) printf("job %d starts\n", job);
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x)
        out[i] = process_value(input_value(i, job), rounds);
    __syncthreads();
    if (trace && threadIdx.x == 0) printf("job %d finishes\n", job);
}

/// Put a tiny operation on the implicit default stream to expose its ordering.
__global__ void marker_kernel(float* value) {
    if (threadIdx.x == 0) *value += 1.0f;
}

/// Process one uploaded chunk with a conventional one-thread-per-element grid.
__global__ void chunk_kernel(const float* in, float* out, std::size_t count, int rounds) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = process_value(in[i], rounds);
}

/// Update a flattened cubic grid while preserving its boundary.
__global__ void laplace_kernel(const float* old, float* next, int side) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(side) * side * side;
    if (i < count) next[i] = jacobi_value(old, i, side);
}

/// Load the worker kernel before the timed section.
inline void warm_worker() {
    worker_kernel<<<1, 64>>>(nullptr, 0, 1, 0, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/// Enqueue one small worker block and immediately check its launch status.
inline void launch_worker(float* out, const Options& opt, int job, cudaStream_t stream = nullptr) {
    worker_kernel<<<1, 64, 0, stream>>>(out, opt.elements, opt.rounds, job, opt.trace);
    CUDA_CHECK(cudaGetLastError());
}
