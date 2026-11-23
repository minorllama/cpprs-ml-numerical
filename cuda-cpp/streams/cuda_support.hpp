#pragma once
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

inline std::atomic<bool> cleanup_failed{false};

/// Throw an error identifying the failed CUDA expression and its source location.
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": "
            + expression + ": " + cudaGetErrorString(status));
}

/// Report cleanup failures while preserving a noexcept destructor contract.
inline void release_cuda(cudaError_t status, const char* expression, const char* file, int line) noexcept {
    if (status == cudaSuccess) return;
    cleanup_failed.store(true);
    std::fprintf(stderr, "%s:%d: cleanup %s: %s\n", file, line, expression, cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)
#define CUDA_RELEASE(expression) release_cuda((expression), #expression, __FILE__, __LINE__)

/// Own a float allocation in device memory.
struct DeviceArray {
    float* data = nullptr;
    std::size_t count;
    /// Allocate count floats.
    explicit DeviceArray(std::size_t n) : count(n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float)));
    }
    /// Release the allocation and report any failure.
    ~DeviceArray() { if (data) CUDA_RELEASE(cudaFree(data)); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

/// Own page-locked host memory for genuinely asynchronous transfers.
struct PinnedArray {
    float* data = nullptr;
    std::size_t count;
    /// Allocate count pinned floats.
    explicit PinnedArray(std::size_t n) : count(n) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), n * sizeof(float)));
    }
    /// Release the page-locked allocation and report any failure.
    ~PinnedArray() { if (data) CUDA_RELEASE(cudaFreeHost(data)); }
    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
};

/// Own a blocking stream so legacy default-stream synchronization stays observable.
struct Stream {
    cudaStream_t value = nullptr;
    /// Create a regular stream participating in legacy default-stream ordering.
    Stream() { CUDA_CHECK(cudaStreamCreate(&value)); }
    /// Destroy the stream and report any failure.
    ~Stream() { if (value) CUDA_RELEASE(cudaStreamDestroy(value)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

/// Own an event used to express data dependencies between streams.
struct Event {
    cudaEvent_t value = nullptr;
    /// Create an event without timing overhead.
    Event() { CUDA_CHECK(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    /// Destroy the event and report any failure.
    ~Event() { if (value) CUDA_RELEASE(cudaEventDestroy(value)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

/// Drain outstanding GPU work before buffers and streams unwind after an error.
struct Drain {
    /// Synchronize before resources declared earlier in the enclosing scope are destroyed.
    ~Drain() { CUDA_RELEASE(cudaDeviceSynchronize()); }
};
