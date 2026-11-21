#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/// Convert a CUDA Runtime status to a descriptive exception.
inline void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

/// Convert a cuBLAS status to an exception.
inline void blas_check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(code)));
}

/// Own one device array and transfer its contents synchronously.
template<class T> struct Buffer {
    T* data = nullptr;
    std::size_t count;
    /// Allocate storage for count elements.
    explicit Buffer(std::size_t size) : count(size) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    /// Release the allocation.
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    /// Upload exactly count values.
    void upload(const std::vector<T>& values) {
        if (values.size() != count) throw std::invalid_argument("upload length mismatch");
        cuda_check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    /// Download the array after preceding default-stream work completes.
    std::vector<T> download() const {
        std::vector<T> values(count);
        cuda_check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

/// Own one CUDA event with timing enabled.
struct Event {
    cudaEvent_t value = nullptr;
    /// Create a timed event.
    Event() { cuda_check(cudaEventCreate(&value)); }
    /// Release the event.
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

/// Own a cuBLAS handle using the default CUDA stream.
struct Blas {
    cublasHandle_t value = nullptr;
    /// Create the handle after the active device has been selected.
    Blas() { blas_check(cublasCreate(&value)); }
    /// Release the library handle.
    ~Blas() { if (value) cublasDestroy(value); }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
};

/// Upload both precisions and retain reference data for the run.
struct Matrices {
    std::vector<float> a, b, c, qa, qb;
    Buffer<float> da, db, seed, out;
    Buffer<half> ha, hb;
    /// Allocate all arrays and upload deterministic inputs.
    explicit Matrices(int n) : a(make_input(n, 1)), b(make_input(n, 2)), c(make_input(n, 3)),
        qa(a.size()), qb(b.size()), da(a.size()), db(b.size()), seed(c.size()),
        out(c.size()), ha(a.size()), hb(b.size()) {
        std::vector<half> ah(a.size()), bh(b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            ah[i] = __float2half_rn(a[i]); bh[i] = __float2half_rn(b[i]);
            qa[i] = __half2float(ah[i]); qb[i] = __half2float(bh[i]);
        }
        da.upload(a); db.upload(b); seed.upload(c); ha.upload(ah); hb.upload(bh);
    }
    /// Restore the initial C matrix before a warm-up or measured GEMM.
    void reset() { cuda_check(cudaMemcpy(out.data, seed.data, c.size() * sizeof(float), cudaMemcpyDeviceToDevice)); }
};

/// Time repeated operations with reset work and one warm-up outside the events.
template<class F> std::vector<float> measure(Matrices& data, int reps, F enqueue) {
    data.reset();
    enqueue();
    cuda_check(cudaDeviceSynchronize());
    Event start, stop;
    std::vector<float> times;
    for (int rep = 0; rep < reps; ++rep) {
        data.reset();
        cuda_check(cudaEventRecord(start.value));
        enqueue();
        cuda_check(cudaEventRecord(stop.value));
        cuda_check(cudaEventSynchronize(stop.value));
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start.value, stop.value));
        times.push_back(ms);
    }
    return times;
}
