#pragma once
#include "kernels.cuh"
#include <chrono>
#include <exception>
#include <memory>
#include <omp.h>

using Clock = std::chrono::steady_clock;

/// Report elapsed wall time after all submitted GPU work completes.
inline void finish_timing(Clock::time_point start) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "elapsed_ms=" << ms << '\n';
}

/// Allocate all worker arrays before the concurrency experiment starts.
inline std::vector<std::unique_ptr<DeviceArray>> worker_arrays(const Options& opt) {
    std::vector<std::unique_ptr<DeviceArray>> arrays;
    for (int i = 0; i < opt.jobs; ++i) arrays.push_back(std::make_unique<DeviceArray>(opt.elements));
    return arrays;
}

/// Download and verify every worker output after timing.
inline void verify_workers(const std::vector<std::unique_ptr<DeviceArray>>& arrays, const Options& opt) {
    const auto table = reference_table(opt.rounds);
    std::vector<float> host(opt.elements);
    for (int job = 0; job < opt.jobs; ++job) {
        CUDA_CHECK(cudaMemcpy(host.data(), arrays[job]->data, opt.elements * sizeof(float), cudaMemcpyDeviceToHost));
        verify_chunk(host.data(), opt.elements, job, table);
    }
    std::cout << "verified " << opt.jobs << " complete worker arrays\n";
}

/// Interleave work on explicit streams with copies and markers on stream zero.
inline void run_streams(const Options& opt) {
    auto arrays = worker_arrays(opt);
    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < opt.jobs; ++i) streams.push_back(std::make_unique<Stream>());
    DeviceArray marker(1);
    PinnedArray scalar(1);
    scalar.data[0] = 1;
    CUDA_CHECK(cudaMemset(marker.data, 0, sizeof(float)));
    Drain drain;
    warm_worker();
    marker_kernel<<<1, 1>>>(marker.data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    for (int job = 0; job < opt.jobs; ++job) {
        launch_worker(arrays[job]->data, opt, job, streams[job]->value);
        CUDA_CHECK(cudaMemcpyAsync(marker.data, scalar.data, sizeof(float), cudaMemcpyHostToDevice));
        marker_kernel<<<1, 1>>>(marker.data);
        CUDA_CHECK(cudaGetLastError());
    }
    finish_timing(start);
    verify_workers(arrays, opt);
}

/// Launch work from OpenMP threads and carry their exceptions back to the main thread.
inline void parallel_launch(const Options& opt, const std::vector<std::unique_ptr<DeviceArray>>& arrays, bool warmup) {
    std::vector<std::exception_ptr> errors(opt.jobs);
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(opt.jobs) schedule(static, 1)
    for (int job = 0; job < opt.jobs; ++job) {
        try {
            CUDA_CHECK(cudaSetDevice(0));
            if (warmup) warm_worker();
            else {
                if (opt.trace) std::fprintf(stderr, "job %d uses host thread %d\n", job, omp_get_thread_num());
                launch_worker(arrays[job]->data, opt, job);
            }
        } catch (...) { errors[job] = std::current_exception(); }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

/// Compare legacy and per-thread default-stream launches from an OpenMP team.
inline void run_multithread(const Options& opt) {
    auto arrays = worker_arrays(opt);
    Drain drain;
    parallel_launch(opt, arrays, true);
    const auto start = Clock::now();
    parallel_launch(opt, arrays, false);
    finish_timing(start);
    verify_workers(arrays, opt);
}
