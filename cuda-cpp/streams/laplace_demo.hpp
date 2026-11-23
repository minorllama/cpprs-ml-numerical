#pragma once
#include "worker_demo.hpp"
#include <utility>

/// Own an independent Jacobi problem and a stream ordering all of its timesteps.
struct Grid {
    DeviceArray first, second;
    Stream stream;
    float* current;
    float* next;
    /// Allocate a cubic domain and upload its fixed-boundary initial state.
    Grid(int side, float boundary) : first(std::size_t(side) * side * side),
        second(first.count), current(first.data), next(second.data) {
        const auto host = initial_grid(side, boundary);
        CUDA_CHECK(cudaMemcpy(first.data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    /// Enqueue one update and swap the logical old/new buffer roles.
    void step(int side, bool serial) {
        const unsigned blocks = static_cast<unsigned>((first.count + 255) / 256);
        laplace_kernel<<<blocks, 256, 0, serial ? nullptr : stream.value>>>(current, next, side);
        CUDA_CHECK(cudaGetLastError());
        std::swap(current, next);
    }
};

/// Verify every node after the final GPU timestep.
inline void verify_grid(const Grid& grid, int side, int steps, float boundary) {
    const auto expected = laplace_reference(side, steps, boundary);
    std::vector<float> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), grid.current, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-5f * (1 + std::abs(expected[i])))
            throw std::runtime_error("Laplace verification failed at node " + std::to_string(i));
}

/// Launch independent small Laplace problems concurrently or on one shared stream.
inline void run_laplace(const Options& opt) {
    std::vector<std::unique_ptr<Grid>> grids;
    for (int job = 0; job < opt.jobs; ++job)
        grids.push_back(std::make_unique<Grid>(opt.side, float(job + 1)));
    Drain drain;
    laplace_kernel<<<1, 256>>>(nullptr, nullptr, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    for (int step = 0; step < opt.steps; ++step)
        for (auto& grid : grids) grid->step(opt.side, opt.serial);
    finish_timing(start);
    for (int job = 0; job < opt.jobs; ++job)
        verify_grid(*grids[job], opt.side, opt.steps, float(job + 1));
    std::cout << "verified " << opt.jobs << " grids of side " << opt.side << '\n';
}
