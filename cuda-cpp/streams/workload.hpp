#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/// Generate a bounded input value determined only by element index and job/chunk.
HD inline float input_value(std::size_t i, int chunk) {
    const auto position = (i + 17 * static_cast<std::size_t>(chunk % 1024)) % 1024;
    return static_cast<float>(position) / 1024.0f + static_cast<float>(chunk % 7) / 16.0f;
}

/// Perform dependent fused multiply-add operations without overflowing.
HD inline float process_value(float value, int rounds) {
    for (int i = 0; i < rounds; ++i) value = fmaf(0.999f, value, 0.001f);
    return value;
}

/// Precompute reference values for every input pattern outside the timed region.
inline std::vector<float> reference_table(int rounds) {
    std::vector<float> table(7 * 1024);
    for (int seed = 0; seed < 7; ++seed)
        for (int i = 0; i < 1024; ++i)
            table[seed * 1024 + i] = process_value(i / 1024.0f + seed / 16.0f, rounds);
    return table;
}

/// Verify every result from a worker or a downloaded pipeline chunk.
inline void verify_chunk(const float* data, std::size_t count, int chunk, const std::vector<float>& table) {
    for (std::size_t i = 0; i < count; ++i) {
        const auto position = (i + 17 * static_cast<std::size_t>(chunk % 1024)) % 1024;
        const float expected = table[(chunk % 7) * 1024 + position];
        if (!std::isfinite(data[i]) || std::abs(data[i] - expected) > 1e-6f * (1 + std::abs(expected)))
            throw std::runtime_error("result mismatch in chunk/job " + std::to_string(chunk));
    }
}

/// Apply one six-neighbor Jacobi update, preserving all Dirichlet boundary nodes.
HD inline float jacobi_value(const float* old, std::size_t i, int side) {
    const std::size_t plane = std::size_t(side) * side;
    const int x = i % side, y = (i / side) % side, z = i / plane;
    if (x == 0 || y == 0 || z == 0 || x == side - 1 || y == side - 1 || z == side - 1)
        return old[i];
    const float sum = old[i - 1] + old[i + 1] + old[i - side] + old[i + side]
                    + old[i - plane] + old[i + plane];
    return sum * (1.0f / 6.0f);
}

/// Construct a cubic domain with a constant boundary and zero interior.
inline std::vector<float> initial_grid(int side, float boundary) {
    std::vector<float> data(std::size_t(side) * side * side, boundary);
    for (int z = 1; z < side - 1; ++z)
        for (int y = 1; y < side - 1; ++y)
            for (int x = 1; x < side - 1; ++x) data[x + side * (y + side * z)] = 0;
    return data;
}

/// Evolve a grid on the CPU to validate the complete final GPU field.
inline std::vector<float> laplace_reference(int side, int steps, float boundary) {
    auto old = initial_grid(side, boundary), next = old;
    for (int step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < old.size(); ++i) next[i] = jacobi_value(old.data(), i, side);
        old.swap(next);
    }
    return old;
}
