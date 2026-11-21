#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string command = "all";
    int n = 256;
    int reps = 3;
    bool tf32 = true;
};

/// Parse a positive decimal integer within the teaching example's allocation limit.
inline int positive(const std::string& text, int limit) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected a positive decimal integer");
    auto value = std::stoull(text);
    if (value == 0 || value > static_cast<unsigned>(limit))
        throw std::invalid_argument("numeric argument is out of range");
    return static_cast<int>(value);
}

/// Parse a mode, matrix side, repetition count, and optional TF32 suppression.
inline Options parse_args(int argc, char** argv) {
    Options opt;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-tf32") opt.tf32 = false;
        else args.emplace_back(argv[i]);
    }
    if (args.size() > 3) throw std::invalid_argument("too many arguments");
    if (!args.empty()) opt.command = args[0];
    if (opt.command != "library" && opt.command != "wmma" &&
        opt.command != "all" && opt.command != "sweep")
        throw std::invalid_argument("mode must be library, wmma, all, or sweep");
    if (args.size() > 1) opt.n = positive(args[1], 8192);
    if (args.size() > 2) opt.reps = positive(args[2], 10000);
    if (opt.n % 16) throw std::invalid_argument("matrix side must be a multiple of 16");
    return opt;
}

/// Generate identical column-major input data in C++ and Rust.
inline std::vector<float> make_input(int n, unsigned salt) {
    std::vector<float> data(std::size_t(n) * n);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>((i * 17 + salt * 13) % 1000) / 1000.0f;
    return data;
}

/// Return every output index for small matrices and distributed samples for large ones.
inline std::vector<std::size_t> check_indices(int n) {
    const std::size_t count = std::size_t(n) * n;
    const std::size_t checks = n <= 128 ? count : 64;
    std::vector<std::size_t> indices;
    for (std::size_t j = 0; j < checks; ++j)
        indices.push_back(j * (count - 1) / (checks - 1));
    return indices;
}

/// Evaluate selected column-major GEMM outputs using double-precision sums.
inline std::vector<double> reference(const std::vector<float>& a,
    const std::vector<float>& b, const std::vector<float>& c, int n, float alpha, float beta) {
    std::vector<double> result;
    for (std::size_t index : check_indices(n)) {
        const std::size_t row = index % n, col = index / n;
        double sum = 0.0;
        for (int k = 0; k < n; ++k) sum += double(a[row + k * n]) * b[k + col * n];
        result.push_back(alpha * sum + beta * c[index]);
    }
    return result;
}

/// Reject non-finite outputs and check maximum error relative to the reference scale.
inline double verify(const std::vector<float>& values, const std::vector<double>& expected, int n) {
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite GPU output");
    const auto indices = check_indices(n);
    double error = 0.0, scale = 0.0;
    for (std::size_t j = 0; j < indices.size(); ++j) {
        error = std::max(error, std::abs(double(values[indices[j]]) - expected[j]));
        scale = std::max(scale, std::abs(expected[j]));
    }
    const double relative = error / std::max(scale, 1e-30);
    if (relative > 1e-3) throw std::runtime_error("GEMM verification failed");
    return relative;
}

/// Report median kernel time, arithmetic throughput, verification error, and coverage.
inline void report(int n, const std::string& mode, std::vector<float> times, double error) {
    std::sort(times.begin(), times.end());
    const double ms = times[times.size() / 2];
    const double gflops = 2.0 * n * n * n / (ms * 1e6);
    std::cout << n << ',' << mode << ',' << ms << ',' << gflops << ','
              << error << ',' << check_indices(n).size() << '\n';
}
