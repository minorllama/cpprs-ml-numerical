#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

struct Options {
    std::size_t elements = 65536;
    int jobs = 8;
    int rounds = 128;
    int chunks = 12;
    int side = 50;
    int steps = 20;
    bool trace = false;
    bool serial = false;
};

/// Parse a strictly positive decimal value without accepting signs or trailing characters.
inline int positive(const std::string& text, int limit) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected a positive decimal integer");
    const auto value = std::stoull(text);
    if (value == 0 || value > static_cast<unsigned>(limit))
        throw std::invalid_argument("numeric option is out of range");
    return static_cast<int>(value);
}

/// Apply a validated numeric option to the configuration.
inline void set_option(Options& opt, const std::string& flag, const std::string& value) {
    if (flag == "--elements") opt.elements = positive(value, 16777216);
    else if (flag == "--jobs") opt.jobs = positive(value, 64);
    else if (flag == "--rounds") opt.rounds = positive(value, 100000);
    else if (flag == "--chunks") opt.chunks = positive(value, 100000);
    else if (flag == "--side") opt.side = positive(value, 256);
    else if (flag == "--steps") opt.steps = positive(value, 100000);
    else throw std::invalid_argument("unknown option: " + flag);
}

/// Parse common options for any of the four experiments.
inline Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--trace") opt.trace = true;
        else if (flag == "--serial") opt.serial = true;
        else {
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
            set_option(opt, flag, argv[++i]);
        }
    }
    if (opt.side < 3) throw std::invalid_argument("side must be at least 3");
    return opt;
}

/// Print the purpose and tunable workload of the selected executable.
inline void print_help(const char* demo, const char* stream_mode) {
    std::cout << "Practical 11: " << demo << ", default stream = " << stream_mode << '\n'
              << "--elements N  floats per worker/chunk (default 65536)\n"
              << "--jobs N      streams, OpenMP tasks, or Laplace grids (default 8)\n"
              << "--rounds N    FMA iterations per element (default 128)\n"
              << "--chunks N    pipeline chunks (default 12)\n"
              << "--side N      cubic Laplace grid side (default 50)\n"
              << "--steps N     Jacobi timesteps (default 20)\n"
              << "--trace       print worker kernel starts and finishes\n"
              << "--serial      use one stream for Laplace or serialize pipeline chunks\n";
}
