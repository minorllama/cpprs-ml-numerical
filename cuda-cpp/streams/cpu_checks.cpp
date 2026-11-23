#include "options.hpp"
#include "workload.hpp"
#include <limits>

/// Fail with a descriptive message when a host regression check does not hold.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

/// Check input cycling, reference lookup, and rejection of corrupted outputs.
void check_chunks() {
    constexpr int rounds = 7;
    const auto table = reference_table(rounds);
    for (int chunk : {0, 1, 2, 1023, 1024, 1025}) {
        std::vector<float> values(1031);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = process_value(input_value(i, chunk), rounds);
        verify_chunk(values.data(), values.size(), chunk, table);
        values[17] = std::numeric_limits<float>::quiet_NaN();
        bool failed = false;
        try { verify_chunk(values.data(), values.size(), chunk, table); }
        catch (const std::runtime_error&) { failed = true; }
        require(failed, "non-finite output was accepted");
    }
}

/// Check a solvable one-interior-node grid and the discrete maximum principle.
void check_laplace() {
    auto one = laplace_reference(3, 1, 2);
    for (float value : one) require(value == 2, "3-cube one-step solution failed");
    auto field = laplace_reference(7, 8, 3);
    for (float value : field) require(value >= 0 && value <= 3, "maximum principle failed");
    const auto initial = initial_grid(7, 3);
    for (std::size_t i = 0; i < initial.size(); ++i)
        if (initial[i] == 3) require(field[i] == 3, "Dirichlet boundary changed");
}

/// Exercise valid and invalid argument parsing without requiring a CUDA installation.
void check_options() {
    char program[] = "cpu_checks", side[] = "--side", value[] = "3";
    char* args[] = {program, side, value};
    require(parse_args(3, args).side == 3, "valid side was rejected");
    for (const std::string text : {"0", "-1", "1x", "", "257"}) {
        bool failed = false;
        try { positive(text, 256); }
        catch (const std::exception&) { failed = true; }
        require(failed, "invalid integer was accepted");
    }
}

/// Run the host checks and return a nonzero status for a failure.
int main() {
    try {
        require(process_value(0.0f, 1) == 0.001f, "one-step FMA check failed");
        check_chunks();
        check_laplace();
        check_options();
        std::cout << "PASS: workload, reference table, verification, Jacobi, CLI\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
