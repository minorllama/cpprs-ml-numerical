#include "pipeline.hpp"
#include "laplace_demo.hpp"
#include <iomanip>

#ifndef DEMO_NAME
#define DEMO_NAME "stream"
#endif
#ifndef STREAM_MODE
#define STREAM_MODE "legacy"
#endif

/// Initialize GPU zero, print its concurrency capabilities, and dispatch the experiment.
void run(const Options& opt) {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "demo=" << DEMO_NAME << "; default_stream=" << STREAM_MODE
              << "; gpu=" << props.name << "; concurrentKernels=" << props.concurrentKernels
              << "; asyncEngineCount=" << props.asyncEngineCount << '\n';
    std::cout << std::fixed << std::setprecision(3);
    const std::string demo = DEMO_NAME;
    if (demo == "stream") run_streams(opt);
    else if (demo == "multithread") run_multithread(opt);
    else if (demo == "pipeline") run_pipeline(opt);
    else if (demo == "laplace") run_laplace(opt);
    else throw std::invalid_argument("unknown compiled experiment");
}

/// Parse arguments, execute the experiment, and return failure for run or cleanup errors.
int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help") {
            print_help(DEMO_NAME, STREAM_MODE);
            return 0;
        }
        run(parse_args(argc, argv));
        return cleanup_failed.load() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
