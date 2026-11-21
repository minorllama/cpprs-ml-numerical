#include "kernels.cuh"

/// Enqueue C = alpha AB + beta C using an explicit cuBLAS compute mode.
void library_gemm(Blas& blas, Matrices& data, int n, const std::string& mode, float alpha, float beta) {
    const bool half_input = mode == "fp16";
    const auto type = half_input ? CUDA_R_16F : CUDA_R_32F;
    const void* a = half_input ? static_cast<void*>(data.ha.data) : data.da.data;
    const void* b = half_input ? static_cast<void*>(data.hb.data) : data.db.data;
    const auto compute = mode == "fp32" ? CUBLAS_COMPUTE_32F_PEDANTIC :
        mode == "tf32" ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    blas_check(cublasGemmEx(blas.value, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, a, type, n, b, type, n, &beta, data.out.data, CUDA_R_32F, n,
        compute, CUBLAS_GEMM_DEFAULT));
}

/// Benchmark a library precision and verify against its appropriate input precision.
void run_library(Blas& blas, Matrices& data, const Options& opt, const std::string& mode,
                 float alpha = 1.0f, float beta = 0.0f) {
    const auto& a = mode == "fp16" ? data.qa : data.a;
    const auto& b = mode == "fp16" ? data.qb : data.b;
    const auto expected = reference(a, b, data.c, opt.n, alpha, beta);
    const auto times = measure(data, opt.reps, [&] { library_gemm(blas, data, opt.n, mode, alpha, beta); });
    const double error = verify(data.out.download(), expected, opt.n);
    report(opt.n, beta == 0 ? mode : "fp16_affine", times, error);
}

/// Compare the direct Tensor Core kernel and cuBLAS with alpha = beta = 2.
void run_wmma(Blas& blas, Matrices& data, const Options& opt) {
    const auto expected = reference(data.qa, data.qb, data.c, opt.n, 2.0f, 2.0f);
    const auto times = measure(data, opt.reps, [&] { launch_tensor(data, opt.n); });
    report(opt.n, "wmma", times, verify(data.out.download(), expected, opt.n));
    run_library(blas, data, opt, "fp16", 2.0f, 2.0f);
}

/// Run the requested comparisons for one matrix dimension.
void run_size(Blas& blas, const Options& opt) {
    Matrices data(opt.n);
    if (opt.command != "wmma") {
        run_library(blas, data, opt, "fp32");
        if (opt.tf32) run_library(blas, data, opt, "tf32");
        run_library(blas, data, opt, "fp16");
    }
    if (opt.command == "all" || opt.command == "wmma") run_wmma(blas, data, opt);
}

/// Select GPU zero and carry out either one dimension or the six-size exercise.
void run(Options opt) {
    cuda_check(cudaSetDevice(0));
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0));
    if (props.major < 8) throw std::runtime_error("this matched project requires compute capability 8.0+");
    Blas blas;
    std::cout << "n,mode,ms,gflops,relative_error,checked\n";
    if (opt.command != "sweep") {
        run_size(blas, opt);
        return;
    }
    opt.command = "library";
    opt.tf32 = false;
    for (int n : {32, 64, 128, 256, 512, 1024}) {
        opt.n = n;
        run_size(blas, opt);
    }
}

/// Print usage or report an exception with a failing exit status.
int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help") {
            auto binary = "tensor_cublas_example";
            std::cout << "Usage: " << binary << " [all|library|wmma|sweep] [n=256] [reps=3] [--no-tf32]\n";
            return 0;
        }
        run(parse_args(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
