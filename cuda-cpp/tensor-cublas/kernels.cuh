#pragma once
#include "resources.hpp"
#include <mma.h>
namespace wm = nvcuda::wmma;

/// Compute one 16×16 output tile per warp using column-major WMMA loads.
__global__ void tensor_gemm(const half* a, const half* b, const float* seed, float* out, int n) {
    const int row = (blockIdx.x * 4 + threadIdx.x / 32) * 16;
    const int col = (blockIdx.y * 4 + threadIdx.y) * 16;
    if (row >= n || col >= n) return;
    wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::col_major> af;
    wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> bf;
    wm::fragment<wm::accumulator, 16, 16, 16, float> acc, cf;
    wm::fill_fragment(acc, 0.0f);
    for (int k = 0; k < n; k += 16) {
        wm::load_matrix_sync(af, a + row + k * n, n);
        wm::load_matrix_sync(bf, b + k + col * n, n);
        wm::mma_sync(acc, af, bf, acc);
    }
    wm::load_matrix_sync(cf, seed + row + col * n, n, wm::mem_col_major);
    for (int j = 0; j < cf.num_elements; ++j) cf.x[j] = 2.0f * acc.x[j] + 2.0f * cf.x[j];
    wm::store_matrix_sync(out + row + col * n, cf, n, wm::mem_col_major);
}

/// Launch the 4×4 warp arrangement used in the practical.
inline void launch_tensor(Matrices& data, int n) {
    tensor_gemm<<<dim3((n + 63) / 64, (n + 63) / 64), dim3(128, 4)>>>(
        data.ha.data, data.hb.data, data.seed.data, data.out.data, n);
    cuda_check(cudaGetLastError());
}
