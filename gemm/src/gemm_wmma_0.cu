#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

#define WMMA_M 16  // 16x16x16 矩阵乘法
#define WMMA_N 16
#define WMMA_K 16

/*
划分方式：将C矩阵分为16x16的小块，每个Block计算一个小块，每个block有32x8个线程，也就是8个warp
*/
// Kernel: Tensor Core GEMM using WMMA API
static __global__ void gemm_wmma_kernel(size_t M, size_t N, size_t K, 
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta) {

    // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，对于内部来说，x是内循环，y是外循环
    // Compute warp indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // Row index
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);             // Column index

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;  // Bounds check

    // WMMA matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);  // Initialize accumulator

    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) 
    {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(b_frag, B + i * N + warpN * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Load C and perform alpha * AB + beta * C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_orig_frag;
    wmma::load_matrix_sync(c_orig_frag, C + warpM * WMMA_M * N + warpN * WMMA_N, N, wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * c_frag.x[i] + beta * c_orig_frag.x[i];
    }

    // Store result back to C
    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
}

template <typename T>
void cuda_gemm_wmma_0(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

// GEMM API: 调用 Kernel 并管理 Stream
template <>
void cuda_gemm_wmma_0<__half>(size_t M, size_t N, size_t K,
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta, cudaStream_t stream) {
    dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M, 1);

    gemm_wmma_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_wmma_0<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
{
    // // Allocate device memory for half-precision matrices
    // half* d_A, * d_B;
    // cudaMalloc(&d_A, m * k * sizeof(half));
    // cudaMalloc(&d_B, k * n * sizeof(half));

    // // Convert A, B from float to half
    // dim3 blockDim(1024);
    // dim3 gridDim((m * k + blockDim.x - 1) / blockDim.x);
    // convert_float_to_half << <gridDim, blockDim, 0, stream >> > (A, d_A, m * k);
    // gridDim = dim3((k * n + blockDim.x - 1) / blockDim.x);
    // convert_float_to_half << <gridDim, blockDim, 0, stream >> > (B, d_B, k * n);

    // // Define grid and block dimensions
    // dim3 blockDimWMMA(16, 16);
    // dim3 gridDimWMMA((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);

    // // Launch WMMA kernel
    // wmma_gemm_kernel << <gridDimWMMA, blockDimWMMA, 0, stream >> > (m, n, k, d_A, d_B, C, alpha, beta);

    // // Cleanup
    // cudaFree(d_A);
    // cudaFree(d_B);
    CHECK_LAST_CUDA_ERROR();
}

template GEMM_EXPORT void cuda_gemm_wmma_0<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_wmma_0<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

