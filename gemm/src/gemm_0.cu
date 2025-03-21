#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    // 最重要的问题是，x 是内部循环
    size_t const row{blockIdx.x * blockDim.x + threadIdx.x}; // H
    size_t const col{blockIdx.y * blockDim.y + threadIdx.y}; // W

    if (row < m && col < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        // 写回的时候也是 H x W 的形式
        C[row * n + col] = alpha * sum + beta * C[row * n + col]; // 这里的row是x方向，col是y方向。但是写回的时候，row是y方向，col是x方向。相当于，写回的时候，结果转置了。
    }
}

template <typename T>
void cuda_gemm_v0(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{ // 对应结果的维度
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x, // H
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U}; // W
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template GEMM_EXPORT void cuda_gemm_v0<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_v0<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);
