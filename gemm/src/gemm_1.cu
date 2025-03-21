#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    size_t const col{blockIdx.x * blockDim.x + threadIdx.x}; // x x是内层循环，而y是外层循环
    size_t const row{blockIdx.y * blockDim.y + threadIdx.y}; // y

    if (row < m && col < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col]; // x 和 y是等价的
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col]; // 这里是xy的不同点，
    }
}

template <typename T>
void cuda_gemm_v1(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template GEMM_EXPORT void cuda_gemm_v1<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_v1<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);
