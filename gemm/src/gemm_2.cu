#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

/*
本方案的思路是：
创建一个 共享内存，用于存储A和B中切下来。

这里为了减少 共享内存的使用，还要将A和B切下来的数据进一步切分，这样可以减少共享内存的使用。

将A 切成 X_tile * K_tile 的块，将B切成 K_tile * Y_tile的块。总共大循环需要计算K/K_tile次。

一个线程组计算出 X * Y 的结果，这里的X和Y是块内的大小。总共需要计算 m/X * n/Y次计算。
*/

/*注意:这里的X_tile和blockDim.x在数值上是相等的，但是在for 循环中，X_tile会被编译器优化，而blockDim.x不会被优化。
原因是X_tile是一个常量，而blockDim.x是运行时常量，运行时在编译之后，无法被编译器优化。
*/
template <typename T, size_t X_tile, size_t Y_tile, size_t K_tile>
__global__ void gemm_v02_vectorized(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    size_t const col{blockIdx.x * blockDim.x + threadIdx.x}; // x x是内层循环，而y是外层循环
    size_t const row{blockIdx.y * blockDim.y + threadIdx.y}; // y

    __shared__ T A_tile[X_tile][K_tile];
    __shared__ T B_tile[K_tile][Y_tile];

    int loop_count = (k + K_tile - 1) / K_tile;
    
    T sum{static_cast<T>(0)};
    for (int i=0; i < loop_count; i++)
    {
        // load A and B into shared memory
        // load A
        for (size_t j = 0; j < K_tile; j += X_tile) // 当 X_tile 大于K_tile时，则有很多thread是浪费掉的。
        {
            if (row < m && i * K_tile + j + threadIdx.x < k && threadIdx.x < K_tile)
            {
                A_tile[threadIdx.y][j + threadIdx.x] = A[row * k + i * K_tile + j + threadIdx.x];
            }
            else
            {
                A_tile[threadIdx.y][j + threadIdx.x] ={static_cast<T>(0)};
            }
        }

        // load B
        for (size_t j = 0; j < K_tile; j += Y_tile)
        {
            if (col < n && i * K_tile + j + threadIdx.y < k && threadIdx.y < K_tile)
            {
                B_tile[j + threadIdx.y][threadIdx.x] = B[(i * K_tile + j + threadIdx.y) * n + col];
            }
            else
            {
                B_tile[j + threadIdx.y][threadIdx.x] = {static_cast<T>(0)};
            }
        }

        __syncthreads();

        // compute C
        if (row < m && col < n)
        {
            for (size_t i = 0; i < K_tile; ++i)
            {
                sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x]; // x 和 y是等价的
            }
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        C[row * n + col] = alpha * sum + beta * C[row * n + col]; // 这里是xy的不同点。
    }
}

template <typename T>
void cuda_gemm_v2(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    const size_t X_tile = 32;
    const size_t Y_tile = 32;
    const size_t K_tile = 16;

    dim3 const block_dim{X_tile, Y_tile, 1U};

    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v02_vectorized<T, X_tile, Y_tile, K_tile><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template GEMM_EXPORT void cuda_gemm_v2<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_v2<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);
