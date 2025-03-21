#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include <stdio.h>
#include <iostream>

/*
本方案的思路在3的基础上将：A_tile[X_tile][K_tile]在拷贝时直接转置成了A_tile[K_tile][X_tile]，这样在计算时，就不需要再转置了。
*/
template <typename T, size_t X_tile, size_t Y_tile, size_t K_tile, size_t Y_num, size_t NUM_THREAD>
__global__ void gemm_v04(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    // gridDim是外层，blockDim是内层
    size_t const w0{blockIdx.x * X_tile}; // x x是内层循环，而y是外层循环
    size_t const h0{blockIdx.y * Y_tile}; // y

    // 这个就是内部thread的坐标，下面坐标用于组织内存拷贝。
    size_t const inner_x_b = threadIdx.x/Y_num; // 64 // 11%
    size_t const inner_y_b = threadIdx.x - inner_x_b * Y_num; // 8

    // size_t const inner_y_b = threadIdx.x/X_tile; // 8 // 37%
    // size_t const inner_x_b = threadIdx.x - inner_y_b * X_tile; // 64

    size_t const inner_x_a = inner_x_b;
    size_t const inner_y_a = inner_y_b;

    // 全局的坐标，包含了block的偏移
    size_t const h1{h0 + inner_x_a};
    size_t const w1{w0 + inner_x_b};

    __shared__ T A_tile[K_tile][X_tile]; // 64 x 8
    __shared__ T B_tile[K_tile][Y_tile]; // 8 x 64

    int loop_count = (k + K_tile - 1) / K_tile;

    T sum[Y_num] = {static_cast<T>(0)};
    for (int i = 0; i < loop_count; i++)
    {
        // load A and B into shared memory
        // load A
        if (h1 < m && i * K_tile < k)
        {
            A_tile[inner_y_a][inner_x_a] = A[h1 * k + i * K_tile + inner_y_a];
        }
        else
        {
            A_tile[inner_y_a][inner_x_a] = {static_cast<T>(0)};
        }

        // load B
        if (w1 < n && i * K_tile < k)
        {
            B_tile[inner_y_b][inner_x_b] = B[(i * K_tile + inner_y_b) * n + w1];
        }
        else
        {
            B_tile[inner_y_b][inner_x_b] = {static_cast<T>(0)};
        }

        __syncthreads();

        // compute C
        #pragma unroll
        for (size_t ii = 0; ii < K_tile; ++ii)
        {
            T b_val = B_tile[ii][inner_x_b];

            for (size_t j = 0; j < Y_num; ++j)
            {
                sum[j] += A_tile[ii][inner_y_b * Y_num + j] * b_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < Y_num; i++)
    {
        size_t h_idx = h0 + inner_y_b * Y_num + i;
        if (h_idx < m && w1 < n)
        {
            C[h_idx * n + w1] = alpha * sum[i] + beta * C[h_idx * n + w1]; // 这里是xy的不同点。
        }
    }
}

template <typename T>
void cuda_gemm_v4(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    const size_t X_tile = 64;
    const size_t Y_tile = 64; // 块大小
    const size_t K_tile = 8;

    const size_t Y_num = 8; // 64x64的块，会被分成Y_num x 64的大小。一个线程每次算出的结果也是Y_num个。

    const size_t NUM_THREAD = DIV_UP(X_tile * Y_tile, Y_num);
    dim3 const block_dim{NUM_THREAD, 1U, 1U}; // x, y, z，只有第一个有效。

    const size_t m_num = DIV_UP(m, X_tile);
    const size_t n_num = DIV_UP(n, Y_tile);

    dim3 const grid_dim{   // 这个内部是threadIdx
        (static_cast<unsigned int>(m_num)),
        (static_cast<unsigned int>(n_num)), 1U}; // 分成m_num * n_num个块
    gemm_v04<T, X_tile, Y_tile, K_tile, Y_num, NUM_THREAD><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}


template GEMM_EXPORT void cuda_gemm_v4<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_v4<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

