#include "softmax.hpp"
#include "utils_function.hpp"
#include <stdio.h>
#include <iostream>


// 输入是m x n的矩阵，一个thread只会处理一行，也就是n个元素。
template <typename T>
__global__ void softmax_v00(size_t m, size_t n, T *_A, T *_C)
{
    std::cout<<"Un-support data type"<<std::endl;
}

template <>
__global__ void softmax_v00<__half>(size_t m, size_t n, __half *_A, __half *_C)
{
    size_t idx = blockIdx.x * 32 + threadIdx.x; // 计算当前线程的索引
    if (idx >= m) return;

    __half* A = _A + idx * n; // 当前行的起始地址
    __half* C = _C + idx * n; // 当前行的起始地址

    __half max_val = -INFINITY;
    for (size_t i = 0; i < n; ++i) { 
        max_val = A[i] > max_val ? A[i] : max_val;
    }

    __half sum = {static_cast<__half>(0)};

    for (size_t i = 0; i < n; ++i) {
        sum += hexp(A[i] - max_val);
    }
  
    // 计算softmax,并传递给output
    for (size_t i = 0; i < n; ++i) {
        C[i] = hexp(A[i] - max_val) / sum;
    }
}

template <>
__global__ void softmax_v00<float>(size_t m, size_t n, float *_A, float *_C)
{
    size_t idx = blockIdx.x * 32 + threadIdx.x; // 计算当前线程的索引
    if (idx >= m) return;

    float* A = _A + idx * n; // 当前行的起始地址
    float* C = _C + idx * n; // 当前行的起始地址

    float max_val = -INFINITY;
    for (size_t i = 0; i < n; ++i) { 
        max_val = A[i] > max_val ? A[i] : max_val;
    }

    float sum = {static_cast<float>(0)};

    for (size_t i = 0; i < n; ++i) {
        sum += expf(A[i] - max_val);
    }
  
    // 计算softmax,并传递给output
    for (size_t i = 0; i < n; ++i) {
        C[i] = expf(A[i] - max_val) / sum;
    }
}

// 每一行thread处理一行数据。
// 将32 作为一个warp，每个warp处理一行数据。
// m/32 = m_num
template <typename T>
void cuda_softmax_v0(size_t m, size_t n, T *A, T *C, cudaStream_t stream)
{
    int m_num = DIV_UP(m, 32); // 每个block处理32行数据

    dim3 const block_dim{static_cast<unsigned int>(m_num), 1U, 1U}; // x, y, z，只有第一个有效。
    dim3 const grid_dim{(static_cast<unsigned int>(32)), 1U, 1U}; // 分成m_num * n_num个块
    softmax_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, A, C);
    CHECK_LAST_CUDA_ERROR();
}

template CUDA_EXPORT void cuda_softmax_v0<float>(size_t m, size_t n, float *A, float *C, cudaStream_t stream);
template CUDA_EXPORT void cuda_softmax_v0<__half>(size_t m, size_t n, __half* A, __half* C, cudaStream_t stream);