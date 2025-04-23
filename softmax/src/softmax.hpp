#ifndef CUDA_SOFTMAX_HPP
#define CUDA_SOFTMAX_HPP

// 包含所有不同版本的softmax实现
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils_function.hpp"


// naive cpu softmax实现
template <typename T>
void cpu_softmax_v0(size_t m, size_t n, T *A, T *C);

// naive GPU kernel
template <typename T>
void cuda_softmax_v0(size_t m, size_t n, T *A, T *C, cudaStream_t stream);

// 优化版本的GPU kernel
template <typename T>
void cuda_softmax_v1(size_t m, size_t n, T *A, T *C, cudaStream_t stream);

// OneFlow 版本的GPU kernel
template <typename T>
void cuda_softmax_v2(size_t m, size_t n, T *A, T *C, cudaStream_t stream);


#endif // CUDA_SOFTMAX_HPP