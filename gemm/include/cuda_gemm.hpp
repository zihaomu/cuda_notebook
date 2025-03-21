#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP

/*
存放所有调用函数的头文件
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined (_MSC_VER) || defined (_WIN32)
#ifndef GEMM_EXPORT 
#define GEMM_EXPORT __declspec(dllexport)
#endif
#endif

// naive gemm kernel
template <typename T>
void cuda_gemm_v0(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// global memory access, 为什么这个算global memory
template <typename T>
void cuda_gemm_v1(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// block tilling vectorized
template <typename T>
void cuda_gemm_v2(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// block tilling for shared memory, and 1d tiled vectorized for register
template <typename T>
void cuda_gemm_v3(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// block tilling for shared memory, and 1d tiled vectorized for register
template <typename T>
void cuda_gemm_v4(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// block tilling for shared memory, and 1d tiled vectorized for register
template <typename T>
void cuda_gemm_v5(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// try to use warp tile to optimize the performance
template <typename T>
void cuda_gemm_v6(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// try to use float2 to optimize the performance
template <typename T>
void cuda_gemm_v7(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// cutlass benchmark
template <typename T>
void cuda_gemm_cutlass(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// tensor core benchmark
template <typename T>
void cuda_gemm_tensor_core(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA naive implementation
template <typename T>
void cuda_gemm_wmma_0(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA naive implementation
template <typename T>
void cuda_gemm_wmma_1(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA naive implementation, 128x128 tiling, and wmma, achieve 23% perfromance of cublas
// 存在问题，bank conflict
template <typename T>
void cuda_gemm_wmma_2(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA fast load C
template <typename T>
void cuda_gemm_wmma_3(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA fixed bank conflict，利用率70%
template <typename T>
void cuda_gemm_wmma_4(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// WMMA 双buffer，没啥效果，利用率40%
template <typename T>
void cuda_gemm_wmma_5(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

// 在5的基础上，尝试增加一个thread block中warp的数量
template <typename T>
void cuda_gemm_wmma_6(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream);

#endif