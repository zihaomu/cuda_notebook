#ifndef CUDA_GEMM_UTILS_CUH
#define CUDA_GEMM_UTILS_CUH

/*
包含辅助函数的头文件
*/

#include <cuda_runtime.h>
#include "cuda_gemm.hpp"

// result check function
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void GEMM_EXPORT check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line);

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void GEMM_EXPORT check_cuda_last(const char* const file, const int line);

// upper bound 
#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }


#endif // CUDA_GEMM_UTILS_CUH
