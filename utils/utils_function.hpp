#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include <cublas_v2.h> // 调用cublas库，对比手写kernel和他的性能。
#include <cuda_runtime.h>

#include <cuda_fp16.h>

#if defined (_MSC_VER) || defined (_WIN32)
#ifndef CUDA_EXPORT 
#define CUDA_EXPORT __declspec(dllexport)
#endif
#endif

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void CUDA_EXPORT check_cuda(cudaError_t err, const char* const func, const char* const file,
                const int line);

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void CUDA_EXPORT check_cuda_last(const char* const file, const int line);

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

// 验证两个矩阵是否相等 
//// @param C 计算结果矩阵
//// @param C_ref 参考矩阵
//// @param m 行数
//// @param n 列数
//// @param ldc 计算结果矩阵的行步长
//// @param abs_tol 绝对容差
///// @param rel_tol 相对容差
template <typename T>
bool all_close(T* C, T* C_ref, size_t m, size_t n, size_t ldc,
               T abs_tol, double rel_tol)
{
    bool status{true};
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            double const C_val{static_cast<double>(C[i * ldc + j])};
            double const C_ref_val{static_cast<double>(C_ref[i * ldc + j])};
            double const diff{C_val - C_ref_val};
            double const diff_val{std::abs(diff)};
            if (diff_val >
                std::max(static_cast<double>(abs_tol),
                         static_cast<double>(std::abs(C_ref_val)) * rel_tol))
            {
                std::cout << "C[" << i << ", " << j << "] = " << C_val
                          << " C_ref[" << i << ", " << j << "] = " << C_ref_val
                          << " Abs Diff: " << diff_val
                          << " Abs Diff Threshold: "
                          << static_cast<double>(abs_tol)
                          << " Rel->Abs Diff Threshold: "
                          << static_cast<double>(
                                 static_cast<double>(std::abs(C_ref_val)) *
                                 rel_tol)
                          << std::endl;
                status = false;
                return status;
            }
        }
    }
    return status;
}

/// @brief 随机初始化矩阵
/// @tparam T 模板类型
/// @tparam type 
/// @param A 指针，需要提前分配好内存
/// @param m 行宽
/// @param n 列
/// @param lda 跨度
/// @param seed 随机种子
template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
void random_initialize_matrix(T* A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U)
{
    std::default_random_engine eng(seed);
    // The best way to verify is to use integer values.
    std::uniform_int_distribution<int> dis(0, 5);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() { return dis(eng); };
    for (size_t i{0U}; i < m; ++i)
    {
        for (size_t j{0U}; j < n; ++j)
        {
            A[i * lda + j] = static_cast<T>(rand());
        }
    }
}

float CUDA_EXPORT measure_performance(std::function<void(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100);

// 打印设备信息
void CUDA_EXPORT print_device_info();

#endif // CUDA_UTILS_HPP