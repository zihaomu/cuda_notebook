#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

/// 输入是M x N，输出也是M x N，输入指针是：A，输出是：C。
template <typename T> // T 是 half，float，double等类型
void cpu_softmax_v0(size_t m, size_t n, T *A, T *C)
{
    for (size_t i = 0; i < m; ++i) 
    {
        T max_val = A[i * n]; // 初始化最大值为当前行的第一个元素
        for (size_t j = 1; j < n; ++j) {
            max_val = std::max(max_val, A[i * n + j]); // 找到当前行的最大值
        }

        T sum_exp = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = std::exp(A[i * n + j] - max_val); // 减去最大值以避免溢出
            sum_exp += C[i * n + j]; // 计算指数和
        }

        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] /= sum_exp; // 归一化
        }
    }
}