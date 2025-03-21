#include "cuda_gemm.hpp"

#include <iostream>

template <typename T>
void cpu_gemm(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            T sum{static_cast<T>(0)};
            for(int l = 0; l < k; l++)
            {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = (T)(alpha * sum + beta * C[i * n + j]);
        }
    }
}