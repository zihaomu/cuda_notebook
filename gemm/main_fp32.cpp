#include <iostream>
#include <cuda_runtime.h>
#include <functional>

#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main()
{
    print_device_info();

    float fp32_abs_tol{1.0e-4f};
    double const fp32_rel_tol{0.0e-4f};

    const size_t num_repeats{1U};
    const size_t num_warmups{1U};
 
    size_t m = 4096;
    size_t n = 4096;
    size_t k = 4096;

    size_t lda{(k + 16U - 1U) / 16U * 16U};
    size_t ldb{(n + 16U - 1U) / 16U * 16U};
    size_t ldc{(n + 16U - 1U) / 16U * 16U};

    // static_assert(lda >= k);
    // static_assert(ldb >= n);
    // static_assert(ldc >= n);

    dim3 const block_dim{32U, 31U, 30U};

    std::cout<<"Block Dim: "<<block_dim.x<<" x "<<block_dim.y<<" x "<<block_dim.z<<std::endl;

    std::cout << "Matrix Size: " << "M = " << m << " N = " << n << " K = " << k
              << std::endl;
    std::cout << "Matrix A: " << m << " x " << k << std::endl;
    std::cout << "Matrix B: " << k << " x " << n << std::endl;
    std::cout << "Matrix C: " << m << " x " << n << std::endl;
    std::cout << std::endl;


    std::vector<std::pair<
    std::string,
    std::function<void(size_t, size_t, size_t, float*, float*,
                       float*, float, float, cudaStream_t)>>> const
    gemm_kernel_launch_functions{
        //{"Custom GEMM Kernel V00", cuda_gemm_v0<float>},
        //{"Custom GEMM Kernel V01", cuda_gemm_v1<float>},
        ////{"Custom GEMM Kernel V02 Tiling", cuda_gemm_v2<float>},
        //{"Custom GEMM Kernel V03 Tiling 1D", cuda_gemm_v3<float>},
        //{"Custom GEMM Kernel V04 Tiling 1D", cuda_gemm_v4<float>},
        //{"Custom GEMM Kernel V05 Tiling 1D", cuda_gemm_v5<float>},
        //{"Custom GEMM Kernel V06 Warp Tiling", cuda_gemm_v6<float>},
        //{"Custom GEMM Kernel V07 Warp Tiling", cuda_gemm_v7<float>},
        {"Custom GEMM Kernel cutlass", cuda_gemm_cutlass<float>},
        {"Custom GEMM Kernel tensor core", cuda_gemm_tensor_core<float>},
        // {"Custom GEMM Kernel WMMA", cuda_gemm_wmma},
    };

    for (auto const& gemm_kernel_launch_function : gemm_kernel_launch_functions)
    {
        std::cout << gemm_kernel_launch_function.first << std::endl; // 输出 kernel 名称
        std::pair<float, float> const gemm_kernel_profile_result{
            profile_gemm<float>(
                m, n, k, lda, ldb, ldc, gemm_kernel_launch_function.second,
                fp32_abs_tol, fp32_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    return(0);
}