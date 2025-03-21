#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_gemm.hpp"
#include "profile_utils.cuh"

int main()
{
    print_device_info();

    constexpr size_t num_repeats{1U};
    constexpr size_t num_warmups{1U};

    __half fp16_abs_tol{__float2half(5.0e-2f)};
    double const fp16_rel_tol{1.0e-1f};

    __half fp16_tensor_core_abs_tol{__float2half(5.0e-2f)};
    double const fp16_tensor_core_rel_tol{1.0e-2f};

    constexpr size_t m{4096U};
    constexpr size_t k{4096U};
    constexpr size_t n{4096U};

    constexpr size_t lda{(k + 16U - 1U) / 16U * 16U};
    constexpr size_t ldb{(n + 16U - 1U) / 16U * 16U};
    constexpr size_t ldc{(n + 16U - 1U) / 16U * 16U};

    // static_assert(lda >= k);
    // static_assert(ldb >= n);
    // static_assert(ldc >= n);

    std::cout << "Matrix Size: " << "M = " << m << " N = " << n << " K = " << k
              << std::endl;
    std::cout << "Matrix A: " << m << " x " << k
              << " Leading Dimension Size = " << lda << std::endl;
    std::cout << "Matrix B: " << k << " x " << n
              << " Leading Dimension Size = " << ldb << std::endl;
    std::cout << "Matrix C: " << m << " x " << n
              << " Leading Dimension Size = " << ldc << std::endl;
    std::cout << std::endl;

    // Define all the GEMM kernel launch functions to be profiled.
    std::vector<std::pair<
        std::string,
        std::function<void(size_t, size_t, size_t, __half *, __half *,
                           __half *, __half,
                           __half, cudaStream_t)>>> const
        gemm_fp16_kernel_launch_functions{
            //{"Custom GEMM Kernel V00", cuda_gemm_v0<__half>},
            //{"Custom GEMM Kernel V01", cuda_gemm_v1<__half>},
            ////{"Custom GEMM Kernel V02 Tiling", cuda_gemm_v2<__half>},
            //{"Custom GEMM Kernel V03 Tiling 1D", cuda_gemm_v3<__half>},
            //{"Custom GEMM Kernel V04 Tiling 1D", cuda_gemm_v4<__half>},
            //{"Custom GEMM Kernel V05 Tiling 1D", cuda_gemm_v5<__half>},
            //{"Custom GEMM Kernel V06 Warp Tiling", cuda_gemm_v6<__half>},
            //{"Custom GEMM Kernel V07 Warp Tiling", cuda_gemm_v7<__half>},
      /*      {"Custom GEMM Kernel cutlass", cuda_gemm_cutlass<__half>},
            {"Custom GEMM Kernel tensor core", cuda_gemm_tensor_core<__half>},*/
            //{"Custom GEMM naive Kernel WMMA", cuda_gemm_wmma_0<__half>},
            //{"Custom GEMM Kernel WMMA Tiling 128x128", cuda_gemm_wmma_1<__half>},
            //{"Custom GEMM Kernel WMMA Tiling 128x128 V2", cuda_gemm_wmma_2<__half>},
            //{"Custom GEMM Kernel WMMA Tiling 128x128 V3", cuda_gemm_wmma_3<__half>},
            // {"Custom GEMM Kernel WMMA Tiling 128x128 bank conflict V4", cuda_gemm_wmma_4<__half>},
            {"Custom GEMM Kernel WMMA Tiling 64x64 4 warp double buffer", cuda_gemm_wmma_5<__half>},
            {"Custom GEMM Kernel WMMA Tiling 64x64 8 warp double buffer", cuda_gemm_wmma_6<__half>},
        };

    for (auto const& gemm_fp16_kernel_launch_function :
         gemm_fp16_kernel_launch_functions)
    {
        std::cout << gemm_fp16_kernel_launch_function.first << std::endl;
        std::pair<__half, __half> const gemm_kernel_profile_result{
            profile_gemm<__half>(
                m, n, k, lda, ldb, ldc, gemm_fp16_kernel_launch_function.second,
                fp16_abs_tol, fp16_rel_tol, num_repeats, num_warmups)};
        std::cout << std::endl;
    }

    // std::vector<std::pair<
    //     std::string,
    //     std::function<void(size_t, size_t, size_t, __half*, __half*,
    //                        size_t, __half*, size_t, __half*,
    //                        __half*, size_t, cudaStream_t)>>> const
    //     gemm_fp16_tensor_core_kernel_launch_functions{
    //         {"Custom GEMM Kernel V07", launch_gemm_kernel_v07<__half>},
    //         {"Custom GEMM Kernel V07 Vectorized",
    //          launch_gemm_kernel_v07_vectorized<__half>},
    //         {"Custom GEMM Kernel V07 Vectorized Double Buffered",
    //          launch_gemm_kernel_v07_vectorized_double_buffered<__half>},
    //     };

    // for (auto const& gemm_fp16_tensor_core_kernel_launch_function :
    //      gemm_fp16_tensor_core_kernel_launch_functions)
    // {
    //     std::cout << gemm_fp16_tensor_core_kernel_launch_function.first
    //               << std::endl;
    //     std::pair<__half, __half> const gemm_kernel_profile_result{
    //         profile_gemm<__half>(
    //             m, n, k, lda, ldb, ldc,
    //             gemm_fp16_tensor_core_kernel_launch_function.second,
    //             fp16_tensor_core_abs_tol, fp16_tensor_core_rel_tol, num_repeats,
    //             num_warmups)};
    //     std::cout << std::endl;
    // }

    return 0;
}