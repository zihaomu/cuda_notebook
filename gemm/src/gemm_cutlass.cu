#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include <stdio.h>
#include <cutlass/gemm/device/gemm.h>
#include <iostream>


// 或者使用 reinterpret_cast 进行指针转换
inline cutlass::half_t convert_half_to_chalf(__half h_ptr)
{
    return *reinterpret_cast<cutlass::half_t*>(&h_ptr);
}


template <typename T>
void cuda_gemm_cutlass(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

template <>
void cuda_gemm_cutlass<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream)
{
    using ElementOutput = float;
    using ElementInputA = float;
    using ElementInputB = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA, 
        ElementInputB, LayoutB, 
        ElementOutput, LayoutC>;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments args(
       problem_size,
       {A, k}, // Pointer to A and leading dimension
       {B, n}, // Pointer to B and leading dimension
       {C, n}, // Pointer to C and leading dimension
       {C, n}, // Pointer to output C and leading dimension
       {alpha, beta} // Scalars
    );

    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM CUTLASS operation failed!" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
    }
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_cutlass<__half>(size_t m, size_t n, size_t k, __half *_A, __half *_B, __half *_C, __half _alpha, __half _beta, cudaStream_t stream)
{
    // convert __half to cutlass::half_t
    cutlass::half_t *A = reinterpret_cast<cutlass::half_t *>(_A);
    cutlass::half_t *B = reinterpret_cast<cutlass::half_t *>(_B);
    cutlass::half_t *C = reinterpret_cast<cutlass::half_t *>(_C);
    cutlass::half_t alpha = convert_half_to_chalf(_alpha);
    cutlass::half_t beta = convert_half_to_chalf(_beta);

    using ElementOutput = cutlass::half_t;
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA, 
        ElementInputB, LayoutB, 
        ElementOutput, LayoutC>;

    Gemm gemm_op;

    // Define epilogue parameters
    using EpilogueOp = typename Gemm::EpilogueOutputOp;
    typename EpilogueOp::Params epilogue_params(alpha, beta);

    cutlass::gemm::GemmCoord problem_size((int)m, (int)n, (int)k);
    typename Gemm::Arguments args(
       problem_size,
       {A, k}, // Pointer to A and leading dimension
       {B, n}, // Pointer to B and leading dimension
       {C, n}, // Pointer to C and leading dimension
       {C, n}, // Pointer to output C and leading dimension
       epilogue_params // Scalars
    );

    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM CUTLASS operation failed!" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
    }
    CHECK_LAST_CUDA_ERROR();
}


template GEMM_EXPORT void cuda_gemm_cutlass<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_cutlass<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);


