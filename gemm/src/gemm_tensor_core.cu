#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/arch/wmma.h>
#include <cuda_runtime.h>
#include <iostream>

//__global__ void convert_float_to_half(const float* input, cutlass::half_t* output, size_t size) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < size) {
//        output[idx] = static_cast<cutlass::half_t>(input[idx]);
//    }
//}

// 或者使用 reinterpret_cast 进行指针转换
inline cutlass::half_t convert_half_to_chalf(__half h_ptr)
{
    return *reinterpret_cast<cutlass::half_t*>(&h_ptr);
}

template <typename T>
void cuda_gemm_tensor_core(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Un-support data type"<<std::endl;
}

template <>
void cuda_gemm_tensor_core<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream)
{
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
       ElementInputA, LayoutA,
       ElementInputB, LayoutB,
       ElementOutput, LayoutC,
       float,
       cutlass::arch::OpClassTensorOp,
       cutlass::arch::Sm80, // Use SM80, supported on RTX 4090 (Ada Lovelace)
       cutlass::gemm::GemmShape<128, 128, 16>, // Optimized block size
       cutlass::gemm::GemmShape<64, 64, 16>, // Optimized warp size
       cutlass::gemm::GemmShape<16, 8, 8> // Tensor core tile size
    >;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments args(
       problem_size,
       { A, k }, // Pointer to converted A and leading dimension
       { B, n }, // Pointer to converted B and leading dimension
       { C, n }, // Pointer to C and leading dimension
       { C, n }, // Pointer to output C and leading dimension
       { alpha, beta } // Scalars
    );

    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM initialization failed" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
       return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM Tensor core operation failed!" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
    }

    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_tensor_core<__half>(size_t m, size_t n, size_t k, __half *_A, __half *_B, __half *_C, __half _alpha, __half _beta, cudaStream_t stream)
{
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    cutlass::half_t *A = reinterpret_cast<cutlass::half_t *>(_A);
    cutlass::half_t *B = reinterpret_cast<cutlass::half_t *>(_B);
    cutlass::half_t *C = reinterpret_cast<cutlass::half_t *>(_C);
    cutlass::half_t alpha = convert_half_to_chalf(_alpha);
    cutlass::half_t beta = convert_half_to_chalf(_beta);

    using Gemm = cutlass::gemm::device::Gemm<
       ElementInputA, LayoutA,
       ElementInputB, LayoutB,
       ElementOutput, LayoutC,
       float,
       cutlass::arch::OpClassTensorOp,
       cutlass::arch::Sm80, // Use SM80, supported on RTX 4090 (Ada Lovelace)
       cutlass::gemm::GemmShape<128, 128, 16>, // Optimized block size
       cutlass::gemm::GemmShape<64, 64, 16>, // Optimized warp size
       cutlass::gemm::GemmShape<16, 8, 8> // Tensor core tile size
    >;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(m, n, k);
    typename Gemm::Arguments args(
       problem_size,
       { A, k }, // Pointer to converted A and leading dimension
       { B, n }, // Pointer to converted B and leading dimension
       { C, n }, // Pointer to C and leading dimension
       { C, n }, // Pointer to output C and leading dimension
       { alpha, beta } // Scalars
    );

    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM initialization failed" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
       return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
       std::cerr << "GEMM Tensor core operation failed!" << std::endl;

       // print details error info
       std::cerr << "Error in " << __FILE__ << " at line " << __LINE__ << ": " << cutlassGetStatusString(status) << std::endl;
    }
    CHECK_LAST_CUDA_ERROR();
}


template GEMM_EXPORT void cuda_gemm_tensor_core<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_tensor_core<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);


