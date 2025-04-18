#ifndef CUDA_SOFTMAX_UTILS_CUH
#define CUDA_SOFTMAX_UTILS_CUH

#include <iostream>
#include <cassert>

#include "./utils_function.hpp"

#define PEAK_BAND_WITH 1008

float compute_effective_bandwidth(size_t m, size_t n, float latency)
{
    return (m * n * 8 * sizeof(float)) / (latency * 1e-3) / 1e9;
}

void print_performance_result(size_t m, size_t n, float latency)
{
    float effective_bandwidth = compute_effective_bandwidth(m, n, latency);
    // float effective_tflops{compute_effective_tflops(m, n, k, latency)};

    std::cout << "Latency: " << latency << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth << " GB/s."

              << std::endl;
    std::cout<<"Achieve "<< effective_bandwidth /PEAK_BAND_WITH * 100<<"% performance of theoretical bandwidth."<<std::endl;
    // std::cout << "Effective TFLOPS: " << effective_tflops << " TFLOPS"
    //           << std::endl;
}

template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
std::pair<float, float> profile_softmax(
    size_t m, size_t n,
    std::function<void(size_t, size_t, T*, T*, cudaStream_t)>
    softmax_kernel_launch_function,
    T abs_tol, double rel_tol, size_t num_repeats = 10, size_t num_warmups = 10,
    unsigned int seed = 0U)
{
    size_t ldm{n};
    T alpha{static_cast<T>(1.0)};
    T beta{static_cast<T>(0.0)};

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocate memory on host.
    T* A_host{nullptr};
    T* C_host{nullptr};
    T* C_host_ref{nullptr};
    T* C_host_from_device{nullptr};
    CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * ldm * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldm * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_ref, m * ldm * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device, m * ldm * sizeof(T)));

    // Initialize matrix A and C.
    random_initialize_matrix(A_host, m, n, ldm);
    random_initialize_matrix(C_host, m, n, ldm);

    // Allocate memory on device.
    T* A_device{nullptr};
    T* C_device{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * ldm * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldm * sizeof(T)));

    // Copy matrix A and B from host to device.
    CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * ldm * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldm * sizeof(T),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_host, m * ldm * sizeof(T),
                                cudaMemcpyHostToHost));

    // Create cuBLAS handle.
    // cublasHandle_t handle;
    // CHECK_CUBLASS_ERROR(cublasCreate(&handle));
    // CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

    // Compute reference output using cuBLAS.
    // 需要官方实现的softmax？
    softmax_kernel_launch_function(m, n, A_device, C_device, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy matrix C from device to host.
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_ref, C_device, m * ldm * sizeof(T),
                                cudaMemcpyDeviceToHost));

    // Launch CUDA GEMM.
    CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldm * sizeof(T), cudaMemcpyHostToDevice));
    // Verify the correctness of CUDA GEMM.
    softmax_kernel_launch_function(m, n, A_device, C_device, stream);

    // launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb,
    // &beta,
    //                       C_device, ldc, handle);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(C_host_from_device, C_device, m * ldm * sizeof(T), cudaMemcpyDeviceToHost));
    assert(all_close<T>(C_host_from_device, C_host_ref, m, n, ldm, abs_tol, rel_tol));

    // Launch cuBLAS GEMM.
    float naive_latency{measure_performance(
        [&](cudaStream_t stream)
        {
            softmax_kernel_launch_function(m, n, A_device, C_device, stream);
            return;
        },
        stream, num_repeats, num_warmups)};

    float boost_cuda_softmax{measure_performance(
        [&](cudaStream_t stream)
        {
            softmax_kernel_launch_function(m, n, A_device, C_device, stream);
            return;
        },
        stream, num_repeats, num_warmups)};

    // Release resources.
    CHECK_CUDA_ERROR(cudaFree(A_device));
    CHECK_CUDA_ERROR(cudaFree(C_device));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_ref));
    CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
    // CHECK_CUBLASS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    std::cout << "naive kernel performance = "<< naive_latency << " ms." << std::endl;
    std::cout << "boost kernel performance = "<< boost_cuda_softmax << " ms." << std::endl;

    std::cout << "softmax Kernel Performance" << std::endl;
    print_performance_result(m, n, boost_cuda_softmax);

    return std::pair<float, float>{naive_latency, boost_cuda_softmax};
}

#endif // CUDA_SOFTMAX_UTILS_CUH