#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

/*
本方案的思路是：
方案3~5都是实现两层循环：blocktiling loop和threadtiling loop。本方案是想在在这两者之间加入一个warp tiling loop。目的是提高缓存利用率。
warp是GPU硬件中的一个概念，一个warp包含32个线程，这32个线程是同时执行的。一个SM中会有多个warp在执行，最好是一部分warp在计算，一部分warp在访存，这样可以隐藏内存访问延迟。

由此可用看出多层为：
1. Block tiling loop: 将数据按照输出划分成小块，block可用可用在不同的SM上执行。
2. Warp tiling loop: 在块的基础上，将数据划分成更小的块，每一个小块能够被一个warp计算。
3. Thread tiling loop: 指令可用在同一个CUDA core上执行，即指令级并行（ILP）
*/

#define WARP_SIZE 32

// load data from global memory to shared memory for A and B
template <typename T, size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN, size_t stride_BM, size_t stride_BK>
__device__ void load_from_gmem(size_t m, size_t n, size_t k, T *A, T *B, T* As, T *Bs, size_t A_m, size_t A_n, size_t B_m, size_t B_n)
{
    // load A
    for (size_t i = 0; i + stride_BM <= BM; i += stride_BM)
    {
        #pragma unroll
        for (size_t j = 0; j < 4; j++)
        {
            As[(A_n * 4 + j) * BM + i + A_m] = A[(A_m + i) * k + A_n * 4 + j];
        }

        // const float4 tmp = reinterpret_cast<float4*>(A + (A_m + i) * k + A_n * 4)[0];
        // As[(A_n * 4 + 0) * BM + i + A_m] = tmp.x;
        // As[(A_n * 4 + 1) * BM + i + A_m] = tmp.y;
        // As[(A_n * 4 + 2) * BM + i + A_m] = tmp.z;
        // As[(A_n * 4 + 3) * BM + i + A_m] = tmp.w;
    }

    // load B
    for (size_t i = 0; i + stride_BK <= BK; i+= stride_BK)
    {
        #pragma unroll
        for (size_t j = 0; j < 4; j++)
        {
            Bs[(i + B_m) * BN + B_n * 4 + j] = B[(B_m + i) * n + B_n * 4 + j];
        }
    }
}

template <size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN, size_t stride_BM, size_t stride_BK>
__device__ void load_from_gmem(size_t m, size_t n, size_t k, float *A, float *B, float* As, float *Bs, size_t A_m, size_t A_n, size_t B_m, size_t B_n)
{
    // load A
    #pragma unroll
    for (size_t i = 0; i + stride_BM <= BM; i += stride_BM)
    {
        // for (size_t j = 0; j < 4; j++)
        // {
        //     As[(A_n * 4 + j) * BM + i + A_m] = A[(A_m + i) * k + A_n * 4 + j];
        // }

        const float4 tmp = reinterpret_cast<float4*>(A + (A_m + i) * k + A_n * 4)[0];
        As[(A_n * 4 + 0) * BM + i + A_m] = tmp.x;
        As[(A_n * 4 + 1) * BM + i + A_m] = tmp.y;
        As[(A_n * 4 + 2) * BM + i + A_m] = tmp.z;
        As[(A_n * 4 + 3) * BM + i + A_m] = tmp.w;
    }

    // load B
    #pragma unroll
    for (size_t i = 0; i + stride_BK <= BK; i+= stride_BK)
    {
        reinterpret_cast<float4*>(&Bs[(i + B_m) * BN + B_n * 4])[0]
         = reinterpret_cast<float4*>(B + (B_m + i) * n + B_n * 4)[0];
    }
}

template <typename T, size_t TM, size_t TN>
__device__ void  post_processing(const size_t n, const size_t M_Warp, const size_t N_Warp, const size_t M_Sub, const size_t N_Sub, const int thread_m_strip, const int thread_n_strip, T* C_in, T* result, T alpha, T beta)
{
    for (int wm = 0; wm < M_Warp; wm++)
    {
        for (int wn = 0; wn < N_Warp; wn++)
        {
            T* C_in_in = C_in + wm * M_Sub * n + wn * N_Sub;
            for (int x = 0; x < TM; x++)
            {
                for (int y = 0; y < TN; y++)
                {
                    C_in_in[(x + thread_m_strip) * n + thread_n_strip + y] =
                    // C_in_in[0] =
                    alpha * result[(wm * N_Warp + wn) * TM * TN + x * TN + y] + 
                    // alpha * result[0] + 
                    beta * C_in_in[(x + thread_m_strip) * n + thread_n_strip + y];
                }
                // float4 tmp = reinterpret_cast<float4*>(C_in_in + (x + thread_m_strip) * n + thread_n_strip)[0];

                // size_t i = (wm * N_Warp + wn) * TM * TN + x * TN;
                
                // tmp.x = alpha * result[i + 0] + beta * tmp.x;
                // tmp.y = alpha * result[i + 1] + beta * tmp.y;
                // tmp.z = alpha * result[i + 2] + beta * tmp.z;
                // tmp.w = alpha * result[i + 3] + beta * tmp.w;

                // // write back
                // reinterpret_cast<float4*>(&C_in_in[(x + thread_m_strip) * n + thread_n_strip])[0] = tmp;
            }
        }
    }
}

template <size_t TM, size_t TN>
__device__ void post_processing<float, TM, TN>(const size_t n, const size_t M_Warp, const size_t N_Warp, const size_t M_Sub, const size_t N_Sub, const int thread_m_strip, const int thread_n_strip, float* C_in, float* result, float alpha, float beta)
{
    for (int wm = 0; wm < M_Warp; wm++)
    {
        for (int wn = 0; wn < N_Warp; wn++)
        {
            float* C_in_in = C_in + wm * M_Sub * n + wn * N_Sub;
            for (int x = 0; x < TM; x++)
            {
                // for (int y = 0; y < TN; y++)
                // {
                //     C_in_in[(x + thread_m_strip) * n + thread_n_strip + y] =
                //     // C_in_in[0] =
                //     alpha * result[(wm * N_Warp + wn) * TM * TN + x * TN + y] + 
                //     // alpha * result[0] + 
                //     beta * C_in_in[(x + thread_m_strip) * n + thread_n_strip + y];
                // }
                float4 tmp = reinterpret_cast<float4*>(C_in_in + (x + thread_m_strip) * n + thread_n_strip)[0];

                size_t i = (wm * N_Warp + wn) * TM * TN + x * TN;
                
                tmp.x = alpha * result[i + 0] + beta * tmp.x;
                tmp.y = alpha * result[i + 1] + beta * tmp.y;
                tmp.z = alpha * result[i + 2] + beta * tmp.z;
                tmp.w = alpha * result[i + 3] + beta * tmp.w;

                // write back
                reinterpret_cast<float4*>(&C_in_in[(x + thread_m_strip) * n + thread_n_strip])[0] = tmp;
            }
        }
    }
}

template <typename T, size_t BM, size_t BN, size_t BK, size_t WM, size_t WN, size_t TM, size_t TN, size_t stride_BM, size_t stride_BK>
__global__ void gemm_v06(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta)
{
    // Block tiling loop
    __shared__ T As[BK * BM];
    __shared__ T Bs[BK * BN];

    const size_t BK_4 = BK/4;
    size_t A_m = threadIdx.x / BK_4;
    size_t A_n = threadIdx.x - A_m * BK_4;

    const size_t BN_4 = BN/4;
    size_t B_m = threadIdx.x / BN_4;
    size_t B_n = threadIdx.x - B_m * BN_4;

    // Warp tiling loop
    int warp_idx = threadIdx.x / WARP_SIZE; // 4
    // 128 分成4个warp，每个warp有32个线程
    int warp_m = warp_idx / 2;          // m 2
    int warp_n = warp_idx - warp_m * 2; // n 2

    // thread tiling loop
    // 32个线程，每个线程计算一个TM * TN的块。32个线程要重新组合成4 x 8的块。
    int warp_inside = threadIdx.x % WARP_SIZE;
    int thread_m = warp_inside % 4; // 4
    int thread_n = warp_inside / 4; // 8

    int thread_m_strip = thread_m * TM;
    int thread_n_strip = thread_n * TN;

    const size_t M_Warp = 2;
    const size_t N_Warp = 2;

    const size_t M_Sub = 32;
    const size_t N_Sub = 32;

    // alloc register
    T result[M_Warp * N_Warp * TM * TN] = {static_cast<T>(0)};
    T register_A[M_Warp * TM];
    T register_B[N_Warp * TN];

    T* C_in = C + (blockIdx.x * BM + warp_m * WM) * n + blockIdx.y * BN + warp_n * WN;
    // float* C_in = C + (blockIdx.x * BM) * n + blockIdx.y * BN;
    T* A_in = A + blockIdx.x * BM * k; // x 对应的是m
    T* B_in = B + blockIdx.y * BN;     // y 对应的是n

    for (int i = 0; i < k; i += BK, A_in += BK, B_in += BK * n)
    {
        // load data from global memory to shared memory for A and B
        load_from_gmem<T, BM, BN, BK, WM, WN, TM, TN, stride_BM, stride_BK>(m, n, k, A_in, B_in, As, Bs, A_m, A_n, B_m, B_n);

        __syncthreads(); // 同步

        // compute C
        for (int j = 0; j < BK; j++)
        {
            // load register
            for (int wm = 0; wm < M_Warp; wm++)
            {
                for (int x = 0; x < TM; x++)
                {
                    register_A[wm * TM + x] = As[j * BM + warp_m * WM + wm * M_Sub + thread_m_strip + x];
                }
            }

            for (int wn = 0; wn < N_Warp; wn++)
            {
                for (int y = 0; y < TN; y++)
                {
                    register_B[wn * TN + y] = Bs[j * BN + warp_n * WN + wn * N_Sub + y + thread_n_strip];
                }
            }

            // compute
            for (int m_idx = 0; m_idx < M_Warp; m_idx++)
            {
                for (int n_idx = 0; n_idx < N_Warp; n_idx++)
                {
                    for (int x = 0; x < TM; x++)
                    {
                        for (int y = 0; y < TN; y++)
                        {
                            // result[(m_idx * TM + n_idx) * TN * 2 + x * TN + y] += register_A[m_idx * TM + x] * register_B[n_idx * TN + y];
                            result[(m_idx * N_Warp + n_idx) * TM * TN + x * TN + y] += register_A[m_idx * TM + x] * register_B[n_idx * TN + y];
                            // result[0] += register_A[0] * register_B[0];
                        }
                    }
                }
            }
        }

        // 代码测试1
        // for (int j = 0; j < BK; j++)
        // {
        //     float val_B = Bs[j * BN + threadIdx.x];

        //     for (int jj = 0; jj < 128; jj++)
        //     {
        //         result[jj] += As[j * BM + jj] * val_B;
        //     }
        // }

        __syncthreads(); // 同步
    }

    // 测试代码2
    // for (int i = 0; i < 128; i++)
    // {
    //     C_in[i*n + threadIdx.x] = result[i];
    // }

    // store result to global memory

    post_processing<T, TM, TN>(n, M_Warp, N_Warp, M_Sub, N_Sub, thread_m_strip, thread_n_strip, C_in, result, alpha, beta);
    // for (int wm = 0; wm < M_Warp; wm++)
    // {
    //     for (int wn = 0; wn < N_Warp; wn++)
    //     {
    //         float* C_in_in = C_in + wm * M_Sub * n + wn * N_Sub;
    //         for (int x = 0; x < TM; x++)
    //         {
    //             // for (int y = 0; y < TN; y++)
    //             // {
    //             //     C_in_in[(x + thread_m_strip) * n + thread_n_strip + y] =
    //             //     // C_in_in[0] =
    //             //     alpha * result[(wm * N_Warp + wn) * TM * TN + x * TN + y] + 
    //             //     // alpha * result[0] + 
    //             //     beta * C_in_in[(x + thread_m_strip) * n + thread_n_strip + y];
    //             // }
    //             float4 tmp = reinterpret_cast<float4*>(C_in_in + (x + thread_m_strip) * n + thread_n_strip)[0];

    //             size_t i = (wm * N_Warp + wn) * TM * TN + x * TN;
                
    //             tmp.x = alpha * result[i + 0] + beta * tmp.x;
    //             tmp.y = alpha * result[i + 1] + beta * tmp.y;
    //             tmp.z = alpha * result[i + 2] + beta * tmp.z;
    //             tmp.w = alpha * result[i + 3] + beta * tmp.w;

    //             // write back
    //             reinterpret_cast<float4*>(&C_in_in[(x + thread_m_strip) * n + thread_n_strip])[0] = tmp;
    //         }
    //     }
    // }
    __syncthreads(); // 同步
}

template <typename T>
void cuda_gemm_v6(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    const size_t BM = 128;
    const size_t BN = 128;
    const size_t BK = 16;

    const size_t WM = 64;
    const size_t WN = 64;

    const size_t TM = 8;
    const size_t TN = 4;

    const size_t BK_4 = BK/4;
    const size_t BN_4 = BN/4;
    const size_t stride_BM = 128/BK_4;
    const size_t stride_BK = 128/BN_4;

    const size_t THREAD_NUM = 128;

    dim3 const block_dim{THREAD_NUM, 1U, 1U};

    dim3 const grid_dim{
        (static_cast<unsigned int>((m + BM - 1U) / BM)),
        (static_cast<unsigned int>((n + BN - 1U) / BN)), 1U};
    gemm_v06<T , BM, BN, BK, WM, WN, TM, TN, stride_BM, stride_BK><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}


template GEMM_EXPORT void cuda_gemm_v6<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_v6<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

