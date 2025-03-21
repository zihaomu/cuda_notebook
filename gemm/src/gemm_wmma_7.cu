#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

/*
问题分析：经过profile发现，内存部分的效率低，尝试使用双缓冲。
优化思路：
基于实验5，在保持MNK仍然是64 64 64的前提下，增加每个thread block中划分的warp数，从之前的4个warp，增加到8个warp。

测试目标，增加SM中活跃的warp数量，是否能够增加效率。
同时：
测试了两种方案减少 bank conflict：
1 使用swizzle，错开warp，减少warp之间的竞争，从而减少bank conflict
2 使用padding，增加shared memory的stride，减少bank conflict
实际测试下来，两者都能很好的减少bank 冲突。

*/

#define BLOCK_SIZE 64
#define BLOCK_SIZE_DIV2 32
#define BLOCK_SIZE_DIV4 16
#define BLOCK_SIZE_DIV8 8

#define WARP_SIZE 32

#define WMMA_M 16  // 16x16x16 矩阵乘法
#define WMMA_N 16
#define WMMA_K 16

/*
一个SM有4个warp，每个warp有32个线程
C输出块大小为64x64，划分为4个块，每块大小为32x32，每块由一个warp计算输出。

对于每个线程，都需要申请一个2x2的空间，用来存储C的wmma fragment。

经过实验，还是有所上升，从52%到55%。
*/

static const int BLOCK_ROW_WARPS = 2;
static const int BLOCK_COL_WARPS = 2;

static const int WARP_ROW_TILES = 2;
static const int WARP_COL_TILES = 1;

static const int BLOCK_TILE_ROW_STRIP = WARP_ROW_TILES * WMMA_M; // 32
static const int BLOCK_TILE_COL_STRIP = WARP_COL_TILES * WMMA_N; // 16

static const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 4
static const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS); // 2

static const int SHMEM_STRIDE = (WMMA_N * BLOCK_ROW_TILES); // 64

static const int SKEW_HALF = 0;
static const int K_TILES = BLOCK_SIZE / WMMA_K; // 4

static const int BLOCK_SKEW_SIZE = BLOCK_SIZE + SKEW_HALF;
static const int BLOCK_SKEW_SIZE_DIV2 = BLOCK_SKEW_SIZE / 2;

static const int SUB_BLOCK_SIZE = DIV_UP(BLOCK_SIZE * BLOCK_SKEW_SIZE, 128) * 128; // 对齐 128

// 函数将A或B从global mem加载到shared memory中，需要给定src和dst的block大小，以及src和dst的leading dimension
// 这个函数由4个warp一起执行，将A或B加载到shared memory中
// src_ldm和dst_ldm都是对应int类型的跨度
static __device__ void load_AB(half* _src, half* _dst, int src_ldm, int dst_ldm)
{
    int* src = (int*)(_src + threadIdx.y * BLOCK_SIZE_DIV8 * src_ldm * 2);
    int* dst = (int*)(_dst + threadIdx.y * BLOCK_SIZE_DIV8 * dst_ldm * 2);
    // A 和 B 在shared memory中的长宽是64x64的half，换成int读取时，一行也就是32，正好对应一个warp
    // src += threadIdx.y * BLOCK_SIZE_DIV4 * src_ldm + threadIdx.x; // threadIdx.x = warpLaneId, threadIdx.y = warpId
    // dst += threadIdx.y * BLOCK_SIZE_DIV4 * dst_ldm + threadIdx.x;
    // src_ldm *= 8;
    // dst_ldm *= 8;
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++)
    {
        int tid = threadIdx.x ^((((threadIdx.y * BLOCK_SIZE_DIV8 + i) &3)<<3));
        
        *(dst + i * dst_ldm + tid) = *(src + i * src_ldm + tid);
    }
}

static __device__ void load_A_and_B(half* _A, half* _B, half* As, half* Bs, int K, int N)
{
    int loadAB_2 = threadIdx.y % 2;  // 0-1 一组，2-3一组 // 2
    int loadAB = threadIdx.y / 2;        // 分成两组，0: load A, 1: load B

    half* src = loadAB == 0 ? _A : _B;

    size_t dst_jump = loadAB == 0 ? 0 : SUB_BLOCK_SIZE;
    half* dst = As + dst_jump;

    size_t b_strip = loadAB == 0 ? 1 : N; // Block strip
    size_t k_strip = loadAB == 0 ? K : N; // inner BLOCK_SIZE strip
    size_t k_strip_DIV2 = k_strip / 2; // inner BLOCK_SIZE strip

    int* src_i = (int*)(src + loadAB_2 * BLOCK_SIZE_DIV2 * k_strip + threadIdx.x * 2);
    int* dst_i = (int*)(dst + loadAB_2 * BLOCK_SIZE_DIV2 * BLOCK_SKEW_SIZE + threadIdx.x * 2);

    for (int i = 0; i < BLOCK_SIZE_DIV2; i++)
    {
        *(dst_i + i * BLOCK_SKEW_SIZE_DIV2) = src_i[i * k_strip_DIV2];
    }
}

static __device__ void compute(half* As, half* Bs, int loadAB, int loadABWarpId, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>* a, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>* b, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>* c)
{
    half* As_wmma = As + loadAB * BLOCK_SKEW_SIZE * BLOCK_TILE_ROW_STRIP;
    half* Bs_wmma = Bs + loadABWarpId * BLOCK_TILE_COL_STRIP;

    for (int k = 0; k < K_TILES; k++)
    {
        // Load A and B
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; i++)
        {
            wmma::load_matrix_sync(a[i], As_wmma + i * WMMA_M * BLOCK_SKEW_SIZE + k * WMMA_K, BLOCK_SKEW_SIZE);
        }

        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++)
        {
            wmma::load_matrix_sync(b[i], Bs_wmma + k * WMMA_K * BLOCK_SKEW_SIZE + i * WMMA_N, BLOCK_SKEW_SIZE);
        }

        // Compute
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; i++)
        {
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++)
            {
                wmma::mma_sync(c[i * WARP_COL_TILES + j], a[i], b[j], c[i * WARP_COL_TILES + j]);
            }
        }
    }
}

// Kernel: Tensor Core GEMM using WMMA API
static __global__ void gemm_wmma_kernel_fp16_v6(size_t M, size_t N, size_t K, 
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta)
{
    extern __shared__ half SMEM[];

    half* As_0 = SMEM;
    half *Bs_0 = As_0 + SUB_BLOCK_SIZE;
    half *As_1 = Bs_0 + SUB_BLOCK_SIZE;
    half *Bs_1 = As_1 + SUB_BLOCK_SIZE;
    half *Cs = SMEM;

    int M_TILES = M / BLOCK_SIZE;
    int N_TILES = N / BLOCK_SIZE;

    int K_block_tile = K / BLOCK_SIZE; // K需要计算的次数

    // 首先确定当前属于哪个block，然后确定当前属于哪个warp
    int warpLaneId = threadIdx.x;   // 32
    int warpId = threadIdx.y;       // 4
    int loadABWarpId = warpId % 4;  // 0-3 一组，4-7一组 // 2
    int loadAB = warpId / 4;        // 分成两组，0: load A, 1: load B

    int blockIdm = blockIdx.y;
    int blockIdn = blockIdx.x;

    size_t K_div2 = K / 2;
    size_t N_div2 = N / 2;

    size_t blockM = blockIdm * BLOCK_SIZE;
    size_t blockN = blockIdn * BLOCK_SIZE;

    // load C: 方案2，用int指针，int = 2 x half，一行宽度64个half，对应32个int，正好一个warp一次读一行
    // TODO 尝试错开warp，不同warp之间读取数据差距不太大。
    int* Csub = (int *)(C + (BLOCK_SIZE_DIV8 * warpId + blockM) * N + blockN) + warpLaneId;
    int* Cs_i = (int *)(Cs + warpId * BLOCK_SIZE_DIV8 * BLOCK_SKEW_SIZE) + warpLaneId;
    const size_t N_DIV2 = N / 2;
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++)
    {
        *(Cs_i + BLOCK_SKEW_SIZE_DIV2 * i) = *(Csub + i * N_DIV2);
    }

    __syncthreads();

    // 将C加载到wmma中
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_wmma[WARP_ROW_TILES * WARP_COL_TILES]; // 2x1

    // 跳转到内部的block tile对应的位置
    half* Csub_0 = Cs + loadAB * BLOCK_TILE_ROW_STRIP * BLOCK_SKEW_SIZE + loadABWarpId * BLOCK_TILE_COL_STRIP;
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            // 跳转到内部sub block tile 对应的位置
            half* Csub_wmma = Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N;
            wmma::load_matrix_sync(c_wmma[i * WARP_COL_TILES + j], Csub_wmma, BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
#pragma unroll
            for (int t = 0; t < c_wmma[i * WARP_COL_TILES + j].num_elements; t++) 
            {
                c_wmma[i * WARP_COL_TILES + j].x[t] *= beta;
            }
        }
    }

    /*
    将warp 分成两组，0组load A，1 组load B
    */
    int idx = 0; // flag for double buffer

    // load A and B to buffer 0
    half* A_ptr = A + blockM * K;
    half* B_ptr = B + blockN;

    // size_t As_offset = 2 * idx * SUB_BLOCK_SIZE;
    // size_t Bs_offset = 2 * idx * SUB_BLOCK_SIZE;
    // half* As = As_0 + As_offset;
    // half* Bs = Bs_0 + Bs_offset;
    // load A and B
    load_AB(A_ptr, As_0, K_div2, BLOCK_SKEW_SIZE_DIV2);
    load_AB(B_ptr, Bs_0, N_div2, BLOCK_SKEW_SIZE_DIV2);

    // load_A_and_B(A_ptr, B_ptr, As_0, Bs_0, K, N);

    __syncthreads();

    // // load As and Bs to wmma tensor core register ? 
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_COL_TILES];

    idx ^= 1; // switch double buffer
    int k = 1;

    do
    {
        // load A and B to buffer 1
        if (k < K_block_tile)
        {
            half* A_ptr_1 = A_ptr + k * BLOCK_SIZE;
            half* B_ptr_1 = B_ptr + k * BLOCK_SIZE * N;

            // As_offset = 2 * idx * SUB_BLOCK_SIZE;
            // Bs_offset = 2 * idx * SUB_BLOCK_SIZE;
            // As = As_0 + As_offset;
            // Bs = Bs_0 + Bs_offset;
            // load A and B
            load_AB(A_ptr_1, As_1, K_div2, BLOCK_SKEW_SIZE_DIV2);
            load_AB(B_ptr_1, Bs_1, N_div2, BLOCK_SKEW_SIZE_DIV2);

            // load_A_and_B(A_ptr_1, B_ptr_1, As_1, Bs_1, K, N);

            k += 1;
        }

        // __syncthreads();
        compute(As_0, Bs_0, loadAB, loadABWarpId, a, b, c_wmma);
        __syncthreads();
        // load A and B to buffer 0
        if (k < K_block_tile)
        {
            half* A_ptr_1 = A_ptr + k * BLOCK_SIZE;
            half* B_ptr_1 = B_ptr + k * BLOCK_SIZE * N;
        
            // load A and B
            load_AB(A_ptr_1, As_0, K_div2, BLOCK_SKEW_SIZE_DIV2);
            load_AB(B_ptr_1, Bs_0, N_div2, BLOCK_SKEW_SIZE_DIV2);

            // load_A_and_B(A_ptr_1, B_ptr_1, As_0, Bs_0, K, N);

            k += 1;
            // idx ^= 1; // switch double buffer
            // __syncthreads();
        }

        compute(As_1, Bs_1, loadAB, loadABWarpId, a, b, c_wmma);
        __syncthreads();
    } while (k < K_block_tile);

    // store c_wmma to Cs
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            wmma::store_matrix_sync(Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N, c_wmma[i * WARP_COL_TILES + j], BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 方案2:store C
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++) 
    {
        *(Csub + i * N_DIV2) = *(Cs_i + BLOCK_SKEW_SIZE_DIV2 * i);
        // *(Csub + i * N_DIV2) = *(Cs_i2 + BLOCK_SKEW_SIZE_DIV2 * i);
    }
}

template <typename T>
void cuda_gemm_wmma_7(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

// GEMM API: 调用 Kernel 并管理 Stream
template <>
void cuda_gemm_wmma_7<__half>(size_t M, size_t N, size_t K,
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta, cudaStream_t stream) {
    dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    size_t value = DIV_UP(SUB_BLOCK_SIZE * 4 * sizeof(__half), 128) * 128;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_wmma_kernel_fp16_v6,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        value));
    // 如果直接调用，会报错，shared memory 超过最大
    // cudaFuncSetAttribute(gemm_wmma_kernel_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, value);

    gemm_wmma_kernel_fp16_v6<<<gridDim, blockDim, value, stream>>>(M, N, K, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_wmma_7<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
{
    // // Allocate device memory for half-precision matrices
    // half* d_A, * d_B;
    // cudaMalloc(&d_A, m * k * sizeof(half));
    // cudaMalloc(&d_B, k * n * sizeof(half));

    // // Convert A, B from float to half
    // dim3 blockDim(1024);
    // dim3 gridDim((m * k + blockDim.x - 1) / blockDim.x);
    // convert_float_to_half << <gridDim, blockDim, 0, stream >> > (A, d_A, m * k);
    // gridDim = dim3((k * n + blockDim.x - 1) / blockDim.x);
    // convert_float_to_half << <gridDim, blockDim, 0, stream >> > (B, d_B, k * n);

    // // Define grid and block dimensions
    // dim3 blockDimWMMA(16, 16);
    // dim3 gridDimWMMA((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);

    // // Launch WMMA kernel
    // wmma_gemm_kernel << <gridDimWMMA, blockDimWMMA, 0, stream >> > (m, n, k, d_A, d_B, C, alpha, beta);

    // // Cleanup
    // cudaFree(d_A);
    // cudaFree(d_B);
    CHECK_LAST_CUDA_ERROR();
}

template GEMM_EXPORT void cuda_gemm_wmma_7<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_wmma_7<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

