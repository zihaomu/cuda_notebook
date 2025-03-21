#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

/*
优化思路：在上一个版本基础上，尝试解决bank conflict。
问题分析：128x128的缓存中，刚好每一行都是32的倍数，下一行每个元素和上一行同一列的元素处于同一个bank上，这就会导致bank conflict。
解决方案：将128x128的块划分成128x(128 + 8)的块，这样就会错开同一列元素在同一个bank上的问题。
性能从36%提升到75%，提升非常大。
*/

#define BLOCK_SIZE 128
#define BLOCK_SIZE_DIV2 64 //128/4
#define BLOCK_SIZE_DIV4 32 //128/4
#define BLOCK_SIZE_DIV8 16 //128/4

#define WARP_SIZE 32

#define WMMA_M 16  // 16x16x16 矩阵乘法
#define WMMA_N 16
#define WMMA_K 16

/*
将 128x128 的块划分成 2x4 的block tile块，每个块一个warp，每块大小是：64x32
进一步将 64x32 的块划分成 4x2 的sub block tile块，每个块一个warp，每块大小是：16x16
8个warp在计算时，也被编排成2x4的组合。
其中，要申请C的wmma fragment大小和block tile块一致，2x4 fragment空间，每个空间被一个warp使用。
*/

static const int BLOCK_ROW_WARPS = 2;
static const int BLOCK_COL_WARPS = 4;

static const int WARP_ROW_TILES = 4;
static const int WARP_COL_TILES = 2;

static const int BLOCK_TILE_ROW_STRIP = WARP_ROW_TILES * WMMA_M; // 64
static const int BLOCK_TILE_COL_STRIP = WARP_COL_TILES * WMMA_N; // 32

static const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 8
static const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS); // 8

static const int SHMEM_STRIDE = (WMMA_N * BLOCK_ROW_TILES); // 128

static const int K_TILES = BLOCK_SIZE / WMMA_K; // 8

static const int SKEW_HALF = 8;

static const int BLOCK_SKEW_SIZE = BLOCK_SIZE + SKEW_HALF;
static const int BLOCK_SKEW_SIZE_DIV4 = BLOCK_SKEW_SIZE / 4;
static const int BLOCK_SKEW_SIZE_DIV2 = BLOCK_SKEW_SIZE / 2;

// Kernel: Tensor Core GEMM using WMMA API
static __global__ void gemm_wmma_kernel_fp16(size_t M, size_t N, size_t K, 
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta) 
{
    int M_TILES = M / BLOCK_SIZE;
    int N_TILES = N / BLOCK_SIZE;

    // 首先确定当前属于哪个block，然后确定当前属于哪个warp
    int warpLaneId = threadIdx.x;   // 32
    int warpId = threadIdx.y;       // 8
    int loadABWarpId = warpId % 4;  // 0-3 一组，4-7一组 // 4
    int loadAB = warpId / 4;  // 分成两组，0: load A, 1: load B, // 2 

    int blockIdm = blockIdx.y;
    int blockIdn = blockIdx.x;

    size_t K_div = K / 2;

    extern __shared__ half SMEM[];

    half* As = SMEM;
    half *Bs = As + BLOCK_SIZE * BLOCK_SKEW_SIZE;
    half *Cs = SMEM;

    size_t blockM = blockIdm * BLOCK_SIZE;
    size_t blockN = blockIdn * BLOCK_SIZE;

    // load C: 方案2，用int2指针，int2 = 4 x half，32个线程，每个线程处理4个half，一次128个，刚好一行
    int2* Csub = (int2 *)(C + (BLOCK_SIZE_DIV8 * warpId + blockM) * N + blockN) + warpLaneId;
    int2* Cs_i2 = (int2 *)(Cs + warpId * BLOCK_SIZE_DIV8 * BLOCK_SKEW_SIZE) + warpLaneId;
    const size_t N_DIV4 = N / 4;
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++) 
    {
        *(Cs_i2 + BLOCK_SKEW_SIZE_DIV4 * i) = *(Csub + i * N_DIV4);
    }

    __syncthreads();

    // 将C加载到wmma中
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_wmma[WARP_ROW_TILES][WARP_COL_TILES]; // 4x2

    // 跳转到内部的block tile对应的位置
    half* Csub_0 = Cs + loadAB * BLOCK_TILE_ROW_STRIP * BLOCK_SKEW_SIZE + loadABWarpId * BLOCK_TILE_COL_STRIP;
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            // 跳转到内部sub block tile 对应的位置
            half* Csub_wmma = Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N;
            wmma::load_matrix_sync(c_wmma[i][j], Csub_wmma, BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    // scale C
// Scale the C matrix.

#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++) {
#pragma unroll
            for (int t = 0; t < c_wmma[i][j].num_elements; t++) {
                c_wmma[i][j].x[t] *= beta;
            }
        }
    }
    /*
    将warp 分成两组，0组load A，1 组load B
    */

    size_t dst_jump = threadIdx.y < 4 ? 0 : BLOCK_SIZE * BLOCK_SKEW_SIZE;
    half* src_ptr = loadAB == 0 ? A + blockM * K : B + blockN;
    half* dst_ptr = As + dst_jump;

    size_t b_strip = loadAB == 0 ? 1 : N; // Block strip
    size_t k_strip = loadAB == 0 ? K : N; // inner BLOCK_SIZE strip
    size_t k_strip_DIV4 = k_strip / 4; // inner BLOCK_SIZE strip
    size_t k_strip_DIV2 = k_strip / 2; // inner BLOCK_SIZE strip

    /*
    对于长度为128x128的half，转换为64x64的int，两个warp正好一次处理一行。
    在将8个warp划分为A和B两组后，一组有4个warp，进一步，再划分两小组，每小组2个warp一次能够处理一行。
    */

    // 替换成float 解决bank conflict

    int loadABWarpId_DIV2 = loadABWarpId % 2; // 将4个warp划分为两组，每组2个warp
    int loadABWarpId_DIV22 = loadABWarpId / 2; // 将4个warp划分为两组，每组2个warp
    src_ptr += loadABWarpId_DIV22 * BLOCK_SIZE_DIV2 * k_strip + (loadABWarpId_DIV2 * WARP_SIZE + warpLaneId) * 2;
    dst_ptr += loadABWarpId_DIV22 * BLOCK_SIZE_DIV2 * BLOCK_SKEW_SIZE + (loadABWarpId_DIV2 * WARP_SIZE + warpLaneId) * 2;

    int* src_ptr_i = reinterpret_cast<int*>(src_ptr);
    int* dst_ptr_i = reinterpret_cast<int*>(dst_ptr);

    // src_ptr_i += loadABWarpId_DIV22 * 64 * k_strip_DIV2 + loadABWarpId_DIV2 * WARP_SIZE + warpLaneId;
    // dst_ptr_i += loadABWarpId_DIV22 * 64 * BLOCK_SKEW_SIZE_DIV2 + loadABWarpId_DIV2 * WARP_SIZE + warpLaneId;

    for (int b = 0; b < K; b += BLOCK_SIZE)
    {
        // load A and B
        int* src_ptr_b = src_ptr_i + b * b_strip / 2;
        int* dst_ptr_b = dst_ptr_i;
        for (int k = 0; k < 64; k++)
        {
            *(dst_ptr_b + k * BLOCK_SKEW_SIZE_DIV2) = src_ptr_b[k * k_strip_DIV2];
        }

        __syncthreads();

        // 使用wmma,能够达到23%的性能提升
        /*
        思路：将128x128的块，划分成2x4的块，每一块一个warp。
        每一块内部，进一步划分成4x2的16x16的小块，每个小块刚好满足一个warp的计算。
        */
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_wmma[WARP_ROW_TILES];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_wmma[WARP_COL_TILES];

        half* As_wmma = As + loadAB * BLOCK_SKEW_SIZE * BLOCK_TILE_ROW_STRIP;
        half* Bs_wmma = Bs + loadABWarpId * BLOCK_TILE_COL_STRIP;

        for (int k = 0; k < K_TILES; k++)
        {
            // Load A and B
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++)
            {
                wmma::load_matrix_sync(a_wmma[i], As_wmma + i * WMMA_M * BLOCK_SKEW_SIZE + k * WMMA_K, BLOCK_SKEW_SIZE);
            }

            #pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++)
            {
                wmma::load_matrix_sync(b_wmma[i], Bs_wmma + k * WMMA_K * BLOCK_SKEW_SIZE + i * WMMA_N, BLOCK_SKEW_SIZE);
            }

            // Compute
            #pragma unroll
            for (int i = 0; i < WARP_ROW_TILES; i++)
            {
                #pragma unroll
                for (int j = 0; j < WARP_COL_TILES; j++)
                {
                    wmma::mma_sync(c_wmma[i][j], a_wmma[i], b_wmma[j], c_wmma[i][j]);
                }
            }
        }

        // __syncthreads();
    }

    // store c_wmma to Cs
    #pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        #pragma unroll
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            wmma::store_matrix_sync(Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N, c_wmma[i][j], BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    __syncthreads();
    
    // 方案2:store C
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++) 
    {
        *(Csub + i * N_DIV4) = *(Cs_i2 + BLOCK_SKEW_SIZE_DIV4 * i);
    }
}

template <typename T>
void cuda_gemm_wmma_4(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

// GEMM API: 调用 Kernel 并管理 Stream
template <>
void cuda_gemm_wmma_4<__half>(size_t M, size_t N, size_t K,
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta, cudaStream_t stream) {
    dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    size_t value = DIV_UP(BLOCK_SIZE * BLOCK_SKEW_SIZE * 2 * sizeof(__half), 64) * 64;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_wmma_kernel_fp16,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        value));
    // 如果直接调用，会报错，shared memory 超过最大
    // cudaFuncSetAttribute(gemm_wmma_kernel_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, value);

    gemm_wmma_kernel_fp16<<<gridDim, blockDim, value, stream>>>(M, N, K, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_wmma_4<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
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

template GEMM_EXPORT void cuda_gemm_wmma_4<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_wmma_4<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

