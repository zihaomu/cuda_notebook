#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

/*
优化思路：相比于上一个，本方案尝试在load C和store C的时候，使用int2指针，每个线程处理4个half，一次处理128个half，刚好一行。
另一个优化点是，本方案只申请了2个128x128的shared memory分别存储C和 A，B。
原因是，C从global memory加载到C shared memory之后，C shared memory的空间只有在最后写回才会需要用到，在计算时，就可以交个A和B来使用。计算完成刚好A和B的shared memory缓存也不需要了。
最后优化结果，从31%性能提升到36%，提升的不是很多。
*/

#define BLOCK_SIZE 128
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
    half *Bs = As + BLOCK_SIZE * BLOCK_SIZE;
    half *Cs = SMEM;

    size_t blockM = blockIdm * BLOCK_SIZE;
    size_t blockN = blockIdn * BLOCK_SIZE;

    // load C：方案1
    // half *Csub = C + blockM * N + blockN;
    // for (int i = warpId; i < BLOCK_SIZE; i += 8) {
    //     for (int j = warpLaneId; j < BLOCK_SIZE; j += 32) 
    //     {
    //         *(Cs + BLOCK_SIZE * i + j) = beta * Csub[i * N + j];
    //     }
    // }

    // load C: 方案2，用int2指针，int2 = 4 x half，32个线程，每个线程处理4个half，一次128个，刚好一行
    // C总共有128x128， 一个warp一次处理一行，一共8个warp，将128行分成8个组，每个组处理16行
    int2* Csub = (int2 *)(C + (BLOCK_SIZE_DIV8 * warpId + blockM) * N + blockN) + warpLaneId;
    int2* Cs_i2 = (int2 *)(Cs + warpId * BLOCK_SIZE_DIV8 * BLOCK_SIZE) + warpLaneId;
    size_t N_DIV4 = N / 4;
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++) 
    {
        *(Cs_i2 + BLOCK_SIZE_DIV4 * i) = *(Csub + i * N_DIV4);
    }

    __syncthreads();

    // 将C加载到wmma中
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_wmma[WARP_ROW_TILES][WARP_COL_TILES]; // 4x2

    // 跳转到内部的block tile对应的位置
    half* Csub_0 = Cs + loadAB * BLOCK_TILE_ROW_STRIP * BLOCK_SIZE + loadABWarpId * BLOCK_TILE_COL_STRIP;
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            // 跳转到内部sub block tile 对应的位置
            half* Csub_wmma = Csub_0 + i * WMMA_M * BLOCK_SIZE + j * WMMA_N;
            wmma::load_matrix_sync(c_wmma[i][j], Csub_wmma, BLOCK_SIZE, wmma::mem_row_major);
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

    size_t dst_jump = threadIdx.y < 4 ? 0 : BLOCK_SIZE * BLOCK_SIZE;
    half* src_ptr = loadAB == 0 ? A + blockM * K : B + blockN;
    half* dst_ptr = As + dst_jump;

    size_t b_strip = loadAB == 0 ? 1 : N; // Block strip
    size_t k_strip = loadAB == 0 ? K : N; // inner BLOCK_SIZE strip
    size_t k_strip_DIV4 = k_strip / 4; // inner BLOCK_SIZE strip

    /*
    每4个warp处理一个128x128的块，将指针转换为float2,128长度的half转换为32长度的float2，正好一个warp处理一行。
    */

    // 替换成float 解决bank conflict
    float2* src_ptr_f2 = reinterpret_cast<float2*>(src_ptr);
    float2* dst_ptr_f2 = reinterpret_cast<float2*>(dst_ptr);

    src_ptr_f2 += loadABWarpId * BLOCK_SIZE_DIV4 * k_strip_DIV4 + warpLaneId;
    dst_ptr_f2 += loadABWarpId * BLOCK_SIZE_DIV4 * BLOCK_SIZE_DIV4 + warpLaneId;

    for (int b = 0; b < K; b += BLOCK_SIZE)
    {
        // load A and B
        float2* src_ptr_b = src_ptr_f2 + b * b_strip / 4;
        float2* dst_ptr_b = dst_ptr_f2;
        for (int k = 0; k < BLOCK_SIZE_DIV4; k++)
        {

            *(dst_ptr_b + k * BLOCK_SIZE_DIV4) = src_ptr_b[k * k_strip_DIV4];
        }

        __syncthreads();

        // 使用wmma,能够达到23%的性能提升
        /*
        思路：将128x128的块，划分成2x4的块，每一块一个warp。
        每一块内部，进一步划分成4x2的16x16的小块，每个小块刚好满足一个warp的计算。
        */
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_wmma[WARP_ROW_TILES];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_wmma[WARP_COL_TILES];

        half* As_wmma = As + loadAB * BLOCK_SIZE * BLOCK_TILE_ROW_STRIP;
        half* Bs_wmma = Bs + loadABWarpId * BLOCK_TILE_COL_STRIP;

        for (int k = 0; k < K_TILES; k++)
        {
            // Load A and B
            for (int i = 0; i < WARP_ROW_TILES; i++)
            {
                wmma::load_matrix_sync(a_wmma[i], As_wmma + i * WMMA_M * BLOCK_SIZE + k * WMMA_K, BLOCK_SIZE);
            }

            for (int i = 0; i < WARP_COL_TILES; i++)
            {
                wmma::load_matrix_sync(b_wmma[i], Bs_wmma + k * WMMA_K * BLOCK_SIZE + i * WMMA_N, BLOCK_SIZE);
            }

            // Compute
            for (int i = 0; i < WARP_ROW_TILES; i++)
            {
                for (int j = 0; j < WARP_COL_TILES; j++)
                {
                    wmma::mma_sync(c_wmma[i][j], a_wmma[i], b_wmma[j], c_wmma[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // store c_wmma to Cs
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            wmma::store_matrix_sync(Csub_0 + i * WMMA_M * BLOCK_SIZE + j * WMMA_N, c_wmma[i][j], BLOCK_SIZE, wmma::mem_row_major);
        }
    }

    __syncthreads();
    
    // 方案1:store C
    // for (int i = warpId; i < BLOCK_SIZE; i += 8) 
    // {
    //     for (int j = warpLaneId; j < BLOCK_SIZE; j += 32) 
    //     {
    //         Csub[i * N + j] = Cs[i * BLOCK_SIZE + j];
    //     }
    // }

    // 方案2:store C
    for (int i = 0; i < BLOCK_SIZE_DIV8; i++) 
    {
        *(Csub + i * N_DIV4) = *(Cs_i2 + BLOCK_SIZE_DIV4 * i);
    }
}

template <typename T>
void cuda_gemm_wmma_3(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

// GEMM API: 调用 Kernel 并管理 Stream
template <>
void cuda_gemm_wmma_3<__half>(size_t M, size_t N, size_t K,
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta, cudaStream_t stream) {
    dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    size_t value = BLOCK_SIZE * BLOCK_SIZE * 2 * sizeof(__half);

    CUDA_CHECK(cudaFuncSetAttribute(gemm_wmma_kernel_fp16,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        value));
    // 如果直接调用，会报错，shared memory 超过最大
    // cudaFuncSetAttribute(gemm_wmma_kernel_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, value);

    gemm_wmma_kernel_fp16<<<gridDim, blockDim, value, stream>>>(M, N, K, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_wmma_3<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
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

template GEMM_EXPORT void cuda_gemm_wmma_3<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_wmma_3<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

