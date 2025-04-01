#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

/*
在9的基础上，使用ldmatrix来加载数据，并且使用wmma来计算数据。
性能从48%下降到37%。

*/

#define BLOCK_SIZE 64
#define BLOCK_SIZE_DIV2 32
#define BLOCK_SIZE_DIV4 16

#define WARP_SIZE 32

#define WMMA_M 16  // 16x16x16 矩阵乘法
#define WMMA_N 16
#define WMMA_K 16

/*
一个SM有4个warp，每个warp有32个线程
C输出块大小为64x64，划分为4个块，每块大小为32x32，每块由一个warp计算输出。

对于每个线程，都需要申请一个2x2的空间，用来存储C的wmma fragment。
*/

static const int BLOCK_ROW_WARPS = 2;
static const int BLOCK_COL_WARPS = 2;

static const int WARP_ROW_TILES = 2;
static const int WARP_COL_TILES = 2;

static const int BLOCK_TILE_ROW_STRIP = WARP_ROW_TILES * WMMA_M; // 32
static const int BLOCK_TILE_COL_STRIP = WARP_COL_TILES * WMMA_N; // 32

static const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 4
static const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS); // 4

static const int SHMEM_STRIDE = (WMMA_N * BLOCK_ROW_TILES); // 64

static const int SKEW_HALF = 8;
static const int K_TILES = BLOCK_SIZE / WMMA_K; // 4

static const int BLOCK_SKEW_SIZE = BLOCK_SIZE + SKEW_HALF;
static const int BLOCK_SKEW_SIZE_DIV2 = BLOCK_SKEW_SIZE / 2;

static const int SUB_BLOCK_SIZE = DIV_UP(BLOCK_SIZE * BLOCK_SKEW_SIZE, 128) * 128;
static const int LDM_NUM = 4*32/8; // 4个warp，每个warp处理4行,用在loadAB中。
static const int LOOP_NUM = BLOCK_SIZE / LDM_NUM; // LoadAB中，for loop的循环次数，用在loadAB中。

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

// 函数将A或B从global mem加载到shared memory中，需要给定src和dst的block大小，以及src和dst的leading dimension
// 这个函数由4个warp一起执行，将A或B加载到shared memory中
// 一个thread 处理int4, 一需要8个thread.
// 每个load AB 分配4 x 32 个thread，一行需要8个thread，一次处理16行。总共64行，需要处理4次
static __device__ void load_AB(half* _src, half* _dst, int src_ldm, int dst_ldm)
{
    int t = threadIdx.x;
    int t_div8 = t / 8;            // 16 个
    int t_mod8 = t - t_div8 * 8; // thread 内部的 8个thread

    int* src = (int*)(_src + (threadIdx.y * 16 + t_div8) * src_ldm * 2 + t_mod8 * 8);
    int* dst = (int*)(_dst + (threadIdx.y * 16 + t_div8) * dst_ldm * 2 + t_mod8 * 8);
    // A 和 B 在shared memory中的长宽是64x64的half，换成int读取时，一行也就是32，正好对应一个warp
    // src += threadIdx.y * BLOCK_SIZE_DIV4 * src_ldm + threadIdx.x; // threadIdx.x = warpLaneId, threadIdx.y = warpId
    // dst += threadIdx.y * BLOCK_SIZE_DIV4 * dst_ldm + threadIdx.x;
    int src_ldm2 = src_ldm * 4;
    int dst_ldm2 = dst_ldm * 4;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        // *dst = *src;
        uint32_t dst_i = __cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst_i), "l"(src), "n"(16));

        src += src_ldm2;
        dst += dst_ldm2;
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

static __device__ void compute(half* As, half* Bs, int loadAB, int loadABWarpId, uint32_t* a_frag, uint32_t* b_frag, uint32_t* c_frag, int warp8, int warp4_row, int warp4_col)
{
    half* As_wmma = As + loadAB * BLOCK_SKEW_SIZE * BLOCK_TILE_ROW_STRIP;
    half* Bs_wmma = Bs + loadABWarpId * BLOCK_TILE_COL_STRIP;

    for (int k = 0; k < K_TILES; k++)
    {
        // Load A and B
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; i++)
        {
            half* ptr = As_wmma + i * WMMA_M * BLOCK_SKEW_SIZE + k * WMMA_K + (warp8 + warp4_row * 8) * BLOCK_SKEW_SIZE + warp4_col * 8;
            uint32_t smem_ptr;
            asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
                "%0, smem_ptr; }\n"
                : "=r"(smem_ptr)
                : "l"(ptr));

            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[i * 4]), "=r"(a_frag[i * 4 + 1]), "=r"(a_frag[i * 4 + 2]), "=r"(a_frag[i * 4 + 3])
                : "r"(smem_ptr));
            // wmma::load_matrix_sync(a[i], As_wmma + i * WMMA_M * BLOCK_SKEW_SIZE + k * WMMA_K, BLOCK_SKEW_SIZE);
        }

        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++)
        {
            half* ptr = Bs_wmma + k * WMMA_K * BLOCK_SKEW_SIZE + i * WMMA_N + (warp8 + warp4_row * 8) * BLOCK_SKEW_SIZE + warp4_col * 8;
            uint32_t smem_ptr;
            asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
                "%0, smem_ptr; }\n"
                : "=r"(smem_ptr)
                : "l"(ptr));

            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[i * 4]), "=r"(b_frag[i * 4 + 1]), "=r"(b_frag[i * 4 + 2]), "=r"(b_frag[i * 4 + 3])
                : "r"(smem_ptr));

            // wmma::load_matrix_sync(b[i], Bs_wmma + k * WMMA_K * BLOCK_SKEW_SIZE + i * WMMA_N, BLOCK_SKEW_SIZE);
        }

        // Compute
        #pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; i++)
        {
            #pragma unroll
            for (int j = 0; j < WARP_COL_TILES; j++)
            {
                // wmma::mma_sync(c[i * WARP_COL_TILES + j], a[i], b[j], c[i * WARP_COL_TILES + j]);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                    : "+r"(c_frag[i * WARP_COL_TILES * 4 + j * 4]), "+r"(c_frag[i * WARP_COL_TILES * 4 + j * 4 + 2])//, "+f"(c_frag_f[i * WARP_COL_TILES * 4 + j * 4 + 2]), "+f"(c_frag_f[i * WARP_COL_TILES * 4 + j * 4 + 3])
                    : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]), 
                    "r"(b_frag[j * 4]), "r"(b_frag[j * 4 + 2])
                    );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                    : "+r"(c_frag[i * WARP_COL_TILES * 4 + j * 4 + 1]), "+r"(c_frag[i * WARP_COL_TILES * 4 + j * 4 + 3])
                    : "r"(a_frag[i * 4]), "r"(a_frag[i * 4 + 1]), "r"(a_frag[i * 4 + 2]), "r"(a_frag[i * 4 + 3]),
                    "r"(b_frag[j * 4 + 1]), "r"(b_frag[j * 4 + 3])
                    );

            }
        }
    }
}

// Kernel: Tensor Core GEMM using WMMA API
static __global__ void gemm_wmma_kernel_fp16_v10(size_t M, size_t N, size_t K, 
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
    int loadABWarpId = warpId % 2;  // 0-1 一组，2-3一组 // 2
    int loadAB = warpId / 2;        // 分成两组，0: load A, 1: load B

    // 继续将一个warp中的thread分成2x2的形式。对应16x16的小块，用来指引ldmatrix 的行起始位置的地址
    int warp4 = warpLaneId/8;  // 将warp 分成4分
    int warp8 = warpLaneId%8;
    int warp4_col = warp4 % 2; // 根据warp的id，确定获取哪个8x8的头。
    int warp4_row = warp4 / 2;

    int warp_div4 = warpLaneId / 4;
    int warp_div44 = warpLaneId % 4;

    int blockIdm = blockIdx.y;
    int blockIdn = blockIdx.x;

    size_t K_div2 = K / 2;
    size_t N_div2 = N / 2;

    size_t blockM = blockIdm * BLOCK_SIZE;
    size_t blockN = blockIdn * BLOCK_SIZE;

    // load C: 方案2，用int指针，int = 2 x half，一行宽度64个half，对应32个int，正好一个warp一次读一行
    // TODO 尝试错开warp，不同warp之间读取数据差距不太大。
    int* Csub = (int *)(C + (BLOCK_SIZE_DIV4 * warpId + blockM) * N + blockN) + warpLaneId;
    int* Cs_i = (int *)(Cs + warpId * BLOCK_SIZE_DIV4 * BLOCK_SKEW_SIZE) + warpLaneId;
    const size_t N_DIV2 = N / 2;
    for (int i = 0; i < BLOCK_SIZE_DIV4; i++)
    {
        *(Cs_i + BLOCK_SKEW_SIZE_DIV2 * i) = *(Csub + i * N_DIV2);
    }

    __syncthreads();

    // 将C加载到wmma中
    uint32_t c_frag[WARP_ROW_TILES * WARP_COL_TILES * 4];
    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_wmma[WARP_ROW_TILES * WARP_COL_TILES]; // 2x2

    // 跳转到内部的block tile对应的位置
    half* Csub_0 = Cs + loadAB * BLOCK_TILE_ROW_STRIP * BLOCK_SKEW_SIZE + loadABWarpId * BLOCK_TILE_COL_STRIP;
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {
            // 跳转到内部sub block tile 对应的位置
            half* ptr = Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N;

            uint32_t smem_ptr;
            asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
                "%0, smem_ptr; }\n"
                : "=r"(smem_ptr)
                : "l"(ptr));

            int ij = i * WARP_COL_TILES * 4;
            asm volatile(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(c_frag[ij + j * 4]), "=r"(c_frag[ij + j * 4 + 1]), "=r"(c_frag[ij + j * 4 + 2]), "=r"(c_frag[ij + j * 4 + 3])
                : "r"(smem_ptr));
            // wmma::load_matrix_sync(c_wmma[i * WARP_COL_TILES + j], Csub_wmma, BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    // Scale the C matrix.
    half* c_half = (half*)c_frag;
    int c_size = WARP_ROW_TILES * WARP_COL_TILES * 4;
#pragma unroll
    for (int i = 0; i < c_size; i++) {
        c_half[i] *= beta;
    }

// #pragma unroll
//     for (int i = 0; i < WARP_ROW_TILES; i++) {
// #pragma unroll
//         for (int j = 0; j < WARP_COL_TILES; j++) {
// #pragma unroll
//             for (int t = 0; t < c_wmma[i * WARP_COL_TILES + j].num_elements; t++) 
//             {
//                 c_wmma[i * WARP_COL_TILES + j].x[t] *= beta;
//             }
//         }
//     }

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
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();

    // // load As and Bs to wmma tensor core register ? 
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_ROW_TILES];
    // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_COL_TILES];
    
    uint32_t a[WARP_ROW_TILES * 4];
    uint32_t b[WARP_COL_TILES * 4];

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
        // CP_ASYNC_COMMIT_GROUP();
        // CP_ASYNC_WAIT_GROUP(0);
        // __syncthreads();
        compute(As_0, Bs_0, loadAB, loadABWarpId, a, b, c_frag, warp8, warp4_row, warp4_col);

        // __syncthreads();
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
            __syncthreads();
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        compute(As_1, Bs_1, loadAB, loadABWarpId, a, b, c_frag, warp8, warp4_row, warp4_col);
        // __syncthreads();
    } while (k < K_block_tile);

    // store c_wmma to Cs
    for (int i = 0; i < WARP_ROW_TILES; i++)
    {
        for (int j = 0; j < WARP_COL_TILES; j++)
        {

            unsigned int* ptr = (unsigned int *)(Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N) + warp_div4 * BLOCK_SKEW_SIZE_DIV2 + warp_div44;
            int innder = i * WARP_COL_TILES * 4;
            *ptr = c_frag[innder + j * 4];
            *(ptr + 4) = c_frag[innder + j * 4 + 1];
            *(ptr + 8 * BLOCK_SKEW_SIZE_DIV2) = c_frag[innder + j * 4 + 2];
            *(ptr + 8 * BLOCK_SKEW_SIZE_DIV2 + 4) = c_frag[innder + j * 4 + 3];

            // wmma::store_matrix_sync(Csub_0 + i * WMMA_M * BLOCK_SKEW_SIZE + j * WMMA_N, c_wmma[i * WARP_COL_TILES + j], BLOCK_SKEW_SIZE, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // 方案2:store C
    for (int i = 0; i < BLOCK_SIZE_DIV4; i++) 
    {
        *(Csub + i * N_DIV2) = *(Cs_i + BLOCK_SKEW_SIZE_DIV2 * i);
        // *(Csub + i * N_DIV2) = *(Cs_i2 + BLOCK_SKEW_SIZE_DIV2 * i);
    }
}

template <typename T>
void cuda_gemm_wmma_10(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
{
    std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
}

// GEMM API: 调用 Kernel 并管理 Stream
template <>
void cuda_gemm_wmma_10<__half>(size_t M, size_t N, size_t K,
    __half *A, __half *B, __half *C, 
    __half alpha, __half beta, cudaStream_t stream) {
    dim3 blockDim(32, 4, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    size_t value = DIV_UP(SUB_BLOCK_SIZE * 4 * sizeof(__half), 128) * 128;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_wmma_kernel_fp16_v10,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        value));
    // 如果直接调用，会报错，shared memory 超过最大
    // cudaFuncSetAttribute(gemm_wmma_kernel_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, value);

    gemm_wmma_kernel_fp16_v10<<<gridDim, blockDim, value, stream>>>(M, N, K, A, B, C, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

template <>
void cuda_gemm_wmma_10<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
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

template GEMM_EXPORT void cuda_gemm_wmma_10<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
template GEMM_EXPORT void cuda_gemm_wmma_10<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

