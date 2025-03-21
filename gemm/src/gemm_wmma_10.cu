// // /*
// //  * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
// //  *
// //  * Please refer to the NVIDIA end user license agreement (EULA) associated
// //  * with this source code for terms and conditions that govern your use of
// //  * this software. Any use, reproduction, disclosure, or distribution of
// //  * this software and related documentation outside the terms of the EULA
// //  * is strictly prohibited.
// //  *
// //  */

// // // CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// // // and Accumulate API introduced in CUDA 9.

// // // In this program, the compute_gemm kernel computes the result of a matrix multiplication
// // // and addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices
// // // are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x K_GLOBAL (row-major), the B matrix
// // // is K_GLOBAL x N_GLOBAL (column-major).
// // // In that kernel, each CTA computes one 128 x 128 tile of the resulting matrix
// // // per iteration. When the tile is computed, the CTA stores it to the global memory
// // // and begins a new iteration, selecting a new 128 x 128 tile to compute.
// // // Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes eight
// // // 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array.
// // // Warps compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// // // moving through the K_GLOBAL dimension of the A and B matrices and accumulating
// // // the intermediate result in the local thread state.

// // // There are a number of simple optimizations used in the algorithm:
// // // - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
// // //   shared memory. After that is done, each warp loads the C matrix fragments from
// // //   shared memory, thus avoiding a random global memory access.
// // // - On each internal iteration, the CTA copies a portion of the A and B matrices from
// // //   global memory to shared memory. After that, all warps in the CTA reuse the A and B
// // //   data from shared memory, thus reducing the number of data copies from global memory.
// // // - The portions of the A and B matrices are stored in shared memory with an additional
// // //   padding (skew) to reduce the number of shared memory access bank conflicts.
// // //   (See a detailed explanation near the SKEW_HALF macro definition.)
// // // - When the CTA finishes computing the tiles of the resulting matrix, each warp stores
// // //   its subtiles to shared memory. The CTA then copies the shared memory contents to
// // //   global memory, again avoiding redundant random global memory accesses.
// // // - Note that the CTA tile size is chosen to maximize the GPU register utilization,
// // //   but carefully enough to avoid local memory use.

// // /*
// // 算法输入：
// // A: MxK矩阵, row_major
// // B: KxN矩阵, col_major
// // C: MxN矩阵, row_major
// // alpha: 系数
// // beta: 系数
// // C = alpha * A * B + beta * C

// // CTA:Cooperative thread arrays，合作线程数组/合作线程阵列
// // 一个CTA由8个warp，32x8个线程组成，计算一个C矩阵(结果矩阵)中的128x128的小块。
// // 在一个128x128的块内，进一步划分成8份16x16x2x4块。对应8个warp，每个warp计算2x4x16x16的子块，组织成2x4的16x16的块。
// // 计算每个16x16的子块时，通过wmma::mma_sync操作，沿着A和B矩阵的K维度移动，将中间结果累积本地线程的缓存中，也就是每个线程需要一个2x4的缓存用来保存结果。

// // 相比于wmma_0中的实现，本算法有有以下优化：
// // 1. CTA将C矩阵的128x128块从全局内存复制到共享内存，然后每个warp从共享内存中加载C矩阵的片段，避免了随机全局内存访问。而wmma_0中每个warp直接从全局内存中加载C矩阵的片段。
// // 2. 在每个内部迭代中，CTA将A和B矩阵的一部分从全局内存复制到共享内存，然后CTA中的所有warp都从共享内存中重用A和B数据，从而减少了从全局内存中复制数据的次数。
// // 3. A和B矩阵的部分存储在共享内存中，带有额外的填充(skew)，以减少共享内存访问冲突?
// // 4. 在写回的时候，同样利用了shared memory，避免了冗余的随机全局内存访问。

// // 注意：CTA的大小被选择为最大化GPU寄存器利用率，需要小心使用，避免用超了使用本地内存。
// // */

// #include "cuda_gemm.hpp"
// #include "cuda_gemm_utils.cuh"

//  #include <cuda_fp16.h>
//  #include <mma.h>
//  #include <cuda_runtime.h>
//  #include <iostream>

// // using namespace nvcuda;

// // #ifndef WARP_SIZE
// // #define WARP_SIZE (32)
// // #endif // WARP_SIZE

// // #define WMMA_M 16  // 16x16x16 矩阵乘法
// // #define WMMA_N 16
// // #define WMMA_K 16

// // // MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32, 16) are
// // // currently supported.
// // static const int M = 16;
// // static const int N = 16;
// // static const int K = 16;

// // // Implementation constants.
// // static const int WARPS_PER_BLOCK   = 8;
// // static const int THREADS_PER_BLOCK = (WARP_SIZE * WARPS_PER_BLOCK);

// // static const int CHUNK_K = 8;

// // static const int BLOCK_ROW_WARPS = 2;
// // static const int BLOCK_COL_WARPS = 4;

// // static const int WARP_ROW_TILES = 4;
// // static const int WARP_COL_TILES = 2;

// // static const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 8
// // static const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS); // 8

// // static const int SHMEM_STRIDE = (N * BLOCK_ROW_TILES); // 128
// // static const int SHMEM_OFFSET = (N * WARP_ROW_TILES);  // 64

// // /*
// // 从上面的定义就可以看出，一个Block有32x8个线程，也就是8个warp，每个warp计算一个2x4的16x16的子块。
// // 一个Block会被分成2x4个Block_Tile,也就是对应的BLOCK_ROW_WARPS x BLOCK_COL_WARPS。
// // 每个Block_Tile会被分成4x2个Warp_Tile，也就是对应的WARP_ROW_TILES x WARP_COL_TILES。

// // 每个warp计算一个Block_Tile的结果。
// // */

// // #define C_LAYOUT wmma::mem_row_major

// // // The macro below is used to shift rows of the A matrix and columns of the B
// // // matrix in shared memory to minimize possible bank conflicts. Before
// // // performing the nvcuda::wmma::mma_sync operation, the warp must load the
// // // matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// // // memory access pattern is not specified for that function, each lane in the
// // // warp can read one or multiple matrix elements from different matrix rows or
// // // columns. For shared memory, such access can result in bank conflicts if
// // // different rows / columns of the matrix map to the same bank. By shifting each
// // // row and column by a few bytes, we make sure that they map to different banks,
// // // thus reducing the number of possible bank conflicts. The number of 8 two-byte
// // // "half" elements is chosen as the minimum possible shift because we must keep
// // // each row and column 128-bit aligned, as required by
// // // nvcuda::wmma::load_matrix_sync.

// // /*
// // 这里通过shift 8个half元素来解决shared memory的bank conflict问题的方法没看懂？
// // */ 
// // static const int SKEW_HALF = 8;

// // // static __global__ void gemm_wmma_kernel_fp32(size_t M_GLOBAL, size_t N_GLOBAL, size_t K_GLOBAL, __half *A, __half *B, __half *C, __half alpha, __half beta) 
// // // {
// // //     extern __shared__ __half shmem[][CHUNK_K * K + SKEW_HALF]; // 为什么要用extern？答：因为这个数组是在每个Block中共享的，所以需要extern声明

// // //     // block tile
// // //     const auto M_TILES = M_GLOBAL / M;
// // //     const auto N_TILES = N_GLOBAL / N;
// // //     const auto K_TILES = K_GLOBAL / K;

// // //     const unsigned int warpId = threadIdx.y; // 8
// // //     const unsigned int laneId = threadIdx.x; // 32

// // //     // offset in shared memory from which the B matrix is stored
// // //     const size_t shmem_idx_b_off = BLOCK_ROW_TILES * M; // 8 * 16 = 128
 
// // //     // This pointer is used to access the C and D matrix tiles this warp computes.
// // //     // 这里注意，128x128，先被切成2x4的block tile，每个block tile被切成4x2的warp tile。这里也就要存储4x2的warp tile大小的数据。

// // //     /*
    
    
    
// // //     */
// // //     half *shmem_warp_tile_ptr = (half *) &shmem[0][0] +
// // //                                 (warpId / 2) * SHMEM_STRIDE * K * 2 +
// // //                                 (warpId % 2) * SHMEM_OFFSET; // 内部跳转，这里是交错存储的吗？


// // //     // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
// // //     half *shmem_warp_stream_ptr = (half *) &shmem[0][0] + warpId * SHMEM_STRIDE * K;
 
// // //     // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
// // //     // each tile computation. Technically this is not generally correct (may
// // //     // result in a loss of precision). Zero still needs to be specially handled
// // //     // though.
// // //     beta /= alpha;
// // // }

// // static __global__ void gemm_wmma_kernel_fp16(size_t M_GLOBAL, size_t N_GLOBAL, size_t K_GLOBAL, __half *A, __half *B, __half *C, __half alpha, __half beta) 
// // {
// //     extern __shared__ __half shmem[][CHUNK_K * K + SKEW_HALF]; // 为什么要用extern？答：因为这个数组是在每个Block中共享的，所以需要extern声明

// //     // block tile
// //     const auto M_TILES = M_GLOBAL / M;
// //     const auto N_TILES = N_GLOBAL / N;
// //     const auto K_TILES = K_GLOBAL / K;

// //     const unsigned int warpId = threadIdx.y; // 8
// //     const unsigned int laneId = threadIdx.x; // 32

// //     // offset in shared memory from which the B matrix is stored
// //     const size_t shmem_idx_b_off = BLOCK_ROW_TILES * M; // 8 * 16 = 128
 
// //     // 将C和D
// //     // This pointer is used to access the C and D matrix tiles this warp computes.
// //     // 这里注意，128x128，先被切成2x4的block tile，每个block tile被切成4x2的warp tile。这里也就要存储4x2的warp tile大小的数据。
// //     half *shmem_warp_tile_ptr = (half *) &shmem[0][0] +
// //                                 (warpId / 2) * SHMEM_STRIDE * K * 2 +
// //                                 (warpId % 2) * SHMEM_OFFSET; // 内部跳转，这里是交错存储的吗？


// //     // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
// //     half *shmem_warp_stream_ptr = (half *) &shmem[0][0] + warpId * SHMEM_STRIDE * K;
 
// //     // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
// //     // each tile computation. Technically this is not generally correct (may
// //     // result in a loss of precision). Zero still needs to be specially handled
// //     // though.
// //     beta /= alpha;

// //     // Each CTA slides along the 128 x 128 tiles from the top left corner of the
// //     // matrix to the right and down, and selects the next tile to compute. Once
// //     // there's no such tile, all warps in this CTA exit.
// //     for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) 
// //     /*
// //     这里是将CTA划分到具体的block中去，gridDim.x是所有的SM数量，代表着一个循环。
// //     block_pos是当前SM的编号。

// //     下面的代码还是按照16x16的块划分？
// //     */
// //     {
// //         const unsigned int block_tile_i =
// //             ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
// //         const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
// //         // 将128x128的C矩阵划分成8x8的tile
        
// //         // Stop when there are no more D matrix tiles to compute in this CTA.
// //         if (block_tile_i >= M_TILES) {
// //         break;
// //         }

// //         // This warp's pointer to the C matrix data to copy memory from to shared
// //         // memory.
// //         const size_t gmem_idx =
// //             (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
// //         const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

// //     // Stream multiple C tiles to shared memory.
// // #pragma unroll
// //         for (int i = 0; i < K; i++) {
// //         typedef int4 copy_t;

// //         *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
// //             *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
// //                 laneId);
// //         }

// //         __syncthreads();

// //         // These fragments will accumulate the result of A and B matrix fragment
// //         // multiplications along the K_GLOBAL dimension.
// //         wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
// //                                                        [WARP_ROW_TILES];

// //         // Load the C matrix tiles into fragments from shared memory.
// // #pragma unroll
// //         for (int i = 0; i < WARP_COL_TILES; i++) {
// // #pragma unroll
// //             for (int j = 0; j < WARP_ROW_TILES; j++) {
// //                 const float *tile_ptr =
// //                     shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

// //                 wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
// //             }
// //         }

// //         __syncthreads();

// //     // Scale the C matrix.
// // #pragma unroll
// //         for (int i = 0; i < WARP_COL_TILES; i++) {
// // #pragma unroll
// //             for (int j = 0; j < WARP_ROW_TILES; j++) {
// // #pragma unroll
// //                 for (int t = 0; t < c[i][j].num_elements; t++) {
// //                 c[i][j].x[t] *= beta;
// //                 }
// //             }
// //         }

// //         // Select what warp copies what matrix to shared memory.
// //         // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
// //         const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
// //                                             M * K_GLOBAL * (warpId % 4) * 2)
// //                                             : (&B[block_tile_j * N * K_GLOBAL] +
// //                                             N * K_GLOBAL * (warpId % 4) * 2);

// //         // Go through the global K dimension by a fixed step at a time.
// // #pragma unroll
// //         for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
// //         // Copy slices of the A and B matrices to shared memory.
// //         // The first half of the warps in the CTA copy the A matrix, the rest copy
// //         // the B matrix.
// //         size_t shmem_idx =
// //             warpId < (WARPS_PER_BLOCK / 2)
// //                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
// //                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

// //         // First half of the warp copies the first row / column of the matrix,
// //         // the second half of the warp copies the next.
// //         int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
// //                                     (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
// //                         (laneId % CHUNK_COPY_LINE_LANES);

// //         // Shift the second half of the warp to the next row / column in the
// //         // shared memory.
// //         shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

// // #pragma unroll
// //         for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;i++) 
// //         {
// //             // Copy 16 bytes at once in each lane.
// //             *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
// //                 *lane_ptr;

// //             // Advance the global memory pointer and the shared memory index.
// //             lane_ptr =
// //                 (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
// //             shmem_idx += CHUNK_COPY_LINES_PER_WARP;
// //         }

// //         __syncthreads();

// //         // Compute a grid of C matrix tiles in each warp.
// // #pragma unroll
// //         for (int k_step = 0; k_step < CHUNK_K; k_step++) {
// //             wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
// //                 a[WARP_COL_TILES];
// //             wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
// //                 b[WARP_ROW_TILES];

// // #pragma unroll
// //             for (int i = 0; i < WARP_COL_TILES; i++) 
// //             {
// //                 size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
// //                 const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

// //                 wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

// // #pragma unroll
// //                 for (int j = 0; j < WARP_ROW_TILES; j++) 
// //                 {
// //                     if (i == 0) 
// //                     {
// //                         // Load the B matrix fragment once, because it is going to be
// //                         // reused against the other A matrix fragments.
// //                         size_t shmem_idx_b = shmem_idx_b_off +
// //                                             (WARP_ROW_TILES * N) * (warpId % 2) +
// //                                             (j * N);
// //                         const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

// //                         wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
// //                     }

// //                     wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
// //                 }
// //             }
// //         }

// //       __syncthreads();
// //     }

// //       // Store the D fragments to shared memory.
// // #pragma unroll
// //     for (int i = 0; i < WARP_COL_TILES; i++) {
// // #pragma unroll
// //         for (int j = 0; j < WARP_ROW_TILES; j++) {
// // #pragma unroll
// //             // Uniform, point-wise transformations of ALL fragment elements by ALL
// //             // threads in the warp are well-defined even though element indices
// //             // within fragment storage are not defined.
// //             for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

// //             float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

// //             wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
// //         }
// //     }

// //     __syncthreads();

// //     // Now that shared memory contains all the D tiles, stream them to global
// //     // memory.
// //     float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

// // #pragma unroll
// //     for (int i = 0; i < K; i++) 
// //     {
// //         *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
// //             *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
// //     }

// //     __syncthreads();
// //   }
// // }

// template <typename T>
// void cuda_gemm_wmma_10(size_t m, size_t n, size_t k, T *A, T *B, T *C, T alpha, T beta, cudaStream_t stream)
// {
//     std::cout<<"Unsupported data type: "<<typeid(T).name()<<std::endl;
// }

// // // GEMM API: 调用 Kernel 并管理 Stream
// // template <>
// // void cuda_gemm_wmma_1<__half>(size_t M, size_t N, size_t K,
// //     __half *A, __half *B, __half *C, 
// //     __half alpha, __half beta, cudaStream_t stream) {

// //     const int BLOCK_SIZE = 128;
// //     dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
// //     dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

// //     gemm_wmma_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
// //     CHECK_LAST_CUDA_ERROR();
// // }

// // template <>
// // void cuda_gemm_wmma_1<float>(size_t m, size_t n, size_t k, float* A, float* B, float* C, float alpha, float beta, cudaStream_t stream)
// // {
// //     const int BLOCK_SIZE = 128;
// //     dim3 blockDim(32, 8, 1);  // 每个 Block 32x8 线程，也就是8个warp，这里的x=32, y=8，x是内循环，y是外循环
// //     dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

// //     gemm_wmma_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, A, B, C, alpha, beta);
// //     CHECK_LAST_CUDA_ERROR();
// //     CHECK_LAST_CUDA_ERROR();
// // }

// template GEMM_EXPORT void cuda_gemm_wmma_10<float>(size_t m, size_t n, size_t k, float *A, float *B, float *C, float alpha, float beta, cudaStream_t stream);
// template GEMM_EXPORT void cuda_gemm_wmma_10<__half>(size_t m, size_t n, size_t k, __half* A, __half* B, __half* C, __half alpha, __half beta, cudaStream_t stream);

