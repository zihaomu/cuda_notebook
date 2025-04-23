#include "softmax.hpp"
#include "utils_function.hpp"
#include <stdio.h>
#include <iostream>
#include <cub/cub.cuh>

#define WARP_SIZE 32
/*
主要思路：
分三种情况，根据输入数据规模来分别走不同的分支。
输入是M x N
当N < 1024时，一个warp处理一行或者多行，利用register；
当4096 > N >= 1024时，一个block处理一行或者多行，利用shared mem；
当N > 4096时，直接处理；
*/

struct MaxOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return max(a, b); }
};

__inline__ __device__ float warpReduceSum(float val, int warp_size) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __syncwarp();
    return val; // 返回当前线程处理的和
}

__inline__ __device__ float warpReduceMax(float val, int warp_size) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    __syncwarp();
    return val; // 返回当前线程处理的最大值
}

template<int BLOCK_SIZE>
__inline__ __device__ float blockReduceSum(float val) {
    // 声明共享内存用于BlockReduce
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float result_broadcast;

    float aggregate = BlockReduce(temp_storage).Sum(val);

    // 仅线程0写入归约结果到共享内存, 避免多线程写入冲突
    if (threadIdx.x == 0) {
        result_broadcast = aggregate; // 将结果广播到所有线程
    }
    __syncthreads(); // 确保所有线程都完成了计算

    return result_broadcast; // 返回结果
}

template<int BLOCK_SIZE>
__inline__ __device__ float blockReduceMax(float val) {
    // 声明共享内存用于BlockReduce
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float result_broadcast;

    float aggregate = BlockReduce(temp_storage).Reduce(val, MaxOp());

    if (threadIdx.x == 0) {
        result_broadcast = aggregate; // 将结果广播到所有线程
    }
    __syncthreads(); // 确保所有线程都完成了计算

    return result_broadcast; // 返回结果
}

// handle when n < 1024
// 实验性：划分成
// 为了加快读取，一次load float2
// 划分线程的方法， block内部分配 N x 32 个线程。
// block 外部，划分成 m_num 个block
// 每个数据的PACK_SIZE数量
// ROW_PER_ACCESS 为一个warp 一次处理的行数
// COL_PER_THREAD 为一个warp 一个thread处理的元素个数
template<int PACK_SIZE, int COL_PER_THREAD, int ROW_PER_ACCESS>
__global__ void softmax_v02_WarpReduce(size_t m, size_t n, float *_A, float *_C)
{
    const int num_packs = COL_PER_THREAD / PACK_SIZE; // 每个线程处理的pack数
    const int lane_id = threadIdx.x; // 一个warp内的线程id
    const int group_id = blockIdx.x * blockDim.y + threadIdx.y; // 定位线程所在的group id
    const int step = blockDim.y * gridDim.x * ROW_PER_ACCESS; // 所有线程一次处理的行数
    // 主要思路：将线程按照group来运行，group内部是warp。

    assert(n % WARP_SIZE == 0); // 确保每个线程处理的列数是warp_size的整数倍
    assert(n / WARP_SIZE == COL_PER_THREAD); // 不考虑边界处理

    float buf[ROW_PER_ACCESS][COL_PER_THREAD]; // 申请buffer

    // 外层循环
    for (int row = group_id * ROW_PER_ACCESS; row < m; row += step)
    {
        // 先定位好哪一行

        float thread_max[ROW_PER_ACCESS]; // 用来保存当前线程处理的ROW_PER_ACCESS行的每行最大值

        // 找出ROW_PER_ACCESS行中，当前thread负责的最大值
        for (int row_id = 0; row_id < ROW_PER_ACCESS; ++row_id) {
            thread_max[row_id] = -FLT_MAX; // 初始化为最小值，目前的最小值

            float* A = _A + (row  + row_id) * n + // 定位到当前行
            lane_id * COL_PER_THREAD// 跳转到对应列的id上
            ; 
            
            float* buf_ptr = buf[row_id]; // 定位到当前线程处理的行

            // TODO 后面支持pack的load和store形式
            #pragma unroll
            for (int i = 0; i < COL_PER_THREAD; i++) 
            {
                buf_ptr[i] = A[i]; // 初始化buffer
                thread_max[row_id] = max(thread_max[row_id], buf_ptr[i]); // 找到当前行的最大值
            }
        }

        // 归约操作，找到当前线程处理的ROW_PER_ACCESS行的最大值
        float warp_max[ROW_PER_ACCESS];

        for (int i = 0; i < ROW_PER_ACCESS; ++i) {
            warp_max[i] = warpReduceMax(thread_max[i], 32); // 初始化为当前线程处理的最大值
        }

        float thread_sum[ROW_PER_ACCESS];
        for (int row_id = 0; row_id < ROW_PER_ACCESS; ++row_id) {
            thread_sum[row_id] = 0; // 初始化为0
            float* buf_ptr = buf[row_id]; // 定位到当前线程处理的行

            for (int i = 0; i < COL_PER_THREAD; i++) 
            {
                buf_ptr[i] = expf(buf_ptr[i] - warp_max[row_id]); // 计算softmax
                thread_sum[row_id] += buf_ptr[i]; // 计算和
            }
        }

        float warp_sum[ROW_PER_ACCESS];

        for (int i = 0; i < ROW_PER_ACCESS; ++i) {
            warp_sum[i] = warpReduceSum(thread_sum[i], 32); // 初始化为当前线程处理的最大值
        }

        for (int row_id = 0; row_id < ROW_PER_ACCESS; ++row_id) {
            float* buf_ptr = buf[row_id]; // 定位到当前线程处理的行
            
            for (int i = 0; i < COL_PER_THREAD; i++) 
            {
                buf_ptr[i] = buf_ptr[i] / warp_sum[row_id]; // 计算softmax
            }
        }

        // 将结果写回到C中
        for (int row_id = 0; row_id < ROW_PER_ACCESS; ++row_id) {
            float* C = _C + (row + row_id) * n + lane_id * COL_PER_THREAD; // 定位到当前行
            float* buf_ptr = buf[row_id]; // 定位到当前线程处理的行

            #pragma unroll
            for (int i = 0; i < COL_PER_THREAD; i++) 
            {
                C[i] = buf_ptr[i]; // 写回结果
            }
        }
    }
}

// handle when 4096 > n >= 1024
// 一个block处理一行。
// 划分线程方式：block 内部分 4 * 32个线程，作为一个block处理1行；为什么这里一个block处理一行？因为一个block内部已经包含多个warp，sm可以达到繁忙状态
// grid 线程划分，总共划分mi个block。
// 一行内，一次处理的数据叫pack，每次跳转pack*block大小。
template<int BLOCK_SIZE, int PACK_SIZE>
__global__ void softmax_v02_BlockReduce(size_t m, size_t n, float *_A, float *_C)
{
    const int num_packs = n / PACK_SIZE; // 每个线程处理的pack数
    const int tid = threadIdx.x; // 一个block内的线程id
    const int group_id = blockIdx.x; // 定位线程所在的group id

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[]; // 申请共享内存
    float* buf = reinterpret_cast<float*>(shared_buf);

    // shared mem 的大小，需要保存一行数据，也就是4096个数据，存下来需要4096 * 4B = 16KB，这种内存的使用量还是太低了，需要需要一次处理多行？

    // 外层循环，这次是一个block处理一行数据。
    for (int row = blockIdx.x; row < m; row += gridDim.x)
    {
        float thread_max = -FLT_MAX;

        // num_pack * pack_size = n
        for (int pack_id = tid; pack_id < num_packs; pack_id++)
        {
            float* A = _A + row * n + pack_id * PACK_SIZE; // 定位到当前行

            float pack[PACK_SIZE]; // pack的大小
            for (int i = 0; i < PACK_SIZE; i++)
            {
                pack[i] = A[i]; // 初始化buffer
            }

            // TODO 增加快速读取pack的方式
            for (int i = 0; i < PACK_SIZE; i++)
            {
                // 错开保存的地址，避免冲突
                buf[i * num_packs + pack_id] = pack[i];
                thread_max = max(thread_max, pack[i]);
            }
        }

        const float row_max = blockReduceMax<BLOCK_SIZE>(thread_max);

        float thread_sum = 0;

        for (int ni = tid; ni < n; ni += BLOCK_SIZE)
        {
            buf[ni] = expf(buf[ni] - row_max);
            thread_sum += buf[ni];
        }

        const float row_sum = blockReduceSum<BLOCK_SIZE>(thread_sum);

        for (int pack_id = tid; pack_id < num_packs; pack_id += BLOCK_SIZE)
        {
            float pack[PACK_SIZE]; // pack的大小

            for (int i = 0; i < PACK_SIZE; i++)
            {
                pack[i] = buf[i * num_packs + pack_id] / row_sum;
            }

            // store
            float* C = _C + row * n + pack_id * PACK_SIZE; // 定位到当前行
            for (int i = 0; i < PACK_SIZE; i++)
            {
                C[i] = pack[i];
            }
        }
    }
}

// handle when n > 4096
__global__ void softmax_v02_Basic(size_t m, size_t n, float *_A, float*_C)
{
    //std::cout<<"Un-support data type"<<std::endl;
}

template <typename T>
void cuda_softmax_v2(size_t m, size_t n, T *A, T *C, cudaStream_t stream)
{
    // 4090 最多有128个sm。每个sm最多设置4个warp

    // check if T is float
    if constexpr (std::is_same<T, float>::value) {
        
        if (n < 1024) {
            dim3 const block_dim{static_cast<unsigned int>(32), 4U, 1U}; // x, y, z，只有第一个有效。
            dim3 const grid_dim{(static_cast<unsigned int>(128)), 1U, 1U}; // 分成m_num * n_num个块
            const int COL_PER_THREAD = n / 32;
            const int ROW_PER_ACCESS = 4; // 每个线程处理的行数
            softmax_v02_WarpReduce<2, 32, 4><<<grid_dim, block_dim, 0U, stream>>>(m, n, A, C);
        }
        else if (n >= 1024 && n <= 4096) {
            dim3 const block_dim{static_cast<unsigned int>(128), 1U, 1U}; // x, y, z，只有第一个有效。
            dim3 const grid_dim{(static_cast<unsigned int>(128)), 1U, 1U}; // 分成m_num * n_num个块

            const int PACK_SIZE = 2;
            const int BLOCK_SIZE = 128; // 每个线程处理的行数

            const int shared_mem_size = sizeof(float) * n; // 共享内存的大小
            softmax_v02_BlockReduce<128, 2><<<grid_dim, block_dim, shared_mem_size, stream>>>(m, n, A, C);
        }
        else 
        {
            softmax_v02_Basic<<<1, 32, 0U, stream>>>(m, n, A, C);
        }
    }

    // dim3 const block_dim{static_cast<unsigned int>(m_num), 1U, 1U}; // x, y, z，只有第一个有效。
    // dim3 const grid_dim{(static_cast<unsigned int>(32)), 1U, 1U}; // 分成m_num * n_num个块
    // softmax_v02<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, A, C);
    CHECK_LAST_CUDA_ERROR();
}

template CUDA_EXPORT void cuda_softmax_v2<float>(size_t m, size_t n, float *A, float *C, cudaStream_t stream);
template CUDA_EXPORT void cuda_softmax_v2<__half>(size_t m, size_t n, __half* A, __half* C, cudaStream_t stream);