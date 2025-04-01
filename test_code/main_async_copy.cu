#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
// 确保对齐到128字节
#define ALIGN_128 __attribute__((aligned(128)))

// 每个块处理128字节（32个float）
constexpr int BLOCK_SIZE = 32;

__global__ void async_copy_kernel(const float* global_in, float* global_out) {
    // 声明共享内存，对齐到128字节
    __shared__ __align__(128) float shmem[BLOCK_SIZE];

    // 计算全局内存地址
    const float* global_addr = global_in + blockIdx.x * BLOCK_SIZE + threadIdx.x * 4;

    // 异步拷贝全局内存到共享内存
    uint32_t shmem_addr = __cvta_generic_to_shared(shmem + threadIdx.x * 4);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(shmem_addr), "l"(global_addr), "n"(16) // 16 个Byte，也就是4个float
    );

    //shmem_addr = __cvta_generic_to_shared(shmem + threadIdx.x * 4 + 4);
    //asm volatile(
    //    "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
    //    :: "r"(shmem_addr), "l"(global_addr + 4), "n"(16) // 16 个Byte，也就是4个float
    //);

    // 提交异步操作组
    asm volatile("cp.async.commit_group;");

    // 等待所有异步拷贝完成
    asm volatile("cp.async.wait_group 0;");

    // 确保所有线程都能访问共享内存
    __syncthreads();

    // 计算结果（示例：直接复制）
    int tid = threadIdx.x;

    for (int i = 0; i < 4; i++)
    {
        global_out[blockIdx.x * BLOCK_SIZE + threadIdx.x * 4 + i] = shmem[threadIdx.x * 4 + i];
    }
}

int main() {
    const int num_elements = 1024;
    const int num_blocks = num_elements / BLOCK_SIZE;
    const int num_thread = BLOCK_SIZE / 4; // 一个thread 处理 4个float


    // 分配对齐的全局内存
    float *h_in, *h_out;
    cudaMallocHost((void**)&h_in, num_elements * sizeof(float));
    cudaMallocHost((void**)&h_out, num_elements * sizeof(float));

    // 初始化数据
    for (int i = 0; i < num_elements; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // 设备指针
    float *d_in, *d_out;
    cudaMalloc(&d_in, num_elements * sizeof(float));
    cudaMalloc(&d_out, num_elements * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_in, h_in, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 启动kernel
    async_copy_kernel<<<num_blocks, num_thread>>>(d_in, d_out);

    // 拷贝结果回主机
    cudaMemcpy(h_out, d_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < num_elements; ++i) {
        std::cout << "out = " << h_out[i] << ", in = " << h_in[i] << std::endl;
        if (h_out[i] != h_in[i]) {
            success = false;
            break;
        }
    }
    printf(success ? "Success!\n" : "Error!\n");

    // 清理资源
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}