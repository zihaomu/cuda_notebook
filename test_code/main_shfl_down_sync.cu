#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

__global__ void kernel() {
    int lane_id = threadIdx.x % warpSize;
    int value = lane_id;
    
    // 从高编号线程获取值（向下移动1个位置）
    int shifted = __shfl_down_sync(0xFFFFFFFF, value, 1); 
    
    printf("Thread %d: original=%d, shifted=%d\n", lane_id, value, shifted);
}

__global__ void kernel_xor() {
    int lane_id = threadIdx.x % warpSize;
    int value = lane_id;
    int shifted = __shfl_xor_sync(0xFFFFFFFF, value, 1);

    printf("Thread %d: original=%d, shifted=%d\n", lane_id, value, shifted);
}

int main() {
    // 启动 kernel
    //kernel << <1, 64 >> > ();
    kernel_xor << <1, 64 >> > ();
    return 0;
}
