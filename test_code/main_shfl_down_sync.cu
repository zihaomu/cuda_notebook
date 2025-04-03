#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda; // 访问 `wmma` 命名空间

#define M 16
#define N 16
#define K 16

__global__ void kernel() {
    int lane_id = threadIdx.x % warpSize;
    int value = lane_id;
    
    // 从高编号线程获取值（向下移动1个位置）
    int shifted = __shfl_down_sync(0xFFFFFFFF, value, 1); 
    // 每个线程获取到下一个线程的值
    // 下一个线程的值实际就是当前lane_id+1的值
    
    printf("Thread %d: original=%d, shifted=%d\n", lane_id, value, shifted);
}

int main() {
    // 启动 kernel
    kernel << <1, 64 >> > ();
    return 0;
}
