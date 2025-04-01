#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda; // 访问 `wmma` 命名空间

#define M 16
#define N 16
#define K 16

__global__ void gemm_mma_sync(half* a, half* b, float* c) {

    unsigned int FragA[4 * 4];      // [4, 4]
    unsigned int FragB[4 * 4];      // [4, 4]
    unsigned int x Accum[4 * 4 * 8] = {0.0}; // [4, 4, 8]

    unsigned int *fragA = FragA; 
    unsigned int *fragB = FragB;
    unsigned int *accum = Accum;

    // asm volatile(
    //     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    //     "{%0,  %1},"
    //     "{%2,  %3,  %4,  %5},"
    //     "{%6,  %7},"
    //     "{%8,  %9};\n"
    //     : "=r"(C[0]), "=r"(C[1])
    //     : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));


    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0,  %1},"
    "{%2, %3, %4,  %5},"
    "{%6,  %7},"
    "{%0,  %1};\n"
    : "+r"(accum[0]), "+r"(accum[1])//, "+f"(accum[4]), "+f"(accum[5])
    : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
      "r"(fragB[0]), "r"(fragB[1])
      //  , "f"(accum[0]), "f"(accum[1]),
      //"f"(accum[4]), "f"(accum[5])
       );

    // // 声明和初始化 WMMA 矩阵
    // wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    // wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // wmma::fill_fragment(c_frag, 0.0f); // 初始化C矩阵

    // // 从全局内存加载数据
    // wmma::load_matrix_sync(a_frag, a, K);
    // wmma::load_matrix_sync(b_frag, b, K);


    // uint32_t t = 0;

    // // 计算矩阵乘法
    // asm volatile (
    //     "mma.sync.aligned.m16n16k16.f32.f16.f16.f32 "
    //     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15};"
    //     : "=f"(c_frag.x[0]), "=f"(c_frag.x[1]), "=f"(c_frag.x[2]), "=f"(c_frag.x[3])
    //     : "r"(t), "r"(t),
    //     "r"(t), "r"(t)
    //     "f"(c_frag.x[0]), "f"(c_frag.x[1]), "f"(c_frag.x[2]), "f"(c_frag.x[3])
    //     );

    // // 存储计算结果到全局内存
    // wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main() {
    half* d_A, * d_B;
    float* d_C;

    // 分配 GPU 内存
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    // 启动 kernel
    gemm_mma_sync << <1, 32 >> > (d_A, d_B, d_C);

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
