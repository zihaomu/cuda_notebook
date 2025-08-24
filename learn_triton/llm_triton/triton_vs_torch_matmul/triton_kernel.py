import torch
import triton
import triton.language as tl
import time

# 定义 Triton matmul kernel
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A_tile = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    # B_tile = tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        A_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k)[None, :] < K, other=0.0)
        B_tile = tl.load(b_ptrs, mask=(k + offs_k)[:, None] < K & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(A_tile, B_tile)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# @triton.jit
# def matmul_fp32_kernel(
#     A_ptr, B_ptr, C_ptr,
#     M, N, K,
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
# ):
#     pid_m = tl.program_id(0) # Program ID for the first dimension (M)
#     pid_n = tl.program_id(1) # Program ID for the second dimension (N)

#     # tl.arange 用来生成一堆线程块。
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # Offsets for the M dimension
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, BLOCK_K)

#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

#     for k in range(0, K, BLOCK_K):
#         A_block_ptr = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
#         B_block_ptr = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

#         a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
#         b = tl.load(B_block_ptr, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

#         acc += tl.dot(a, b)

#     C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#     tl.store(C_block_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


# Benchmark 主函数
def benchmark():
    sizes = [1024 * i for i in range(1, 9)]  # 1024 到 8192
    print(f"{'Size':>10} | {'Torch (ms)':>12} | {'Triton (ms)':>13} | {'Speedup':>8}")
    print("-" * 50)

    for size in sizes:
        A = torch.randn((16, size), device='cuda', dtype=torch.float32)
        B = torch.randn((size, size), device='cuda', dtype=torch.float32)

        # Torch benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # warmup
        C_torch = torch.matmul(A, B)
        start.record()
        for _ in range(10):
            C_torch = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end)/10  # 平均时间

        # Triton benchmark
        torch.cuda.synchronize()
        C_triton = triton_matmul(A, B)
        
        start.record()
        for _ in range(10):
            C_triton = triton_matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 10  # 平均时间

        # 验证结果正确性（误差容忍 1e-3）
        max_error = (C_torch - C_triton).abs().max().item()
        print(f"Max error: {max_error:.6f}")
        # assert max_error < 1e-2, f"Max error too high: {max_error}"

        print(f"{size:10} | {torch_time:12.3f} | {triton_time:13.3f} | {torch_time / triton_time:8.2f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    benchmark()
