import torch
import triton
import triton.language as tl
import time

# Triton kernel: compute C = A @ B
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_block_ptr = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_block_ptr, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_block_ptr, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0], "Shape mismatch"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c


def benchmark():
    torch.manual_seed(0)
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

    # PyTorch baseline
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        C_torch = A @ B
    torch.cuda.synchronize()
    t1 = time.time()
    torch_time = (t1 - t0) / 100 * 1000

    # Triton
    torch.cuda.synchronize()
    
    C_triton = triton_matmul(A, B)
    t0 = time.time()
    for _ in range(100):
        C_triton = triton_matmul(A, B)
    torch.cuda.synchronize()
    t1 = time.time()
    triton_time = (t1 - t0) / 100 * 1000

    print(f"Torch matmul:   {torch_time:.3f} ms")
    print(f"Triton matmul:  {triton_time:.3f} ms")
    print("Close?:", torch.allclose(C_torch, C_triton, atol=1e-1, rtol=1e-1))

benchmark()

'''
Device:4090
测试结果：
Torch matmul:   0.775 ms
Triton matmul:  0.070 ms
Close?: True
'''