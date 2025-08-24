import torch
import triton
import triton.language as tl
import time

from typing import List, Tuple

# def broadcast_matmul_shape(a_shape: List[int], b_shape: List[int]) -> List[int]:
#     def shape_rank(shape): return len(shape)
#     a_rank, b_rank = shape_rank(a_shape), shape_rank(b_shape)

#     # Step 1: Adjust 1D shapes to 2D for matmul semantics
#     a_adjusted = False
#     b_adjusted = False
#     if a_rank == 1:
#         a_shape = [1, a_shape[0]]  # Treat as row vector
#         a_adjusted = True
#     if b_rank == 1:
#         b_shape = [b_shape[0], 1]  # Treat as column vector
#         b_adjusted = True

#     # Step 2: Validate matrix dims
#     if len(a_shape) < 2 or len(b_shape) < 2:
#         raise ValueError("Invalid shapes after adjustment")

#     a_batch, a_m, a_k = a_shape[:-2], a_shape[-2], a_shape[-1]
#     b_batch, b_k, b_n = b_shape[:-2], b_shape[-2], b_shape[-1]
#     if a_k != b_k:
#         raise ValueError(f"Incompatible matrix dimensions {a_k} vs {b_k} for matmul")

#     # Step 3: Broadcast batch dimensions
#     def broadcast_batch_shape(a_batch, b_batch):
#         result = []
#         for a_dim, b_dim in zip(reversed(a_batch[::-1] + [1] * (len(b_batch) - len(a_batch))),
#                                 reversed(b_batch[::-1] + [1] * (len(a_batch) - len(b_batch)))):
#             if a_dim == 1:
#                 result.append(b_dim)
#             elif b_dim == 1:
#                 result.append(a_dim)
#             elif a_dim == b_dim:
#                 result.append(a_dim)
#             else:
#                 raise ValueError(f"Incompatible batch dimensions: {a_dim} vs {b_dim}")
#         return result[::-1]

#     batch_shape = broadcast_batch_shape(a_batch, b_batch)
#     output_shape = batch_shape + [a_m, b_n]

#     # Step 4: Squeeze if original A or B was 1D
#     if a_adjusted:
#         output_shape.pop(-2)  # remove a_m dim
#     if b_adjusted:
#         output_shape.pop(-1)  # remove b_n dim

#     return output_shape

@triton.jit
def matmul_fp32_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0) # Program ID for the first dimension (M)
    pid_n = tl.program_id(1) # Program ID for the second dimension (N)

    # tl.arange 用来生成一堆线程块。
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # Offsets for the M dimension
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A_block_ptr = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        B_block_ptr = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_block_ptr, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_block_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matmul_add_fp32_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A_block_ptr = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        B_block_ptr = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_block_ptr, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # Load bias and add
    bias_ptr = Bias_ptr + offs_n
    bias = tl.load(bias_ptr, mask=offs_n < N, other=0.0)
    acc += bias[None, :]  # broadcast over rows

    C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_block_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_fp32(A: torch.Tensor, B: torch.Tensor):
    
    C_shape = list(A.shape)
    C_shape[-1] = B.shape[-1]  # Adjust C_shape to match B's last dimension
    # C_shape = broadcast_matmul_shape(A.shape, B.shape) 
    # A and B maybe a 1D, 2D tensor or 3D tensor, reshape them to 2D
    if A.ndim == 1:
        A = A.view(1, -1)
    elif A.ndim == 3:
        A = A.view(-1, A.shape[-1])
        
    assert B.ndim == 2, "B must be a 2D tensor for matmul"
    
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_fp32_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    # triton.sync()
    # according to the original code, if A or B is 1D, we need to reshape C
    C = C.view(C_shape)
    return C


def triton_matmul_add_fp32(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):

    # assign C_shape to A.shape, and convert the C_shape to list
    C_shape = list(A.shape)
    C_shape[-1] = B.shape[-1]  # Adjust C_shape to match B's last dimension

    # A and B maybe a 1D, 2D tensor or 3D tensor, reshape them to 2D
    if A.ndim == 1:
        A = A.view(1, -1)
    elif A.ndim == 3:
        A = A.view(-1, A.shape[-1])
    assert B.ndim == 2, "B must be a 2D tensor for matmul"
    
    assert A.dtype == B.dtype == bias.dtype == torch.float32
    assert A.shape[1] == B.shape[0]
    assert B.shape[1] == bias.shape[0]

    M, K = A.shape
    K2, N = B.shape
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_add_fp32_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # triton sycrhonize
    # triton.cuda.sync()
    return C.view(C_shape)  # Reshape C to match the output shape after broadcasting
