from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _matmul_kernel(x_desc, y_desc, out_desc, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = tl.cdiv(512, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    offset_0 = pid_1 * _BLOCK_SIZE_0
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in tl.range(0, 512, _BLOCK_SIZE_2, warp_specialize=True):
        acc_copy = acc
        acc_copy_0 = acc_copy
        load = x_desc.load([offset_0, offset_2])
        load_1 = y_desc.load([offset_2, offset_1])
        acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
    out_desc.store([offset_0, offset_1], acc)

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(512, _BLOCK_SIZE_1) * triton.cdiv(512, _BLOCK_SIZE_0),](TensorDescriptor.from_tensor(x, [_BLOCK_SIZE_0, _BLOCK_SIZE_2]), TensorDescriptor.from_tensor(y, [_BLOCK_SIZE_2, _BLOCK_SIZE_1]), TensorDescriptor.from_tensor(out, [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), _BLOCK_SIZE_1, _BLOCK_SIZE_0, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
    return out

