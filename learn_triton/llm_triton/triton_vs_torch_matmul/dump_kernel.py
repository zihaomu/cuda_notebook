import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    tl.store(C + offsets, a + b)

# 创建一些 dummy 数据
x = torch.randn(1024, device='cuda', dtype=torch.float32)
y = torch.randn(1024, device='cuda', dtype=torch.float32)
z = torch.empty(1024, device='cuda', dtype=torch.float32)

# 执行一次 kernel，触发编译
add_kernel[(1024 // 64,)](x, y, z, BLOCK_SIZE=64)

# 获取缓存中的 CompiledKernel 对象
compiled = list(add_kernel.fn.cache.values())[0]

print("=== PTX ===")
print(compiled.asm['ptx'])
