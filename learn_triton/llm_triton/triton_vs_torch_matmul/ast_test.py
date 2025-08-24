import triton
import triton.language as tl
import torch
import inspect
import ast

# 关键：BLOCK_SIZE 必须声明为 tl.constexpr！
@triton.jit
def add_kernel(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    tl.store(C + offsets, a + b)

# 准备输入数据
x = torch.randn(128, device='cuda')
y = torch.randn(128, device='cuda')
z = torch.empty(128, device='cuda')

# 调用 kernel，触发 JIT 编译（必要步骤）
add_kernel[(2,)](x, y, z, BLOCK_SIZE=64)

# 现在 add_kernel 是 JITFunction，有 .fn 属性了
print(type(add_kernel))  # triton.runtime.jit.JITFunction

# ✅ 提取 Python AST（语法树）
source = inspect.getsource(add_kernel.fn)
tree = ast.parse(source)
print(ast.dump(tree, indent=4))
