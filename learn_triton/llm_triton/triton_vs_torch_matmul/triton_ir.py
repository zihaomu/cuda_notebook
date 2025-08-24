import triton
import triton.language as tl
import torch
import inspect
import ast
import os
import os
import inspect
from typing import Union
from pathlib import Path

os.environ["TRITON_DEBUG"] = "1"

# 关键：BLOCK_SIZE 必须声明为 tl.constexpr！
@triton.jit
def add_kernel(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    tl.store(C + offsets, a + b)



class TritonIRInspector:
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
        self.cache = kernel_fn.cache
        self._ensure_compiled()

    def _ensure_compiled(self):
        """确保至少有一个 specialization 被编译了"""
        if len(self.cache) == 0:
            raise RuntimeError("Kernel hasn't been compiled yet. Run it at least once with data.")

    def get_all_ir(self):
        ir_list = []
        for compiled in self.cache.values():
            ir_entry = {}
            if isinstance(compiled, dict):  # Triton >=3.2 pip版本
                ir_entry = compiled
            elif hasattr(compiled, "asm"):  # CompiledKernel
                ir_entry = compiled.asm
            else:
                continue
            ir_list.append(ir_entry)
        return ir_list

    def dump_all(self, out_dir: Union[str, Path]):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ir_list = self.get_all_ir()
        for i, ir in enumerate(ir_list):
            for key in ir:
                ext = {
                    'ttgir': 'mlir',
                    'llir': 'll',
                    'ptx': 'ptx',
                    'cubin': 'cubin'
                }.get(key, key)
                filename = Path(out_dir) / f"kernel_{i}.{ext}"
                mode = 'wb' if key == 'cubin' else 'w'
                with open(filename, mode) as f:
                    f.write(ir[key] if mode == 'w' else ir[key])
                print(f"[✓] Dumped {key} to {filename}")

    def print_summary(self):
        ir_list = self.get_all_ir()
        for i, ir in enumerate(ir_list):
            print(f"\n=== Compiled Kernel {i} ===")
            for k in ir:
                print(f"  - {k}: {len(ir[k])} bytes" if isinstance(ir[k], (str, bytes)) else f"  - {k}")


# 准备输入数据
x = torch.randn(128, device='cuda')
y = torch.randn(128, device='cuda')
z = torch.empty(128, device='cuda')

# 调用 kernel，触发 JIT 编译（必要步骤）
add_kernel[(2,)](x, y, z, BLOCK_SIZE=64)

inspector = TritonIRInspector(add_kernel)
inspector.print_summary()

# 现在 add_kernel 是 JITFunction，有 .fn 属性了
print(type(add_kernel))  # triton.runtime.jit.JITFunction

# Step 2: 获取编译 cache
compiled = list(add_kernel.cache.values())[0]  # CompiledKernel 对象

print("llir = ",compiled.get('llir', 'No LLIR'))  # 如果 compiled 是 dict

print("=== CompiledKernel ===")
print("Function Name:", compiled)

# Step 3: 打印不同类型 IR
print("=== Triton TTGIR (MLIR-style IR) ===")
print(compiled.asm.get('ttgir', 'No TTGIR'))

print("\n=== PTX ===")
print(compiled.asm.get('ptx', 'No PTX'))

print("\n=== LLVM IR ===")
print(compiled.asm.get('llir', 'No LLVM IR'))

print("\n=== Binary cubin (length) ===")
print(len(compiled.asm.get('cubin', b'')), "bytes")
