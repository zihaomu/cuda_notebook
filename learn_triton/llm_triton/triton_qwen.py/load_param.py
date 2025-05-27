'''
本文件存储层的定义和实现
模型结构
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
model： Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)

# 下面是实际的模型参数
# lm_head.weight: shape=torch.Size([151936, 1536]), dtype=torch.bfloat16
# model.embed_tokens.weight: shape=torch.Size([151936, 1536]), dtype=torch.bfloat16

# model.layers.9.input_layernorm.weight: shape=torch.Size([1536]), dtype=torch.bfloat16
# model.layers.9.mlp.down_proj.weight: shape=torch.Size([1536, 8960]), dtype=torch.bfloat16
# model.layers.9.mlp.gate_proj.weight: shape=torch.Size([8960, 1536]), dtype=torch.bfloat16
# model.layers.9.mlp.up_proj.weight: shape=torch.Size([8960, 1536]), dtype=torch.bfloat16
# model.layers.9.post_attention_layernorm.weight: shape=torch.Size([1536]), dtype=torch.bfloat16
# model.layers.9.self_attn.k_proj.bias: shape=torch.Size([256]), dtype=torch.bfloat16
# model.layers.9.self_attn.k_proj.weight: shape=torch.Size([256, 1536]), dtype=torch.bfloat16
# model.layers.9.self_attn.o_proj.weight: shape=torch.Size([1536, 1536]), dtype=torch.bfloat16
# model.layers.9.self_attn.q_proj.bias: shape=torch.Size([1536]), dtype=torch.bfloat16
# model.layers.9.self_attn.q_proj.weight: shape=torch.Size([1536, 1536]), dtype=torch.bfloat16
# model.layers.9.self_attn.v_proj.bias: shape=torch.Size([256]), dtype=torch.bfloat16
# model.layers.9.self_attn.v_proj.weight: shape=torch.Size([256, 1536]), dtype=torch.bfloat16

# model.norm.weight: shape=torch.Size([1536]), dtype=torch.bfloat16
'''

from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch

path = "E:\my_project\DeepSeek-R1-Distill-Qwen-1.5B\model.safetensors"
# path = "E:\my_project\llava-fastvithd_0.5b_stage2\model.safetensors"
state_dict = load_file(path)

def load_param(param_name):
    # find if param_name in state_dict
    if param_name in state_dict:
        bf16_tensor = state_dict[param_name]
        return bf16_tensor.to(torch.float32)

    else:
        raise ValueError(f"Parameter {param_name} not found in state_dict")


if __name__ == "__main__":
    # print all param names and shapes
    for name, tensor in state_dict.items():
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")