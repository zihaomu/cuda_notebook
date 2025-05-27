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
'''

from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn

import triton
import torch
import numpy as np

from sdpa_attention import sdpa_attention_forward
from utils_qwen import apply_rotary_pos_emb
from load_param import load_param

# convert "model.layers.*.self_attn.q_proj" to "model.layers.0.self_attn.q_proj"
def get_idx_from_name(name, idx):
    """
    Convert a name with wildcards to a specific index.
    :param name: The name with wildcards.
    :param idx: The index to replace the wildcard.
    :return: The name with the wildcard replaced by the index.
    """
    return name.replace("*", str(idx))

    
class Qwen2Config:
    model_type = "qwen2"
    # q_proj_weights, 
    # k_proj_weights, 
    # v_proj_weights, 
    # o_proj_weights,
    # gate_proj_weights, 
    # up_proj_weights, 
    # down_proj_weights, 
    # input_layernorm_weight, 
    # post_attention_layernorm_weight,  
    # q_proj_bias=None, 
    # k_proj_bias=None, 
    # v_proj_bias=None

    base_model_tp_plan = [
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
        "model.layers.*.self_attn.o_proj.weight",
        
        "model.layers.*.mlp.gate_proj.weight",
        "model.layers.*.mlp.up_proj.weight",
        "model.layers.*.mlp.down_proj.weight",
        
        "model.layers.*.input_layernorm.weight",
        "model.layers.*.post_attention_layernorm.weight",

        "model.layers.*.self_attn.q_proj.bias",
        "model.layers.*.self_attn.k_proj.bias",
        "model.layers.*.self_attn.v_proj.bias",
    ]
    
    embding_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    last_norm_name = "model.norm.weight"
    
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size,
                 hidden_act, layer_norm_eps, max_position_embeddings):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window=False
        self.sliding_window=4096
        self.max_window_layers=28
        self.rope_theta=10000


class Linear:
    '''
    全连接层
    '''
    def __init__(self, in_features, out_features, has_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.weights = None
        self.bias = None
        
    def load_weights(self, weights, bias=None):
        '''
        加载权重
        :param weights: 权重
        :param bias: 偏置
        :return:
        '''
        assert self.weights is None, "linear not initialized"
        self.weights = weights
        if self.has_bias:
            assert self.bias is None, "bias not initialized"
            self.bias = bias

    def forward(self, x):
        # 检查维度是否正确
        # TODO 使用triton实现linear
        assert self.weights is not None, "linear not initialized"
        if self.has_bias:
            # use torch matmul instead of np.matmul
            return torch.matmul(x, self.weights.T) + self.bias
            # return np.matmul(x, self.weights.T) + self.bias
        else:
            return torch.matmul(x, self.weights.T)


class WordEmbedding:
    '''
    词嵌入层
    '''
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = None

    def load_weights(self, weights):
        '''
        加载权重
        :param weights: 权重
        :return:
        '''
        assert self.weights is None, "embedding not initialized"
        self.weights = weights

    def forward(self, input_ids):

        data = self.weights[input_ids].unsqueeze(0)

        if data.dim() == 2:
            data = data.unsqueeze(0)

        return data


class Softmax:
    '''
    softmax层
    '''
    def __init__(self, dim):
        self.dim = dim

    def forward(self, input):
        # TODO 使用triton实现softmax
        return input.softmax(dim=-1)


class Qwen2RMSNorm:
    '''
    RMSNorm层
    '''
    def __init__(self, dim, eps=1e-6):
        self.dim = dim
        self.eps = eps
        self.weight = None

    def _norm(self, x):
        # TODO 使用triton实现rmsnorm
        return x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()

    def load_weights(self, weight):
        '''
        加载权重
        :param weight: 权重
        :return:
        '''
        assert self.weight is None, "rmsnorm not initialized"
        self.weight = weight
                
        # print(f"Qwen2RMSNorm:, {self.weight.shape}")
        

    def forward(self, x):
        # TODO 使用triton实现rmsnorm
        # print self.weight.shape
        
        # print(f"Qwen2RMSNorm: {x.shape}, {self.weight.shape}")
        d = self._norm(x)
        return d * self.weight


class SiLU:
    def __init__(self):
        pass
    
    '''
    SiLU激活函数
    '''
    def forward(self, x):
        # TODO 使用triton实现silu
        return x * x.sigmoid()


class Qwen2MLP:
    def __init__(self, config:Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, has_bias=False)
        self.norm = Qwen2RMSNorm(self.intermediate_size)
        assert config.hidden_act == "silu", "only silu is supported"
        self.act_fn = SiLU()
        
    def load_weights(self, gate_proj_weights, up_proj_weights, down_proj_weights):
        '''
        加载权重
        :param gate_proj_weights: gate_proj权重
        :param up_proj_weights: up_proj权重
        :param down_proj_weights: down_proj权重
        :return:
        '''
        self.gate_proj.load_weights(gate_proj_weights)
        self.up_proj.load_weights(up_proj_weights)
        self.down_proj.load_weights(down_proj_weights)

    def forward(self, x):
        x_up = self.up_proj.forward(x)
        x_gate = self.act_fn.forward(self.gate_proj.forward(x))
        down_proj = self.down_proj.forward(x_up * x_gate)
        return down_proj

# TODO 实现 用于实现 kv cache
class Cache:
    def __init__(self):
        self.layer_id = 0
        self.key_cache = None
        self.value_cache = None

    '''
    保存当前的k和v的同时，将组合出完整的k和v
    '''
    def update(self, key_states, value_states):
        if self.key_cache is None:
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache, self.value_cache


class Qwen2Attention:
    def __init__(self, config:Qwen2Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = Linear(config.hidden_size, config.num_attention_heads * self.head_dim, has_bias=True)
        self.k_proj = Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=True)
        self.v_proj = Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, has_bias=True)
        self.o_proj = Linear(config.num_attention_heads * self.head_dim, config.hidden_size, has_bias=False)
        self.scaling = self.head_dim**-0.5
        self.cache = Cache()

    def load_weights(self, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights, q_proj_bias=None, k_proj_bias=None, v_proj_bias=None):
        '''
        加载权重
        :param q_proj_weights: q_proj权重
        :param k_proj_weights: k_proj权重
        :param v_proj_weights: v_proj权重
        :param o_proj_weights: o_proj权重
        :return:
        '''
        self.q_proj.load_weights(q_proj_weights, q_proj_bias)
        self.k_proj.load_weights(k_proj_weights, k_proj_bias)
        self.v_proj.load_weights(v_proj_weights, v_proj_bias)
        self.o_proj.load_weights(o_proj_weights, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_positions: torch.Tensor,
        sin_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: Optional[Cache] = None, # 种点，KV cache
        cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_positions, sin_positions)

        # if past_key_value is not None:
        #     # TODO 增加策略
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin_positions, "cos": cos_positions, "cache_position": cache_position}
        key_states, value_states = self.cache.update(key_states, value_states)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = sdpa_attention_forward(
            self.config,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj.forward(attn_output)
        return attn_output, attn_weights

class Qwen2DecoderLayer:
    def __init__(self, config:Qwen2Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def load_weights(self, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights,
                     gate_proj_weights, up_proj_weights, down_proj_weights, input_layernorm_weight, post_attention_layernorm_weight,  q_proj_bias=None, k_proj_bias=None, v_proj_bias=None):
        '''
        加载权重
        :param q_proj_weights: q_proj权重
        :param k_proj_weights: k_proj权重
        :param v_proj_weights: v_proj权重

        :param o_proj_weights: o_proj权重
        :param gate_proj_weights: gate_proj权重
        :param up_proj_weights: up_proj权重
        :param down_proj_weights: down_proj权重
        :return:
        '''
        self.self_attn.load_weights(q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights, q_proj_bias, k_proj_bias, v_proj_bias)
        self.mlp.load_weights(gate_proj_weights, up_proj_weights, down_proj_weights)
        self.input_layernorm.load_weights(input_layernorm_weight)
        self.post_attention_layernorm.load_weights(post_attention_layernorm_weight)


    def forward(self, hidden_states, cos_positions, sin_positions, attention_mask, past_key_value=None, cache_position=None):
        # TODO 检查实现是否正确。
        residual = hidden_states
        # TODO 使用triton实现

        # 1. layernorm
        hidden_states = self.input_layernorm.forward(hidden_states)

        # 2. self attention
        attn_output, attn_weights = self.self_attn.forward(hidden_states, cos_positions, sin_positions, attention_mask, past_key_value, cache_position)
        hidden_states = attn_output + residual

        # 3. layernorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)

        # 4. mlp
        mlp_output = self.mlp.forward(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states        


def _compute_default_rope_parameters(
    config: Optional[Qwen2Config] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor



class Qwen2RotaryEmbedding:
    def __init__(self, config: Qwen2Config, device=None):
        # super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        # self.original_inv_freq = self.inv_freq

    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model:
    def __init__(self, config:Qwen2Config):
        self.config = config
        self.vocab_size = config.vocab_size
        self.word_embedding = WordEmbedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        self.layer_norm = Qwen2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, has_bias=False)

    def load(self):
        '''
        加载权重
        :param weights: 权重
        :return:
        '''
        
        # load word embedding
        self.word_embedding.load_weights(load_param(Qwen2Config.embding_name))
        
        # load layer norm
        self.layer_norm.load_weights(load_param(Qwen2Config.last_norm_name))
        
        # load lm_head
        self.lm_head.load_weights(load_param(Qwen2Config.lm_head_name))

        # load param layer by layer
        for layer_idx in range(self.config.num_hidden_layers):
            t0 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[0], layer_idx))
            t1 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[1], layer_idx))
            t2 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[2], layer_idx))
            t3 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[3], layer_idx))
            t4 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[4], layer_idx))
            t5 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[5], layer_idx))
            t6 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[6], layer_idx))
            t7 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[7], layer_idx))
            t8 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[8], layer_idx))
            t9 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[9], layer_idx))
            t10 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[10], layer_idx))
            t11 = load_param(get_idx_from_name(Qwen2Config.base_model_tp_plan[11], layer_idx))

            self.layers[layer_idx].load_weights(t0, t1, t2, t3,
                                                t4, t5, t6, t7,
                                                t8, t9, t10, t11)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        
        # word embedding
        hidden_states = self.word_embedding.forward(input_ids)

        # TODO 检查 hidden_state

        # rotary embedding
        cos_positions, sin_positions = self.rotary_emb.forward(hidden_states, position_ids)
        # TODO 检查 cos_positions


        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            # layer forward
            hidden_states = layer.forward(hidden_states, cos_positions, sin_positions, attention_mask)
        
        # layer norm
        hidden_states = self.layer_norm.forward(hidden_states)
        
        # lm head
        lm_logits = self.lm_head.forward(hidden_states)
        
        
        # print(f"hidden_states: {hidden_states.shape}")  # torch.Size([1, 12, 1536])
        # print(f"lm_logits: {lm_logits.shape}")
        return lm_logits[:, -1, :]

