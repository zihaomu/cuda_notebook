import torch
from typing import Optional, Tuple
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def functional_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=None, is_causal=False):
    """
    Manual implementation of scaled dot-product attention similar to PyTorch's F.scaled_dot_product_attention.

    Args:
        query: (batch, num_heads, seq_len_q, head_dim)
        key:   (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        attn_mask: optional attention mask, broadcastable to (batch, num_heads, seq_len_q, seq_len_k)
        dropout_p: dropout probability
        scale: custom scaling factor. If None, use 1/sqrt(head_dim)
        is_causal: whether to apply causal mask

    Returns:
        attn_output: (batch, num_heads, seq_len_q, head_dim)
    """

    B, H, Lq, D = query.shape
    _, _, Lk, _ = key.shape

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Step 1: Compute raw attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # shape: (B, H, Lq, Lk)

    # Step 2: Apply masks
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

    if is_causal:
        causal_mask = torch.tril(torch.ones(Lq, Lk, device=query.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

    # Step 3: Softmax over last dim (Lk)
    attn_weights = F.softmax(scores, dim=-1)

    # Step 4: Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Step 5: Compute output
    attn_output = torch.matmul(attn_weights, value)  # shape: (B, H, Lq, D)

    return attn_output


def sdpa_attention_forward(
    config,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(config, "num_key_value_groups"):
        key = repeat_kv(key, config.num_key_value_groups)
        value = repeat_kv(value, config.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # print("causal_mask = ", causal_mask, "is_causal = ", is_causal, "scaling = ", scaling, "dropout = ", dropout)
    # print = ausal_mask =  None is_causal =  False scaling =  0.08838834764831845 dropout =  0.0
    attn_output = functional_scaled_dot_product_attention(query, key, value, attn_mask=causal_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal)
    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query,
    #     key,
    #     value,
    #     attn_mask=causal_mask,
    #     dropout_p=dropout,
    #     scale=scaling,
    #     is_causal=is_causal,
    # )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
