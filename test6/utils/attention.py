from utils.pos_enc import RoPE


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)

    def repeat_kv(self, x):
        if self.num_queries_per_kv == 1:
            return x
        b, t, kv_heads, d = x.shape
        return x[:, :, :, None, :].expand(
            b, t, kv_heads, self.num_queries_per_kv, d
        ).reshape(b, t, self.num_attention_heads, d)

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q), self.rope(k)
        q = q.transpose(1, 2)
        k = self.repeat_kv(k).transpose(1, 2)
        v = self.repeat_kv(v).transpose(1, 2)

        attn_mask = None
        is_causal = attention_mask is None
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(f"attention_mask must have shape {(b, t)}")
            if attention_mask.device != x.device:
                attention_mask = attention_mask.to(device=x.device)
            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            elif attention_mask.is_floating_point() or attention_mask.dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                valid = attention_mask != 0
            else:
                raise TypeError("attention_mask must be boolean or numeric")
            causal = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
            attn_mask = (
                causal[None, None, :, :] & valid[:, None, None, :] & valid[:, None, :, None]
            )
            is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        return self.o_proj(y)
