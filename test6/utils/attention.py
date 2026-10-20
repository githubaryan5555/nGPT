import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.rope = RoPE(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

    def repeat_kv(self, x):
        if self.num_queries_per_kv == 1:
            return x

        b, t, kv_heads, d = x.shape

        return (
            x[:, :, :, None, :]
            .expand(b, t, kv_heads, self.num_queries_per_kv, d)
            .reshape(b, t, self.num_attention_heads, d)
        )

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape

        # --------------------------------------------------
        # 1. Q, K, V projections
        # --------------------------------------------------
        q = self.q_proj(x).view(
            b, t, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            b, t, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            b, t, self.num_key_value_heads, self.head_dim
        )

        # --------------------------------------------------
        # 2. Rotary positional embeddings
        # --------------------------------------------------
        q, k = self.rope(q), self.rope(k)

        q = q.transpose(1, 2)
        k = self.repeat_kv(k).transpose(1, 2)
        v = self.repeat_kv(v).transpose(1, 2)

        # Shapes: [B, H, T, D]

        # --------------------------------------------------
        # 3. Explicit attention score calculation
        # --------------------------------------------------
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(self.head_dim))

        # Shape: [B, H, T, T]

        # --------------------------------------------------
        # 4. Explicit causal and padding masks
        # --------------------------------------------------
        causal = torch.ones(
            (t, t), device=x.device, dtype=torch.bool
        ).tril()

        allowed = causal[None, None, :, :]

        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}"
                )

            if attention_mask.device != x.device:
                attention_mask = attention_mask.to(x.device)

            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            elif (
                attention_mask.is_floating_point()
                or attention_mask.dtype in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ):
                valid = attention_mask != 0
            else:
                raise TypeError(
                    "attention_mask must be boolean or numeric"
                )

            allowed = (
                allowed
                & valid[:, None, None, :]
                & valid[:, None, :, None]
            )

        # --------------------------------------------------
        # 5. Explicit, numerically stable base-2 softmax
        # --------------------------------------------------
        # Compute scores in FP32 for numerical stability.
        scores = scores.float()

        # Mask invalid positions BEFORE finding row maxima.
        scores = scores.masked_fill(~allowed, float("-inf"))

        # Standard softmax equivalence:
        # exp(x) = 2 ** (x * log2(e))
        scores_base2 = scores * math.log2(math.e)

        # Stable normalization: subtract maximum per row.
        row_max = scores_base2.amax(dim=-1, keepdim=True)

        # Handle rows where every position is masked.
        row_max = torch.where(
            torch.isfinite(row_max),
            row_max,
            torch.zeros_like(row_max),
        )

        shifted = scores_base2 - row_max

        # Explicit exponentiation: this is the operation
        # you can replace with your own approximation.
        weights = torch.exp2(shifted)

        # Explicitly zero all masked positions.
        weights = weights.masked_fill(~allowed, 0.0)

        # Explicit denominator and normalization.
        denominator = weights.sum(dim=-1, keepdim=True)

        attention_weights = weights / denominator.clamp_min(
            torch.finfo(weights.dtype).tiny
        )

        # --------------------------------------------------
        # 6. Attention dropout
        # --------------------------------------------------
        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        # --------------------------------------------------
        # 7. Weighted sum of values
        # --------------------------------------------------
        y = torch.matmul(attention_weights.to(v.dtype), v)

        # --------------------------------------------------
        # 8. Merge heads and output projection
        # --------------------------------------------------
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, self.hidden_size)
        )

        return self.o_proj(y)
