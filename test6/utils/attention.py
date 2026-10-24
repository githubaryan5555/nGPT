
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pos_enc import RoPE


class RMSNorm(nn.Module):
    """RMSNorm used by DeepSeek MLA for the compressed KV latent."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class GQAAttention(nn.Module):
    """
    DeepSeek-style Multi-Head Latent Attention.

    Compatible interface with the existing GQAAttention:
        attention = MLAAttention(config)
        y = attention(x, attention_mask=None)

    Input:  (batch, sequence, hidden_size)
    Output: (batch, sequence, hidden_size)

    Uses pure PyTorch operations and the compressed-KV MLA
    computation path. No explicit full K/V reconstruction.

    Required config attributes:
        hidden_size
        num_attention_heads
        max_seq_len
        rope_theta

    Optional config attributes:
        q_lora_rank
        kv_lora_rank
        qk_nope_head_dim
        qk_rope_head_dim
        v_head_dim
        attention_dropout
        rms_norm_eps
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.max_seq_len = config.max_seq_len
        self.attention_dropout = getattr(
            config, "attention_dropout", 0.0
        )

        # Small defaults for an initial experiment.
        self.q_lora_rank = getattr(config, "q_lora_rank", 64)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 64)

        # The head dimension is split into non-rotary and rotary parts.
        head_dim = getattr(
            config,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )

        self.qk_nope_head_dim = getattr(
            config, "qk_nope_head_dim", head_dim // 2
        )
        self.qk_rope_head_dim = getattr(
            config, "qk_rope_head_dim", head_dim - self.qk_nope_head_dim
        )
        self.v_head_dim = getattr(config, "v_head_dim", head_dim)

        if self.qk_nope_head_dim <= 0:
            raise ValueError("qk_nope_head_dim must be positive")

        if self.qk_rope_head_dim <= 0:
            raise ValueError("qk_rope_head_dim must be positive")

        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for RoPE")

        if self.q_lora_rank < 0 or self.kv_lora_rank <= 0:
            raise ValueError("Invalid MLA low-rank dimensions")

        if self.v_head_dim <= 0:
            raise ValueError("v_head_dim must be positive")

        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")

        if self.q_lora_rank == 0:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads
                * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=False,
            )
        else:
            self.q_down_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=False
            )
            self.q_norm = RMSNorm(
                self.q_lora_rank,
                getattr(config, "rms_norm_eps", 1e-6),
            )
            self.q_up_proj = nn.Linear(
                self.q_lora_rank,
                self.num_attention_heads
                * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=False,
            )

        # Compress the input into a shared KV latent and a shared
        # rotary key component.
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )

        self.kv_norm = RMSNorm(
            self.kv_lora_rank,
            getattr(config, "rms_norm_eps", 1e-6),
        )

        # Expands the latent into per-head non-rotary keys and values.
        # During attention, its key portion is used in transposed form
        # to transform queries into latent space.
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_attention_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )

        self.softmax_scale = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        ) ** -0.5

        # Apply RoPE only to the rotary subspace, not to the entire head.
        self.rope = RoPE(
            self.qk_rope_head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape

        if t > self.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        # ---------------------------------------------------------
        # 1. Query projection and optional query compression.
        # ---------------------------------------------------------
        if self.q_lora_rank == 0:
            q = self.q_proj(x)
        else:
            q = self.q_up_proj(
                self.q_norm(self.q_down_proj(x))
            )

        q = q.view(
            b,
            t,
            self.num_attention_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )

        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        q_pe = self.rope(q_pe)

        # ---------------------------------------------------------
        # 2. Compress K/V into a shared latent representation.
        # ---------------------------------------------------------
        kv = self.kv_down_proj(x)

        kv, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.kv_norm(kv)

        # One shared rotary key component across attention heads.
        k_pe = self.rope(k_pe.unsqueeze(2)).squeeze(2)

        # ---------------------------------------------------------
        # 3. Transform non-rotary queries into KV latent space.
        #
        # Original DeepSeek optimized MLA:
        # q_nope @ W_K  compared with  compressed KV latent.
        #
        # No full non-rotary K tensor is materialized.
        # ---------------------------------------------------------
        kv_weight = self.kv_up_proj.weight.reshape(
            self.num_attention_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )

        w_k = kv_weight[:, :self.qk_nope_head_dim, :]

        q_latent = torch.einsum(
            "bshd,hdc->bshc",
            q_nope,
            w_k,
        )

        # ---------------------------------------------------------
        # 4. Attention scores.
        #
        # Non-rotary contribution:
        #     transformed query @ compressed KV
        #
        # Rotary contribution:
        #     rotary query @ shared rotary key
        # ---------------------------------------------------------
        scores = torch.einsum(
            "bshc,btc->bsht",
            q_latent,
            kv,
        )

        scores = scores + torch.einsum(
            "bshr,btr->bsht",
            q_pe,
            k_pe,
        )

        scores = scores * self.softmax_scale

        # ---------------------------------------------------------
        # 5. Causal and padding masks.
        #
        # A boolean attention_mask uses True/1 for valid tokens,
        # matching the existing GQAAttention interface.
        # ---------------------------------------------------------
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

            causal = torch.ones(
                (t, t), device=x.device, dtype=torch.bool
            ).tril()

            # Shape: (B, 1, T, T).
            allowed = (
                causal[None, None, :, :]
                & valid[:, None, None, :]
            )

            scores = scores.masked_fill(
                ~allowed,
                torch.finfo(scores.dtype).min,
            )
        else:
            # No padding mask: causal attention only.
            causal = torch.ones(
                (t, t), device=x.device, dtype=torch.bool
            ).tril()

            scores = scores.masked_fill(
                ~causal[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )

        # Float32 softmax improves numerical stability.
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)

        if attention_mask is not None:
            # Avoid producing meaningful outputs for padded queries.
            attn = attn * valid[:, None, :, None].to(attn.dtype)

        attn = F.dropout(
            attn,
            p=self.attention_dropout,
            training=self.training,
        )

        # ---------------------------------------------------------
        # 6. Aggregate compressed KV latents, then project directly
        #    into value space.
        #
        # No per-head full V tensor is materialized.
        # ---------------------------------------------------------
        out_latent = torch.einsum(
            "bsht,btc->bshc",
            attn,
            kv,
        )

        w_v = kv_weight[
            :, self.qk_nope_head_dim:, :
        ]

        out = torch.einsum(
            "bshc,hvc->bshv",
            out_latent,
            w_v,
        )

        # ---------------------------------------------------------
        # 7. Merge heads and apply the output projection.
        # ---------------------------------------------------------
        out = out.contiguous().view(
            b, t, self.num_attention_heads * self.v_head_dim
        )

        return self.o_proj(out)


# Optional alias if you want to keep the existing class name
# in your model code.
