
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pos_enc import RoPE


class RMSNorm(nn.Module):
    """RMSNorm for MLA's compressed query and KV representations."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float * self.weight.float()).to(input_dtype)


class MLAAttention(nn.Module):
    """
    DeepSeek-style Multi-Head Latent Attention.

    Compatible with:
        attention = MLAAttention(config)
        output = attention(x, attention_mask=None)

    Input/output:
        x:      (B, T, hidden_size)
        output: (B, T, hidden_size)

    Required config:
        hidden_size
        num_attention_heads
        max_seq_len
        rope_theta

    Optional config:
        q_lora_rank
        kv_lora_rank
        qk_nope_head_dim
        qk_rope_head_dim
        v_head_dim
        head_dim
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

        self.q_lora_rank = getattr(config, "q_lora_rank", 64)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 64)

        head_dim = getattr(
            config,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )

        self.qk_nope_head_dim = getattr(
            config, "qk_nope_head_dim", head_dim // 2
        )

        self.qk_rope_head_dim = getattr(
            config,
            "qk_rope_head_dim",
            head_dim - self.qk_nope_head_dim,
        )

        self.v_head_dim = getattr(config, "v_head_dim", head_dim)

        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError("hidden_size and num_attention_heads must be positive")

        if self.q_lora_rank < 0 or self.kv_lora_rank <= 0:
            raise ValueError("Invalid MLA low-rank dimensions")

        if self.qk_nope_head_dim <= 0:
            raise ValueError("qk_nope_head_dim must be positive")

        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive and even")

        if self.v_head_dim <= 0:
            raise ValueError("v_head_dim must be positive")

        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError("attention_dropout must be between 0 and 1")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        qk_head_dim = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )

        # ---------------------------------------------------------
        # 1. Query projection.
        # ---------------------------------------------------------
        if self.q_lora_rank == 0:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads * qk_head_dim,
                bias=False,
            )
        else:
            self.q_down_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
            )

            self.q_norm = RMSNorm(
                self.q_lora_rank,
                self.rms_norm_eps,
            )

            self.q_up_proj = nn.Linear(
                self.q_lora_rank,
                self.num_attention_heads * qk_head_dim,
                bias=False,
            )

        # ---------------------------------------------------------
        # 2. Shared compressed KV latent and rotary key projection.
        # ---------------------------------------------------------
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )

        self.kv_norm = RMSNorm(
            self.kv_lora_rank,
            self.rms_norm_eps,
        )

        # Produces each head's non-rotary key and value components
        # from the same compressed KV latent.
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_attention_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # ---------------------------------------------------------
        # 3. Output projection.
        # ---------------------------------------------------------
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )

        self.softmax_scale = qk_head_dim ** -0.5

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
        # 1. Project queries and split rotary/non-rotary dimensions.
        # ---------------------------------------------------------
        if self.q_lora_rank == 0:
            q = self.q_proj(x)
        else:
            q = self.q_up_proj(
                self.q_norm(self.q_down_proj(x))
            )

        q = q.reshape(
            b,
            t,
            self.num_attention_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )

        q_nope, q_pe = torch.split(
            q,
            [
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        q_pe = self.rope(q_pe)

        # ---------------------------------------------------------
        # 2. Compress KV. The latent is shared across all heads.
        # ---------------------------------------------------------
        kv = self.kv_down_proj(x)

        kv, k_pe = torch.split(
            kv,
            [
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )

        kv = self.kv_norm(kv)

        # Apply RoPE to the shared rotary key component.
        k_pe = self.rope(k_pe.unsqueeze(2)).squeeze(2)

        # ---------------------------------------------------------
        # 3. Transform non-rotary queries into latent space.
        #
        # kv_up_proj.weight has shape:
        # (heads * (key_dim + value_dim), kv_lora_rank)
        # ---------------------------------------------------------
        kv_weight = self.kv_up_proj.weight.reshape(
            self.num_attention_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )

        w_k = kv_weight[:, :self.qk_nope_head_dim, :]

        q_latent = torch.einsum(
            "bthd,hdc->bthc",
            q_nope,
            w_k,
        )

        # ---------------------------------------------------------
        # 4. Calculate attention scores.
        #
        # Shape convention: (B, query_T, heads, key_T)
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
        # 5. Construct the causal mask.
        #
        # FIX: scores are (B, T, H, T), so the causal mask must
        # broadcast as (1, T, 1, T), NOT (1, 1, T, T).
        # ---------------------------------------------------------
        causal = torch.ones(
            (t, t),
            device=x.device,
            dtype=torch.bool,
        ).tril()

        # (1, query_T, 1, key_T)
        allowed = causal[None, :, None, :]

        valid = None

        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

            if attention_mask.device != x.device:
                attention_mask = attention_mask.to(device=x.device)

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

            # Mask invalid keys and padded query positions.
            allowed = (
                allowed
                & valid[:, None, None, :]
                & valid[:, :, None, None]
            )

        scores = scores.masked_fill(
            ~allowed,
            torch.finfo(scores.dtype).min,
        )

        # ---------------------------------------------------------
        # 6. Softmax in float32 for improved numerical stability.
        # ---------------------------------------------------------
        attn = F.softmax(
            scores.float(),
            dim=-1,
        ).to(dtype=x.dtype)

        if valid is not None:
            attn = attn * valid[:, :, None, None].to(attn.dtype)

        attn = F.dropout(
            attn,
            p=self.attention_dropout,
            training=self.training,
        )

        # ---------------------------------------------------------
        # 7. Aggregate the compressed KV latent.
        # No full per-head K or V tensors are materialized.
        # ---------------------------------------------------------
        out_latent = torch.einsum(
            "bsht,btc->bshc",
            attn,
            kv,
        )

        # Direct latent-to-value projection.
        w_v = kv_weight[:, self.qk_nope_head_dim:, :]

        out = torch.einsum(
            "bshc,hvc->bshv",
            out_latent,
            w_v,
        )

        # ---------------------------------------------------------
        # 8. Merge heads and project to hidden_size.
        # ---------------------------------------------------------
        out = out.contiguous().reshape(
            b,
            t,
            self.num_attention_heads * self.v_head_dim,
        )

        return self.o_proj(out)


# Keep compatibility with model.py if it imports GQAAttention.
GQAAttention = MLAAttention
