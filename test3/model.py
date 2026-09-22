import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CENTRALIZED MODEL CONFIG
# ============================================================

# Vocabulary / Model
vocab_size = 8192
hidden_size = 256
num_hidden_layers = 6

# MLP / SwiGLU
intermediate_size = 768

# Attention / GQA
num_attention_heads = 8
num_key_value_heads = 2
attention_dropout = 0.0

# Model dropout
hidden_dropout = 0.0

# RMSNorm
rms_norm_eps = 1e-5

# RoPE
rope_theta = 10000.0
max_seq_len = 256

# Output
tie_word_embeddings = True

# Initialization
initializer_range = 0.02


# ============================================================
# DERIVED VALUES
# ============================================================

head_dim = hidden_size // num_attention_heads
num_queries_per_kv = num_attention_heads // num_key_value_heads


# ============================================================
# SANITY CHECKS
# ============================================================

assert vocab_size > 0
assert hidden_size > 0
assert num_hidden_layers > 0

assert hidden_size % num_attention_heads == 0
assert num_attention_heads % num_key_value_heads == 0
assert head_dim % 2 == 0

assert intermediate_size > hidden_size

assert max_seq_len > 0

assert rms_norm_eps > 0

assert attention_dropout >= 0.0
assert attention_dropout < 1.0

assert hidden_dropout >= 0.0
assert hidden_dropout < 1.0

assert rope_theta > 0

assert initializer_range > 0

assert isinstance(tie_word_embeddings, bool)


# ============================================================
# RMSNORM
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=rms_norm_eps):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight


# ============================================================
# ROPE
# ============================================================

class RoPE(nn.Module):
    def __init__(
        self,
        dim=head_dim,
        max_seq_len=max_seq_len,
        theta=rope_theta,
    ):
        super().__init__()

        freqs = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(
            positions,
            freqs,
        )

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False,
        )

    def forward(self, x):
        # x:
        # [batch, sequence, heads, head_dim]

        seq_len = x.shape[1]

        assert seq_len <= self.cos.shape[0], (
            f"Sequence length {seq_len} exceeds "
            f"max_seq_len {self.cos.shape[0]}"
        )

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # [S, D/2]
        # ->
        # [1, S, 1, D/2]

        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotated = torch.stack(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos,
            ],
            dim=-1,
        )

        return rotated.flatten(-2)


# ============================================================
# GQA ATTENTION
# ============================================================

class GQAAttention(nn.Module):
    def __init__(
        self,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        attention_dropout=attention_dropout,
    ):
        super().__init__()

        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

        self.hidden_size = hidden_size

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_queries_per_kv = (
            num_attention_heads // num_key_value_heads
        )

        self.head_dim = (
            hidden_size // num_attention_heads
        )

        # ----------------------------------------------------
        # Query projection
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias=False,
        )

        # ----------------------------------------------------
        # Key projection
        # ----------------------------------------------------

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )

        # ----------------------------------------------------
        # Value projection
        # ----------------------------------------------------

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias=False,
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.o_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Rotary Position Embedding
        # ----------------------------------------------------

        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )

        self.attention_dropout = attention_dropout

    def repeat_kv(self, x):
        # x:
        # [batch, sequence, kv_heads, head_dim]

        if self.num_queries_per_kv == 1:
            return x

        batch, seq_len, heads, dim = x.shape

        x = x[:, :, :, None, :]

        # [B, S, KV, 1, D]
        #
        # ->
        #
        # [B, S, KV, queries_per_KV, D]

        x = x.expand(
            batch,
            seq_len,
            heads,
            self.num_queries_per_kv,
            dim,
        )

        return x.reshape(
            batch,
            seq_len,
            self.num_attention_heads,
            dim,
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # ----------------------------------------------------
        # Project Q, K, V
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------------------------
        # Split into heads
        # ----------------------------------------------------

        q = q.view(
            batch,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        )

        k = k.view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        v = v.view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # ----------------------------------------------------
        # Apply RoPE
        # ----------------------------------------------------

        q = self.rope(q)
        k = self.rope(k)

        # ----------------------------------------------------
        # Expand K/V for GQA
        # ----------------------------------------------------

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # ----------------------------------------------------
        # [B, S, H, D]
        # ->
        # [B, H, S, D]
        # ----------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------------------------------------------
        # Causal self-attention
        # ----------------------------------------------------

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.attention_dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        # ----------------------------------------------------
        # [B, H, S, D]
        # ->
        # [B, S, H, D]
        # ----------------------------------------------------

        y = y.transpose(1, 2).contiguous()

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        y = y.view(
            batch,
            seq_len,
            self.hidden_size,
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        return self.o_proj(y)


# ============================================================
# SWIGLU
# ============================================================

class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x))
            * self.up_proj(x)
        )


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Pre-attention RMSNorm
        # ----------------------------------------------------

        self.input_layernorm = RMSNorm(
            hidden_size,
            rms_norm_eps,
        )

        # ----------------------------------------------------
        # GQA self-attention
        # ----------------------------------------------------

        self.self_attn = GQAAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
        )

        # ----------------------------------------------------
        # Pre-MLP RMSNorm
        # ----------------------------------------------------

        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            rms_norm_eps,
        )

        # ----------------------------------------------------
        # SwiGLU MLP
        # ----------------------------------------------------

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x):

        # Attention residual
        x = x + self.self_attn(
            self.input_layernorm(x)
        )

        # MLP residual
        x = x + self.mlp(
            self.post_attention_layernorm(x)
        )

        return x


# ============================================================
# MODEL 5555
# ============================================================

class Model5555(nn.Module):
    def __init__(
        self,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        hidden_dropout=hidden_dropout,
        tie_word_embeddings=tie_word_embeddings,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Store model configuration
        # ----------------------------------------------------

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len

        self.tie_word_embeddings = tie_word_embeddings

        # ----------------------------------------------------
        # Token embedding
        #
        # [B, S]
        # ->
        # [B, S, hidden_size]
        # ----------------------------------------------------

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        # ----------------------------------------------------
        # Input dropout
        # ----------------------------------------------------

        self.embed_dropout = nn.Dropout(
            hidden_dropout
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                Block()
                for _ in range(num_hidden_layers)
            ]
        )

        # ----------------------------------------------------
        # Final RMSNorm
        # ----------------------------------------------------

        self.final_layernorm = RMSNorm(
            hidden_size,
            rms_norm_eps,
        )

        # ----------------------------------------------------
        # Language-model head
        #
        # [B, S, hidden_size]
        # ->
        # [B, S, vocab_size]
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ----------------------------------------------------
        # Initialize weights
        # ----------------------------------------------------

        self.apply(self._init_weights)

        # Re-tie after initialization for clarity.
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=initializer_range,
            )

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, input_ids):

        # ----------------------------------------------------
        # Input:
        #
        # [B, S]
        #
        # ----------------------------------------------------

        batch_size, seq_len = input_ids.shape

        # ----------------------------------------------------
        # Model-level input checks
        # ----------------------------------------------------

        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds "
            f"max_seq_len {self.max_seq_len}"
        )

        assert input_ids.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ), (
            "input_ids must contain integer token IDs"
        )

        assert torch.all(input_ids >= 0), (
            "input_ids contains a negative token ID"
        )

        assert torch.all(input_ids < self.vocab_size), (
            "input_ids contains a token ID >= vocab_size"
        )

        # ----------------------------------------------------
        # Token embedding
        #
        # [B, S]
        # ->
        # [B, S, D]
        # ----------------------------------------------------

        x = self.embed_tokens(input_ids)

        # ----------------------------------------------------
        # Input dropout
        # ----------------------------------------------------

        x = self.embed_dropout(x)

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        for layer in self.layers:
            x = layer(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_layernorm(x)

        # ----------------------------------------------------
        # Language-model head
        #
        # [B, S, D]
        # ->
        # [B, S, V]
        # ----------------------------------------------------

        logits = self.lm_head(x)

        return logits
