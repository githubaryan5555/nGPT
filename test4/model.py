"""
Small decoder-only language model used by ``train.py``.

Contract
--------
model(input_ids):
    input_ids: torch.long tensor [B, T]
    returns:   logits tensor [B, T, vocab_size]

train.py is responsible for shifting labels and calculating cross entropy.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    vocab_size: int = cfg.vocab_size
    hidden_size: int = cfg.hidden_size
    num_hidden_layers: int = cfg.num_hidden_layers
    intermediate_size: int = cfg.intermediate_size

    num_attention_heads: int = cfg.num_attention_heads
    num_key_value_heads: int = cfg.num_key_value_heads

    attention_dropout: float = cfg.attention_dropout
    hidden_dropout: float = cfg.hidden_dropout

    rms_norm_eps: float = cfg.rms_norm_eps
    rope_theta: float = cfg.rope_theta
    max_seq_len: int = cfg.max_seq_len

    tie_word_embeddings: bool = cfg.tie_word_embeddings
    initializer_range: float = cfg.initializer_range

    def __post_init__(self):
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "max_seq_len",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")

            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE"
            )

        if self.intermediate_size <= self.hidden_size:
            raise ValueError(
                "intermediate_size must be greater than hidden_size"
            )

        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be > 0")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")

        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be > 0")

        for name in ("attention_dropout", "hidden_dropout"):
            value = getattr(self, name)

            if not 0 <= value < 1:
                raise ValueError(
                    f"{name} must be in [0, 1)"
                )

        if not isinstance(self.tie_word_embeddings, bool):
            raise TypeError(
                "tie_word_embeddings must be a bool"
            )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self):
        return self.num_attention_heads // self.num_key_value_heads


# ============================================================
# RMS NORM
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the variance in fp32 for numerical stability.
        variance = x.float().pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)

        return x * self.weight


# ============================================================
# ROPE
# ============================================================

class RoPE(nn.Module):
    """
    Rotary positional embeddings.

    Input:
        [B, T, H, D]

    Output:
        [B, T, H, D]
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        angles = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos",
            angles.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            angles.sin(),
            persistent=False,
        )

    def forward(self, x):
        # x: [B, T, H, D]

        seq_len = x.size(1)

        if seq_len > self.cos.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"RoPE limit {self.cos.size(0)}"
            )

        cos = self.cos[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = self.sin[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        # [1, T, 1, D/2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)


# ============================================================
# GQA ATTENTION
# ============================================================

class GQAAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv

        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout

        # Query has all attention heads.
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )

        # K/V use fewer heads for GQA.
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
        """
        Expand K/V heads for GQA.

        Input:
            [B, T, KV_heads, D]

        Output:
            [B, T, attention_heads, D]
        """

        if self.num_queries_per_kv == 1:
            return x

        b, t, kv_heads, d = x.shape

        x = x[:, :, :, None, :]

        x = x.expand(
            b,
            t,
            kv_heads,
            self.num_queries_per_kv,
            d,
        )

        return x.reshape(
            b,
            t,
            self.num_attention_heads,
            d,
        )

    def forward(self, x):
        b, t, _ = x.shape

        # ----------------------------------------------------
        # Q / K / V projections
        # ----------------------------------------------------

        q = self.q_proj(x).view(
            b,
            t,
            self.num_attention_heads,
            self.head_dim,
        )

        k = self.k_proj(x).view(
            b,
            t,
            self.num_key_value_heads,
            self.head_dim,
        )

        v = self.v_proj(x).view(
            b,
            t,
            self.num_key_value_heads,
            self.head_dim,
        )

        # ----------------------------------------------------
        # Rotary position embeddings
        # ----------------------------------------------------

        q = self.rope(q)
        k = self.rope(k)

        # ----------------------------------------------------
        # Repeat K/V for GQA
        # ----------------------------------------------------

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # ----------------------------------------------------
        # [B, T, H, D] -> [B, H, T, D]
        # ----------------------------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------------------------------------------
        # PyTorch SDPA
        #
        # is_causal=True gives decoder-style masking.
        # PyTorch can select an efficient attention kernel
        # depending on device/dtype/backend.
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

        # [B, H, T, D] -> [B, T, H, D]
        y = y.transpose(1, 2)

        # Merge heads.
        y = y.contiguous().view(
            b,
            t,
            self.hidden_size,
        )

        return self.o_proj(y)


# ============================================================
# SWIGLU
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        return self.down_proj(gate * up)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.self_attn = GQAAttention(config)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.mlp = SwiGLU(config)

    def forward(self, x):
        # Pre-LN attention residual.
        x = x + self.self_attn(
            self.input_layernorm(x)
        )

        # Pre-LN MLP residual.
        x = x + self.mlp(
            self.post_attention_layernorm(x)
        )

        return x


# ============================================================
# MODEL
# ============================================================

class Model5555LM(nn.Module):
    """
    Causal language model.

    Input:
        [B, T] torch.long

    Output:
        [B, T, vocab_size]
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        **overrides,
    ):
        super().__init__()

        if config is None:
            config = Config()

        if not isinstance(config, Config):
            raise TypeError(
                "config must be an instance of Config"
            )

        # Allow train.py / experiments to override config.
        if overrides:
            values = asdict(config)
            values.update(overrides)
            config = Config(**values)

        self.config = config

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.embed_dropout = nn.Dropout(
            config.hidden_dropout
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        # ----------------------------------------------------
        # Language-model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        self.apply(self._init_weights)

        # Weight tying.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Depth-scaled residual initialization.
        #
        # This helps keep residual branches controlled as the
        # number of layers increases.
        residual_std = (
            config.initializer_range
            / (
                (2 * config.num_hidden_layers) ** 0.5
            )
        )

        for layer in self.layers:
            nn.init.normal_(
                layer.self_attn.o_proj.weight,
                mean=0.0,
                std=residual_std,
            )

            nn.init.normal_(
                layer.mlp.down_proj.weight,
                mean=0.0,
                std=residual_std,
            )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, input_ids):
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "input_ids must be a torch.Tensor"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [B, T], "
                f"got {tuple(input_ids.shape)}"
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids must be torch.long, "
                f"got {input_ids.dtype}"
            )

        batch_size, seq_len = input_ids.shape

        if batch_size <= 0:
            raise ValueError(
                "batch size must be > 0"
            )

        if seq_len <= 0:
            raise ValueError(
                "sequence length must be > 0"
            )

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )

        if input_ids.numel():
            min_id = input_ids.min()
            max_id = input_ids.max()

            if min_id < 0 or max_id >= self.config.vocab_size:
                raise ValueError(
                    "input_ids contains a token outside "
                    f"[0, {self.config.vocab_size})"
                )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        x = self.embed_tokens(input_ids)

        x = self.embed_dropout(x)

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for layer in self.layers:
            x = layer(x)

        # ----------------------------------------------------
        # Final LM head
        # ----------------------------------------------------

        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits

    # ========================================================
    # PARAMETER INFORMATION
    # ========================================================

    def get_num_params(self):
        """
        Number of unique model parameters.

        Shared/tied parameters are counted only once because
        PyTorch's parameters() iterator deduplicates them.
        """
        return sum(
            p.numel()
            for p in self.parameters()
        )

    def get_trainable_params(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def get_model_size_mb(self, dtype_bytes=2):
        if dtype_bytes <= 0:
            raise ValueError(
                "dtype_bytes must be > 0"
            )

        return (
            self.get_num_params()
            * dtype_bytes
            / 1024**2
        )

    def get_model_size_gb(self, dtype_bytes=2):
        if dtype_bytes <= 0:
            raise ValueError(
                "dtype_bytes must be > 0"
            )

        return (
            self.get_num_params()
            * dtype_bytes
            / 1024**3
        )

    # ========================================================
    # FLOPs
    # ========================================================

    def get_flops_per_token(self, seq_len=None):
        """
        Approximate forward-pass FLOPs per token.

        Includes:
            - Q/K/V/O projections
            - SwiGLU
            - attention score/value operations
            - final vocabulary projection

        This is an approximation, not a hardware profiler.
        """

        if seq_len is None:
            seq_len = self.config.max_seq_len

        if not isinstance(seq_len, int):
            raise TypeError(
                "seq_len must be an int"
            )

        if seq_len <= 0 or seq_len > self.config.max_seq_len:
            raise ValueError(
                "seq_len is outside the model context"
            )

        d = self.config.hidden_size
        f = self.config.intermediate_size
        h = self.config.num_attention_heads
        kv = self.config.num_key_value_heads
        v = self.config.vocab_size
        layers = self.config.num_hidden_layers

        head_dim = d // h

        # ----------------------------------------------------
        # Linear projections
        #
        # Q: d -> d
        # K: d -> kv * head_dim
        # V: d -> kv * head_dim
        # O: d -> d
        #
        # Multiply + add ~= 2 FLOPs.
        # ----------------------------------------------------

        projection_flops = 2 * (
            d * d
            + d * (kv * head_dim)
            + d * (kv * head_dim)
            + d * d
        )

        # ----------------------------------------------------
        # SwiGLU
        #
        # gate projection
        # up projection
        # down projection
        # ----------------------------------------------------

        mlp_flops = 6 * d * f

        # ----------------------------------------------------
        # Attention.
        #
        # QK^T:
        #   ~2 * T * T * d
        #
        # Attention @ V:
        #   ~2 * T * T * d
        #
        # Total sequence FLOPs:
        #   ~4 * T^2 * d
        #
        # Per token:
        #   ~4 * T * d
        # ----------------------------------------------------

        attention_flops_per_token = 4 * seq_len * d

        # ----------------------------------------------------
        # Vocabulary projection.
        # ----------------------------------------------------

        lm_head_flops = 2 * d * v

        per_layer = (
            projection_flops
            + mlp_flops
            + attention_flops_per_token
        )

        return (
            layers * per_layer
            + lm_head_flops
        )

    def estimate_mfu(
        self,
        tokens_per_second,
        peak_flops,
    ):
        """
        Approximate Model FLOPs Utilization percentage.
        """

        if tokens_per_second < 0:
            raise ValueError(
                "tokens_per_second must be >= 0"
            )

        if peak_flops <= 0:
            raise ValueError(
                "peak_flops must be > 0"
            )

        model_flops = self.get_flops_per_token()

        return (
            tokens_per_second
            * model_flops
            / peak_flops
            * 100.0
        )

    # ========================================================
    # CONFIG
    # ========================================================

    def get_config(self):
        return asdict(self.config)

    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        text,
        tokenizer,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
        do_sample=True,
    ):
        """
        Generate text from a string prompt.

        The full generated sequence is preserved even when the
        model has to truncate its attention context.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not isinstance(max_new_tokens, int):
            raise TypeError(
                "max_new_tokens must be an int"
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be >= 0"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        if top_k is not None:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(
                    "top_k must be a positive integer"
                )

        if top_p is not None:
            if not 0 < top_p <= 1:
                raise ValueError(
                    "top_p must be in (0, 1]"
                )

        if eos_token_id is not None:
            if not isinstance(eos_token_id, int):
                raise TypeError(
                    "eos_token_id must be an int"
                )

            if not 0 <= eos_token_id < self.config.vocab_size:
                raise ValueError(
                    "eos_token_id is outside vocabulary"
                )

        # ----------------------------------------------------
        # Tokenize prompt
        # ----------------------------------------------------

        encoded = tokenizer.encode(text)

        if isinstance(encoded, torch.Tensor):
            ids = encoded.to(
                dtype=torch.long
            )
        else:
            ids = torch.tensor(
                encoded,
                dtype=torch.long,
            )

        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        if ids.ndim != 2:
            raise ValueError(
                "tokenizer.encode must return one sequence"
            )

        if ids.size(0) != 1:
            raise ValueError(
                "generate() only supports one prompt"
            )

        if ids.size(1) == 0:
            raise ValueError(
                "prompt must contain at least one token"
            )

        # Keep the complete sequence separately.
        output_ids = ids.clone()

        device = self.embed_tokens.weight.device

        output_ids = output_ids.to(
            device=device,
            dtype=torch.long,
        )

        self.eval()

        # ----------------------------------------------------
        # Generation loop
        # ----------------------------------------------------

        for _ in range(max_new_tokens):

            # Only the latest context is fed into the model.
            context_ids = output_ids[
                :,
                -self.config.max_seq_len:,
            ]

            logits = self(context_ids)

            # Only the final position predicts the next token.
            logits = logits[:, -1, :]

            if not do_sample:
                next_token = logits.argmax(
                    dim=-1,
                    keepdim=True,
                )

            else:
                # --------------------------------------------
                # Temperature
                # --------------------------------------------

                logits = logits / temperature

                # --------------------------------------------
                # Top-k
                # --------------------------------------------

                if top_k is not None:
                    k = min(
                        top_k,
                        logits.size(-1),
                    )

                    top_values = torch.topk(
                        logits,
                        k,
                        dim=-1,
                    ).values

                    threshold = top_values[:, [-1]]

                    logits = logits.masked_fill(
                        logits < threshold,
                        float("-inf"),
                    )

                # --------------------------------------------
                # Top-p / nucleus sampling
                # --------------------------------------------

                if top_p is not None and top_p < 1.0:

                    sorted_logits, sorted_indices = torch.sort(
                        logits,
                        descending=True,
                        dim=-1,
                    )

                    sorted_probs = F.softmax(
                        sorted_logits,
                        dim=-1,
                    )

                    cumulative_probs = torch.cumsum(
                        sorted_probs,
                        dim=-1,
                    )

                    remove = cumulative_probs > top_p

                    # Keep the first token above the threshold.
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    sorted_logits = sorted_logits.masked_fill(
                        remove,
                        float("-inf"),
                    )

                    logits = torch.full_like(logits, float("-inf"))

                    logits.scatter_(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_logits,
                    )

                # --------------------------------------------
                # Sample
                # --------------------------------------------

                probs = F.softmax(
                    logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            # Append to complete sequence.
            output_ids = torch.cat(
                (
                    output_ids,
                    next_token,
                ),
                dim=1,
            )

            # Stop on EOS.
            if (
                eos_token_id is not None
                and bool(
                    (next_token == eos_token_id).all()
                )
            ):
                break

        # ----------------------------------------------------
        # Decode complete sequence
        # ----------------------------------------------------

        return tokenizer.decode(
            output_ids[0].tolist()
        )
