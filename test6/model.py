"""Small decoder-only language model used by ``train.py``."""

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import config as cfg
except ImportError:  # Support running ``python test4/model.py`` directly.
    import config as cfg


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
            "vocab_size", "hidden_size", "num_hidden_layers",
            "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "max_seq_len",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must be <= num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.intermediate_size <= self.hidden_size:
            raise ValueError("intermediate_size must be greater than hidden_size")

        for name in ("rms_norm_eps", "rope_theta", "initializer_range"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("attention_dropout", "hidden_dropout"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not isinstance(self.tie_word_embeddings, bool):
            raise TypeError("tie_word_embeddings must be a bool")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self):
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).to(x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
        angles = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32), inv_freq
        )
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, position_offset=0):
        seq_len = x.size(1)
        end = position_offset + seq_len
        if position_offset < 0 or end > self.cos.size(0):
            raise ValueError(
                f"position range [{position_offset}, {end}) exceeds RoPE limit "
                f"{self.cos.size(0)}"
            )
        cos = self.cos[position_offset:end].to(device=x.device, dtype=x.dtype)
        sin = self.sin[position_offset:end].to(device=x.device, dtype=x.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


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


class SwiGLU(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


                                                * 100.0
        

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

        self.hidden_dropout = nn.Dropout(
            config.hidden_dropout,
        )

        # ------------------------------------------------------------
        # 64 x 64 learned multiplicative modulation matrices.
        #
        # Each modulation matrix is repeated over its corresponding
        # weight matrix:
        #
        #     W_new = W * tiled(A)
        #
        # A is initialized near 1 so the initial behavior remains
        # close to the original model.
        # ------------------------------------------------------------

        modulation_size = 64

        self.q_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.k_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.v_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.o_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.gate_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.up_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

        self.down_mod = nn.Parameter(
            torch.ones(
                modulation_size,
                modulation_size,
            )
        )

    @staticmethod
    def _modulate_weight(weight, modulation):
        """
        Apply a repeating 64x64 multiplicative pattern to weight.

        Mathematically:

            W_new[i, j] =
                W[i, j] * A[i % 64, j % 64]

        No full tiled modulation matrix is materialized.
        """

        rows, cols = weight.shape
        m_rows, m_cols = modulation.shape

        if rows % m_rows != 0 or cols % m_cols != 0:
            raise ValueError(
                f"Weight shape {tuple(weight.shape)} must be divisible "
                f"by modulation shape {tuple(modulation.shape)}"
            )

        row_groups = rows // m_rows
        col_groups = cols // m_cols

        modulation_view = modulation.reshape(
            1,
            m_rows,
            1,
            m_cols,
        )

        weight_view = weight.reshape(
            row_groups,
            m_rows,
            col_groups,
            m_cols,
        )

        return (
            weight_view * modulation_view
        ).reshape_as(weight)

    def _modulated_linear(self, x, linear, modulation):
        weight = self._modulate_weight(
            linear.weight,
            modulation,
        )

        return F.linear(
            x,
            weight,
            linear.bias,
        )

    def forward(self, x, attention_mask=None):

        # ------------------------------------------------------------
        # Attention
        #
        # Apply modulation to q/k/v/o weights.
        # ------------------------------------------------------------

        norm_x = self.input_layernorm(x)

        attn = self.self_attn

        b, t, _ = norm_x.shape

        q = self._modulated_linear(
            norm_x,
            attn.q_proj,
            self.q_mod,
        ).view(
            b,
            t,
            attn.num_attention_heads,
            attn.head_dim,
        )

        k = self._modulated_linear(
            norm_x,
            attn.k_proj,
            self.k_mod,
        ).view(
            b,
            t,
            attn.num_key_value_heads,
            attn.head_dim,
        )

        v = self._modulated_linear(
            norm_x,
            attn.v_proj,
            self.v_mod,
        ).view(
            b,
            t,
            attn.num_key_value_heads,
            attn.head_dim,
        )

        q, k = attn.rope(q), attn.rope(k)

        q = q.transpose(1, 2)

        k = attn.repeat_kv(k).transpose(1, 2)

        v = attn.repeat_kv(v).transpose(1, 2)

        attn_mask = None
        is_causal = attention_mask is None

        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}"
                )

            if attention_mask.device != norm_x.device:
                attention_mask = attention_mask.to(
                    device=norm_x.device
                )

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
                (t, t),
                device=norm_x.device,
                dtype=torch.bool,
            ).tril()

            attn_mask = (
                causal[None, None, :, :]
                & valid[:, None, None, :]
                & valid[:, None, :, None]
            )

            is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=(
                attn.attention_dropout
                if self.training
                else 0.0
            ),
            is_causal=is_causal,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(b, t, attn.hidden_size)
        )

        # Modulated output projection.
        y = self._modulated_linear(
            y,
            attn.o_proj,
            self.o_mod,
        )

        x = x + self.hidden_dropout(y)

        # ------------------------------------------------------------
        # SwiGLU
        #
        # Modulate gate, up and down projections.
        # ------------------------------------------------------------

        norm_x = self.post_attention_layernorm(x)

        mlp = self.mlp

        gate = self._modulated_linear(
            norm_x,
            mlp.gate_proj,
            self.gate_mod,
        )

        up = self._modulated_linear(
            norm_x,
            mlp.up_proj,
            self.up_mod,
        )

        mlp_out = self._modulated_linear(
            F.silu(gate) * up,
            mlp.down_proj,
            self.down_mod,
        )

        x = x + self.hidden_dropout(mlp_out)

        return x


class Model5555LM(nn.Module):
    """Causal language model with input/output shapes ``[B, T]`` and ``[B, T, V]``."""

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

        if overrides:
            values = asdict(config)
            values.update(overrides)
            config = Config(**values)

        self.config = config

        # ------------------------------------------------------------
        # Input embedding
        # ------------------------------------------------------------

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.embed_dropout = nn.Dropout(
            config.hidden_dropout,
        )

        # ------------------------------------------------------------
        # ONE physical block.
        #
        # The same block, including the same 64x64 modulation
        # matrices, is reused for every computational repetition.
        # ------------------------------------------------------------

        self.block = Block(config)

        self.num_repeats = config.num_hidden_layers

        # ------------------------------------------------------------
        # Output
        # ------------------------------------------------------------

        self.final_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # ------------------------------------------------------------
        # Standard initialization
        # ------------------------------------------------------------

        self.apply(self._init_weights)

        # ------------------------------------------------------------
        # Initialize modulation matrices AFTER normal initialization.
        #
        # A = 1 + small noise
        #
        # Therefore:
        #
        #     W_new = W * A
        #
        # starts approximately as W.
        # ------------------------------------------------------------

        modulation_std = 0.02

        with torch.no_grad():
            for modulation in (
                self.block.q_mod,
                self.block.k_mod,
                self.block.v_mod,
                self.block.o_mod,
                self.block.gate_mod,
                self.block.up_mod,
                self.block.down_mod,
            ):
                modulation.add_(
                    torch.randn_like(modulation)
                    * modulation_std
                )

        # ------------------------------------------------------------
        # Optional weight tying
        # ------------------------------------------------------------

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ------------------------------------------------------------
        # Residual projection initialization.
        #
        # Scale according to TOTAL computational depth even though
        # there is only one physical block.
        # ------------------------------------------------------------

        residual_std = (
            config.initializer_range
            / math.sqrt(
                2 * config.num_hidden_layers
            )
        )

        nn.init.normal_(
            self.block.self_attn.o_proj.weight,
            std=residual_std,
        )

        nn.init.normal_(
            self.block.mlp.down_proj.weight,
            std=residual_std,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                std=self.config.initializer_range,
            )

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
    ):
        # ------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "input_ids must be a torch.Tensor"
            )

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], "
                f"got {tuple(input_ids.shape)}"
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                f"input_ids must be torch.long, "
                f"got {input_ids.dtype}"
            )

        b, t = input_ids.shape

        if b <= 0 or t <= 0:
            raise ValueError(
                "batch size and sequence length must be > 0"
            )

        if t > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )

        # ------------------------------------------------------------
        # Attention mask validation
        # ------------------------------------------------------------

        if attention_mask is not None:
            if not isinstance(
                attention_mask,
                torch.Tensor,
            ):
                raise TypeError(
                    "attention_mask must be a torch.Tensor"
                )

            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same "
                    "shape as input_ids"
                )

            if (
                attention_mask.device
                != input_ids.device
            ):
                raise ValueError(
                    "attention_mask and input_ids "
                    "must be on the same device"
                )

            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(
                    torch.bool
                )

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
                attention_mask = (
                    attention_mask != 0
                )

            else:
                raise TypeError(
                    "attention_mask must be boolean "
                    "or numeric"
                )

        # ------------------------------------------------------------
        # Token ID validation
        # ------------------------------------------------------------

        if (
            input_ids.min() < 0
            or input_ids.max()
            >= self.config.vocab_size
        ):
            raise ValueError(
                f"input_ids contains a token outside "
                f"[0, {self.config.vocab_size})"
            )

        # ------------------------------------------------------------
        # Embedding
        # ------------------------------------------------------------

        x = self.embed_dropout(
            self.embed_tokens(input_ids)
        )

        hidden_states = (
            [] if output_hidden_states else None
        )

        # ------------------------------------------------------------
        # REUSE THE SAME MODULATED BLOCK.
        #
        # Every repetition uses:
        #
        #   the same W matrices
        #   the same 64x64 modulation matrices
        #
        # Only x changes from repetition to repetition.
        # ------------------------------------------------------------

        for _ in range(self.num_repeats):
            x = self.block(
                x,
                attention_mask,
            )

            if output_hidden_states:
                hidden_states.append(x)

        # ------------------------------------------------------------
        # Final normalization + LM head
        # ------------------------------------------------------------

        logits = self.lm_head(
            self.final_layernorm(x)
        )

        if output_hidden_states:
            return logits, hidden_states

        return logits

    def get_num_params(self):
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
        if (
            not isinstance(dtype_bytes, Real)
            or not math.isfinite(dtype_bytes)
            or dtype_bytes <= 0
        ):
            raise ValueError(
                "dtype_bytes must be finite and > 0"
            )

        return (
            self.get_num_params()
            * dtype_bytes
            / 1024**2
        )

    def get_model_size_gb(self, dtype_bytes=2):
        return (
            self.get_model_size_mb(dtype_bytes)
            / 1024
        )

    def get_flops_per_token(self, seq_len=None):
        if seq_len is None:
            seq_len = self.config.max_seq_len

        if (
            isinstance(seq_len, bool)
            or not isinstance(seq_len, int)
        ):
            raise TypeError(
                "seq_len must be an int"
            )

        if not 0 < seq_len <= self.config.max_seq_len:
            raise ValueError(
                "seq_len is outside the model context"
            )

        d = self.config.hidden_size
        f = self.config.intermediate_size
        h = self.config.num_attention_heads
        kv = self.config.num_key_value_heads
        v = self.config.vocab_size

        # Computationally, the shared block is still executed
        # num_hidden_layers times.
        layers = self.config.num_hidden_layers

        head_dim = d // h

        projection = 2 * (
            d * d
            + 2 * d * kv * head_dim
        )

        mlp = 6 * d * f

        return (
            layers
            * (
                projection
                + mlp
                + 4 * seq_len * d
            )
            + 2 * d * v
        )

    def estimate_mfu(
        self,
        tokens_per_second,
        peak_flops,
    ):
        for name, value in (
            ("tokens_per_second", tokens_per_second),
            ("peak_flops", peak_flops),
        ):
            if (
                not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise TypeError(
                    f"{name} must be a finite number"
                )

        if tokens_per_second < 0:
            raise ValueError(
                "tokens_per_second must be >= 0"
            )

        if peak_flops <= 0:
            raise ValueError(
                "peak_flops must be > 0"
            )

        return (
            tokens_per_second
            * self.get_flops_per_token()
            / peak_flops
            * 100.0
        )

    def get_config(self):
        return asdict(self.config)

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
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
        ):
            raise TypeError(
                "max_new_tokens must be an int"
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be >= 0"
            )

        if not isinstance(do_sample, bool):
            raise TypeError(
                "do_sample must be a bool"
            )

        # ------------------------------------------------------------
        # Keep the remainder of your existing generate() method
        # EXACTLY as it is.
        # ------------------------------------------------------------


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
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
        ):
            raise TypeError(
                "max_new_tokens must be an int"
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be >= 0"
            )

        if not isinstance(do_sample, bool):
            raise TypeError(
                "do_sample must be a bool"
            )

        if not isinstance(temperature, Real):
            raise TypeError(
                "temperature must be a real number"
            )

        if not math.isfinite(temperature):
            raise ValueError(
                "temperature must be finite"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        if top_k is not None:
            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
            ):
                raise TypeError(
                    "top_k must be an int or None"
                )

            if top_k <= 0:
                raise ValueError(
                    "top_k must be > 0"
                )

        if top_p is not None:
            if not isinstance(top_p, Real):
                raise TypeError(
                    "top_p must be a real number or None"
                )

            if not math.isfinite(top_p):
                raise ValueError(
                    "top_p must be finite"
                )

            if not 0 < top_p <= 1:
                raise ValueError(
                    "top_p must be in (0, 1]"
                )

        # ------------------------------------------------------------
        # Tokenize prompt
        # ------------------------------------------------------------

        if hasattr(tokenizer, "encode"):
            input_ids = tokenizer.encode(text)
        else:
            raise TypeError(
                "tokenizer must provide an encode() method"
            )

        if not isinstance(input_ids, (list, tuple)):
            input_ids = list(input_ids)

        if len(input_ids) == 0:
            raise ValueError(
                "tokenizer.encode(text) returned no tokens"
            )

        device = next(self.parameters()).device

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        # ------------------------------------------------------------
        # Generate autoregressively
        # ------------------------------------------------------------

        for _ in range(max_new_tokens):

            # Keep only the most recent context window.
            if input_ids.size(1) > self.config.max_seq_len:
                model_input = input_ids[
                    :, -self.config.max_seq_len:
                ]
            else:
                model_input = input_ids

            logits = self(
                model_input
            )

            # Only the final position predicts the next token.
            logits = logits[:, -1, :]

            # --------------------------------------------------------
            # Temperature
            # --------------------------------------------------------

            logits = logits / temperature

            # --------------------------------------------------------
            # Top-k filtering
            # --------------------------------------------------------

            if top_k is not None:
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                    dim=-1,
                )

                cutoff = values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            # --------------------------------------------------------
            # Top-p / nucleus filtering
            # --------------------------------------------------------

            if top_p is not None:
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

                sorted_remove = (
                    cumulative_probs > top_p
                )

                # Keep the first token that crosses top_p.
                sorted_remove[:, 1:] = (
                    sorted_remove[:, :-1].clone()
                )
                sorted_remove[:, 0] = False

                remove_mask = torch.zeros_like(
                    logits,
                    dtype=torch.bool,
                )

                remove_mask.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_remove,
                )

                logits = logits.masked_fill(
                    remove_mask,
                    float("-inf"),
                )

            # --------------------------------------------------------
            # Select next token
            # --------------------------------------------------------

            if do_sample:
                probs = F.softmax(
                    logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            else:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            # Append token.
            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )

            # --------------------------------------------------------
            # EOS stopping
            # --------------------------------------------------------

            if eos_token_id is not None:
                if (
                    next_token.item()
                    == eos_token_id
                ):
                    break

        # ------------------------------------------------------------
        # Decode
        # ------------------------------------------------------------

        output_ids = input_ids[0].tolist()

        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(output_ids)

        raise TypeError(
            "tokenizer must provide a decode() method"
        )
