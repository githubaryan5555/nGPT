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


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.hidden_dropout(
            self.self_attn(self.input_layernorm(x), attention_mask)
        )
        x = x + self.hidden_dropout(
            self.mlp(self.post_attention_layernorm(x))
        )
        return x


class Model5555LM(nn.Module):
    """Causal language model with input/output shapes ``[B, T]`` and ``[B, T, V]``."""

    def __init__(self, config: Optional[Config] = None, **overrides):
        super().__init__()
        if config is None:
            config = Config()
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if overrides:
            values = asdict(config)
            values.update(overrides)
            config = Config(**values)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.final_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        residual_std = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        b, t = input_ids.shape
        if b <= 0 or t <= 0:
            raise ValueError("batch size and sequence length must be > 0")
        if t > self.config.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.config.max_seq_len}")
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("attention_mask must be a torch.Tensor")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must be on the same device")
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.to(torch.bool)
            elif attention_mask.is_floating_point() or attention_mask.dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
            ):
                attention_mask = attention_mask != 0
            else:
                raise TypeError("attention_mask must be boolean or numeric")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError(f"input_ids contains a token outside [0, {self.config.vocab_size})")
        x = self.embed_dropout(self.embed_tokens(input_ids))
        hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            x = layer(x, attention_mask)
            if output_hidden_states:
                hidden_states.append(x)
        logits = self.lm_head(self.final_layernorm(x))
        if output_hidden_states:
            return logits, hidden_states
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self, dtype_bytes=2):
        if not isinstance(dtype_bytes, Real) or not math.isfinite(dtype_bytes) or dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be finite and > 0")
        return self.get_num_params() * dtype_bytes / 1024**2

    def get_model_size_gb(self, dtype_bytes=2):
        return self.get_model_size_mb(dtype_bytes) / 1024

    def get_flops_per_token(self, seq_len=None):
        if seq_len is None:
            seq_len = self.config.max_seq_len
        if isinstance(seq_len, bool) or not isinstance(seq_len, int):
            raise TypeError("seq_len must be an int")
        if not 0 < seq_len <= self.config.max_seq_len:
            raise ValueError("seq_len is outside the model context")
        d, f = self.config.hidden_size, self.config.intermediate_size
        h, kv, v, layers = self.config.num_attention_heads, self.config.num_key_value_heads, self.config.vocab_size, self.config.num_hidden_layers
        head_dim = d // h
        projection = 2 * (d*d + 2*d*kv*head_dim)
        mlp = 6 * d * f
        return layers * (projection + mlp + 4 * seq_len * d) + 2 * d * v

    def estimate_mfu(self, tokens_per_second, peak_flops):
        for name, value in (("tokens_per_second", tokens_per_second), ("peak_flops", peak_flops)):
            if not isinstance(value, Real) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number")
        if tokens_per_second < 0:
            raise ValueError("tokens_per_second must be >= 0")
        if peak_flops <= 0:
            raise ValueError("peak_flops must be > 0")
        return tokens_per_second * self.get_flops_per_token() / peak_flops * 100.0

    def get_config(self):
        return asdict(self.config)

    @torch.no_grad()
    def generate(self, text, tokenizer, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None, do_sample=True):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an int")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a bool")
        if not isinstance(temperature, Real) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        if top_p is not None and (not isinstance(top_p, Real) or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be finite and in (0, 1]")
        if eos_token_id is not None:
            if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
                raise TypeError("eos_token_id must be an int")
            if not 0 <= eos_token_id < self.config.vocab_size:
                raise ValueError("eos_token_id is outside vocabulary")

        encoded = tokenizer.encode(text)
        ids = encoded.to(dtype=torch.long) if isinstance(encoded, torch.Tensor) else torch.tensor(encoded, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.size(0) != 1:
            raise ValueError("tokenizer.encode must return one sequence")
        if ids.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("tokenizer produced a token outside the model vocabulary")

        output_ids = ids.to(device=self.embed_tokens.weight.device, dtype=torch.long)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = output_ids[:, -self.config.max_seq_len:]
                logits = self(context)[:, -1, :]
                if not do_sample:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        remove = cumulative > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.full_like(logits, float("-inf"))
                        logits.scatter_(-1, sorted_indices, sorted_logits)
                    if not torch.isfinite(logits).any(dim=-1).all():
                        raise RuntimeError("model produced no finite logits for sampling")
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                output_ids = torch.cat((output_ids, next_token), dim=1)
                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break
        finally:
            self.train(was_training)
        return tokenizer.decode(output_ids[0].tolist())
