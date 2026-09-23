"""Small decoder-only language model used by ``train.py``.

Contract
--------
``model(input_ids)`` accepts a ``torch.long`` tensor of shape ``[B, T]`` and
returns logits of shape ``[B, T, vocab_size]``.  ``train.py`` is responsible
for shifting labels and calculating cross entropy.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an int")
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.intermediate_size <= self.hidden_size:
            raise ValueError("intermediate_size must be greater than hidden_size")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0 or self.initializer_range <= 0:
            raise ValueError("eps, rope_theta, and initializer_range must be positive")
        for name in ("attention_dropout", "hidden_dropout"):
            value = getattr(self, name)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
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
        # Keep the reduction in fp32 for stable fp16/bf16 training.
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        # x: [batch, sequence, heads, head_dim]
        seq_len = x.size(1)
        if seq_len > self.cos.size(0):
            raise ValueError(f"sequence length {seq_len} exceeds RoPE limit {self.cos.size(0)}")
        cos = self.cos[:seq_len].to(device=x.device, dtype=x.dtype)[None, :, None, :]
        sin = self.sin[:seq_len].to(device=x.device, dtype=x.dtype)[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


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
        b, t, h, d = x.shape
        return x[:, :, :, None, :].expand(b, t, h, self.num_queries_per_kv, d).reshape(b, t, -1, d)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q), self.rope(k)
        q, k, v = (z.transpose(1, 2) for z in (q, self.repeat_kv(k), self.repeat_kv(v)))
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, self.hidden_size))


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

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Model5555LM(nn.Module):
    """Causal language model with a stable ``[B,T] -> [B,T,V]`` interface."""
    def __init__(self, config: Optional[Config] = None, **overrides):
        super().__init__()
        config = Config() if config is None else config
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")
        if overrides:
            values = asdict(config)
            values.update(overrides)
            config = Config(**values)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
        if input_ids.size(1) == 0 or input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside [0, vocab_size)")
        x = self.embed_dropout(self.embed_tokens(input_ids))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.final_layernorm(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self, dtype_bytes=2):
        return self.get_num_params() * dtype_bytes / 1024**2

    def get_model_size_gb(self, dtype_bytes=2):
        return self.get_num_params() * dtype_bytes / 1024**3

    def get_flops_per_token(self, seq_len=None):
        """Approximate forward FLOPs; attention cost depends on context length."""
        t = self.config.max_seq_len if seq_len is None else seq_len
        if t <= 0 or t > self.config.max_seq_len:
            raise ValueError("seq_len is outside the model context")
        d, f, h, kv, v, l = (self.config.hidden_size, self.config.intermediate_size,
                              self.config.num_attention_heads, self.config.num_key_value_heads,
                              self.config.vocab_size, self.config.num_hidden_layers)
        projections = 2 * d * (d + 2 * kv * (d // h)) + 2 * d * d
        mlp = 6 * d * f
        attention = 4 * h * t * (d // h)
        return l * (projections + mlp + attention) + 2 * d * v

    def estimate_mfu(self, tokens_per_second, peak_flops):
        if tokens_per_second < 0 or peak_flops <= 0:
            raise ValueError("tokens_per_second must be >= 0 and peak_flops must be > 0")
        return tokens_per_second * self.get_flops_per_token() / peak_flops * 100.0

    def get_config(self):
        return asdict(self.config)

    @torch.no_grad()
    def generate(self, text, tokenizer, max_new_tokens=100, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None, do_sample=True):
        if not isinstance(text, str) or max_new_tokens < 0 or temperature <= 0:
            raise ValueError("text must be a string, max_new_tokens >= 0, temperature > 0")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        device = self.embed_tokens.weight.device
        ids = tokenizer.encode(text)
        ids = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.size(0) != 1 or ids.size(1) == 0:
            raise ValueError("tokenizer.encode must return one non-empty sequence")
        ids = ids.to(device=device, dtype=torch.long)[:, -self.config.max_seq_len:]
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.max_seq_len:])[:, -1, :]
            if not do_sample:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                if top_p is not None and top_p < 1:
                    values, indices = torch.sort(logits, descending=True, dim=-1)
                    remove = torch.cumsum(F.softmax(values, dim=-1), dim=-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(1, indices, values.masked_fill(remove, float("-inf")))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat((ids, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return tokenizer.decode(ids[0].tolist())
