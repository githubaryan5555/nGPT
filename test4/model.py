import math
from dataclasses import dataclass, asdict

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
        assert isinstance(self.vocab_size, int)
        assert isinstance(self.hidden_size, int)
        assert isinstance(self.num_hidden_layers, int)
        assert isinstance(self.intermediate_size, int)
        assert isinstance(self.num_attention_heads, int)
        assert isinstance(self.num_key_value_heads, int)
        assert isinstance(self.max_seq_len, int)

        assert self.vocab_size > 0
        assert self.hidden_size > 0
        assert self.num_hidden_layers > 0
        assert self.intermediate_size > 0

        assert self.num_attention_heads > 0
        assert self.num_key_value_heads > 0

        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0

        head_dim = self.hidden_size // self.num_attention_heads

        assert head_dim % 2 == 0

        assert self.intermediate_size > self.hidden_size

        assert self.max_seq_len > 0

        assert self.rms_norm_eps > 0

        assert 0.0 <= self.attention_dropout < 1.0
        assert 0.0 <= self.hidden_dropout < 1.0

        assert self.rope_theta > 0
        assert self.initializer_range > 0

        assert isinstance(self.tie_word_embeddings, bool)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self):
        return self.num_attention_heads // self.num_key_value_heads


# ============================================================
# RMSNORM
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )

        output = output.to(dtype=x.dtype)

        return output * self.weight


# ============================================================
# ROPE
# ============================================================

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len, theta):
        super().__init__()

        freqs = 1.0 / (
            theta ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        freqs = torch.outer(positions, freqs)

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
        seq_len = x.shape[1]

        assert seq_len <= self.cos.shape[0]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

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
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv
        self.head_dim = config.head_dim
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
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def repeat_kv(self, x):
        if self.num_queries_per_kv == 1:
            return x

        batch, seq_len, heads, dim = x.shape

        x = x[:, :, :, None, :]

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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

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

        q = self.rope(q)
        k = self.rope(k)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            batch,
            seq_len,
            self.hidden_size,
        )

        return self.o_proj(y)


# ============================================================
# SWIGLU
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, config):
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
        return self.down_proj(
            F.silu(self.gate_proj(x))
            * self.up_proj(x)
        )


# ============================================================
# BLOCK
# ============================================================

class Block(nn.Module):
    def __init__(self, config):
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
        x = x + self.self_attn(
            self.input_layernorm(x)
        )

        x = x + self.mlp(
            self.post_attention_layernorm(x)
        )

        return x


# ============================================================
# MODEL 5555
# ============================================================

class Model5555LM(nn.Module):
    def __init__(self, config=None, **overrides):
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

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.max_seq_len = config.max_seq_len

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.embed_dropout = nn.Dropout(
            config.hidden_dropout
        )

        self.layers = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.final_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
        assert input_ids.ndim == 2

        batch_size, seq_len = input_ids.shape

        assert seq_len <= self.config.max_seq_len

        assert input_ids.dtype == torch.long

        #assert torch.all(input_ids >= 0)
        #assert torch.all(input_ids < self.config.vocab_size)

        x = self.embed_tokens(input_ids)

        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

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
        return (
            self.get_num_params()
            * dtype_bytes
            / (1024 ** 2)
        )

    def get_model_size_gb(self, dtype_bytes=2):
        return (
            self.get_num_params()
            * dtype_bytes
            / (1024 ** 3)
        )

    def get_flops_per_token(self):
        d = self.config.hidden_size
        v = self.config.vocab_size
        l = self.config.num_hidden_layers
        f = self.config.intermediate_size
        h = self.config.num_attention_heads
        kv = self.config.num_key_value_heads

        embedding_output = 0

        attention_projection = (
            2 * d * (
                d
                + 2 * kv * (d // h)
            )
        )

        attention_output = 2 * d * d

        mlp = (
            2 * d * f
            + 2 * d * f
            + 2 * f * d
        )

        attention_scores = (
            2 * h * self.config.max_seq_len * (d // h)
        )

        attention_value = attention_scores

        block_flops = (
            attention_projection
            + attention_output
            + mlp
            + attention_scores
            + attention_value
        )

        lm_head = 2 * d * v

        return (
            l * block_flops
            + lm_head
            + embedding_output
        )

    def estimate_vram(
        self,
        batch_size=1,
        seq_len=None,
        dtype_bytes=2,
        optimizer="adamw",
        gradients=True,
        activations=True,
    ):
        if seq_len is None:
            seq_len = self.config.max_seq_len

        assert batch_size > 0
        assert seq_len > 0
        assert dtype_bytes > 0

        params = self.get_num_params()

        parameter_memory = (
            params * dtype_bytes
        )

        gradient_memory = (
            params * dtype_bytes
            if gradients
            else 0
        )

        if optimizer.lower() == "adamw":
            optimizer_memory = params * 8
        elif optimizer.lower() in ("sgd", "none"):
            optimizer_memory = 0
        else:
            raise ValueError(
                "optimizer must be 'adamw', 'sgd', or 'none'"
            )

        activation_elements = (
            batch_size
            * seq_len
            * self.config.hidden_size
            * self.config.num_hidden_layers
            * 4
        )

        activation_memory = (
            activation_elements * dtype_bytes
            if activations
            else 0
        )

        total = (
            parameter_memory
            + gradient_memory
            + optimizer_memory
            + activation_memory
        )

        return {
            "parameters_gb": parameter_memory / (1024 ** 3),
            "gradients_gb": gradient_memory / (1024 ** 3),
            "optimizer_gb": optimizer_memory / (1024 ** 3),
            "activations_gb": activation_memory / (1024 ** 3),
            "total_gb": total / (1024 ** 3),
        }

    def estimate_mfu(
        self,
        tokens_per_second,
        peak_flops,
    ):
        assert tokens_per_second >= 0
        assert peak_flops > 0

        achieved_flops = (
            tokens_per_second
            * self.get_flops_per_token()
        )

        return (
            achieved_flops
            / peak_flops
            * 100.0
        )

    def get_config(self):
        return asdict(self.config)

    def count_parameters_by_component(self):
        result = {}

        result["embed_tokens"] = (
            self.embed_tokens.weight.numel()
        )

        result["attention"] = 0
        result["mlp"] = 0
        result["norm"] = 0

        for layer in self.layers:
            result["attention"] += sum(
                p.numel()
                for p in layer.self_attn.parameters()
            )

            result["mlp"] += sum(
                p.numel()
                for p in layer.mlp.parameters()
            )

            result["norm"] += (
                layer.input_layernorm.weight.numel()
                + layer.post_attention_layernorm.weight.numel()
            )

        result["final_layernorm"] = (
            self.final_layernorm.weight.numel()
        )

        if self.config.tie_word_embeddings:
            result["lm_head"] = 0
        else:
            result["lm_head"] = (
                self.lm_head.weight.numel()
            )

        return result

    def get_parameter_report(self):
        components = self.count_parameters_by_component()

        total = self.get_num_params()

        report = {}

        for name, count in components.items():
            report[name] = {
                "parameters": count,
                "percentage": (
                    count / total * 100.0
                    if total > 0
                    else 0.0
                ),
            }

        report["total"] = {
            "parameters": total,
            "percentage": 100.0,
        }

        return report

    def print_model_info(
        self,
        batch_size=1,
        seq_len=None,
        dtype_bytes=2,
        optimizer="adamw",
        peak_flops=None,
        tokens_per_second=None,
    ):
        if seq_len is None:
            seq_len = self.config.max_seq_len

        params = self.get_num_params()

        print("=" * 64)
        print("MODEL5555LM")
        print("=" * 64)

        print(
            f"Parameters       : {params:,}"
        )

        print(
            f"Trainable params : "
            f"{self.get_trainable_params():,}"
        )

        print(
            f"Model size       : "
            f"{self.get_model_size_mb(dtype_bytes):.16f} MB"
        )

        print(
            f"Model size       : "
            f"{self.get_model_size_gb(dtype_bytes):.16f} GB"
        )

        print(
            f"Hidden size      : "
            f"{self.config.hidden_size}"
        )

        print(
            f"Layers           : "
            f"{self.config.num_hidden_layers}"
        )

        print(
            f"Intermediate     : "
            f"{self.config.intermediate_size}"
        )

        print(
            f"Attention heads  : "
            f"{self.config.num_attention_heads}"
        )

        print(
            f"KV heads         : "
            f"{self.config.num_key_value_heads}"
        )

        print(
            f"Head dimension   : "
            f"{self.config.head_dim}"
        )

        print(
            f"Context length   : "
            f"{self.config.max_seq_len}"
        )

        print(
            f"Vocab size       : "
            f"{self.config.vocab_size}"
        )

        print(
            f"FLOPs/token      : "
            f"{self.get_flops_per_token():.16f}"
        )

        vram = self.estimate_vram(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=dtype_bytes,
            optimizer=optimizer,
        )

        print(
            f"VRAM parameters  : "
            f"{vram['parameters_gb']:.16f} GB"
        )

        print(
            f"VRAM gradients   : "
            f"{vram['gradients_gb']:.16f} GB"
        )

        print(
            f"VRAM optimizer   : "
            f"{vram['optimizer_gb']:.16f} GB"
        )

        print(
            f"VRAM activations : "
            f"{vram['activations_gb']:.16f} GB"
        )

        print(
            f"VRAM estimate    : "
            f"{vram['total_gb']:.16f} GB"
        )

        if (
            peak_flops is not None
            and tokens_per_second is not None
        ):
            mfu = self.estimate_mfu(
                tokens_per_second=tokens_per_second,
                peak_flops=peak_flops,
            )

            print(
                f"MFU              : "
                f"{mfu:.16f} %"
            )

        print("=" * 64)
    


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
    ):
        self.eval()

        assert isinstance(text, str)
        assert max_new_tokens >= 0
        assert temperature > 0

        device = next(self.parameters()).device

        input_ids = tokenizer.encode(text)

        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_new_tokens):
            input_ids_cond = input_ids[:, -self.config.max_seq_len:]

            logits = self(input_ids_cond)
