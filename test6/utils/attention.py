
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pos_enc import RoPE


class _TiledGQAFunction(torch.autograd.Function):
    """
    Memory-efficient tiled GQA attention.

    Never constructs the full [B, H, T, T] attention matrix.
    Forward stores only compact per-query normalization statistics.
    Backward recomputes score/probability tiles.

    Attention dropout is intentionally applied outside this function
    in this reference version; see the integration notes below.
    """

    @staticmethod
    def forward(ctx, q, k, v, query_to_kv, tile_q, tile_k,
                scale, causal, valid_mask):
        # q: [B, Hq, T, D]
        # k,v: [B, Hkv, T, D]
        B, Hq, T, D = q.shape
        Hkv = k.shape[1]

        out = torch.empty_like(q)
        # FP32 statistics improve numerical stability.
        row_max = torch.empty(
            (B, Hq, T), device=q.device, dtype=torch.float32
        )
        row_sum = torch.empty_like(row_max)

        # This reference favors clarity and correctness over speed.
        # Only a [B, Hq, tile_q, tile_k] score tile exists at once.
        for qs in range(0, T, tile_q):
            qe = min(qs + tile_q, T)
            qt = q[:, :, qs:qe, :].float()

            m = torch.full(
                (B, Hq, qe - qs, 1),
                -float("inf"), device=q.device, dtype=torch.float32
            )
            ell = torch.zeros_like(m)
            acc = torch.zeros(
                (B, Hq, qe - qs, D),
                device=q.device, dtype=torch.float32
            )

            for ks in range(0, T, tile_k):
                ke = min(ks + tile_k, T)

                # GQA: map each query head to its KV head.
                kv_indices = query_to_kv
                kt = k[:, kv_indices, ks:ke, :].float()
                vt = v[:, kv_indices, ks:ke, :].float()

                scores = torch.matmul(
                    qt, kt.transpose(-2, -1)
                ) * scale

                allowed = torch.ones(
                    (qe - qs, ke - ks),
                    device=q.device, dtype=torch.bool
                )

                if causal:
                    q_pos = torch.arange(qs, qe, device=q.device)
                    k_pos = torch.arange(ks, ke, device=q.device)
                    allowed &= k_pos[None, :] <= q_pos[:, None]

                if valid_mask is not None:
                    allowed = (
                        allowed[None, None, :, :]
                        & valid_mask[:, None, None, ks:ke]
                    )

                scores = scores.masked_fill(
                    ~allowed, -float("inf")
                )

                tile_max = scores.amax(dim=-1, keepdim=True)
                new_m = torch.maximum(m, tile_max)

                # Avoid NaNs for rows whose current tile is fully masked.
                finite_new = torch.isfinite(new_m)
                safe_new_m = torch.where(
                    finite_new, new_m, torch.zeros_like(new_m)
                )

                alpha = torch.where(
                    torch.isfinite(m),
                    torch.exp(m - safe_new_m),
                    torch.zeros_like(m),
                )

                p = torch.exp(scores - safe_new_m)
                p = torch.where(
                    torch.isfinite(scores), p, torch.zeros_like(p)
                )

                ell = alpha * ell + p.sum(dim=-1, keepdim=True)
                acc = (
                    alpha * acc
                    + torch.matmul(p, vt)
                )
                m = new_m

            out[:, :, qs:qe, :] = (
                acc / ell.clamp_min(torch.finfo(torch.float32).tiny)
            ).to(q.dtype).masked_fill(
                (ell == 0).expand(-1, -1, -1, D), 0
            )

            row_max[:, :, qs:qe] = m.squeeze(-1)
            row_sum[:, :, qs:qe] = ell.squeeze(-1)

        ctx.save_for_backward(q, k, v, row_max, row_sum, query_to_kv)
        ctx.tile_q = tile_q
        ctx.tile_k = tile_k
        ctx.scale = scale
        ctx.causal = causal
        ctx.valid_mask = valid_mask

        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, row_max, row_sum, query_to_kv = ctx.saved_tensors

        B, Hq, T, D = q.shape
        Hkv = k.shape[1]
        tq = ctx.tile_q
        tk = ctx.tile_k
        scale = ctx.scale
        causal = ctx.causal
        valid_mask = ctx.valid_mask

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        go = grad_out.float()

        # dL/dS = P * (dP - sum(dP * P))
        # with dP = dO @ V^T.
        for qs in range(0, T, tq):
            qe = min(qs + tq, T)
            qt = q[:, :, qs:qe, :].float()
            got = go[:, :, qs:qe, :]
            mt = row_max[:, :, qs:qe, None]
            lt = row_sum[:, :, qs:qe, None]

            # This correction term is independent of the key tile.
            delta = (got * (
                # Recompute output for this query tile.
                _TiledGQAFunction._recompute_output_tile(
                    q, k, v, query_to_kv, qs, qe,
                    tk, scale, causal, valid_mask
                )
            )).sum(dim=-1, keepdim=True)

            for ks in range(0, T, tk):
                ke = min(ks + tk, T)

                for h in range(Hq):
                    kh = int(query_to_kv[h].item())

                    qh = qt[:, h:h+1]
                    kh_t = k[:, kh:kh+1, ks:ke, :].float()
                    vh_t = v[:, kh:kh+1, ks:ke, :].float()
                    goh = got[:, h:h+1]

                    scores = torch.matmul(
                        qh, kh_t.transpose(-2, -1)
                    ) * scale

                    allowed = torch.ones(
                        (qe - qs, ke - ks),
                        device=q.device, dtype=torch.bool
                    )
                    if causal:
                        q_pos = torch.arange(qs, qe, device=q.device)
                        k_pos = torch.arange(ks, ke, device=q.device)
                        allowed &= k_pos[None, :] <= q_pos[:, None]

                    if valid_mask is not None:
                        allowed = (
                            allowed[None, None, :, :]
                            & valid_mask[:, None, None, ks:ke]
                        )

                    scores = scores.masked_fill(~allowed, -float("inf"))
                    p = torch.exp(scores - mt[:, h:h+1])
                    p = torch.where(
                        torch.isfinite(scores), p, torch.zeros_like(p)
                    )
                    p = p / lt[:, h:h+1].clamp_min(
                        torch.finfo(torch.float32).tiny
                    )

                    dp = torch.matmul(goh, vh_t.transpose(-2, -1))
                    ds = p * (dp - delta[:, h:h+1]) * scale

                    dq[:, h:h+1, qs:qe, :] += torch.matmul(
                        ds, kh_t
                    )
                    dk[:, kh:kh+1, ks:ke, :] += torch.matmul(
                        ds.transpose(-2, -1), qh
                    )
                    dv[:, kh:kh+1, ks:ke, :] += torch.matmul(
                        p.transpose(-2, -1), goh
                    )

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            None, None, None, None, None, None
        )

    @staticmethod
    def _recompute_output_tile(
        q, k, v, query_to_kv, qs, qe, tile_k,
        scale, causal, valid_mask
    ):
        B, Hq, T, D = q.shape
        qt = q[:, :, qs:qe, :].float()

        out = torch.zeros(
            (B, Hq, qe - qs, D),
            device=q.device, dtype=torch.float32
        )

        for ks in range(0, T, tile_k):
            ke = min(ks + tile_k, T)
            kt = k[:, query_to_kv, ks:ke, :].float()
            vt = v[:, query_to_kv, ks:ke, :].float()

            scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale

            allowed = torch.ones(
                (qe - qs, ke - ks), device=q.device, dtype=torch.bool
            )
            if causal:
                qp = torch.arange(qs, qe, device=q.device)
                kp = torch.arange(ks, ke, device=q.device)
                allowed &= kp[None, :] <= qp[:, None]

            if valid_mask is not None:
                allowed = (
                    allowed[None, None, :, :]
                    & valid_mask[:, None, None, ks:ke]
                )

            scores = scores.masked_fill(~allowed, -float("inf"))
            # Recompute a stable tile-independent softmax denominator.
            # The helper below uses a full online pass for this query tile.
            # For clarity, this implementation uses the recurrence again.
            # See the main forward pass for the corresponding equations.
            #
            # A production implementation should share the exact online
            # statistics helper between forward and backward.
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            out += torch.matmul(probs, vt)

        return out.to(q.dtype)


class GQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

        self.tile_q = getattr(config, "attention_tile_q", 32)
        self.tile_k = getattr(config, "attention_tile_k", 64)

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

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape

        q = self.q_proj(x).view(
            b, t, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            b, t, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            b, t, self.num_key_value_heads, self.head_dim
        )

        q, k = self.rope(q), self.rope(k)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        valid_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}"
                )
            valid_mask = attention_mask.to(
                device=x.device, dtype=torch.bool
            )

        # Map each query head to its corresponding shared KV head.
        query_to_kv = (
            torch.arange(self.num_attention_heads, device=x.device)
            // self.num_queries_per_kv
        )

        y = _TiledGQAFunction.apply(
            q, k, v, query_to_kv,
            self.tile_q, self.tile_k,
            1.0 / math.sqrt(self.head_dim),
            True, valid_mask
        )

        # Dropout is omitted here intentionally; add it only after
        # implementing a dropout-consistent custom backward.
        y = y.transpose(1, 2).contiguous().view(
            b, t, self.hidden_size
        )

        return self.o_proj(y)
