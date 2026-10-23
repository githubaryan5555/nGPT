
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pos_enc import RoPE


class GQAAttention(nn.Module):
    """
    Pure-PyTorch Gated Linear Attention.

    Input:
        x:              (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len), where nonzero means valid

    Output:
        y:              (batch, seq_len, hidden_size)

    The recurrent state has shape:
        (batch, heads, head_dim, head_dim)

    At each token:
        S_t = gate_t * S_(t-1) + outer(K_t, V_t)
        O_t = Q_t @ S_t

    gate_t decays the old state independently for each key dimension.
    """

    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

        self.inner_size = (
            self.num_attention_heads * self.head_dim
        )

        if self.inner_size != self.hidden_size:
            raise ValueError(
                "GLAAttention requires "
                "num_attention_heads * head_dim == hidden_size, "
                f"got {self.num_attention_heads} * "
                f"{self.head_dim} != {self.hidden_size}."
            )

        # Q, K and V projections.
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.inner_size,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.inner_size,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.inner_size,
            bias=False,
        )

        # Input-dependent decay gate.
        # One gate per attention head and key dimension.
        self.g_proj = nn.Linear(
            self.hidden_size,
            self.inner_size,
            bias=True,
        )

        # Start with relatively slow decay.
        # sigmoid(2) ~= 0.881.
        nn.init.constant_(self.g_proj.bias, 2.0)

        # Output projection.
        self.o_proj = nn.Linear(
            self.inner_size,
            self.hidden_size,
            bias=False,
        )

        # Same positional-encoding interface as GQAAttention.
        self.rope = RoPE(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        h = self.num_attention_heads
        d = self.head_dim

        if t == 0:
            return x

        # Validate and normalize the padding mask.
        valid = None

        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError(
                    f"attention_mask must have shape {(b, t)}, "
                    f"got {tuple(attention_mask.shape)}"
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

        # Project hidden states: (B, T, H, D).
        q = self.q_proj(x).view(b, t, h, d)
        k = self.k_proj(x).view(b, t, h, d)
        v = self.v_proj(x).view(b, t, h, d)

        # Each token produces its own decay vector.
        # sigmoid ensures every gate lies in (0, 1).
        gate = torch.sigmoid(
            self.g_proj(x).view(b, t, h, d)
        )

        # Apply RoPE to queries and keys.
        q = self.rope(q)
        k = self.rope(k)

        # Recurrent state accumulation is more numerically stable
        # in float32 when model activations use fp16 or bfloat16.
        state_dtype = (
            torch.float32
            if q.dtype in (torch.float16, torch.bfloat16)
            else q.dtype
        )

        q = q.to(state_dtype)
        k = k.to(state_dtype)
        v = v.to(state_dtype)
        gate = gate.to(state_dtype)

        # One independent state matrix per batch item and head.
        state = torch.zeros(
            (b, h, d, d),
            device=x.device,
            dtype=state_dtype,
        )

        outputs = []

        # Causal recurrent GLA.
        # No T x T attention or decay matrix is constructed.
        for i in range(t):
            q_i = q[:, i]       # (B, H, D)
            k_i = k[:, i]       # (B, H, D)
            v_i = v[:, i]       # (B, H, D)
            gate_i = gate[:, i] # (B, H, D)

            # Decay old state and write the current key-value pair.
            new_state = (
                state * gate_i.unsqueeze(-1)
                + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
            )

            if valid is not None:
                valid_i = valid[:, i, None, None, None]

                # Padding positions neither update nor decay state.
                state = torch.where(
                    valid_i,
                    new_state,
                    state,
                )
            else:
                state = new_state

            # Read the updated state with the current query.
            o_i = torch.einsum(
                "bhd,bhdv->bhv",
                q_i,
                state,
            )

            if valid is not None:
                o_i = o_i * valid[:, i, None, None].to(
                    o_i.dtype
                )

            outputs.append(o_i)

        # (B, T, H, D).
        y = torch.stack(outputs, dim=1)

        # Restore the original input dtype.
        y = y.to(x.dtype)

        # The recurrent form has no explicit attention-probability
        # matrix, so dropout is applied to its output activations.
        y = F.dropout(
            y,
            p=self.attention_dropout,
            training=self.training,
        )

        # Merge heads and apply the output projection.
        y = y.contiguous().view(
            b,
            t,
            self.hidden_size,
        )

        return self.o_proj(y)
