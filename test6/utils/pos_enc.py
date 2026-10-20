import math
import torch
import torch.nn as nn
import torch.nn.functional as F



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
