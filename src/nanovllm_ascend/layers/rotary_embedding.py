from __future__ import annotations

import torch
import torch_npu
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        position_ids = torch.arange(
            max_position_embeddings,
            dtype=torch.float32,
        )

        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            emb.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            emb.sin(),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, T]，当前项目里通常是 [1, total_tokens]
        flat_position_ids = position_ids.reshape(-1)

        cos = self.cos_cached.index_select(0, flat_position_ids)
        sin = self.sin_cached.index_select(0, flat_position_ids)

        return (
            cos.view(*position_ids.shape, self.head_dim),
            sin.view(*position_ids.shape, self.head_dim),
        )


def apply_rotary_pos_emb_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    # q:   [T, num_heads, head_dim]
    # k:   [T, num_key_value_heads, head_dim]
    # cos: [T, head_dim]
    # sin: [T, head_dim]
    #
    # npu_rotary_mul 要求 4D 输入。
    # 这里使用 S B N D：
    # q/k -> [T, 1, N, D]
    # cos/sin -> [T, 1, 1, D]
    total_tokens = q.shape[0]

    cos_q = cos.to(dtype=q.dtype).view(total_tokens, 1, 1, q.shape[-1])
    sin_q = sin.to(dtype=q.dtype).view(total_tokens, 1, 1, q.shape[-1])

    q_embed = torch_npu.npu_rotary_mul(
        q.unsqueeze(1),
        cos_q,
        sin_q,
        rotary_mode="half",
    ).squeeze(1)

    if k.dtype == q.dtype:
        cos_k = cos_q
        sin_k = sin_q
    else:
        cos_k = cos.to(dtype=k.dtype).view(total_tokens, 1, 1, k.shape[-1])
        sin_k = sin.to(dtype=k.dtype).view(total_tokens, 1, 1, k.shape[-1])

    k_embed = torch_npu.npu_rotary_mul(
        k.unsqueeze(1),
        cos_k,
        sin_k,
        rotary_mode="half",
    ).squeeze(1)

    return q_embed, k_embed