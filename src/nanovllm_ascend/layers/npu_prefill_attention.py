from __future__ import annotations

import torch
import torch_npu
from torch import nn


class NpuBatchPrefillAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        block_size: int,
        max_mask_len: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_mask_len = max_mask_len
        self.scale = head_dim ** -0.5
        attn_mask = torch.triu(
            torch.ones(max_mask_len, max_mask_len, dtype=torch.int8),
            diagonal=1,
        )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        total_tokens, num_heads, head_dim = q.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(f"query shape mismatch: got {tuple(q.shape)}")
        if k.shape[0] != total_tokens or v.shape[0] != total_tokens:
            raise ValueError(f"k/v token mismatch: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

        max_prompt_len = 0
        prev = 0
        for end in attn_metadata.actual_seq_lengths_q:
            max_prompt_len = max(max_prompt_len, end - prev)
            prev = end
        if max_prompt_len > self.max_mask_len:
            raise ValueError(f"prompt length {max_prompt_len} exceeds max_mask_len {self.max_mask_len}")

        target_dtype = v.dtype
        attn_out, _ = torch_npu.npu_fused_infer_attention_score(
            query=q.to(dtype=target_dtype).contiguous(),
            key=k.to(dtype=target_dtype).contiguous(),
            value=v.contiguous(),
            atten_mask=self.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_kv,
            num_key_value_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        return attn_out.view(total_tokens, self.num_heads, self.head_dim)

