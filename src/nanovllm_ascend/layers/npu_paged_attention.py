from __future__ import annotations

import torch
import torch_npu
from torch import nn


class NpuPagedAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens, num_heads, head_dim = q.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(f"query shape mismatch: got {tuple(q.shape)}")

        query = q.to(dtype=key_cache.dtype).contiguous()
        block_tables = block_tables.to(device=query.device, dtype=torch.int32).contiguous()
        context_lens = context_lens.to(device="cpu", dtype=torch.int32).contiguous()
        output = torch.empty_like(query)

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache.contiguous(),
            value_cache=value_cache.contiguous(),
            num_kv_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=block_tables,
            context_lens=context_lens,
            out=output,
        )
        return output
