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
        self.fia_mask_len = max(max_mask_len, 2048)
        self.scale = head_dim ** -0.5
        attn_mask = torch.triu(
            torch.ones(self.fia_mask_len, self.fia_mask_len, dtype=torch.int8),
            diagonal=1,
        )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _paged_cache_for_fia(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_blocks, cache_block_size, _, _ = key_cache.shape
        key = key_cache.contiguous().view(num_blocks, cache_block_size, -1)
        value = value_cache.contiguous().view(num_blocks, cache_block_size, -1)
        return key, value, cache_block_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata,
        key_cache: torch.Tensor | None = None,
        value_cache: torch.Tensor | None = None,
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
        query = q.to(dtype=target_dtype).contiguous()
        if attn_metadata.use_paged_prefill:
            if key_cache is None or value_cache is None:
                raise ValueError("paged prefill requires physical key/value cache")
            key, value, cache_block_size = self._paged_cache_for_fia(
                key_cache=key_cache,
                value_cache=value_cache,
            )
            block_table = attn_metadata.block_tables.to(
                device=query.device,
                dtype=torch.int32,
            ).contiguous()
            actual_seq_lengths_kv = (
                attn_metadata.context_lens.to(device="cpu", dtype=torch.int32)
                .contiguous()
                .tolist()
            )
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=self.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        else:
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
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
