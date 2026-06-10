from __future__ import annotations

import torch
import torch_npu
from torch import nn

from nanovllm_ascend.npu.acl_graph import (
    PagedAttentionGraphTask,
    current_decode_graph_entry,
    record_paged_attention_task,
)


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

        output = torch.empty_like(q)
        entry = current_decode_graph_entry()
        if entry is not None:
            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=q,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=block_tables,
                context_lens=context_lens,
                out=output,
            )
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=q,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=block_tables,
                context_lens=context_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            record_paged_attention_task(
                PagedAttentionGraphTask(
                    query=q,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    num_kv_heads=self.num_key_value_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    output=output,
                    workspace=workspace,
                    handle=handle,
                    event=event,
                )
            )
            return output

        torch_npu._npu_paged_attention(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_key_value_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=block_tables,
            context_lens=context_lens,
            out=output,
        )
        return output
