from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any, Iterator

import torch

from .block_manager import PagedKVCacheMetadata


_active_decode_graph_entry: "DecodeGraphEntry | None" = None


@dataclass
class PagedAttentionGraphTask:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    output: torch.Tensor
    workspace: Any
    handle: Any
    event: Any


@dataclass
class DecodeGraphEntry:
    batch_size: int
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    graph: Any
    update_stream: Any
    logits: torch.Tensor | None = None
    tasks: list[PagedAttentionGraphTask] = field(default_factory=list)

    def copy_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: PagedKVCacheMetadata,
    ) -> None:
        self.input_ids.copy_(input_ids)
        self.position_ids.copy_(position_ids)
        copy_block_tables(self.block_tables, attn_metadata.block_tables)
        self.context_lens.copy_(attn_metadata.context_lens)
        self.slot_mapping.copy_(attn_metadata.slot_mapping)


class DecodeGraphStats:
    def __init__(self) -> None:
        self.captures = 0
        self.replays = 0
        self.updates = 0
        self.fallbacks = 0
        self.capture_failures = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "captures": self.captures,
            "replays": self.replays,
            "updates": self.updates,
            "fallbacks": self.fallbacks,
            "capture_failures": self.capture_failures,
        }


def current_decode_graph_entry() -> DecodeGraphEntry | None:
    return _active_decode_graph_entry


@contextmanager
def capture_decode_graph_tasks(entry: DecodeGraphEntry) -> Iterator[None]:
    global _active_decode_graph_entry
    previous = _active_decode_graph_entry
    _active_decode_graph_entry = entry
    try:
        yield
    finally:
        _active_decode_graph_entry = previous


def record_paged_attention_task(task: PagedAttentionGraphTask) -> None:
    entry = current_decode_graph_entry()
    if entry is None:
        raise RuntimeError("paged attention graph task recorded outside graph capture")
    entry.tasks.append(task)


def _clone_decode_metadata(attn_metadata: PagedKVCacheMetadata) -> PagedKVCacheMetadata:
    return PagedKVCacheMetadata(
        block_tables=attn_metadata.block_tables.clone(),
        context_lens=attn_metadata.context_lens.clone(),
        slot_mapping=attn_metadata.slot_mapping.clone(),
        actual_seq_lengths_q=attn_metadata.actual_seq_lengths_q,
        actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_kv,
        query_lens=attn_metadata.query_lens,
        use_paged_prefill=attn_metadata.use_paged_prefill,
    )


class DecodeGraphRunner:
    def __init__(
        self,
        model,
        kv_cache,
        batch_sizes: list[int],
        max_model_len: int | None = None,
        block_size: int | None = None,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.batch_sizes = set(batch_sizes)
        self.max_blocks_per_seq = None
        if max_model_len is not None and block_size is not None:
            self.max_blocks_per_seq = math.ceil(max_model_len / block_size)
        self.entries: dict[int, DecodeGraphEntry] = {}
        self.disabled_batch_sizes: set[int] = set()
        self.stats = DecodeGraphStats()

    def supports(self, batch_size: int) -> bool:
        return batch_size in self.batch_sizes and batch_size not in self.disabled_batch_sizes

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: PagedKVCacheMetadata,
    ) -> torch.Tensor | None:
        batch_size = int(input_ids.numel())
        if not self.supports(batch_size):
            self.stats.fallbacks += 1
            return None

        entry = self.entries.get(batch_size)
        if entry is None:
            entry = self._capture(batch_size, input_ids, position_ids, attn_metadata)
            if entry is None:
                self.stats.fallbacks += 1
                return None

        entry.copy_inputs(input_ids, position_ids, attn_metadata)
        self._update_paged_attention_tasks(entry)
        entry.graph.replay()
        self.stats.replays += 1
        return entry.logits

    def _capture(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: PagedKVCacheMetadata,
    ) -> DecodeGraphEntry | None:
        graph = torch.npu.NPUGraph()
        update_stream = torch.npu.Stream()
        block_tables = self._make_static_block_tables(
            batch_size=batch_size,
            block_tables=attn_metadata.block_tables,
        )
        entry = DecodeGraphEntry(
            batch_size=batch_size,
            input_ids=input_ids.clone(),
            position_ids=position_ids.clone(),
            block_tables=block_tables,
            context_lens=attn_metadata.context_lens.clone(),
            slot_mapping=attn_metadata.slot_mapping.clone(),
            graph=graph,
            update_stream=update_stream,
        )
        static_metadata = _clone_decode_metadata(attn_metadata)
        static_metadata.block_tables = entry.block_tables
        static_metadata.context_lens = entry.context_lens
        static_metadata.slot_mapping = entry.slot_mapping

        try:
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                with capture_decode_graph_tasks(entry):
                    outputs = self.model(
                        input_ids_flat=entry.input_ids,
                        position_ids_flat=entry.position_ids,
                        kv_cache=self.kv_cache,
                        attn_metadata=static_metadata,
                        is_prefill=False,
                    )
                    entry.logits = outputs.logits
            if not entry.tasks:
                raise RuntimeError("decode graph captured no paged attention tasks")
        except Exception as exc:
            self.disabled_batch_sizes.add(batch_size)
            self.stats.capture_failures += 1
            print(f"[decode_graph] disable batch_size={batch_size}: {exc}")
            return None

        self.entries[batch_size] = entry
        self.stats.captures += 1
        return entry

    def _make_static_block_tables(
        self,
        batch_size: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        if self.max_blocks_per_seq is None:
            return block_tables.clone()
        static_block_tables = block_tables.new_zeros(
            (batch_size, self.max_blocks_per_seq)
        )
        copy_block_tables(static_block_tables, block_tables)
        return static_block_tables

    def _update_paged_attention_tasks(self, entry: DecodeGraphEntry) -> None:
        if not entry.tasks:
            return

        import torch_npu

        current_stream = torch.npu.current_stream()
        with torch.npu.stream(entry.update_stream):
            for task in entry.tasks:
                workspace = task.workspace
                torch.npu.graph_task_update_begin(entry.update_stream, task.handle)
                torch_npu._npu_paged_attention(
                    query=task.query,
                    key_cache=task.key_cache,
                    value_cache=task.value_cache,
                    num_kv_heads=task.num_kv_heads,
                    num_heads=task.num_heads,
                    scale_value=task.scale,
                    block_table=task.block_tables,
                    context_lens=task.context_lens,
                    out=task.output,
                    workspace=workspace,
                )
                torch.npu.graph_task_update_end(entry.update_stream)
                task.event.record(entry.update_stream)
        current_stream.wait_stream(entry.update_stream)
        self.stats.updates += 1

    def stats_dict(self) -> dict[str, int]:
        return self.stats.as_dict()


def copy_block_tables(target: torch.Tensor, source: torch.Tensor) -> None:
    target_shape = getattr(target, "shape", None)
    source_shape = getattr(source, "shape", None)
    if target_shape is None or source_shape is None:
        target.copy_(source)
        return
    if len(target_shape) != 2 or len(source_shape) != 2:
        target.copy_(source)
        return
    if source_shape[1] > target_shape[1]:
        raise RuntimeError(
            f"decode graph block table width {target_shape[1]} is smaller than runtime width {source_shape[1]}"
        )
    target.zero_()
    target[:, : source_shape[1]].copy_(source)
