# src/nanovllm_ascend/npu/block_manager.py
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PagedKVCacheMetadata:
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    actual_seq_lengths_q: list[int] | None = None
    actual_seq_lengths_kv: list[int] | None = None


class BlockManager:
    """
    Simple block manager for paged KV cache.

    Only manages logical block allocation.
    Does not own key/value cache tensors.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        device: torch.device | str,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device

        self.free_block_ids: list[int] = list(range(num_blocks))
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}

    def _alloc_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("BlockManager out of free blocks")
        return self.free_block_ids.pop(0)

    def _num_required_blocks(self, end_pos: int) -> int:
        if end_pos <= 0:
            return 0
        return math.ceil(end_pos / self.block_size)

    def ensure_blocks(self, slot: int, end_pos: int) -> None:
        if slot not in self.block_tables:
            self.block_tables[slot] = []
            self.seq_lens[slot] = 0

        table = self.block_tables[slot]
        required_blocks = self._num_required_blocks(end_pos)

        while len(table) < required_blocks:
            table.append(self._alloc_block())

    def free_slot(self, slot: int) -> None:
        table = self.block_tables.pop(slot, [])
        self.seq_lens.pop(slot, None)
        self.free_block_ids.extend(table)

    def reset(self) -> None:
        self.free_block_ids = list(range(self.num_blocks))
        self.block_tables.clear()
        self.seq_lens.clear()

    def _physical_slot(self, slot: int, logical_pos: int) -> int:
        table = self.block_tables[slot]

        block_idx = logical_pos // self.block_size
        block_offset = logical_pos % self.block_size

        if block_idx >= len(table):
            raise RuntimeError(
                f"missing block: slot={slot}, logical_pos={logical_pos}"
            )

        block_id = table[block_idx]
        return block_id * self.block_size + block_offset

    def get_block_tables_tensor(self, slots: list[int]) -> torch.Tensor:
        max_blocks = max(len(self.block_tables[slot]) for slot in slots)

        block_tables = torch.zeros(
            (len(slots), max_blocks),
            dtype=torch.int32,
            device=self.device,
        )

        for i, slot in enumerate(slots):
            table = self.block_tables[slot]
            block_tables[i, : len(table)] = torch.tensor(
                table,
                dtype=torch.int32,
                device=self.device,
            )

        return block_tables

    def get_context_lens_tensor(self, slots: list[int]) -> torch.Tensor:
        return torch.tensor(
            [self.seq_lens[slot] for slot in slots],
            dtype=torch.int32,
            device="cpu",
        )

    def get_slot_mapping(
        self,
        slots: list[int],
        start_positions: list[int],
        num_tokens_per_seq: list[int],
    ) -> torch.Tensor:
        mapping: list[int] = []

        for slot, start_pos, num_tokens in zip(
            slots,
            start_positions,
            num_tokens_per_seq,
        ):
            for offset in range(num_tokens):
                logical_pos = start_pos + offset
                mapping.append(self._physical_slot(slot, logical_pos))

        return torch.tensor(mapping, dtype=torch.int64, device=self.device)

    def prepare_prefill_metadata(
        self,
        slots: list[int],
        seq_lens: list[int],
        start_positions: list[int] | None = None,
    ) -> PagedKVCacheMetadata:
        if start_positions is None:
            start_positions = [0] * len(slots)

        if not (len(slots) == len(seq_lens) == len(start_positions)):
            raise ValueError("slots, seq_lens and start_positions length mismatch")

        for slot, seq_len, start_pos in zip(slots, seq_lens, start_positions):
            end_pos = start_pos + seq_len
            self.ensure_blocks(slot, end_pos)
            self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)

        actual_seq_lengths: list[int] = []
        total = 0
        for seq_len in seq_lens:
            total += seq_len
            actual_seq_lengths.append(total)

        return PagedKVCacheMetadata(
            block_tables=self.get_block_tables_tensor(slots),
            context_lens=self.get_context_lens_tensor(slots),
            slot_mapping=self.get_slot_mapping(
                slots=slots,
                start_positions=start_positions,
                num_tokens_per_seq=seq_lens,
            ),
            actual_seq_lengths_q=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths.copy(),
        )

    def prepare_decode_metadata(
        self,
        slots: list[int],
        start_positions: list[int],
        q_len: int = 1,
    ) -> PagedKVCacheMetadata:
        if len(slots) != len(start_positions):
            raise ValueError("slots and start_positions length mismatch")

        num_tokens_per_seq = [q_len] * len(slots)

        for slot, start_pos in zip(slots, start_positions):
            end_pos = start_pos + q_len
            self.ensure_blocks(slot, end_pos)
            self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)

        return PagedKVCacheMetadata(
            block_tables=self.get_block_tables_tensor(slots),
            context_lens=self.get_context_lens_tensor(slots),
            slot_mapping=self.get_slot_mapping(
                slots=slots,
                start_positions=start_positions,
                num_tokens_per_seq=num_tokens_per_seq,
            ),
        )