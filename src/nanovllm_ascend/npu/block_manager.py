from __future__ import annotations

import math
from dataclasses import dataclass

import torch


BlockHash = tuple["BlockHash | None", tuple[int, ...]]


@dataclass
class PagedKVCacheMetadata:
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    actual_seq_lengths_q: list[int] | None = None
    actual_seq_lengths_kv: list[int] | None = None
    query_lens: list[int] | None = None
    use_paged_prefill: bool = False


@dataclass
class BlockState:
    block_id: int
    ref_count: int = 0
    block_hash: BlockHash | None = None


class BlockManager:
    """
    Unified block pool for runtime allocation and prefix cache reuse.

    Prefix-cached blocks stay in the free queue when ref_count reaches zero.
    They are removed from the hash map only when the physical block is reused.
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

        self.blocks = [BlockState(block_id=i) for i in range(num_blocks)]
        self.free_block_ids: list[int] = list(range(num_blocks))
        self.cached_block_hash_to_ids: dict[BlockHash, list[int]] = {}
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}

    def _as_token_tuple(self, token_ids, start: int, end: int) -> tuple[int, ...]:
        chunk = token_ids[start:end]
        if hasattr(chunk, "tolist"):
            values = chunk.tolist()
        elif hasattr(token_ids, "values"):
            values = token_ids.values[start:end]
        else:
            values = list(chunk)
        return tuple(int(value) for value in values)

    def _make_block_hash(
        self,
        token_ids,
        block_idx: int,
        parent_hash: BlockHash | None,
    ) -> BlockHash:
        start = block_idx * self.block_size
        end = start + self.block_size
        return (parent_hash, self._as_token_tuple(token_ids, start, end))

    def _remove_from_free_queue(self, block_id: int) -> None:
        try:
            self.free_block_ids.remove(block_id)
        except ValueError:
            pass

    def _pop_cached_block(self, block_hash: BlockHash, block_id: int) -> None:
        block_ids = self.cached_block_hash_to_ids.get(block_hash)
        if block_ids is None:
            return
        try:
            block_ids.remove(block_id)
        except ValueError:
            return
        if not block_ids:
            self.cached_block_hash_to_ids.pop(block_hash, None)

    def _evict_block_hash(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.block_hash is None:
            return
        self._pop_cached_block(block.block_hash, block_id)
        block.block_hash = None

    def _alloc_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("BlockManager out of free blocks")

        block_id = self.free_block_ids.pop(0)
        block = self.blocks[block_id]
        self._evict_block_hash(block_id)
        if block.ref_count != 0:
            raise RuntimeError(f"block {block_id} is not free")
        block.ref_count = 1
        return block_id

    def _retain_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self._remove_from_free_queue(block_id)
        block.ref_count += 1

    def _release_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"block {block_id} ref_count underflow")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)

    def _num_required_blocks(self, end_pos: int) -> int:
        if end_pos <= 0:
            return 0
        return math.ceil(end_pos / self.block_size)

    def _init_slot(
        self,
        slot: int,
        start_pos: int,
        prefix_block_ids: list[int] | None,
    ) -> None:
        if slot in self.block_tables:
            return

        table = list(prefix_block_ids or [])
        for block_id in table:
            self._retain_block(block_id)
        self.block_tables[slot] = table
        self.seq_lens[slot] = start_pos

    def ensure_blocks(self, slot: int, end_pos: int) -> None:
        if slot not in self.block_tables:
            self.block_tables[slot] = []
            self.seq_lens[slot] = 0

        table = self.block_tables[slot]
        required_blocks = self._num_required_blocks(end_pos)

        while len(table) < required_blocks:
            table.append(self._alloc_block())

    def find_longest_prefix_blocks(
        self,
        token_ids,
        max_cache_blocks: int,
    ) -> list[int]:
        prefix_blocks: list[int] = []
        parent_hash: BlockHash | None = None

        for block_idx in range(max_cache_blocks):
            block_hash = self._make_block_hash(token_ids, block_idx, parent_hash)
            block_ids = self.cached_block_hash_to_ids.get(block_hash)
            if not block_ids:
                break
            prefix_blocks.append(block_ids[0])
            parent_hash = block_hash

        return prefix_blocks

    def cache_full_blocks(
        self,
        slot: int,
        token_ids,
        num_cached_blocks: int = 0,
        num_full_blocks: int | None = None,
    ) -> None:
        table = self.block_tables[slot]
        total_full_blocks = num_full_blocks
        if total_full_blocks is None:
            total_full_blocks = int(token_ids.numel()) // self.block_size

        parent_hash: BlockHash | None = None
        if num_cached_blocks > 0:
            parent_hash = self.blocks[table[num_cached_blocks - 1]].block_hash

        for block_idx in range(num_cached_blocks, total_full_blocks):
            block_id = table[block_idx]
            block = self.blocks[block_id]
            if block.block_hash is not None:
                parent_hash = block.block_hash
                continue
            block_hash = self._make_block_hash(token_ids, block_idx, parent_hash)
            block.block_hash = block_hash
            self.cached_block_hash_to_ids.setdefault(block_hash, []).append(block_id)
            parent_hash = block_hash

    def free_slot(self, slot: int) -> None:
        table = self.block_tables.pop(slot, [])
        self.seq_lens.pop(slot, None)
        for block_id in reversed(table):
            self._release_block(block_id)

    def reset(self) -> None:
        if self.block_tables:
            raise RuntimeError("cannot reset BlockManager with active slots")
        self.free_block_ids = list(range(self.num_blocks))
        self.cached_block_hash_to_ids.clear()
        self.seq_lens.clear()
        for block in self.blocks:
            block.ref_count = 0
            block.block_hash = None

    def clear_prefix_cache(self) -> None:
        if self.block_tables or any(block.ref_count > 0 for block in self.blocks):
            raise RuntimeError("cannot clear prefix cache with active sequences")
        self.cached_block_hash_to_ids.clear()
        for block in self.blocks:
            block.block_hash = None

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
        prefix_block_ids: list[list[int]] | None = None,
    ) -> PagedKVCacheMetadata:
        if start_positions is None:
            start_positions = [0] * len(slots)
        if prefix_block_ids is None:
            prefix_block_ids = [[] for _ in slots]

        if not (
            len(slots)
            == len(seq_lens)
            == len(start_positions)
            == len(prefix_block_ids)
        ):
            raise ValueError(
                "slots, seq_lens, start_positions and prefix_block_ids length mismatch"
            )

        actual_seq_lengths_q: list[int] = []
        actual_seq_lengths_kv: list[int] = []
        total_q = 0
        total_kv = 0
        use_paged_prefill = any(start_pos > 0 for start_pos in start_positions)

        for slot, seq_len, start_pos, prefix_ids in zip(
            slots,
            seq_lens,
            start_positions,
            prefix_block_ids,
        ):
            self._init_slot(slot, start_pos, prefix_ids)
            end_pos = start_pos + seq_len
            self.ensure_blocks(slot, end_pos)
            self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)
            total_q += seq_len
            total_kv += end_pos
            actual_seq_lengths_q.append(total_q)
            actual_seq_lengths_kv.append(total_kv)

        return PagedKVCacheMetadata(
            block_tables=self.get_block_tables_tensor(slots),
            context_lens=self.get_context_lens_tensor(slots),
            slot_mapping=self.get_slot_mapping(
                slots=slots,
                start_positions=start_positions,
                num_tokens_per_seq=seq_lens,
            ),
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_lens=seq_lens,
            use_paged_prefill=use_paged_prefill,
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
