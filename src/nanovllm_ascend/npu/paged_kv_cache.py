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


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        max_num_seqs: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_cache = torch.empty(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.empty_like(self.key_cache)
        self.block_tables: list[list[int]] = [[] for _ in range(max_num_seqs)]
        self.seq_lens = [0 for _ in range(max_num_seqs)]
        self.free_blocks = list(range(num_blocks))

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("PagedKVCache out of free blocks")
        return self.free_blocks.pop(0)

    def _ensure_blocks(self, seq_slot: int, end_pos: int) -> None:
        required_blocks = math.ceil(end_pos / self.block_size)
        table = self.block_tables[seq_slot]
        while len(table) < required_blocks:
            table.append(self._alloc_block())

    def _normalize_seq_slots(self, seq_slots) -> list[int]:
        if isinstance(seq_slots, int):
            return [seq_slots]
        if isinstance(seq_slots, torch.Tensor):
            return [int(x) for x in seq_slots.detach().cpu().tolist()]
        return [int(x) for x in seq_slots]

    def _normalize_start_positions(self, start_pos, batch_size: int) -> list[int]:
        if isinstance(start_pos, int):
            return [start_pos for _ in range(batch_size)]
        if isinstance(start_pos, torch.Tensor):
            values = start_pos.detach().cpu().tolist()
            if isinstance(values, int):
                values = [values]
            return [int(x) for x in values]
        values = [int(x) for x in start_pos]
        if len(values) != batch_size:
            raise ValueError(f"len(start_pos) {len(values)} != batch_size {batch_size}")
        return values

    def get_block_tables_tensor(self, seq_slots) -> torch.Tensor:
        slots = self._normalize_seq_slots(seq_slots)
        max_num_blocks = max(len(self.block_tables[slot]) for slot in slots)
        block_tables = torch.zeros(
            (len(slots), max_num_blocks),
            dtype=torch.int32,
            device=self.device,
        )
        for i, slot in enumerate(slots):
            table = self.block_tables[slot]
            if table:
                block_tables[i, : len(table)] = torch.tensor(table, dtype=torch.int32, device=self.device)
        return block_tables

    def get_context_lens_tensor(self, seq_slots) -> torch.Tensor:
        slots = self._normalize_seq_slots(seq_slots)
        return torch.tensor([self.seq_lens[slot] for slot in slots], dtype=torch.int32, device="cpu")

    def _slot_mapping_one(self, seq_slot: int, start_pos: int, num_tokens: int) -> list[int]:
        table = self.block_tables[seq_slot]
        mapping = []
        for offset in range(num_tokens):
            logical_pos = start_pos + offset
            block_idx = logical_pos // self.block_size
            block_offset = logical_pos % self.block_size
            mapping.append(table[block_idx] * self.block_size + block_offset)
        return mapping

    def get_prefill_slot_mapping(self, seq_slots, seq_lens: list[int], start_pos) -> torch.Tensor:
        slots = self._normalize_seq_slots(seq_slots)
        starts = self._normalize_start_positions(start_pos, len(slots))
        mapping: list[int] = []
        for slot, seq_len, start in zip(slots, seq_lens, starts):
            mapping.extend(self._slot_mapping_one(slot, start, seq_len))
        return torch.tensor(mapping, dtype=torch.int64, device=self.device)

    def get_slot_mapping(self, seq_slots, start_pos, q_len: int) -> torch.Tensor:
        slots = self._normalize_seq_slots(seq_slots)
        starts = self._normalize_start_positions(start_pos, len(slots))
        mapping: list[int] = []
        for slot, start in zip(slots, starts):
            mapping.extend(self._slot_mapping_one(slot, start, q_len))
        return torch.tensor(mapping, dtype=torch.int64, device=self.device)

    def prepare_prefill_metadata(self, seq_slots, seq_lens: list[int], start_pos=None) -> PagedKVCacheMetadata:
        slots = self._normalize_seq_slots(seq_slots)
        if len(seq_lens) != len(slots):
            raise ValueError(f"len(seq_lens) {len(seq_lens)} != batch_size {len(slots)}")
        starts = [0 for _ in slots] if start_pos is None else self._normalize_start_positions(start_pos, len(slots))

        for slot, seq_len, start in zip(slots, seq_lens, starts):
            end_pos = start + seq_len
            self._ensure_blocks(seq_slot=slot, end_pos=end_pos)
            self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)

        actual_seq_lengths = []
        total = 0
        for seq_len in seq_lens:
            total += int(seq_len)
            actual_seq_lengths.append(total)

        return PagedKVCacheMetadata(
            block_tables=self.get_block_tables_tensor(slots),
            context_lens=self.get_context_lens_tensor(slots),
            slot_mapping=self.get_prefill_slot_mapping(slots, seq_lens, starts),
            actual_seq_lengths_q=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths.copy(),
        )

    def prepare_metadata(self, seq_slots, start_pos, q_len: int) -> PagedKVCacheMetadata:
        slots = self._normalize_seq_slots(seq_slots)
        starts = self._normalize_start_positions(start_pos, len(slots))
        for slot, start in zip(slots, starts):
            end_pos = start + q_len
            self._ensure_blocks(seq_slot=slot, end_pos=end_pos)
            self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)
        return PagedKVCacheMetadata(
            block_tables=self.get_block_tables_tensor(slots),
            context_lens=self.get_context_lens_tensor(slots),
            slot_mapping=self.get_slot_mapping(slots, starts, q_len),
        )

    def write_prefill(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        total_tokens, num_kv_heads, head_dim = key_states.shape
        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(f"kv shape mismatch: got {tuple(key_states.shape)}")
        if slot_mapping.numel() != total_tokens:
            raise ValueError(f"slot_mapping numel {slot_mapping.numel()} != total_tokens {total_tokens}")

        key_cache_flat = self.key_cache[layer_idx].view(
            self.num_blocks * self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        value_cache_flat = self.value_cache[layer_idx].view(
            self.num_blocks * self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        slot_mapping = slot_mapping.to(device=self.device, dtype=torch.long).contiguous()
        key_cache_flat.index_copy_(0, slot_mapping, key_states.contiguous())
        value_cache_flat.index_copy_(0, slot_mapping, value_states.contiguous())

    def write_decode(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if key_states.dim() == 4 and key_states.shape[2] == 1:
            key_to_write = key_states[:, :, 0, :].contiguous()
            value_to_write = value_states[:, :, 0, :].contiguous()
        elif key_states.dim() == 3:
            key_to_write = key_states.contiguous()
            value_to_write = value_states.contiguous()
        else:
            raise ValueError(f"unsupported key_states shape: {tuple(key_states.shape)}")

        batch_size, num_kv_heads, head_dim = key_to_write.shape
        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(f"kv shape mismatch: got {tuple(key_to_write.shape)}")
        if slot_mapping.numel() != batch_size:
            raise ValueError(f"slot_mapping numel {slot_mapping.numel()} != batch_size {batch_size}")

        key_cache_flat = self.key_cache[layer_idx].view(
            self.num_blocks * self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        value_cache_flat = self.value_cache[layer_idx].view(
            self.num_blocks * self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )
        slot_mapping = slot_mapping.to(device=self.device, dtype=torch.long).contiguous()
        key_cache_flat.index_copy_(0, slot_mapping, key_to_write)
        value_cache_flat.index_copy_(0, slot_mapping, value_to_write)

    def get_physical_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

