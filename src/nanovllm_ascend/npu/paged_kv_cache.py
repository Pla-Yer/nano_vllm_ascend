from __future__ import annotations

import math
from dataclasses import dataclass

import torch

class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
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
        self.value_cache = torch.empty(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
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
    

