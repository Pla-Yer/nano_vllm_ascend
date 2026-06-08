# src/nanovllm_ascend/sequence.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from .sampling_params import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt: str
    max_new_tokens: int
    prompt_token_ids: "torch.Tensor | Any"
    sampling_params: "SamplingParams"

    status: SequenceStatus = SequenceStatus.WAITING
    estimated_prompt_len: int = 0
    reserved_blocks: int = 0
    finish_reason: str | None = None
    cached_prefix_len: int = 0
    cached_block_ids: list[int] = field(default_factory=list)
    runtime_prompt_token_ids: "torch.Tensor | Any | None" = None
    runtime_prompt_len: int = 0

    # Filled after prefill
    prompt_len: int = 0
    cache_position: int = 0
    next_token_id: int | None = None

    # Generated output tokens, not including prompt tokens
    generated_token_ids: list[int] = field(default_factory=list)

    def set_prefill_result(self, next_token_id: int, prompt_len: int) -> None:
        self.next_token_id = next_token_id
        self.prompt_len = prompt_len
        self.cache_position = prompt_len

    def append_next_token(self) -> int:
        if self.next_token_id is None:
            raise RuntimeError(f"seq {self.seq_id} has no next_token_id")

        token_id = self.next_token_id
        self.generated_token_ids.append(token_id)
        return token_id

    def set_decode_result(self, next_token_id: int) -> None:
        self.next_token_id = next_token_id
        self.cache_position += 1

    def reach_max_tokens(self) -> bool:
        return len(self.generated_token_ids) >= self.max_new_tokens

    def finish(self, reason: str | None = None) -> None:
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason
