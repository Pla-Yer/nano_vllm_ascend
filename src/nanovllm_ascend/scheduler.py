# src/nanovllm_ascend/scheduler.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .sequence import Sequence, SequenceStatus


@dataclass
class SchedulerStep:
    prefill_seqs: list[Sequence] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    finish_seqs: list[Sequence] = field(default_factory=list)


class MiniScheduler:
    def __init__(self, max_num_seqs: int, block_size: int, total_num_blocks: int):
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        self.total_num_blocks = total_num_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: dict[int, Sequence] = {}
        self.finished: dict[int, Sequence] = {}
        self.seqs: dict[int, Sequence] = {}
        self.reserved_blocks_total = 0
        self._next_seq_id = 0

    def reset(self) -> None:
        self.waiting.clear()
        self.running.clear()
        self.finished.clear()
        self.seqs.clear()
        self.reserved_blocks_total = 0
        self._next_seq_id = 0

    def next_seq_id(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        return seq_id

    def add_request(self, seq: Sequence) -> Sequence:
        self.seqs[seq.seq_id] = seq
        self.waiting.append(seq)
        return seq

    def compute_required_blocks(self, runtime_prompt_len: int, max_new_tokens: int) -> int:
        total_tokens = runtime_prompt_len + max_new_tokens
        if total_tokens <= 0:
            return 0
        return math.ceil(total_tokens / self.block_size)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _can_admit(self, seq: Sequence) -> bool:
        if len(self.running) >= self.max_num_seqs:
            return False
        return self.reserved_blocks_total + seq.reserved_blocks <= self.total_num_blocks

    def _finish_sequence(self, seq: Sequence, reason: str) -> None:
        self.running.pop(seq.seq_id, None)
        self.reserved_blocks_total -= seq.reserved_blocks
        seq.finish(reason)
        self.finished[seq.seq_id] = seq

    def finish_sequences(
        self,
        seqs: list[Sequence],
        reason: str,
    ) -> list[Sequence]:
        finished: list[Sequence] = []
        for seq in seqs:
            if seq.seq_id not in self.running:
                continue
            self._finish_sequence(seq, reason)
            finished.append(seq)
        return finished

    def _should_finish(self, seq: Sequence, eos_token_id: int) -> str | None:
        if seq.next_token_id is not None and seq.next_token_id == eos_token_id:
            return "eos"
        if seq.reach_max_tokens():
            return "max_new_tokens"
        return None

    def plan_next_step(self, eos_token_id: int) -> SchedulerStep:
        step = SchedulerStep()

        for seq in list(self.running.values()):
            finish_reason = self._should_finish(seq, eos_token_id)
            if finish_reason is not None:
                self._finish_sequence(seq, finish_reason)
                step.finish_seqs.append(seq)

        while self.waiting and self._can_admit(self.waiting[0]):
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running[seq.seq_id] = seq
            self.reserved_blocks_total += seq.reserved_blocks
            step.prefill_seqs.append(seq)

        if self.waiting and not self.running:
            seq = self.waiting[0]
            if seq.reserved_blocks > self.total_num_blocks:
                self.waiting.popleft()
                seq.finish("aborted_no_capacity")
                self.finished[seq.seq_id] = seq
                step.finish_seqs.append(seq)

        for seq in self.running.values():
            if seq.next_token_id is not None:
                step.decode_seqs.append(seq)

        return step

    def abort_sequences(
        self,
        seqs: list[Sequence],
        reason: str = "aborted_runtime_error",
    ) -> list[Sequence]:
        aborted: list[Sequence] = []
        for seq in seqs:
            if seq.seq_id not in self.running:
                continue
            self._finish_sequence(seq, reason)
            aborted.append(seq)
        return aborted
