# src/nanovllm_ascend/scheduler.py
from __future__ import annotations

from collections import deque

from .sequence import Sequence, SequenceStatus


class MiniScheduler:
    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: dict[int, Sequence] = {}
        self._next_seq_id = 0

    def add_request(self, prompt: str, max_new_tokens: int) -> Sequence:
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        self._next_seq_id += 1
        self.waiting.append(seq)
        return seq

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule_prefill(self) -> list[Sequence]:
        available = self.max_num_seqs - len(self.running)
        if available <= 0:
            return []

        scheduled: list[Sequence] = []

        while available > 0 and self.waiting:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running[seq.seq_id] = seq
            scheduled.append(seq)
            available -= 1

        return scheduled

    def schedule_decode(self) -> list[Sequence]:
        return list(self.running.values())

    def finish_sequence(self, seq: Sequence) -> None:
        self.running.pop(seq.seq_id, None)
        seq.finish()