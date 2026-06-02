from __future__ import annotations

from .model_runner import ModelRunner
from .sequence import Sequence, SequenceStatus
from .scheduler import MiniScheduler
class LLM:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        block_size: int = 128,
        num_blocks: int = 12,
        max_num_seqs: int = 4,
        device_id: int = 0,
    ):
        self.runner = ModelRunner(
            model_path=model_path,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            block_size=block_size,
            device_id=device_id,
        )
        self.scheduler = MiniScheduler(max_num_seqs=max_num_seqs)

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        if not prompts:
            return []

        scheduler = MiniScheduler(max_num_seqs=len(prompts))

        seqs = [
            scheduler.add_request(prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]

        eos_token_id = int(self.runner.tokenizer.eos_token_id)

        while scheduler.has_unfinished():
            # 1. Admit waiting requests to prefill
            prefill_seqs = scheduler.schedule_prefill()
            if prefill_seqs:
                self.runner.prefill(prefill_seqs)

            # 2. Select running requests for decode
            running_seqs = scheduler.schedule_decode()

            decode_seqs: list[Sequence] = []

            for seq in running_seqs:
                if seq.next_token_id is None:
                    continue

                if seq.next_token_id == eos_token_id:
                    self.runner.free_seq(seq)
                    scheduler.finish_sequence(seq)
                    continue

                if seq.reach_max_tokens():
                    self.runner.free_seq(seq)
                    scheduler.finish_sequence(seq)
                    continue

                decode_seqs.append(seq)

            # 3. Decode one token for active sequences
            if decode_seqs:
                self.runner.decode(decode_seqs)
        return [
            {
                "texts": self.runner.tokenizer.decode(
                    seq.generated_token_ids,
                    skip_special_tokens=True,
                ),
                "token_ids": list(seq.generated_token_ids),
            }
            for seq in seqs
        ]
    
        
