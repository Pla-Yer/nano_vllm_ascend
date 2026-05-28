from __future__ import annotations

from .model_runner import ModelRunner


class LLM:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        block_size: int = 128,
        device_id: int = 0,
    ):
        self.runner = ModelRunner(
            model_path=model_path,
            max_model_len=max_model_len,
            block_size=block_size,
            device_id=device_id,
        )

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        if not prompts:
            return []

        batch_size = len(prompts)
        generated_token_ids = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]

        prefill = self.runner.prefill(prompts)
        next_tokens = prefill.next_tokens
        cache_positions = prefill.seq_lens.copy()

        for _ in range(max_new_tokens):
            active_slots: list[int] = []
            active_tokens: list[int] = []
            active_positions: list[int] = []

            for seq_slot in range(batch_size):
                if finished[seq_slot]:
                    continue

                token_id = next_tokens[seq_slot]
                if token_id == prefill.eos_token_id:
                    finished[seq_slot] = True
                    continue

                generated_token_ids[seq_slot].append(token_id)
                active_slots.append(seq_slot)
                active_tokens.append(token_id)
                active_positions.append(cache_positions[seq_slot])

            if not active_slots:
                break

            new_next_tokens = self.runner.decode(
                active_slots=active_slots,
                token_ids=active_tokens,
                cache_positions=active_positions,
            )

            for i, seq_slot in enumerate(active_slots):
                next_tokens[seq_slot] = new_next_tokens[i]
                cache_positions[seq_slot] += 1

        return self.runner.decode_token_ids(generated_token_ids)

