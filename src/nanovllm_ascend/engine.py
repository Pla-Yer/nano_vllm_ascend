from __future__ import annotations

from .sequence import Sequence
from .scheduler import MiniScheduler
from .sampling_params import SamplingParams


class EngineCore:
    def __init__(self, runner, scheduler: MiniScheduler):
        self.runner = runner
        self.scheduler = scheduler

    def _free_finished(self, seqs) -> None:
        for seq in seqs:
            try:
                self.runner.free_seq(seq)
            except Exception:
                pass

    def _abort_and_free(self, seqs) -> None:
        for seq in self.scheduler.abort_sequences(seqs):
            try:
                self.runner.free_seq(seq)
            except Exception:
                pass

    def _commit_final_tokens(self, seqs) -> None:
        final_seqs = []
        for seq in seqs:
            seq.append_next_token()
            final_seqs.append(seq)
        finished = self.scheduler.finish_sequences(final_seqs, "max_new_tokens")
        self._free_finished(finished)

    def step(self, eos_token_id: int) -> None:
        step = self.scheduler.plan_next_step(eos_token_id)
        self._free_finished(step.finish_seqs)

        if step.prefill_seqs:
            try:
                self.runner.prefill(step.prefill_seqs)
            except RuntimeError:
                self._abort_and_free(step.prefill_seqs)

        if step.decode_seqs:
            final_seqs = [
                seq
                for seq in step.decode_seqs
                if len(seq.generated_token_ids) + 1 >= seq.max_new_tokens
            ]
            decode_seqs = [
                seq
                for seq in step.decode_seqs
                if len(seq.generated_token_ids) + 1 < seq.max_new_tokens
            ]
            if final_seqs:
                self._commit_final_tokens(final_seqs)
            if not decode_seqs:
                return
            try:
                self.runner.decode(decode_seqs)
            except RuntimeError:
                self._abort_and_free(decode_seqs)

    def run(self, eos_token_id: int) -> None:
        while self.scheduler.has_unfinished():
            self.step(eos_token_id)


class LLM:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        block_size: int = 128,
        num_blocks: int = 12,
        max_num_seqs: int = 4,
        device_id: int = 0,
        enable_prefix_cache: bool = False,
    ):
        from .model_runner import ModelRunner

        self.runner = ModelRunner(
            model_path=model_path,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            block_size=block_size,
            device_id=device_id,
            enable_prefix_cache=enable_prefix_cache,
        )
        self.scheduler = MiniScheduler(
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            total_num_blocks=num_blocks,
        )
        self.engine_core = EngineCore(self.runner, self.scheduler)

    def _get_engine_core(self) -> EngineCore:
        engine_core = getattr(self, "engine_core", None)
        if engine_core is None:
            engine_core = EngineCore(self.runner, self.scheduler)
            self.engine_core = engine_core
        return engine_core

    def _resolve_sampling_params(
        self,
        sampling_params: SamplingParams | None,
        *,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
    ) -> SamplingParams:
        base = sampling_params or SamplingParams()
        return base.with_overrides(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        sampling_params: SamplingParams | None = None,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[dict[str, object]]:
        if not prompts:
            return []

        resolved_sampling_params = self._resolve_sampling_params(
            sampling_params,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        self.scheduler.reset()
        prompt_token_ids = self.runner.tokenize_prompts(prompts)

        seqs = [
            Sequence(
                seq_id=self.scheduler.next_seq_id(),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                prompt_token_ids=token_ids,
                sampling_params=resolved_sampling_params,
            )
            for prompt, token_ids in zip(prompts, prompt_token_ids)
        ]
        self.runner.prepare_sequences(seqs)
        for seq in seqs:
            seq.reserved_blocks = self.scheduler.compute_required_blocks(
                runtime_prompt_len=seq.runtime_prompt_len,
                max_new_tokens=max_new_tokens,
            )
            self.scheduler.add_request(seq)

        eos_token_id = int(self.runner.tokenizer.eos_token_id)
        self._get_engine_core().run(eos_token_id)

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

    def clear_prefix_cache(self) -> None:
        self.runner.clear_prefix_cache()
