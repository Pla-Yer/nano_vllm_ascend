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
            self.runner.free_seq(seq)

    def _commit_final_tokens(self, seqs):
        final_seqs = []
        for seq in seqs:
            seq.append_next_token()
            final_seqs.append(seq)
        finished = self.scheduler.finish_sequences(final_seqs, "max_new_tokens")
        self._free_finished(finished)
        return finished

    def step(self, eos_token_id: int):
        step = self.scheduler.plan_next_step(eos_token_id)
        self._free_finished(step.finish_seqs)
        finished_seqs = list(step.finish_seqs)

        if step.prefill_seqs:
            self.runner.prefill(step.prefill_seqs)

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
                finished_seqs.extend(self._commit_final_tokens(final_seqs))
            if not decode_seqs:
                return finished_seqs
            self.runner.decode(decode_seqs)

        return finished_seqs

    def run(self, eos_token_id: int) -> None:
        while self.scheduler.has_unfinished():
            self.step(eos_token_id)


class LLM:
    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        block_size: int = 128,
        num_blocks: int | None = None,
        max_num_seqs: int = 4,
        device_id: int = 0,
        npu_memory_utilization: float = 0.8,
        enable_prefix_cache: bool = False,
        enable_decode_graph: bool = False,
        decode_graph_batch_sizes: list[int] | None = None,
    ):
        from .model_runner import ModelRunner

        if decode_graph_batch_sizes is None:
            decode_graph_batch_sizes = list(range(1, max_num_seqs + 1))

        self.runner = ModelRunner(
            model_path=model_path,
            max_model_len=max_model_len,
            num_blocks=num_blocks,
            block_size=block_size,
            device_id=device_id,
            npu_memory_utilization=npu_memory_utilization,
            enable_prefix_cache=enable_prefix_cache,
            enable_decode_graph=enable_decode_graph,
            decode_graph_batch_sizes=decode_graph_batch_sizes,
        )
        self.scheduler = MiniScheduler(
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            total_num_blocks=self.runner.num_blocks,
        )
        self.engine_core = EngineCore(self.runner, self.scheduler)

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
        request_ids = [
            self.submit(
                prompt,
                max_new_tokens=max_new_tokens,
                sampling_params=resolved_sampling_params,
            )
            for prompt in prompts
        ]

        outputs_by_request_id = {}
        while self.has_unfinished():
            for output in self.step():
                outputs_by_request_id[output["request_id"]] = output

        return [
            {
                "texts": outputs_by_request_id[request_id]["texts"],
                "token_ids": outputs_by_request_id[request_id]["token_ids"],
            }
            for request_id in request_ids
        ]

    def submit(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        sampling_params: SamplingParams | None = None,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> int:
        resolved_sampling_params = self._resolve_sampling_params(
            sampling_params,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_ids = self.runner.tokenize_prompts([prompt])[0]
        seq = Sequence(
            seq_id=self.scheduler.next_seq_id(),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=token_ids,
            sampling_params=resolved_sampling_params,
        )
        self.runner.prepare_sequences([seq])
        seq.reserved_blocks = self.scheduler.compute_required_blocks(
            runtime_prompt_len=seq.runtime_prompt_len,
            max_new_tokens=max_new_tokens,
        )
        self.scheduler.add_request(seq)
        return seq.seq_id

    def step(self) -> list[dict[str, object]]:
        eos_token_id = int(self.runner.tokenizer.eos_token_id)
        finished = self.engine_core.step(eos_token_id)
        return [
            {
                "request_id": seq.seq_id,
                "texts": self.runner.tokenizer.decode(
                    seq.generated_token_ids,
                    skip_special_tokens=True,
                ),
                "token_ids": list(seq.generated_token_ids),
            }
            for seq in finished
        ]

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def clear_prefix_cache(self) -> None:
        self.runner.clear_prefix_cache()

    def warm(
        self,
        prompt: str = "warm",
        max_new_tokens: int = 1,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        self.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )
