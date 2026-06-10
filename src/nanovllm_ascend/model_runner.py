from __future__ import annotations

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .layers import Sampler
from .models.qwen3 import Qwen3ForCausalLM
from .npu.paged_kv_cache import PagedKVCache
from .npu.block_manager import BlockManager
from .npu.acl_graph import DecodeGraphRunner
from .sequence import Sequence


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        num_blocks: int | None,
        block_size: int,
        device_id: int,
        npu_memory_utilization: float,
        enable_prefix_cache: bool = False,
        enable_decode_graph: bool = False,
        decode_graph_batch_sizes: list[int] | None = None,
    ):
        torch.npu.set_device(device_id)

        self.model_path = model_path
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.device = "npu"
        self.dtype = torch.bfloat16
        self.sampler = Sampler()
        self.enable_prefix_cache = enable_prefix_cache
        self.enable_decode_graph = enable_decode_graph

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.config.nanovllm_block_size = block_size
        self.config.nanovllm_max_model_len = max_model_len

        self.model = self._load_model()
        self.num_blocks = num_blocks or self._num_blocks_from_memory(
            npu_memory_utilization
        )

        self.block_manager = BlockManager(
            num_blocks=self.num_blocks,
            block_size=block_size,
            device=self.device,
        )

        self.kv_cache = PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.decode_graph_runner = None
        if enable_decode_graph:
            self.decode_graph_runner = DecodeGraphRunner(
                model=self.model,
                kv_cache=self.kv_cache,
                batch_sizes=decode_graph_batch_sizes or [1],
                max_model_len=self.max_model_len,
                block_size=self.block_size,
            )

    @torch.inference_mode()
    def _load_model(self) -> Qwen3ForCausalLM:
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        ).to(self.device)
        hf_model.eval()

        model = Qwen3ForCausalLM(self.config).to(
            dtype=self.dtype,
            device=self.device,
        )
        missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=False)

        if getattr(self.config, "tie_word_embeddings", False):
            model.tie_weights()
            missing = [key for key in missing if key != "lm_head.weight"]

        if missing or unexpected:
            print(f"[load] missing={missing}, unexpected={unexpected}")

        model.eval()
        del hf_model
        return model

    def _num_blocks_from_memory(self, npu_memory_utilization: float) -> int:
        free_memory, total_memory = torch.npu.mem_get_info()
        used_memory = total_memory - free_memory
        kv_cache_budget = int(total_memory * npu_memory_utilization) - used_memory
        bytes_per_block = (
            2
            * self.config.num_hidden_layers
            * self.block_size
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )
        return kv_cache_budget // bytes_per_block

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs["input_ids"][0]

    def tokenize_prompts(self, prompts: list[str]) -> list[torch.Tensor]:
        prompt_token_ids = [self._tokenize_prompt(prompt) for prompt in prompts]
        seq_lens = [int(token_ids.numel()) for token_ids in prompt_token_ids]
        if max(seq_lens) > self.max_model_len:
            raise ValueError(f"prompt length {max(seq_lens)} exceeds max_model_len {self.max_model_len}")
        return prompt_token_ids

    def prepare_sequences(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            seq.estimated_prompt_len = int(seq.prompt_token_ids.numel())
            seq.cached_block_ids = []
            seq.cached_prefix_len = 0

            if self.enable_prefix_cache:
                full_blocks = seq.estimated_prompt_len // self.block_size
                max_cache_blocks = full_blocks
                if seq.estimated_prompt_len % self.block_size == 0 and max_cache_blocks > 0:
                    max_cache_blocks -= 1
                if max_cache_blocks > 0:
                    seq.cached_block_ids = self.block_manager.find_longest_prefix_blocks(
                        seq.prompt_token_ids,
                        max_cache_blocks=max_cache_blocks,
                    )
                    seq.cached_prefix_len = len(seq.cached_block_ids) * self.block_size

            seq.runtime_prompt_token_ids = seq.prompt_token_ids[seq.cached_prefix_len :]
            seq.runtime_prompt_len = int(seq.runtime_prompt_token_ids.numel())
            if seq.runtime_prompt_len <= 0:
                raise ValueError(f"seq {seq.seq_id} has empty runtime prompt suffix")

    def _prepare_prefill_inputs(self, seqs: list[Sequence]):
        input_ids_list = [seq.runtime_prompt_token_ids for seq in seqs]
        seq_lens = [seq.runtime_prompt_len for seq in seqs]

        input_ids_flat = torch.cat(input_ids_list, dim=0).to(self.device)
        position_ids_flat = torch.cat(
            [
                torch.arange(
                    seq.cached_prefix_len,
                    seq.cached_prefix_len + seq.runtime_prompt_len,
                    dtype=torch.long,
                )
                for seq in seqs
            ],
            dim=0,
        ).to(self.device)

        last_token_indices = []
        total = 0
        for seq_len in seq_lens:
            total += seq_len
            last_token_indices.append(total - 1)

        return (
            input_ids_flat,
            position_ids_flat,
            seq_lens,
            torch.tensor(last_token_indices, dtype=torch.long, device=self.device),
        )

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence]) -> None:
        if not seqs:
            return

        seq_ids = [seq.seq_id for seq in seqs]

        input_ids_flat, position_ids_flat, seq_lens, last_token_indices = (
            self._prepare_prefill_inputs(seqs)
        )

        attn_metadata = self.block_manager.prepare_prefill_metadata(
            slots=seq_ids,
            seq_lens=seq_lens,
            start_positions=[seq.cached_prefix_len for seq in seqs],
            prefix_block_ids=[seq.cached_block_ids for seq in seqs],
        )

        outputs = self.model(
            input_ids_flat=input_ids_flat,
            position_ids_flat=position_ids_flat,
            kv_cache=self.kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=True,
        )

        last_logits = outputs.logits.index_select(0, last_token_indices)
        for seq, logits in zip(seqs, last_logits.unbind(0)):
            next_token = int(self.sampler.sample(logits.unsqueeze(0), seq.sampling_params).item())
            seq.set_prefill_result(
                next_token_id=next_token,
                prompt_len=seq.estimated_prompt_len,
            )
            if self.enable_prefix_cache:
                self.block_manager.cache_full_blocks(
                    slot=seq.seq_id,
                    token_ids=seq.prompt_token_ids,
                    num_cached_blocks=len(seq.cached_block_ids),
                    num_full_blocks=seq.estimated_prompt_len // self.block_size,
                )

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> None:
        if not seqs:
            return

        token_ids = [seq.append_next_token() for seq in seqs]
        cache_positions = [seq.cache_position for seq in seqs]
        seq_ids = [seq.seq_id for seq in seqs]

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.tensor(
            cache_positions,
            dtype=torch.long,
            device=self.device,
        )

        attn_metadata = self.block_manager.prepare_decode_metadata(
            slots=seq_ids,
            start_positions=cache_positions,
            q_len=1,
        )

        logits = None
        if self.decode_graph_runner is not None:
            logits = self.decode_graph_runner.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )

        if logits is None:
            outputs = self.model(
                input_ids_flat=input_ids,
                position_ids_flat=position_ids,
                kv_cache=self.kv_cache,
                attn_metadata=attn_metadata,
                is_prefill=False,
            )
            logits = outputs.logits

        for seq, seq_logits in zip(seqs, logits.unbind(0)):
            next_token = int(self.sampler.sample(seq_logits.unsqueeze(0), seq.sampling_params).item())
            seq.set_decode_result(next_token)

    def decode_graph_stats(self) -> dict[str, int]:
        if self.decode_graph_runner is None:
            return {}
        return self.decode_graph_runner.stats_dict()

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free_slot(seq.seq_id)

    def clear_prefix_cache(self) -> None:
        self.block_manager.clear_prefix_cache()
