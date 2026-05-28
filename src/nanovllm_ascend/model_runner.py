from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .models.qwen3 import Qwen3ForCausalLM
from .npu.paged_kv_cache import PagedKVCache


@dataclass
class PrefillResult:
    next_tokens: list[int]
    seq_lens: list[int]
    eos_token_id: int


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        block_size: int,
        device_id: int,
    ):
        torch.npu.set_device(device_id)

        self.model_path = model_path
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.device = "npu"
        self.dtype = torch.bfloat16

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
        self.kv_cache: PagedKVCache | None = None

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

    def _new_kv_cache(self, batch_size: int) -> PagedKVCache:
        num_blocks_per_seq = (self.max_model_len + self.block_size - 1) // self.block_size
        return PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=batch_size * num_blocks_per_seq,
            block_size=self.block_size,
            max_num_seqs=batch_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def _tokenize_prompts(self, prompts: list[str]):
        input_ids_list = []
        seq_lens = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            ids = inputs["input_ids"][0]
            input_ids_list.append(ids)
            seq_lens.append(ids.numel())

        if max(seq_lens) > self.max_model_len:
            raise ValueError(f"prompt length {max(seq_lens)} exceeds max_model_len {self.max_model_len}")

        input_ids_flat = torch.cat(input_ids_list, dim=0).to(self.device)
        position_ids_flat = torch.cat(
            [torch.arange(seq_len, dtype=torch.long) for seq_len in seq_lens],
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
    def prefill(self, prompts: list[str]) -> PrefillResult:
        batch_size = len(prompts)
        self.kv_cache = self._new_kv_cache(batch_size)

        input_ids_flat, position_ids_flat, seq_lens, last_token_indices = self._tokenize_prompts(prompts)
        seq_slots = torch.arange(batch_size, dtype=torch.long, device=self.device)
        attn_metadata = self.kv_cache.prepare_prefill_metadata(
            seq_slots=seq_slots,
            seq_lens=seq_lens,
        )

        outputs = self.model.forward_flat(
            input_ids_flat=input_ids_flat,
            position_ids_flat=position_ids_flat,
            kv_cache=self.kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=True,
        )
        last_logits = outputs.logits.index_select(0, last_token_indices)
        next_tokens_tensor = torch.argmax(last_logits, dim=-1)
        next_tokens = [int(x) for x in next_tokens_tensor.detach().cpu().tolist()]

        return PrefillResult(
            next_tokens=next_tokens,
            seq_lens=seq_lens,
            eos_token_id=int(self.tokenizer.eos_token_id),
        )

    @torch.inference_mode()
    def decode(
        self,
        active_slots: list[int],
        token_ids: list[int],
        cache_positions: list[int],
    ) -> list[int]:
        if self.kv_cache is None:
            raise RuntimeError("prefill must run before decode")

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        seq_slots = torch.tensor(active_slots, dtype=torch.long, device=self.device)
        cache_position = torch.tensor(cache_positions, dtype=torch.long, device=self.device)

        attn_metadata = self.kv_cache.prepare_metadata(
            seq_slots=seq_slots,
            start_pos=cache_position,
            q_len=1,
        )

        outputs = self.model.forward_flat(
            input_ids_flat=input_ids,
            position_ids_flat=cache_position,
            kv_cache=self.kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=False,
        )
        next_tokens = torch.argmax(outputs.logits, dim=-1)
        return [int(x) for x in next_tokens.detach().cpu().tolist()]

    def decode_token_ids(self, generated_token_ids: list[list[int]]) -> list[str]:
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_token_ids
        ]
