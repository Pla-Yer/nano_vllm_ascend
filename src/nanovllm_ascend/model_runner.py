from __future__ import annotations

import torch
import torch_npu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .layers import Sampler
from .models.qwen3 import Qwen3ForCausalLM
from .npu.paged_kv_cache import PagedKVCache
from .npu.block_manager import BlockManager
from .sampling_params import SamplingParams
from .sequence import Sequence

class ModelRunner:
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        num_blocks: int,
        block_size: int,
        device_id: int,
    ):
        torch.npu.set_device(device_id)

        self.model_path = model_path
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.device = "npu"
        self.dtype = torch.bfloat16
        self.sampler = Sampler()

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
        self.num_blocks =  num_blocks #  should be tuned based on NPU memory and model size

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
    def prefill(self, seqs: list[Sequence], sampling_params: SamplingParams) -> None:
        if not seqs:
            return

        prompts = [seq.prompt for seq in seqs]
        seq_ids = [seq.seq_id for seq in seqs]

        input_ids_flat, position_ids_flat, seq_lens, last_token_indices = (
            self._tokenize_prompts(prompts)
        )

        attn_metadata = self.block_manager.prepare_prefill_metadata(
            slots=seq_ids,
            seq_lens=seq_lens,
        )

        outputs = self.model(
            input_ids_flat=input_ids_flat,
            position_ids_flat=position_ids_flat,
            kv_cache=self.kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=True,
        )

        last_logits = outputs.logits.index_select(0, last_token_indices)
        next_tokens_tensor = self.sampler.sample(last_logits, sampling_params)
        next_tokens = [int(x) for x in next_tokens_tensor.detach().cpu().tolist()]

        for seq, next_token, prompt_len in zip(seqs, next_tokens, seq_lens):
            seq.set_prefill_result(
                next_token_id=next_token,
                prompt_len=prompt_len,
            )

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence], sampling_params: SamplingParams) -> None:
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

        outputs = self.model(
            input_ids_flat=input_ids,
            position_ids_flat=position_ids,
            kv_cache=self.kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=False,
        )

        next_tokens_tensor = self.sampler.sample(outputs.logits, sampling_params)
        next_tokens = [int(x) for x in next_tokens_tensor.detach().cpu().tolist()]

        for seq, next_token in zip(seqs, next_tokens):
            seq.set_decode_result(next_token)
    
    def free_seq(self, seq: Sequence) -> None:
      self.block_manager.free_slot(seq.seq_id)
