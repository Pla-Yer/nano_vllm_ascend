from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nanovllm_ascend.layers import (
    LMHead,
    Linear,
    NpuBatchPrefillAttention,
    NpuPagedAttention,
    RMSNorm,
    RotaryEmbedding,
    SiLUActivation,
    VocabParallelEmbedding,
    apply_rotary_pos_emb_tnd,
)


@dataclass
class CausalLMOutput:
    logits: torch.Tensor


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        block_size = getattr(config, "nanovllm_block_size", 128)
        max_model_len = getattr(config, "nanovllm_max_model_len", 2048)
        self.prefill_attn = NpuBatchPrefillAttention(
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            block_size=block_size,
            max_mask_len=max_model_len,
        )
        self.paged_attn = NpuPagedAttention(
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache,
        attn_metadata,
        is_prefill: bool,
    ) -> torch.Tensor:
        total_tokens, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(total_tokens, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_tokens, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_tokens, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb_tnd(
            query_states,
            key_states,
            *position_embeddings,
        )

        if is_prefill:
            kv_cache.write_prefill(
                layer_idx=self.layer_idx,
                key_states=key_states,
                value_states=value_states,
                slot_mapping=attn_metadata.slot_mapping,
            )
            attn_output = self.prefill_attn(
                q=query_states,
                k=key_states,
                v=value_states,
                attn_metadata=attn_metadata,
            )
        else:
            kv_cache.write_decode(
                layer_idx=self.layer_idx,
                key_states=key_states,
                value_states=value_states,
                slot_mapping=attn_metadata.slot_mapping,
            )
            key_cache_layer, value_cache_layer = kv_cache.get_physical_cache(layer_idx=self.layer_idx)
            attn_output = self.paged_attn(
                q=query_states,
                key_cache=key_cache_layer,
                value_cache=value_cache_layer,
                block_tables=attn_metadata.block_tables,
                context_lens=attn_metadata.context_lens,
            )

        return self.o_proj(attn_output.reshape(total_tokens, self.num_heads * self.head_dim))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache,
        attn_metadata,
        is_prefill: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

    def forward_flat(
        self,
        input_ids_flat: torch.Tensor,
        position_ids_flat: torch.Tensor,
        kv_cache,
        attn_metadata,
        is_prefill: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids_flat)
        cos, sin = self.rotary_emb(position_ids_flat.unsqueeze(0))
        position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                is_prefill=is_prefill,
            )
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward_flat(
        self,
        input_ids_flat: torch.Tensor,
        position_ids_flat: torch.Tensor,
        kv_cache,
        attn_metadata,
        is_prefill: bool,
    ) -> CausalLMOutput:
        hidden_states = self.model.forward_flat(
            input_ids_flat=input_ids_flat,
            position_ids_flat=position_ids_flat,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
        )
        return CausalLMOutput(logits=self.lm_head(hidden_states))
