__all__ = [
    "SiLUActivation",
    "LMHead",
    "VocabParallelEmbedding",
    "RMSNorm",
    "Linear",
    "NpuPagedAttention",
    "NpuBatchPrefillAttention",
    "RotaryEmbedding",
    "apply_rotary_pos_emb_tnd",
    "Sampler",
    "sample_tokens",
]


def __getattr__(name: str):
    if name == "SiLUActivation":
        from .activation import SiLUActivation

        return SiLUActivation
    if name in {"LMHead", "VocabParallelEmbedding"}:
        from .embed_head import LMHead, VocabParallelEmbedding

        return {"LMHead": LMHead, "VocabParallelEmbedding": VocabParallelEmbedding}[name]
    if name == "RMSNorm":
        from .layernorm import RMSNorm

        return RMSNorm
    if name == "Linear":
        from .linear import Linear

        return Linear
    if name == "NpuPagedAttention":
        from .npu_paged_attention import NpuPagedAttention

        return NpuPagedAttention
    if name == "NpuBatchPrefillAttention":
        from .npu_prefill_attention import NpuBatchPrefillAttention

        return NpuBatchPrefillAttention
    if name in {"RotaryEmbedding", "apply_rotary_pos_emb_tnd"}:
        from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb_tnd

        return {
            "RotaryEmbedding": RotaryEmbedding,
            "apply_rotary_pos_emb_tnd": apply_rotary_pos_emb_tnd,
        }[name]
    if name in {"Sampler", "sample_tokens"}:
        from .sampler import Sampler, sample_tokens

        return {
            "Sampler": Sampler,
            "sample_tokens": sample_tokens,
        }[name]
    raise AttributeError(name)
