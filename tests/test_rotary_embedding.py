import torch

from nanovllm_ascend.layers.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb_tnd


def test_rotary_tnd_shapes():
    rotary = RotaryEmbedding(head_dim=8, max_position_embeddings=16, rope_theta=10000.0)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(position_ids)

    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    q_out, k_out = apply_rotary_pos_emb_tnd(q, k, cos.squeeze(0), sin.squeeze(0))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype

