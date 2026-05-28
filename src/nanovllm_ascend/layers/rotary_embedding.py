import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, :].float()
        freqs = torch.matmul(inv_freq, position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos_q = cos.to(device=q.device, dtype=q_dtype).unsqueeze(1)
    sin_q = sin.to(device=q.device, dtype=q_dtype).unsqueeze(1)
    cos_k = cos.to(device=k.device, dtype=k_dtype).unsqueeze(1)
    sin_k = sin.to(device=k.device, dtype=k_dtype).unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)
