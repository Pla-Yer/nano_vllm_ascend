import torch
from torch import nn

from .linear import Linear


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, dtype=dtype, device=device))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.weight, input_ids)


class LMHead(Linear):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=bias,
            dtype=dtype,
            device=device,
        )

