import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

