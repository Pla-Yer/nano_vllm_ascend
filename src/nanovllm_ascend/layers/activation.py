import torch
import torch.nn.functional as F
from torch import nn


class SiLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

