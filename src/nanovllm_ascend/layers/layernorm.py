import torch
import torch_npu
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(
            hidden_states,
            self.weight,
            epsilon=self.variance_epsilon,
        )[0]