## 端到端性能一般，暂未使用
from __future__ import annotations

import torch
import torch_npu
from torch import nn

from nanovllm_ascend.layers.linear import Linear


class Qwen3FusedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)

        hidden_states = torch_npu.npu_swiglu(
            gate_up.contiguous(),
            dim=-1,
        )

        return self.down_proj(hidden_states)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        gate_key = prefix + "gate_proj.weight"
        up_key = prefix + "up_proj.weight"
        gate_up_key = prefix + "gate_up_proj.weight"

        if gate_up_key not in state_dict and gate_key in state_dict and up_key in state_dict:
            state_dict[gate_up_key] = torch.cat(
                [state_dict[gate_key], state_dict[up_key]],
                dim=0,
            )

        state_dict.pop(gate_key, None)
        state_dict.pop(up_key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )