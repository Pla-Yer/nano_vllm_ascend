from __future__ import annotations

import torch

from nanovllm_ascend.sampling_params import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, sampling_params: SamplingParams) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")

        if sampling_params.is_greedy():
            return torch.argmax(logits, dim=-1)

        filtered_logits = logits / sampling_params.temperature
        filtered_logits = self._apply_top_k(filtered_logits, sampling_params.top_k)
        filtered_logits = self._apply_top_p(filtered_logits, sampling_params.top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits

        topk_values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = topk_values[..., -1:].expand_as(logits)
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        filtered_logits = logits.clone()
        filtered_logits.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits.masked_fill(sorted_mask, float("-inf")),
        )
        return filtered_logits
