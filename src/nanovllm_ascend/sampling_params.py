from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")

    def is_greedy(self) -> bool:
        return self.temperature <= 0

    def with_overrides(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> "SamplingParams":
        return replace(
            self,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            top_p=self.top_p if top_p is None else top_p,
        )
