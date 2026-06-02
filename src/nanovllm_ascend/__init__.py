def __getattr__(name: str):
    if name == "LLM":
        from .engine import LLM

        return LLM
    if name == "SamplingParams":
        from .sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(name)


__all__ = ["LLM", "SamplingParams"]
