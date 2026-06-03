def __getattr__(name: str):
    if name in {"LLM", "EngineCore"}:
        from .engine import EngineCore, LLM

        return {"LLM": LLM, "EngineCore": EngineCore}[name]
    if name == "SamplingParams":
        from .sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(name)


__all__ = ["LLM", "EngineCore", "SamplingParams"]
