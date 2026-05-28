def __getattr__(name: str):
    if name == "LLM":
        from .engine import LLM

        return LLM
    raise AttributeError(name)


__all__ = ["LLM"]

