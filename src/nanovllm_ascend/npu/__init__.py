def __getattr__(name: str):
    if name in {"BlockManager", "PagedKVCacheMetadata"}:
        from .block_manager import BlockManager, PagedKVCacheMetadata

        return {
            "BlockManager": BlockManager,
            "PagedKVCacheMetadata": PagedKVCacheMetadata,
        }[name]
    if name == "PagedKVCache":
        from .paged_kv_cache import PagedKVCache

        return PagedKVCache
    raise AttributeError(name)


__all__ = ["PagedKVCache", "PagedKVCacheMetadata", "BlockManager"]
