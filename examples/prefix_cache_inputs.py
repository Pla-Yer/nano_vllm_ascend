PREFIX_HEADER = (
    "You are a technical assistant specializing in large language model inference systems, "
    "KV cache management, paged attention, prefix cache, continuous batching, decode scheduling, "
    "block tables, and NPU acceleration.\n\n"
    "The following context is shared by multiple requests. It is intentionally long so that "
    "prefix cache can reuse several complete cache blocks. The content below must remain exactly "
    "the same for cache-hit prompts.\n\n"
)

PREFIX_PARAGRAPH = (
    "Prefix cache stores the key and value tensors of a previously processed prompt prefix. "
    "When another request starts with exactly the same token prefix, the inference engine can skip "
    "recomputing those prefix tokens during prefill. Instead, it reuses the cached KV blocks and only "
    "computes the remaining suffix tokens. Correct prefix cache implementation requires exact token "
    "matching, block-aligned cache reuse, valid block tables, correct context lengths, correct position ids, "
    "and a causal attention mask that works when query length is smaller than key-value length. "
    "In paged attention, cached prefix blocks and newly allocated suffix blocks are connected through "
    "the block table, so the attention backend can see the complete logical sequence. During decode, "
    "newly generated tokens must be appended to the correct physical block with the correct offset. "
    "Prefix cache is most useful when the shared prefix is long and the generated continuation is short, "
    "because the optimization mainly reduces prefill cost rather than decode cost.\n\n"
)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_shared_prefix(
    tokenizer,
    min_tokens: int,
    case_id: str | None = None,
) -> str:
    prefix = PREFIX_HEADER
    if case_id is not None:
        prefix = f"CASE_ID: {case_id}\n\n" + prefix

    while count_tokens(tokenizer, prefix) < min_tokens:
        prefix += PREFIX_PARAGRAPH

    return prefix
