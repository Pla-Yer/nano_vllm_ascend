from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--shared-prefix-min-tokens", type=int, default=466)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def cleanup():
    gc.collect()
    if hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_long_shared_prefix(tokenizer, min_tokens: int) -> str:
    header = (
        "You are a technical assistant specializing in large language model inference systems, "
        "KV cache management, paged attention, prefix cache, continuous batching, decode scheduling, "
        "block tables, and NPU acceleration.\n\n"
        "The following context is shared by multiple requests. It is intentionally long so that "
        "prefix cache can reuse several complete cache blocks. The content below must remain exactly "
        "the same for all cache-hit prompts.\n\n"
    )

    paragraph = (
        "Prefix cache stores the key and value tensors of a previously processed prompt prefix. "
        "When another request starts with exactly the same token prefix, the inference engine can skip "
        "recomputing those prefix tokens during prefill. Instead, it reuses the cached KV blocks and only "
        "computes the remaining suffix tokens. Correct prefix cache implementation requires exact token "
        "matching, block-aligned cache reuse, valid block tables, correct context lengths, correct position ids, "
        "and a causal attention mask that works when query length is smaller than key-value length. "
        "In paged attention, cached prefix blocks and newly allocated suffix blocks are connected through "
        "the block table, so the attention backend can see the complete logical sequence. During decode, "
        "newly generated tokens must be appended to the correct physical block with the correct offset.\n\n"
    )

    prefix = header
    while count_tokens(tokenizer, prefix) < min_tokens:
        prefix += paragraph

    return prefix


def build_prompts(tokenizer, min_prefix_tokens: int):
    shared_prefix = build_long_shared_prefix(tokenizer, min_prefix_tokens)

    warm_prompt = (
        shared_prefix
        + "\n\nQuestion: Briefly explain what prefix cache stores. Answer in one short paragraph."
    )

    hit_prompt = (
        shared_prefix
        + "\n\nQuestion: Explain how cached prefix blocks affect the prefill path. "
          "Focus on Q length, KV length, block table usage, context length, and decode continuation."
    )

    miss_prompt = (
        "This is a completely different document about rotary embeddings, position ids, tensor layouts, "
        "and sequence batching. It deliberately starts with different tokens, so it should not match the "
        "previously warmed prefix cache.\n\n"
        "Question: Explain rotary position embedding in one short paragraph."
    )

    return shared_prefix, warm_prompt, hit_prompt, miss_prompt


def run_batch_no_cache(args, prompts, sampling_params):
    print("\n===== no_cache_batch =====")

    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        device_id=args.device_id,
    )

    t0 = time.time()
    outputs = llm.generate(
        prompts,
        max_new_tokens=args.max_new_tokens,
        sampling_params=sampling_params,
    )
    dt = time.time() - t0

    total = sum(len(out["token_ids"]) for out in outputs)
    print(f"time={dt:.2f}s generated={total} throughput={total / dt:.2f} tok/s")

    for i, out in enumerate(outputs):
        print(f"\n----- no_cache output {i} -----")
        print(f"generated_tokens={len(out['token_ids'])}")
        print(f"token_ids={out['token_ids']}")
        print(out["texts"])

    del llm
    cleanup()
    return outputs


def run_batch_cache_hit(args, warm_prompt, prompts, sampling_params):
    print("\n===== cache_hit_batch_after_warmup =====")

    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        device_id=args.device_id,
    )

    print("\nWarming prefix cache...")
    t0 = time.time()
    warm_outputs = llm.generate(
        [warm_prompt],
        max_new_tokens=8,
        sampling_params=sampling_params,
    )
    warm_dt = time.time() - t0

    print(f"warm_time={warm_dt:.2f}s")
    print(f"warm_generated_tokens={len(warm_outputs[0]['token_ids'])}")
    print(f"warm_text={warm_outputs[0]['texts']}")

    print("\nRunning batch after warm-up...")
    t1 = time.time()
    outputs = llm.generate(
        prompts,
        max_new_tokens=args.max_new_tokens,
        sampling_params=sampling_params,
    )
    dt = time.time() - t1

    total = sum(len(out["token_ids"]) for out in outputs)
    print(f"time={dt:.2f}s generated={total} throughput={total / dt:.2f} tok/s")

    for i, out in enumerate(outputs):
        print(f"\n----- cache_hit output {i} -----")
        print(f"generated_tokens={len(out['token_ids'])}")
        print(f"token_ids={out['token_ids']}")
        print(out["texts"])

    del llm
    cleanup()
    return outputs


def compare_one(name: str, ids_a: list[int], ids_b: list[int]) -> bool:
    n = min(len(ids_a), len(ids_b))

    for i in range(n):
        if ids_a[i] != ids_b[i]:
            print(f"\n{name}: FAIL")
            print(f"first_diff_index={i}")
            print(f"no_cache[{i}]={ids_a[i]}")
            print(f"cache_hit[{i}]={ids_b[i]}")

            left = max(0, i - 8)
            right = min(n, i + 8)

            print("no_cache window:")
            print(ids_a[left:right])

            print("cache_hit window:")
            print(ids_b[left:right])
            return False

    if len(ids_a) != len(ids_b):
        print(f"\n{name}: FAIL length mismatch")
        print(f"no_cache length={len(ids_a)}")
        print(f"cache_hit length={len(ids_b)}")
        return False

    print(f"\n{name}: PASS")
    return True


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    shared_prefix, warm_prompt, hit_prompt, miss_prompt = build_prompts(
        tokenizer=tokenizer,
        min_prefix_tokens=args.shared_prefix_min_tokens,
    )

    shared_prefix_tokens = count_tokens(tokenizer, shared_prefix)
    warm_tokens = count_tokens(tokenizer, warm_prompt)
    hit_tokens = count_tokens(tokenizer, hit_prompt)
    miss_tokens = count_tokens(tokenizer, miss_prompt)

    expected_cached_len = (shared_prefix_tokens // args.block_size) * args.block_size
    expected_cached_blocks = expected_cached_len // args.block_size

    print("Prefix cache batch equivalence test")
    print(f"block_size={args.block_size}")
    print(f"shared_prefix_tokens={shared_prefix_tokens}")
    print(f"expected_cached_len_floor={expected_cached_len}")
    print(f"expected_cached_blocks_floor={expected_cached_blocks}")
    print(f"warm_prompt_tokens={warm_tokens}")
    print(f"hit_prompt_tokens={hit_tokens}")
    print(f"miss_prompt_tokens={miss_tokens}")
    print(f"hit_prompt_tokens % block_size = {hit_tokens % args.block_size}")
    print(f"tokens_until_next_block = {(args.block_size - hit_tokens % args.block_size) % args.block_size}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    batch_prompts = [hit_prompt, miss_prompt]

    no_cache_outputs = run_batch_no_cache(
        args=args,
        prompts=batch_prompts,
        sampling_params=sampling_params,
    )

    cache_outputs = run_batch_cache_hit(
        args=args,
        warm_prompt=warm_prompt,
        prompts=batch_prompts,
        sampling_params=sampling_params,
    )

    print("\n===== Compare batch outputs =====")

    ok0 = compare_one(
        name="seq0 hit_prompt",
        ids_a=no_cache_outputs[0]["token_ids"],
        ids_b=cache_outputs[0]["token_ids"],
    )

    ok1 = compare_one(
        name="seq1 miss_prompt",
        ids_a=no_cache_outputs[1]["token_ids"],
        ids_b=cache_outputs[1]["token_ids"],
    )

    print("\n===== Final Result =====")
    if ok0 and ok1:
        print("PASS: batch cache-hit path is token-equivalent to batch no-cache path.")
    else:
        print("FAIL: batch cache-hit path is not token-equivalent to batch no-cache path.")
        print("Focus on batch metadata: actual_seq_lengths_q, actual_seq_lengths_kv, context_lens, block_tables, seq slots, and logits indexing.")


if __name__ == "__main__":
    main()