from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import LLM, SamplingParams
from nanovllm_ascend.sequence import Sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--device-id", type=int, default=0)

    parser.add_argument("--shared-prefix-min-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Measure tokenization, prepare_sequences, and runner.prefill separately.",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)

    return parser.parse_args()


def resolved_max_num_seqs(args) -> int:
    return args.max_num_seqs or args.batch_size


def make_llm(args, *, enable_prefix_cache: bool) -> LLM:
    return LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=resolved_max_num_seqs(args),
        device_id=args.device_id,
        enable_prefix_cache=enable_prefix_cache,
    )


def sync_npu():
    if hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def cleanup():
    gc.collect()
    if hasattr(torch, "npu"):
        try:
            torch.npu.empty_cache()
        except Exception:
            pass


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def stat(xs: list[float]) -> dict[str, float]:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def percentile(p: float) -> float:
        if n == 1:
            return xs_sorted[0]
        idx = int(round((n - 1) * p))
        return xs_sorted[idx]

    return {
        "avg": statistics.mean(xs_sorted),
        "min": min(xs_sorted),
        "max": max(xs_sorted),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
    }


def build_shared_prefix(tokenizer, min_tokens: int, case_id: str) -> str:
    header = (
        f"CASE_ID: {case_id}\n\n"
        "You are a technical assistant specializing in large language model inference systems, "
        "KV cache management, paged attention, prefix cache, continuous batching, decode scheduling, "
        "block tables, and NPU acceleration.\n\n"
        "The following context is shared by multiple requests. It is intentionally long so that "
        "prefix cache can reuse several complete cache blocks. The content below must remain exactly "
        "the same for the warm request and the cache-hit request.\n\n"
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
        "newly generated tokens must be appended to the correct physical block with the correct offset. "
        "Prefix cache is most useful when the shared prefix is long and the generated continuation is short, "
        "because the optimization mainly reduces prefill cost rather than decode cost.\n\n"
    )

    prefix = header
    while count_tokens(tokenizer, prefix) < min_tokens:
        prefix += paragraph

    return prefix


def build_batch_cases(tokenizer, min_prefix_tokens: int, iter_id: int, batch_size: int):
    warm_prompts = []
    hit_prompts = []

    for seq_id in range(batch_size):
        case_id = f"iter_{iter_id}_seq_{seq_id}"
        shared_prefix = build_shared_prefix(
            tokenizer=tokenizer,
            min_tokens=min_prefix_tokens,
            case_id=case_id,
        )

        warm_prompts.append(
            shared_prefix
            + "\n\nQuestion: Briefly explain what prefix cache stores. Answer in one short paragraph."
        )
        hit_prompts.append(
            shared_prefix
            + "\n\nQuestion: Explain how cached prefix blocks affect the prefill path. "
            "Focus on Q length, KV length, block table usage, and context length."
        )

    return warm_prompts, hit_prompts


def timed_generate(llm, prompts, max_new_tokens, sampling_params):
    sync_npu()
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params,
    )
    sync_npu()

    dt = time.perf_counter() - t0
    total_gen_tokens = sum(len(out["token_ids"]) for out in outputs)
    return dt, total_gen_tokens, outputs


def count_runtime_prompt_tokens(llm, prompts: list[str]) -> int:
    token_ids = llm.runner.tokenize_prompts(prompts)
    return sum(int(ids.numel()) for ids in token_ids)


def inspect_prefix_cache(llm, prompts: list[str], max_new_tokens: int):
    token_ids = llm.runner.tokenize_prompts(prompts)
    seqs = [
        Sequence(
            seq_id=i,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=ids,
            sampling_params=SamplingParams(),
        )
        for i, (prompt, ids) in enumerate(zip(prompts, token_ids))
    ]
    llm.runner.prepare_sequences(seqs)
    return {
        "prompt_tokens": sum(seq.estimated_prompt_len for seq in seqs),
        "cached_tokens": sum(seq.cached_prefix_len for seq in seqs),
        "runtime_prompt_tokens": sum(seq.runtime_prompt_len for seq in seqs),
        "cached_blocks": sum(len(seq.cached_block_ids) for seq in seqs),
    }


def timed_prefill_only(llm, prompts, max_new_tokens, sampling_params):
    t0 = time.perf_counter()
    token_ids = llm.runner.tokenize_prompts(prompts)
    tokenize_dt = time.perf_counter() - t0

    seqs = [
        Sequence(
            seq_id=i,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_token_ids=ids,
            sampling_params=sampling_params,
        )
        for i, (prompt, ids) in enumerate(zip(prompts, token_ids))
    ]

    t1 = time.perf_counter()
    llm.runner.prepare_sequences(seqs)
    prepare_dt = time.perf_counter() - t1

    cache_status = {
        "prompt_tokens": sum(seq.estimated_prompt_len for seq in seqs),
        "cached_tokens": sum(seq.cached_prefix_len for seq in seqs),
        "runtime_prompt_tokens": sum(seq.runtime_prompt_len for seq in seqs),
        "cached_blocks": sum(len(seq.cached_block_ids) for seq in seqs),
    }

    _, _, seq_lens, _ = llm.runner._prepare_prefill_inputs(seqs)
    attn_metadata = llm.runner.block_manager.prepare_prefill_metadata(
        slots=[seq.seq_id for seq in seqs],
        seq_lens=seq_lens,
        start_positions=[seq.cached_prefix_len for seq in seqs],
        prefix_block_ids=[seq.cached_block_ids for seq in seqs],
    )
    cache_status["prefill_route"] = (
        "paged_prefill" if attn_metadata.use_paged_prefill else "dense_prefill"
    )

    sync_npu()
    t2 = time.perf_counter()
    llm.runner.prefill(seqs)
    sync_npu()
    prefill_dt = time.perf_counter() - t2

    for seq in seqs:
        llm.runner.free_seq(seq)

    return {
        "tokenize_time": tokenize_dt,
        "prepare_time": prepare_dt,
        "prefill_time": prefill_dt,
        **cache_status,
    }


def engine_warmup(llm, sampling_params):
    timed_generate(
        llm=llm,
        prompts=["Hello, briefly say hello."],
        max_new_tokens=1,
        sampling_params=sampling_params,
    )


def run_no_cache_bench(args, tokenizer, sampling_params):
    print("\n===== no-cache bench =====")
    llm = make_llm(args, enable_prefix_cache=False)
    engine_warmup(llm, sampling_params)

    times = []
    gen_tokens = []
    prompt_tokens_list = []

    for i in range(args.iters):
        _, hit_prompts = build_batch_cases(
            tokenizer=tokenizer,
            min_prefix_tokens=args.shared_prefix_min_tokens,
            iter_id=i,
            batch_size=args.batch_size,
        )

        prompt_tokens = count_runtime_prompt_tokens(llm, hit_prompts)
        dt, total_gen_tokens, _ = timed_generate(
            llm=llm,
            prompts=hit_prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )

        times.append(dt)
        gen_tokens.append(total_gen_tokens)
        prompt_tokens_list.append(prompt_tokens)

        print(
            f"iter={i} "
            f"time={dt:.4f}s "
            f"prompt_tokens={prompt_tokens} "
            f"gen_tokens={total_gen_tokens} "
            f"total_tok/s={(prompt_tokens + total_gen_tokens) / dt:.2f} "
            f"gen_tok/s={total_gen_tokens / dt:.2f}"
        )

    del llm
    cleanup()
    return {
        "times": times,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens_list,
    }


def run_cache_hit_bench(args, tokenizer, sampling_params):
    print("\n===== cache-hit bench =====")
    llm = make_llm(args, enable_prefix_cache=True)
    engine_warmup(llm, sampling_params)

    warm_times = []
    hit_times = []
    hit_gen_tokens = []
    hit_prompt_tokens_list = []
    cached_tokens_list = []
    runtime_prompt_tokens_list = []

    for i in range(args.iters):
        warm_prompts, hit_prompts = build_batch_cases(
            tokenizer=tokenizer,
            min_prefix_tokens=args.shared_prefix_min_tokens,
            iter_id=i,
            batch_size=args.batch_size,
        )

        warm_dt, warm_gen, _ = timed_generate(
            llm=llm,
            prompts=warm_prompts,
            max_new_tokens=1,
            sampling_params=sampling_params,
        )

        cache_status = inspect_prefix_cache(
            llm=llm,
            prompts=hit_prompts,
            max_new_tokens=args.max_new_tokens,
        )

        hit_dt, hit_gen, _ = timed_generate(
            llm=llm,
            prompts=hit_prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )

        warm_times.append(warm_dt)
        hit_times.append(hit_dt)
        hit_gen_tokens.append(hit_gen)
        hit_prompt_tokens_list.append(cache_status["prompt_tokens"])
        cached_tokens_list.append(cache_status["cached_tokens"])
        runtime_prompt_tokens_list.append(cache_status["runtime_prompt_tokens"])

        if cache_status["cached_tokens"] == 0:
            print("WARNING: cache-hit request did not hit any cached prefix blocks.")

        print(
            f"iter={i} "
            f"warm_time={warm_dt:.4f}s "
            f"hit_time={hit_dt:.4f}s "
            f"hit_prompt_tokens={cache_status['prompt_tokens']} "
            f"actual_cached_tokens={cache_status['cached_tokens']} "
            f"actual_runtime_prompt_tokens={cache_status['runtime_prompt_tokens']} "
            f"actual_cached_blocks={cache_status['cached_blocks']} "
            f"gen_tokens={hit_gen} "
            f"online_total_tok/s={(cache_status['runtime_prompt_tokens'] + hit_gen) / hit_dt:.2f} "
            f"online_gen_tok/s={hit_gen / hit_dt:.2f}"
        )

    del llm
    cleanup()
    return {
        "warm_times": warm_times,
        "hit_times": hit_times,
        "hit_gen_tokens": hit_gen_tokens,
        "hit_prompt_tokens": hit_prompt_tokens_list,
        "cached_tokens": cached_tokens_list,
        "runtime_prompt_tokens": runtime_prompt_tokens_list,
    }


def run_no_cache_prefill_bench(args, tokenizer, sampling_params):
    print("\n===== no-cache prefill-only bench =====")
    llm = make_llm(args, enable_prefix_cache=False)
    engine_warmup(llm, sampling_params)

    rows = []
    for i in range(args.iters):
        _, hit_prompts = build_batch_cases(
            tokenizer=tokenizer,
            min_prefix_tokens=args.shared_prefix_min_tokens,
            iter_id=i,
            batch_size=args.batch_size,
        )
        row = timed_prefill_only(
            llm=llm,
            prompts=hit_prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )
        rows.append(row)
        print(
            f"iter={i} "
            f"route={row['prefill_route']} "
            f"tokenize={row['tokenize_time']:.4f}s "
            f"prepare={row['prepare_time']:.4f}s "
            f"prefill={row['prefill_time']:.4f}s "
            f"prompt_tokens={row['prompt_tokens']} "
            f"runtime_prompt_tokens={row['runtime_prompt_tokens']}"
        )

    del llm
    cleanup()
    return rows


def run_cache_hit_prefill_bench(args, tokenizer, sampling_params):
    print("\n===== cache-hit prefill-only bench =====")
    llm = make_llm(args, enable_prefix_cache=True)
    engine_warmup(llm, sampling_params)

    rows = []
    for i in range(args.iters):
        warm_prompts, hit_prompts = build_batch_cases(
            tokenizer=tokenizer,
            min_prefix_tokens=args.shared_prefix_min_tokens,
            iter_id=i,
            batch_size=args.batch_size,
        )
        warm_row = timed_prefill_only(
            llm=llm,
            prompts=warm_prompts,
            max_new_tokens=1,
            sampling_params=sampling_params,
        )
        hit_row = timed_prefill_only(
            llm=llm,
            prompts=hit_prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )
        hit_row["warm_prefill_time"] = warm_row["prefill_time"]
        rows.append(hit_row)
        print(
            f"iter={i} "
            f"route={hit_row['prefill_route']} "
            f"warm_prefill={warm_row['prefill_time']:.4f}s "
            f"tokenize={hit_row['tokenize_time']:.4f}s "
            f"prepare={hit_row['prepare_time']:.4f}s "
            f"prefill={hit_row['prefill_time']:.4f}s "
            f"prompt_tokens={hit_row['prompt_tokens']} "
            f"cached_tokens={hit_row['cached_tokens']} "
            f"runtime_prompt_tokens={hit_row['runtime_prompt_tokens']} "
            f"cached_blocks={hit_row['cached_blocks']}"
        )

    del llm
    cleanup()
    return rows


def print_prefill_only_summary(no_cache_rows, cache_hit_rows):
    no_cache_prefill = [row["prefill_time"] for row in no_cache_rows]
    cache_hit_prefill = [row["prefill_time"] for row in cache_hit_rows]
    no_cache_online = [
        row["tokenize_time"] + row["prepare_time"] + row["prefill_time"]
        for row in no_cache_rows
    ]
    cache_hit_online = [
        row["tokenize_time"] + row["prepare_time"] + row["prefill_time"]
        for row in cache_hit_rows
    ]

    avg_no_cache_prefill = statistics.mean(no_cache_prefill)
    avg_cache_hit_prefill = statistics.mean(cache_hit_prefill)
    avg_no_cache_online = statistics.mean(no_cache_online)
    avg_cache_hit_online = statistics.mean(cache_hit_online)

    print("\n" + "=" * 80)
    print("Prefill-Only Summary")
    print("=" * 80)
    print_stats("no_cache_prefill_time", no_cache_prefill)
    print_stats("cache_hit_prefill_time", cache_hit_prefill)
    print_stats("no_cache_tokenize_prepare_prefill_time", no_cache_online)
    print_stats("cache_hit_tokenize_prepare_prefill_time", cache_hit_online)
    print()
    print(f"prefill_compute_speedup={avg_no_cache_prefill / avg_cache_hit_prefill:.3f}x")
    print(f"tokenize_prepare_prefill_speedup={avg_no_cache_online / avg_cache_hit_online:.3f}x")


def print_stats(name: str, times: list[float]):
    s = stat(times)
    print(
        f"{name}: "
        f"avg={s['avg']:.4f}s "
        f"p50={s['p50']:.4f}s "
        f"p90={s['p90']:.4f}s "
        f"min={s['min']:.4f}s "
        f"max={s['max']:.4f}s"
    )


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print("Prefix cache speedup benchmark")
    print(f"model_path={args.model_path}")
    print(f"block_size={args.block_size}")
    print(f"num_blocks={args.num_blocks}")
    print(f"max_num_seqs={resolved_max_num_seqs(args)}")
    print(f"batch_size={args.batch_size}")
    print(f"iters={args.iters}")
    print(f"shared_prefix_min_tokens={args.shared_prefix_min_tokens}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"temperature={args.temperature}")
    print(f"prefill_only={args.prefill_only}")
    print()
    print("Notes:")
    print("- no-cache uses an LLM with prefix cache disabled.")
    print("- cache-hit uses an LLM with prefix cache enabled, warms the same prefixes, then times only the hit request.")
    print("- token accounting uses the runner chat-template tokenization, not raw tokenizer.encode.")
    print("- num_blocks must be large enough to keep all warmed prefix blocks resident.")
    print("- max_new_tokens=1 is now a prefill-dominant path because the engine skips the unused final decode forward.")

    if args.prefill_only:
        no_cache_prefill = run_no_cache_prefill_bench(args, tokenizer, sampling_params)
        cache_hit_prefill = run_cache_hit_prefill_bench(args, tokenizer, sampling_params)
        print_prefill_only_summary(no_cache_prefill, cache_hit_prefill)
        return

    no_cache = run_no_cache_bench(args, tokenizer, sampling_params)
    cache_hit = run_cache_hit_bench(args, tokenizer, sampling_params)

    no_cache_times = no_cache["times"]
    cache_hit_times = cache_hit["hit_times"]
    warm_times = cache_hit["warm_times"]

    avg_no_cache = statistics.mean(no_cache_times)
    avg_cache_hit = statistics.mean(cache_hit_times)
    avg_warm = statistics.mean(warm_times)

    online_speedup = avg_no_cache / avg_cache_hit
    online_latency_reduction = (avg_no_cache - avg_cache_hit) / avg_no_cache * 100.0

    no_cache_total = sum(no_cache_times)
    cache_hit_total = sum(cache_hit_times)
    warm_total = sum(warm_times)
    amortized_total = warm_total + cache_hit_total

    amortized_speedup = no_cache_total / amortized_total
    amortized_latency_change = (no_cache_total - amortized_total) / no_cache_total * 100.0

    avg_prompt_tokens = statistics.mean(no_cache["prompt_tokens"])
    avg_hit_prompt_tokens = statistics.mean(cache_hit["hit_prompt_tokens"])
    avg_cached_tokens = statistics.mean(cache_hit["cached_tokens"])
    avg_runtime_prompt_tokens = statistics.mean(cache_hit["runtime_prompt_tokens"])
    avg_saved_prompt_ratio = (
        avg_cached_tokens / avg_hit_prompt_tokens * 100.0
        if avg_hit_prompt_tokens
        else 0.0
    )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print_stats("no_cache_time", no_cache_times)
    print_stats("cache_hit_online_time", cache_hit_times)
    print_stats("warm_time", warm_times)

    print()
    print(f"avg_no_cache_time={avg_no_cache:.4f}s")
    print(f"avg_cache_hit_online_time={avg_cache_hit:.4f}s")
    print(f"avg_warm_time={avg_warm:.4f}s")

    print()
    print(f"online_speedup={online_speedup:.3f}x")
    print(f"online_latency_reduction={online_latency_reduction:.2f}%")

    print()
    print(f"no_cache_total_time={no_cache_total:.4f}s")
    print(f"cache_hit_online_total_time={cache_hit_total:.4f}s")
    print(f"warm_total_time={warm_total:.4f}s")
    print(f"amortized_total_time=warm_total + cache_hit_online_total = {amortized_total:.4f}s")
    print(f"amortized_speedup={amortized_speedup:.3f}x")
    print(f"amortized_latency_change={amortized_latency_change:.2f}%")

    print()
    print("Token accounting per batch, actual runtime tokenization:")
    print(f"avg_no_cache_prompt_tokens={avg_prompt_tokens:.1f}")
    print(f"avg_hit_prompt_tokens={avg_hit_prompt_tokens:.1f}")
    print(f"avg_actual_cached_tokens={avg_cached_tokens:.1f}")
    print(f"avg_actual_runtime_prompt_tokens={avg_runtime_prompt_tokens:.1f}")
    print(f"avg_saved_prompt_ratio={avg_saved_prompt_ratio:.2f}%")

    print()
    print("Interpretation:")
    print("- online_speedup is the latency gain after the prefix cache is already warmed.")
    print("- amortized_speedup includes warm cost; it improves as the same prefix is reused more times.")
    print("- if actual_cached_tokens is 0 or much lower than expected, inspect block size, token matching, and num_blocks.")


if __name__ == "__main__":
    main()
