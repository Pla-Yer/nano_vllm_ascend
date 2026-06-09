from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--npu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--enable-prefix-cache", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sync_npu() -> None:
    import torch

    torch.npu.synchronize()


def build_prompts(args) -> list[str]:
    base_prompts = args.prompt or [
        "Explain how paged KV cache improves batched LLM inference on Ascend NPU."
    ]
    prompts = [
        base_prompts[i % len(base_prompts)]
        for i in range(args.batch_size)
    ]
    return [prompt * args.prompt_repeat for prompt in prompts]


def stat(values: list[float]) -> dict[str, float]:
    xs = sorted(values)

    def percentile(p: float) -> float:
        idx = int(round((len(xs) - 1) * p))
        return xs[idx]

    return {
        "avg": statistics.mean(xs),
        "min": xs[0],
        "max": xs[-1],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
    }


def timed_run(
    llm: LLM,
    prompts: list[str],
    max_new_tokens: int,
    sampling_params: SamplingParams,
) -> dict[str, float | int]:
    llm.scheduler.reset()

    prepare_t0 = time.perf_counter()
    request_ids = [
        llm.submit(
            prompt,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )
        for prompt in prompts
    ]
    sync_npu()
    prepare_s = time.perf_counter() - prepare_t0

    prompt_tokens = sum(
        llm.scheduler.seqs[request_id].estimated_prompt_len
        for request_id in request_ids
    )

    t0 = time.perf_counter()
    llm.step()
    sync_npu()
    prefill_s = time.perf_counter() - t0

    decode_t0 = time.perf_counter()
    outputs_by_request_id = {}
    while llm.has_unfinished():
        for output in llm.step():
            outputs_by_request_id[output["request_id"]] = output
    sync_npu()
    decode_s = time.perf_counter() - decode_t0

    output_tokens = sum(
        len(outputs_by_request_id[request_id]["token_ids"])
        for request_id in request_ids
    )
    decode_tokens = max(output_tokens - len(prompts), 0)
    total_s = prefill_s + decode_s

    return {
        "prepare_s": prepare_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "decode_tokens": decode_tokens,
        "prefill_tok_s": prompt_tokens / prefill_s,
        "decode_tok_s": decode_tokens / decode_s if decode_tokens else 0.0,
        "output_tok_s": output_tokens / total_s,
        "total_tok_s": (prompt_tokens + output_tokens) / total_s,
    }


def make_summary(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    keys = [
        "prepare_s",
        "prefill_s",
        "decode_s",
        "total_s",
        "prefill_tok_s",
        "decode_tok_s",
        "output_tok_s",
        "total_tok_s",
    ]
    return {key: stat([float(row[key]) for row in rows]) for key in keys}


def print_row(iter_id: int, row: dict[str, float | int]) -> None:
    print(
        f"iter={iter_id} "
        f"prepare={row['prepare_s']:.4f}s "
        f"prefill={row['prefill_s']:.4f}s "
        f"decode={row['decode_s']:.4f}s "
        f"total={row['total_s']:.4f}s "
        f"prompt_tokens={row['prompt_tokens']} "
        f"output_tokens={row['output_tokens']} "
        f"prefill_tok/s={row['prefill_tok_s']:.2f} "
        f"decode_tok/s={row['decode_tok_s']:.2f} "
        f"output_tok/s={row['output_tok_s']:.2f} "
        f"total_tok/s={row['total_tok_s']:.2f}"
    )


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\n===== summary =====")
    for key, values in summary.items():
        print(
            f"{key}: "
            f"avg={values['avg']:.4f} "
            f"p50={values['p50']:.4f} "
            f"p90={values['p90']:.4f} "
            f"min={values['min']:.4f} "
            f"max={values['max']:.4f}"
        )


def main() -> None:
    args = parse_args()
    import torch

    prompts = build_prompts(args)
    max_num_seqs = args.max_num_seqs or args.batch_size

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=max_num_seqs,
        device_id=args.device_id,
        npu_memory_utilization=args.npu_memory_utilization,
        enable_prefix_cache=args.enable_prefix_cache,
    )

    torch.npu.reset_peak_memory_stats()
    print("nanovllm_ascend inference benchmark")
    print(f"model_path={args.model_path}")
    print(f"batch_size={args.batch_size}")
    print(f"max_num_seqs={max_num_seqs}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"max_model_len={args.max_model_len}")
    print(f"block_size={args.block_size}")
    print(f"num_blocks={llm.runner.num_blocks}")
    print(f"npu_memory_utilization={args.npu_memory_utilization}")
    print(f"enable_prefix_cache={args.enable_prefix_cache}")
    print(f"prompt_repeat={args.prompt_repeat}")
    print(f"warmup_iters={args.warmup_iters}")
    print(f"iters={args.iters}")

    for _ in range(args.warmup_iters):
        timed_run(
            llm=llm,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )

    rows = []
    for iter_id in range(1, args.iters + 1):
        row = timed_run(
            llm=llm,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            sampling_params=sampling_params,
        )
        rows.append(row)
        print_row(iter_id, row)

    summary = make_summary(rows)
    print_summary(summary)
    peak_hbm_gb = torch.npu.max_memory_allocated() / 1024**3
    print(f"peak_hbm_gb={peak_hbm_gb:.3f}")

    result = {
        "config": {
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "max_num_seqs": max_num_seqs,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "block_size": args.block_size,
            "num_blocks": llm.runner.num_blocks,
            "npu_memory_utilization": args.npu_memory_utilization,
            "enable_prefix_cache": args.enable_prefix_cache,
            "prompt_repeat": args.prompt_repeat,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "peak_hbm_gb": peak_hbm_gb,
        },
        "summary": summary,
        "iterations": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote_json={args.output_json}")


if __name__ == "__main__":
    main()
