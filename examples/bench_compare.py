#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline throughput benchmark for nano_vllm_ascend / nanovllm / vLLM / vLLM-Ascend.

Goals:
  1) Keep the workload close to vLLM's random-token benchmark style.
  2) Use the same random prompt_token_ids and requested output lengths across backends.
  3) Avoid tokenizer/chat-template overhead so Ascend NPU and GPU vLLM results are more comparable.

Examples:
  # nano_vllm_ascend on Ascend NPU
  python examples/bench_compare.py \
      --backend nano \
      --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
      --num-prompts 256 --input-len 1024 --output-len 128 \
      --max-num-seqs 256 --ignore-eos \
      --output-json bench_outputs/nano.json

  # nanovllm on GPU
  python examples/bench_compare.py \
      --backend nanovllm \
      --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
      --num-prompts 256 --input-len 1024 --output-len 128 \
      --max-num-seqs 256 --ignore-eos --enforce-eager \
      --output-json bench_outputs/nanovllm.json

  # vLLM-Ascend on Ascend NPU, or vLLM on GPU in a GPU vLLM environment
  python examples/bench_compare.py \
      --backend vllm \
      --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
      --num-prompts 256 --input-len 1024 --output-len 128 \
      --max-num-seqs 256 --ignore-eos --dtype bfloat16 \
      --output-json bench_outputs/vllm.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nano", "nanovllm", "vllm"), required=True)
    parser.add_argument("--model-path", required=True)

    # Workload. Fixed lengths are best for cross-machine comparison.
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--random-input-len", action="store_true",
                        help="Use randint(min_input_len, input_len) per request.")
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--random-output-len", action="store_true",
                        help="Use randint(min_output_len, output_len) per request.")
    parser.add_argument("--min-output-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-id-high", type=int, default=10000,
                        help="Upper bound for random token ids before clamping to vocab_size - 1.")

    # Engine knobs shared as much as possible.
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="If omitted, use num_prompts.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="For vLLM GPU/vLLM-Ascend/nanovllm.")
    parser.add_argument("--npu-memory-utilization", type=float, default=0.8,
                        help="For nano_vllm_ascend.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Mainly for vLLM/vLLM-Ascend.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Mainly for vLLM/vLLM-Ascend.")
    parser.add_argument("--dtype", default="auto",
                        help="vLLM dtype, e.g. auto/half/float16/bfloat16.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Pass enforce_eager=True to vLLM/nanovllm if supported.")

    # nano_vllm_ascend optional features. They are passed only if the local LLM supports them.
    parser.add_argument("--enable-decode-graph", action="store_true")
    parser.add_argument("--decode-graph-batch-sizes", default=None,
                        help="Comma-separated batch sizes, passed only if supported by local nano LLM.")
    parser.add_argument("--enable-prefix-cache", action="store_true")

    # Sampling. Greedy + ignore_eos is recommended for stable requested output length.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0,
                        help="Used by nano; passed to vLLM only if accepted by SamplingParams.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="vLLM: SamplingParams(ignore_eos=True). nano: monkey-patch eos_token_id to -1.")

    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--print-outputs", action="store_true")
    return parser.parse_args()


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass optional kwargs only when the installed backend accepts them."""
    sig = inspect.signature(callable_obj)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if v is not None and k in params}


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x) for x in value.replace(",", " ").split()]


def sync_device() -> None:
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def reset_peak_memory() -> None:
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.reset_peak_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_peak_memory() -> dict[str, float | None]:
    stats: dict[str, float | None] = {
        "cuda_peak_allocated_gb": None,
        "cuda_peak_reserved_gb": None,
        "npu_peak_allocated_gb": None,
        "npu_peak_reserved_gb": None,
    }
    gib = 1024 ** 3
    try:
        import torch
        if torch.cuda.is_available():
            stats["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / gib
            stats["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / gib
        if hasattr(torch, "npu") and torch.npu.is_available():
            # torch-npu versions differ; keep this best-effort.
            if hasattr(torch.npu, "max_memory_allocated"):
                stats["npu_peak_allocated_gb"] = torch.npu.max_memory_allocated() / gib
            if hasattr(torch.npu, "max_memory_reserved"):
                stats["npu_peak_reserved_gb"] = torch.npu.max_memory_reserved() / gib
    except Exception:
        pass
    return stats


def load_vocab_size(model_path: str) -> int | None:
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        return int(getattr(config, "vocab_size"))
    except Exception:
        return None


def make_dataset(args: argparse.Namespace) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    rng = random.Random(args.seed)
    vocab_size = load_vocab_size(args.model_path)
    token_high = args.token_id_high
    if vocab_size is not None:
        token_high = min(token_high, vocab_size - 1)
    if token_high <= 0:
        raise ValueError(f"invalid token-id upper bound: {token_high}")

    input_lens: list[int] = []
    output_lens: list[int] = []
    prompts: list[list[int]] = []
    for _ in range(args.num_prompts):
        in_len = (rng.randint(args.min_input_len, args.input_len)
                  if args.random_input_len else args.input_len)
        out_len = (rng.randint(args.min_output_len, args.output_len)
                   if args.random_output_len else args.output_len)
        input_lens.append(in_len)
        output_lens.append(out_len)
        prompts.append([rng.randint(0, token_high) for _ in range(in_len)])

    meta = {
        "vocab_size": vocab_size,
        "token_id_high_used": token_high,
        "input_lens_min": min(input_lens),
        "input_lens_max": max(input_lens),
        "output_lens_min": min(output_lens),
        "output_lens_max": max(output_lens),
    }
    return prompts, output_lens, meta


def make_nano_llm(args: argparse.Namespace):
    from nanovllm_ascend import LLM

    kwargs = {
        "model_path": args.model_path,
        "max_model_len": args.max_model_len,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "max_num_seqs": args.max_num_seqs or args.num_prompts,
        "device_id": args.device_id,
        "npu_memory_utilization": args.npu_memory_utilization,
        "enable_prefix_cache": args.enable_prefix_cache,
        "enable_decode_graph": args.enable_decode_graph,
        "decode_graph_batch_sizes": parse_int_list(args.decode_graph_batch_sizes),
    }
    return LLM(**filter_supported_kwargs(LLM, kwargs))


def make_nano_sampling_params(args: argparse.Namespace):
    from nanovllm_ascend import SamplingParams

    kwargs = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    return SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs))


def nano_generate_from_token_ids(
    llm: Any,
    prompt_token_ids: list[list[int]],
    output_lens: list[int],
    sampling_params: Any,
) -> list[dict[str, Any]]:
    """Direct token-id submit path for nano_vllm_ascend.

    This intentionally uses internal Sequence/scheduler APIs so the benchmark
    can avoid tokenizer/chat-template overhead and match vLLM's prompt_token_ids
    workload.
    """
    import torch
    from nanovllm_ascend.sequence import Sequence

    llm.scheduler.reset()
    request_ids: list[int] = []
    for ids, max_new_tokens in zip(prompt_token_ids, output_lens):
        seq_id = llm.scheduler.next_seq_id()
        seq = Sequence(
            seq_id=seq_id,
            prompt="",
            max_new_tokens=max_new_tokens,
            prompt_token_ids=torch.tensor(ids, dtype=torch.long),
            sampling_params=sampling_params,
        )
        llm.runner.prepare_sequences([seq])
        seq.reserved_blocks = llm.scheduler.compute_required_blocks(
            runtime_prompt_len=seq.runtime_prompt_len,
            max_new_tokens=max_new_tokens,
        )
        llm.scheduler.add_request(seq)
        request_ids.append(seq_id)

    outputs_by_request_id: dict[int, dict[str, Any]] = {}
    while llm.has_unfinished():
        for output in llm.step():
            outputs_by_request_id[int(output["request_id"])] = output

    return [
        outputs_by_request_id.get(req_id, {"request_id": req_id, "texts": "", "token_ids": []})
        for req_id in request_ids
    ]


def run_nano(args: argparse.Namespace, prompts: list[list[int]], output_lens: list[int]) -> tuple[list[int], Any, float]:
    llm = make_nano_llm(args)
    sampling_params = make_nano_sampling_params(args)
    if args.ignore_eos:
        # Matches vLLM SamplingParams(ignore_eos=True) well enough for throughput tests.
        try:
            llm.runner.tokenizer.eos_token_id = -1
        except Exception:
            pass

    for _ in range(args.warmup_iters):
        nano_generate_from_token_ids(llm, prompts, output_lens, sampling_params)
        sync_device()

    outputs = None
    sync_device()
    reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(args.profile_iters):
        outputs = nano_generate_from_token_ids(llm, prompts, output_lens, sampling_params)
        sync_device()
    elapsed = time.perf_counter() - t0
    assert outputs is not None

    generated_lens = [len(out.get("token_ids", [])) for out in outputs]
    if args.print_outputs:
        for i, out in enumerate(outputs[:8]):
            print(f"[nano output {i}] len={len(out.get('token_ids', []))} text={out.get('texts', '')[:160]!r}")
    return generated_lens, llm, elapsed


def make_nanovllm_llm(args: argparse.Namespace):
    """Create upstream nanovllm LLM without affecting nano_vllm_ascend.

    The public nanovllm benchmark instantiates LLM as::

        LLM(path, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.7)

    Different forks may name the first argument `model`, `model_path`, or keep it
    positional, so this wrapper detects the signature and only passes supported
    optional kwargs.
    """
    from nanovllm import LLM

    optional_kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs or args.num_prompts,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
    }
    optional_kwargs = filter_supported_kwargs(LLM, optional_kwargs)

    sig = inspect.signature(LLM)
    params = sig.parameters
    if "model_path" in params:
        return LLM(model_path=args.model_path, **optional_kwargs)
    if "model" in params:
        return LLM(model=args.model_path, **optional_kwargs)
    return LLM(args.model_path, **optional_kwargs)


def make_nanovllm_sampling_params(args: argparse.Namespace, output_lens: list[int]):
    from nanovllm import SamplingParams

    params = []
    for out_len in output_lens:
        kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": out_len,
            "ignore_eos": args.ignore_eos,
        }
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
        params.append(SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs)))
    return params


def call_nanovllm_generate(llm: Any, prompts: list[list[int]], sampling_params: Any) -> Any:
    # nanovllm accepts raw list[list[int]] token ids directly, unlike vLLM's
    # [{"prompt_token_ids": ...}] wrapper. Keep this path separate.
    kwargs = filter_supported_kwargs(llm.generate, {"use_tqdm": False})
    return llm.generate(prompts, sampling_params, **kwargs)


def call_nanovllm_warmup(llm: Any) -> None:
    """Warm up nanovllm in the same style as upstream bench.py.

    Do not warm up with the measured random-token workload here. Upstream
    nanovllm has prefix caching enabled internally, and replaying the exact same
    full-block prompts in the measured run can reuse hashed prompt blocks. Some
    versions then hit ``BlockManager.may_append`` assertions during decode, for
    example when ``input_len`` is exactly one KV-cache block such as 256 tokens.
    Using a tiny unrelated string prompt matches nanovllm's own benchmark and
    keeps the measured workload independent from warmup cache state.
    """
    from nanovllm import SamplingParams

    kwargs = filter_supported_kwargs(llm.generate, {"use_tqdm": False})
    llm.generate(["Benchmark: "], SamplingParams(), **kwargs)


def extract_generated_len(output: Any) -> int | None:
    """Best-effort extraction for vLLM/nanovllm-like output objects."""
    if output is None:
        return None

    if isinstance(output, dict):
        for key in ("token_ids", "output_token_ids", "generated_token_ids"):
            value = output.get(key)
            if value is not None:
                return len(value)
        nested = output.get("outputs")
        if nested:
            return extract_generated_len(nested[0])

    if isinstance(output, (list, tuple)):
        if not output:
            return 0
        if all(isinstance(x, int) for x in output):
            return len(output)

    for attr in ("token_ids", "output_token_ids", "generated_token_ids"):
        value = getattr(output, attr, None)
        if value is not None:
            return len(value)

    nested = getattr(output, "outputs", None)
    if nested:
        return extract_generated_len(nested[0])

    return None


def run_nanovllm(args: argparse.Namespace, prompts: list[list[int]], output_lens: list[int]) -> tuple[list[int], Any, float]:
    llm = make_nanovllm_llm(args)
    sampling_params = make_nanovllm_sampling_params(args, output_lens)

    for _ in range(args.warmup_iters):
        call_nanovllm_warmup(llm)
        sync_device()

    outputs = None
    sync_device()
    reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(args.profile_iters):
        outputs = call_nanovllm_generate(llm, prompts, sampling_params)
        sync_device()
    elapsed = time.perf_counter() - t0
    assert outputs is not None

    outputs_list = [outputs] if isinstance(outputs, (str, bytes)) else list(outputs)
    if len(outputs_list) != len(output_lens):
        print(
            f"[warn] nanovllm returned {len(outputs_list)} outputs for {len(output_lens)} requests; "
            "missing generated lengths will fall back to requested max_tokens."
        )

    generated_lens: list[int] = []
    fallback_count = 0
    for i, requested_len in enumerate(output_lens):
        output = outputs_list[i] if i < len(outputs_list) else None
        generated_len = extract_generated_len(output)
        if generated_len is None:
            # Some nanovllm variants return only text. With --ignore-eos this is
            # normally equal to max_tokens; otherwise this is an approximation.
            generated_len = requested_len
            fallback_count += 1
        generated_lens.append(generated_len)

    if fallback_count:
        print(
            f"[warn] nanovllm outputs did not expose token_ids for {fallback_count} requests; "
            "using requested max_tokens as generated length. Use --ignore-eos for accurate throughput."
        )

    if args.print_outputs:
        for i, out in enumerate(outputs_list[:8]):
            text = getattr(out, "text", out if isinstance(out, str) else "")
            print(f"[nanovllm output {i}] len={generated_lens[i]} text={str(text)[:160]!r}")
    return generated_lens, llm, elapsed


def make_vllm_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs = {
        "model": args.model_path,
        "tokenizer": args.model_path,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs or args.num_prompts,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "block_size": args.block_size,
    }
    return LLM(**filter_supported_kwargs(LLM, kwargs))


def make_vllm_sampling_params(args: argparse.Namespace, output_lens: list[int]):
    from vllm import SamplingParams

    params = []
    for out_len in output_lens:
        kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": out_len,
            "ignore_eos": args.ignore_eos,
        }
        # vLLM versions differ on top_k validation/defaults.
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
        params.append(SamplingParams(**filter_supported_kwargs(SamplingParams, kwargs)))
    return params


def run_vllm(args: argparse.Namespace, prompts: list[list[int]], output_lens: list[int]) -> tuple[list[int], Any, float]:
    llm = make_vllm_llm(args)
    sampling_params = make_vllm_sampling_params(args, output_lens)
    vllm_prompts = [{"prompt_token_ids": ids} for ids in prompts]

    for _ in range(args.warmup_iters):
        llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=False)
        sync_device()

    outputs = None
    sync_device()
    reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(args.profile_iters):
        outputs = llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=False)
        sync_device()
    elapsed = time.perf_counter() - t0
    assert outputs is not None

    generated_lens: list[int] = []
    for out in outputs:
        # RequestOutput.outputs[0].token_ids
        generated_lens.append(len(out.outputs[0].token_ids))
    if args.print_outputs:
        for i, out in enumerate(outputs[:8]):
            text = getattr(out.outputs[0], "text", "")
            print(f"[vllm output {i}] len={len(out.outputs[0].token_ids)} text={text[:160]!r}")
    return generated_lens, llm, elapsed


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(round((len(xs) - 1) * pct / 100.0))
    return xs[k]


def main() -> None:
    args = parse_args()
    if args.profile_iters != 1:
        # The JSON reports one aggregate timing. Multiple measured iters are possible,
        # but using 1 avoids hiding graph/cache behavior behind averaging.
        print(f"[warn] profile-iters={args.profile_iters}; elapsed is total over all profile iters.")

    prompts, output_lens, dataset_meta = make_dataset(args)
    total_input_tokens = sum(len(p) for p in prompts) * args.profile_iters
    requested_output_tokens = sum(output_lens) * args.profile_iters

    print(
        f"backend={args.backend} num_prompts={args.num_prompts} "
        f"input_len=[{dataset_meta['input_lens_min']},{dataset_meta['input_lens_max']}] "
        f"output_len=[{dataset_meta['output_lens_min']},{dataset_meta['output_lens_max']}] "
        f"max_num_seqs={args.max_num_seqs or args.num_prompts}"
    )

    if args.backend == "nano":
        generated_lens, llm, elapsed = run_nano(args, prompts, output_lens)
    elif args.backend == "nanovllm":
        generated_lens, llm, elapsed = run_nanovllm(args, prompts, output_lens)
    else:
        generated_lens, llm, elapsed = run_vllm(args, prompts, output_lens)

    total_output_tokens = sum(generated_lens) * args.profile_iters
    total_tokens = total_input_tokens + total_output_tokens

    result = {
        "backend": args.backend,
        "model_path": os.path.abspath(os.path.expanduser(args.model_path)),
        "num_prompts": args.num_prompts,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs or args.num_prompts,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_eos": args.ignore_eos,
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "elapsed_s": elapsed,
        "total_input_tokens": total_input_tokens,
        "requested_output_tokens": requested_output_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "request_throughput_req_s": (args.num_prompts * args.profile_iters) / elapsed,
        "output_throughput_tok_s": total_output_tokens / elapsed,
        "total_throughput_tok_s": total_tokens / elapsed,
        "avg_input_len": statistics.mean([len(p) for p in prompts]),
        "avg_requested_output_len": statistics.mean(output_lens),
        "avg_actual_output_len": statistics.mean(generated_lens) if generated_lens else 0.0,
        "actual_output_len_p50": percentile([float(x) for x in generated_lens], 50),
        "actual_output_len_p90": percentile([float(x) for x in generated_lens], 90),
        "actual_output_len_min": min(generated_lens) if generated_lens else 0,
        "actual_output_len_max": max(generated_lens) if generated_lens else 0,
        "dataset": dataset_meta,
        "memory": get_peak_memory(),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
