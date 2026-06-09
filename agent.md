# Agent Guide for nanovllm_ascend_m6

This repository is a minimal Ascend NPU runtime for Qwen3. Keep it small,
direct, and aligned with the current engine path.

## Project Boundary

- Qwen3 only.
- Ascend NPU only.
- bf16 only.
- One runtime path: batch prefill with `npu_fused_infer_attention_score`,
  paged decode with `_npu_paged_attention`.
- Do not revive M1-M5 code, historical benchmark variants, naive attention,
  `ContinuousKVCache`, `NaivePaged`, or `qwen3_with_*` files.
- Do not add fallback inference paths for non-NPU, non-Qwen, old tokenizer
  behavior, or alternate attention backends unless the user explicitly asks.

## Current Public API

```python
from nanovllm_ascend import LLM, SamplingParams

llm = LLM(
    model_path,
    max_model_len=2048,
    block_size=128,
    num_blocks=None,
    max_num_seqs=4,
    device_id=0,
    npu_memory_utilization=0.8,
    enable_prefix_cache=False,
)
llm.warm()
outputs = llm.generate(["Hello"], max_new_tokens=128)
```

`num_blocks=None` means KV cache blocks are computed from
`npu_memory_utilization` after model load. Passing `num_blocks` is an explicit
override.

## Core Call Chain

The real engine path is:

```text
LLM.generate()
  -> scheduler.reset()
  -> runner.tokenize_prompts()
  -> runner.prepare_sequences()
  -> scheduler.add_request()
  -> LLM.step()
  -> EngineCore.step()
  -> scheduler.plan_next_step()
  -> runner.prefill() / runner.decode()
```

Use this path for end-to-end correctness. Manual helper calls such as direct
`runner.prefill()` are useful for low-level inspection but do not prove full
engine behavior.

## Module Responsibilities

- `src/nanovllm_ascend/engine.py`: public `LLM`, request submission, step loop,
  warmup, output shaping.
- `src/nanovllm_ascend/scheduler.py`: continuous batching admission and KV block
  reservation.
- `src/nanovllm_ascend/model_runner.py`: tokenizer, Qwen3 model loading,
  sequence preparation, prefill/decode tensor preparation.
- `src/nanovllm_ascend/models/qwen3.py`: flat TND model forward for both prefill
  and decode, selected by `is_prefill`.
- `src/nanovllm_ascend/npu/block_manager.py`: block tables, slot mapping,
  prefix-cache block reuse, attention metadata.
- `src/nanovllm_ascend/npu/paged_kv_cache.py`: physical paged KV tensors and
  the single TND `write()` path.
- `src/nanovllm_ascend/layers/npu_prefill_attention.py`: dense prefill and
  paged cache-hit prefill through FIA.
- `src/nanovllm_ascend/layers/npu_paged_attention.py`: decode paged attention.
- `src/nanovllm_ascend/openai_server.py`: minimal OpenAI-compatible chat server.

## Prefix Cache Rules

- `enable_prefix_cache=False` must keep `cached_prefix_len=0` and use dense
  prefill.
- Cache-hit prefill is selected by `BlockManager.prepare_prefill_metadata()` via
  `start_positions > 0`.
- Cache-hit prefill should stay on FIA-style paged prefill over the physical KV
  cache.
- Validate prefix-cache changes against no-cache generated token ids or text,
  not only whether a kernel runs.
- Keep full-block reuse simple. Do not add partial-block reuse unless requested.

## Coding Style for Future Changes

- Prefer direct, readable code over generic frameworks or speculative helpers.
- Keep interfaces explicit and self-explanatory.
- For internal trusted inputs, avoid defensive `try/except`, compatibility
  branches, and broad `if` guards.
- Keep necessary errors where they diagnose real engine misuse, such as shape
  mismatches, block exhaustion, and active-sequence cache clearing.
- Do not preserve code because it might be useful later.
- If a public or script interface changes, update all call sites, README, and
  examples in the same change.
- Do not introduce a second prefill/decode tensor layout. Keep TND as the single
  model-internal shape.
- Do not add nested abstractions for one-line operations.

## Refactor Checklist

Before editing:

```powershell
git status --short
rg -n "m1_|m2_|m3_|m4_|m5_|qwen3_with_|ContinuousKVCache|NaivePaged|fallback|naive" .
```

After editing:

```powershell
python -B -c "from pathlib import Path; import ast; [ast.parse(p.read_text(encoding='utf-8')) for root in ['src','examples','tests'] for p in Path(root).glob('**/*.py')]; print('ast ok')"
```

If `pytest`, `torch`, and the Ascend runtime are installed:

```powershell
python -m pytest
python examples/generate.py --model-path <model> --prompt "Hello" --warm
python examples/bench.py --model-path <model> --batch-size 1 --max-new-tokens 128 --iters 5 --output-json bench_outputs/baseline.json
python examples/bench_prefix_cache_speedup.py --model-path <model> --prefill-only
```

In this Windows workspace, lightweight syntax validation may be the only
available check when `pytest`, `torch`, or `torch_npu` are missing. State that
boundary clearly instead of implying runtime validation.

## Files That Are Mostly Diagnostic

- `examples/bench.py` is the baseline inference performance script for NPU
  layer optimization. Keep its metrics stable across layer changes.
- `examples/bench_prefix_cache_speedup.py` is for route and timing diagnosis.
  Keep route reporting explicit.
- `examples/check_prefix_cache_equivalence.py` is for comparing cache-hit output
  against no-cache output.
- `examples/prefix_cache_inputs.py` holds shared long-prefix text for diagnostic
  scripts. Keep it plain and dependency-free.

## Worktree Discipline

- The user may have uncommitted edits. Do not revert files you did not modify.
- Generated cache files such as `__pycache__` are disposable.
- Keep docs and examples synchronized with interface changes.
