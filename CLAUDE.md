# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal LLM inference engine for Huawei Ascend NPU. Loads Qwen3 weights from HuggingFace, replaces the model with a custom implementation using Ascend NPU kernels, and runs continuous-batched inference with paged KV cache.

**Hard constraints — do not violate:**
- Qwen3 only
- Ascend NPU only (requires `torch_npu` + CANN toolkit)
- bf16 only
- Single runtime path: batch prefill via `npu_fused_infer_attention_score`, paged decode via `_npu_paged_attention`
- Do not revive M1–M5 code, historical variants, naive attention, `ContinuousKVCache`, `NaivePaged`, or `qwen3_with_*` files
- Do not add fallback paths for non-NPU, non-Qwen, or alternate attention backends unless explicitly asked
- Keep TND as the single model-internal tensor layout

## Commands

```powershell
# Install (editable)
pip install -e .

# Tests
python -m pytest

# Syntax-only validation (no runtime needed)
python -B -c "from pathlib import Path; import ast; [ast.parse(p.read_text(encoding='utf-8')) for root in ['src','examples','tests'] for p in Path(root).glob('**/*.py')]; print('ast ok')"

# Generate
python examples/generate.py --model-path <model> --prompt "Hello" --warm

# Benchmark
python examples/bench.py --model-path <model> --batch-size 1 --max-new-tokens 128 --iters 5 --output-json bench_outputs/baseline.json
python examples/bench.py --model-path <model> --batch-size 1 --max-new-tokens 128 --enable-decode-graph

# Prefix cache equivalence check
python examples/bench_prefix_cache_speedup.py --model-path <model> --prefill-only

# OpenAI-compatible server
python -m nanovllm_ascend.openai_server --model-path <model> --warm
```

No Makefile, linting config, or CI exists. On this Windows workspace without NPU, syntax validation is often the only available check — state that boundary clearly.

## Architecture

```
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

**Core components (all in `src/nanovllm_ascend/`):**

| File | Component | Purpose |
|------|-----------|---------|
| `engine.py` | `LLM`, `EngineCore` | Public API, request submission, step loop, warmup |
| `scheduler.py` | `MiniScheduler` | Continuous batching admission, KV block reservation |
| `model_runner.py` | `ModelRunner` | Tokenizer, Qwen3 weight loading, prefill/decode tensor prep |
| `models/qwen3.py` | `Qwen3ForCausalLM` | Custom model forward; `is_prefill` flag switches attention path |
| `npu/block_manager.py` | `BlockManager` | Block tables, slot mapping, prefix-cache block reuse |
| `npu/paged_kv_cache.py` | `PagedKVCache` | Physical KV tensor storage, TND `write()` path |
| `npu/acl_graph.py` | `DecodeGraphRunner` | NPU graph capture/replay for decode acceleration |
| `layers/npu_prefill_attention.py` | `NpuBatchPrefillAttention` | Dense + cache-hit prefill via FIA kernel |
| `layers/npu_paged_attention.py` | `NpuPagedAttention` | Decode paged attention, ACL graph integration |
| `openai_server.py` | `OpenAIChatService` | FastAPI `/v1/chat/completions` endpoint |

**Key NPU kernels:** `npu_fused_infer_attention_score` (prefill), `_npu_paged_attention` (decode), `npu_rms_norm`, `npu_rotary_mul`

**Prefix cache:** When `enable_prefix_cache=True`, `BlockManager.find_longest_prefix_blocks()` matches content-addressed block hashes before prefill, reusing matched blocks and only prefilling the suffix. Validate cache changes against no-cache output token IDs, not just kernel execution.

## Public API

```python
from nanovllm_ascend import LLM, SamplingParams

llm = LLM(
    model_path,
    max_model_len=2048,
    block_size=128,
    num_blocks=None,        # auto-computed from npu_memory_utilization
    max_num_seqs=4,
    device_id=0,
    npu_memory_utilization=0.8,
    enable_prefix_cache=False,
    enable_decode_graph=False,
)
llm.warm()
outputs = llm.generate(["Hello"], max_new_tokens=128)
```

`enable_decode_graph=True` captures decode graphs for batch sizes `[1, 2, 4, 8, 16]` by default; pass `decode_graph_batch_sizes=[...]` to customize.

## Coding Style

- Direct, readable code over generic frameworks or speculative helpers
- Keep interfaces explicit and self-explanatory
- For internal trusted inputs, avoid defensive `try/except`, compatibility branches, and broad `if` guards
- Keep necessary errors where they diagnose real engine misuse (shape mismatches, block exhaustion, etc.)
- Do not preserve code because it might be useful later
- If a public or script interface changes, update all call sites, README, and examples in the same change
- Do not add nested abstractions for one-line operations

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

If NPU runtime is available:
```powershell
python -m pytest
python examples/generate.py --model-path <model> --prompt "Hello" --warm
```

## Worktree Discipline

- The user may have uncommitted edits. Do not revert files you did not modify.
- Generated cache files such as `__pycache__` are disposable.
- Keep docs and examples synchronized with interface changes.
