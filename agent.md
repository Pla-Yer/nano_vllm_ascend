# Agent Guide

This project is the minimal m6 extraction from `nanovllm_ascend`.

## Project Intent

- Keep one runtime path only: Qwen3 + Ascend NPU + bf16 + batch prefill + NPU paged decode.
- Keep the model inference path flat/TND: hidden states are `[T, hidden]`, Q/K/V are `[T, heads, dim]`, and both prefill and decode call `Qwen3ForCausalLM.forward_flat(...)`.
- Prefer small, direct code over compatibility layers.
- Preserve clear boundaries so this can later grow into a service framework without carrying the old experimental baseline structure.

## Boundaries

- `engine.py` owns Python-level request state: prompts, generated token ids, active slots, finished flags, and the decode loop.
- `model_runner.py` is the only bridge that creates tensors, tokenizes prompts, prepares prefill/decode calls, owns the model, and owns the KV cache.
- `models/`, `layers/`, and `npu/` own tensor operations, Qwen3 modules, KV cache layout, NPU attention calls, and device-specific details.
- Do not let Engine construct `torch.Tensor` objects or know about `slot_mapping`, `block_tables`, `context_lens`, or `actual_seq_lengths`.
- Do not add a separate 4D decode model path. Decode may receive one token per active sequence, but it should be flattened to TND before entering the transformer layers.

## Scope Rules

- Do not reintroduce M1-M5 baselines, benchmark harnesses, CPU fallback, CUDA fallback, naive attention, `ContinuousKVCache`, or historical `qwen3_with_*` variants.
- Do not add broad robustness for unsupported backends. Fail clearly when an assumption is violated.
- Keep greedy argmax sampling unless the task explicitly asks for a sampling feature.
- Keep `block_size=128`, `max_model_len=2048`, and `torch.bfloat16` as defaults unless a change is explicitly requested.

## Public API

The supported API is:

```python
from nanovllm_ascend import LLM

llm = LLM(model_path, max_model_len=2048, block_size=128, device_id=0)
texts = llm.generate(prompts, max_new_tokens=128)
```

The CLI example is:

```powershell
python examples/generate.py --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ --prompt "Hello" --prompt "Explain KV cache"
```

## Verification

- Run syntax validation without creating `.pyc` files when Windows permissions block `__pycache__`:

```powershell
python -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').rglob('*.py')]; print('ast ok')"
```

- In an environment with `torch`, run the unit tests:

```powershell
$env:PYTHONPATH='src'
python -B -m pytest tests
```

- In an Ascend environment with `torch_npu`, run the integration example with a local Qwen3 model path and a small `--max-new-tokens` first.

## Editing Guidance

- Keep changes localized to the layer that owns the behavior.
- If new scheduler or service features are added later, keep scheduler/block-manager state logical and Python-only; translate execution plans to NPU metadata inside ModelRunner or lower.
- Add tests for KV metadata shape and slot mapping when changing cache behavior.
- Before finishing, scan for old baseline names:

```powershell
rg -n "m1_|m2_|m3_|m4_|m5_|benchmark|qwen3_with_|ContinuousKVCache|NaivePaged" .
```
