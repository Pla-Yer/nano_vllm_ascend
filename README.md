# nano_vllm_ascend

A simaple nano vllm for Ascend NPU

## Current Support
- Qwen3 only
- Ascend NPU only
- bf16 only
- prefill with `npu_fused_infer_attention_score`
- decode with `_npu_paged_attention`

## Run

```powershell
pip install -e .
python examples/generate.py --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ --prompt "Hello" --prompt "Explain KV cache" --warm
```

## Benchmark

```powershell
python examples/bench.py --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ --batch-size 1 --max-new-tokens 128 --iters 5 --output-json bench_outputs/baseline.json
```

Use `examples/bench.py` as the baseline script for NPU layer optimization. It
reports prepare, prefill, decode, total latency, token throughput, and peak HBM.

## Python API

```python
from nanovllm_ascend import LLM, SamplingParams

llm = LLM("/home/player/models/Qwen3/Qwen/Qwen3-0___6B/")
llm.warm()
texts = llm.generate(["Hello", "Explain KV cache"], max_new_tokens=128)
sampled = llm.generate(
    ["Write a short slogan"],
    max_new_tokens=64,
    sampling_params=SamplingParams(temperature=0.8, top_k=20, top_p=0.9),
)
```

`LLM` computes KV cache blocks from `npu_memory_utilization=0.8` by default.
Pass `num_blocks` to override the automatic KV cache size.
