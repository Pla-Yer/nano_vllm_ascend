# nanovllm-ascend-m6

Minimal extraction of the current m6 path:

- Qwen3 only
- Ascend NPU only
- bf16 only
- batch prefill with `npu_fused_infer_attention_score`
- decode with `_npu_paged_attention`

## Run

```powershell
pip install -e .
python examples/generate.py --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ --prompt "Hello" --prompt "Explain KV cache"
```

## Python API

```python
from nanovllm_ascend import LLM

llm = LLM("/home/player/models/Qwen3/Qwen/Qwen3-0___6B/")
texts = llm.generate(["Hello", "Explain KV cache"], max_new_tokens=128)
```

