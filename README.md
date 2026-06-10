# nano_vllm_ascend

A minimal LLM inference engine for Huawei Ascend NPU.

## Features

- **Qwen3 only** · **Ascend NPU only** · **bf16 only**
- Prefill via `npu_fused_infer_attention_score`
- Decode via `_npu_paged_attention`
- Optional decode-only ACL graph replay with paged-attention task updates
- Prefix cache (content-addressed block reuse)
- OpenAI-compatible chat completions API

## Install

```powershell
pip install -e .
```

---

## Python API

```python
from nanovllm_ascend import LLM, SamplingParams

llm = LLM("/home/player/models/Qwen3/Qwen/Qwen3-0___6B/")
llm.warm()
texts = llm.generate(["Hello", "Explain KV cache"], max_new_tokens=128)

# Sampling
sampled = llm.generate(
    ["Write a short slogan"],
    max_new_tokens=64,
    sampling_params=SamplingParams(temperature=0.8, top_k=20, top_p=0.9),
)

# Decode graph acceleration
graph_llm = LLM(
    "/home/player/models/Qwen3/Qwen/Qwen3-0___6B/",
    max_num_seqs=4,
    enable_decode_graph=True,
)
```

`LLM` computes KV cache blocks from `npu_memory_utilization=0.8` by default.
Pass `num_blocks` to override the automatic KV cache size.
`enable_decode_graph=True` captures exact decode batch sizes `[1, 2, 4, 8, 16, 32]`
by default. Pass `decode_graph_batch_sizes=[...]` to restrict capture to
specific decode batch sizes.

---

## Examples

### generate.py — Text Generation

Basic text generation. Load a model, run one or more prompts, report throughput.

```powershell
python examples/generate.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --prompt "Hello" \
    --prompt "Explain KV cache" \
    --warm
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to the model directory |
| `--prompt` | (required) | One or more prompts (repeat the flag) |
| `--max-new-tokens` | 128 | Maximum new tokens to generate |
| `--warm` | off | Run a warm-up generation first |
| `--enable-decode-graph` | off | Enable ACL graph replay for decode |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--top-k` | 0 | Top-k sampling |
| `--top-p` | 1.0 | Nucleus sampling probability |
| `--max-model-len` | 2048 | Maximum model context length |
| `--block-size` | 128 | KV cache block size |
| `--npu-memory-utilization` | 0.8 | Fraction of NPU HBM for KV cache |
| `--device-id` | 0 | NPU device ID |

---

### bench_e2e.py — End-to-End Benchmark

Measure prepare, prefill, decode, and total latency per iteration. Reports token throughput, peak HBM memory, and avg/p50/p90/min/max statistics. Optionally writes results to JSON.

```powershell
# Baseline
python examples/bench_e2e.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --batch-size 1 --max-new-tokens 128 --iters 5 \
    --output-json bench_outputs/baseline.json

# With decode graph
python examples/bench_e2e.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --batch-size 1 --max-new-tokens 128 --enable-decode-graph

# With prefix cache
python examples/bench_e2e.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --batch-size 4 --enable-prefix-cache --iters 10
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 1 | Number of concurrent requests |
| `--max-new-tokens` | 128 | Max tokens to generate per request |
| `--iters` | 5 | Number of measured iterations |
| `--warmup-iters` | 1 | Warmup iterations (not timed) |
| `--enable-prefix-cache` | off | Enable prefix caching |
| `--enable-decode-graph` | off | Enable ACL graph replay |
| `--prompt` | None | Base prompts (cycled to fill batch) |
| `--prompt-repeat` | 1 | Repeat each prompt N times |
| `--output-json` | None | Path to write JSON results |

---

### bench_layers.py — Layer Micro-Benchmark

Profile individual model layers on NPU. Measures latency (ms) for RMSNorm, Linear, MLP, LMHead, and Rotary Embedding across various token counts. Reports avg/p50/min/max per layer.

```powershell
python examples/bench_layers.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --tokens 1,16,128,512 --iters 200

python examples/bench_layers.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --dtype fp16 --tokens 1,64,256 --output-json bench_outputs/layers.json
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--tokens` | 1,16,128,512 | Comma-separated token counts to benchmark |
| `--dtype` | bf16 | Data type (bf16/fp16/fp32) |
| `--iters` | 200 | Measured iterations per layer |
| `--warmup-iters` | 20 | Warmup iterations per layer |
| `--output-json` | None | Path to write JSON results |

---

### bench_compare.py — Cross-Backend Throughput Comparison

Offline throughput comparison across multiple backends: `nano` (nano_vllm_ascend on Ascend NPU), `nanovllm` (upstream nanovllm on GPU), and `vllm` (vLLM / vLLM-Ascend). Uses random token-id workloads (no tokenizer overhead) for fair comparison.

```powershell
# nano_vllm_ascend on Ascend NPU
python examples/bench_compare.py --backend nano \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --num-prompts 256 --input-len 1024 --output-len 128

# vLLM on GPU
python examples/bench_compare.py --backend vllm \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --num-prompts 256 --input-len 1024 --output-len 128

# nanovllm on GPU
python examples/bench_compare.py --backend nanovllm \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
    --num-prompts 256 --input-len 1024 --output-len 128
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | (required) | `nano`, `nanovllm`, or `vllm` |
| `--num-prompts` | 256 | Number of requests in the workload |
| `--input-len` | 1024 | Fixed input token length per request |
| `--output-len` | 128 | Fixed requested output length |
| `--random-input-len` | off | Use random input lengths |
| `--random-output-len` | off | Use random output lengths |
| `--enable-decode-graph` | off | ACL decode graph (nano only) |
| `--enable-prefix-cache` | off | Prefix caching (nano only) |
| `--enforce-eager` | off | Disable CUDA graphs (vLLM/nanovllm) |
| `--output-json` | None | Path to write JSON results |
| `--print-outputs` | off | Print first 8 generated outputs |

---

## OpenAI-Compatible API

### Start Server

```powershell
python examples/api/start_server.py \
    --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ --warm
```

Exposes `/v1/chat/completions` on port 8000 by default. Pass `--port` to change.

### Interactive Chat Client

Multi-turn chat with streaming output and conversation history.

```powershell
python examples/api/chat.py
python examples/api/chat.py --system "你是一个有用的助手" --temperature 0.8
python examples/api/chat.py --no-stream
```

Built-in commands: `/clear` (reset conversation), `/history` (view past messages), `/quit` (exit).

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | http://127.0.0.1:8000/v1/chat/completions | Server endpoint |
| `--model` | nanovllm-ascend | Model name |
| `--max-tokens` | 512 | Max tokens per response |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling |
| `--system` | None | System prompt |
| `--no-stream` | off | Disable streaming |

### Test Client

Concurrent HTTP test client for load-testing or validating the server.

```powershell
# Single request
python examples/api/test_chat_completions.py --prompt "Hello"

# 8 concurrent streaming requests
python examples/api/test_chat_completions.py \
    --prompt "Hello" --prompt "What is AI?" \
    --concurrent 8 --stream
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | http://127.0.0.1:7000/v1/chat/completions | Server endpoint |
| `--prompt` | (repeatable) | One or more prompts |
| `--concurrent` | 1 | Number of concurrent requests (threads) |
| `--stream` | off | Use streaming (SSE) mode |
| `--max-tokens` | 256 | Max tokens to generate |

### curl Examples

Non-streaming:

```powershell
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nanovllm-ascend",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "temperature": 0.7
}'
```

Streaming (SSE):

```powershell
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "nanovllm-ascend",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128,
  "stream": true
}'
```
