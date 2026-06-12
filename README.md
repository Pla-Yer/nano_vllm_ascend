# nano_vllm_ascend

A minimal LLM inference engine for Huawei Ascend NPU.

## Features

- **Qwen3 only** · **Ascend NPU only** · **bf16 only**
- Prefill via `npu_fused_infer_attention_score`
- Decode via `_npu_paged_attention`
- Optional decode-only ACL graph replay with paged-attention task updates
- Prefix cache (content-addressed block reuse)
- OpenAI-compatible chat completions API

## Performance

> Qwen3-0.6B · Ascend 910B3 vs vllm-ascend vs vllm (RTX 5090 Laptop) · 随机 token-id 工作负载 · 详细数据见 [docs](docs/nano_vllm_ascend_VS_vllm_ascend_VS_vllm_VS_nano_vlm.md)

### 跨后端吞吐量（32 req × 256 in / 2048 out）

| 后端 | 设备 | output tok/s | total tok/s |
|------|------|-------------|-------------|
| **nano-vllm-ascend** | 910B3 | **1296** | **1458** |
| vllm-ascend | 910B3 | 1255 | 1412 |
| vllm | 5090 Laptop | 2263 | 2546 |
| nano-vllm | 5090 Laptop | 2258 | 2540 |

### Ascend 同平台多工作负载对比

| 工作负载 | nano output tok/s | vllm-ascend output tok/s | nano 优势 |
|----------|-------------------|-------------------------|----------|
| 32×128/1024 (decode-heavy) | 1453 | 1343 | **+8.2%** |
| 32×256/2048 (均衡) | 1296 | 1255 | **+3.2%** |
| 32×2048/2048 (均衡) | 1033 | 975 | **+6.0%** |
| 32×1024/128 (prefill-heavy) | 1228 | 1313 | -6.5% |
| 100×2048/2048 (大批量) | 1410 | 1856 | -24.0% |

### ACL Graph 加速效果

| 配置 | decode tok/s | 加速比 |
|------|-------------|--------|
| baseline (bs=1, 无 graph) | 18.9 | 1.0× |
| ACL Graph (bs=1) | 100.8 | **5.3×** |
| ACL Graph (bs=4) | 372.4 | **19.7×** |

---

## Install

建议使用 vllm-ascend 官方 Docker 容器，已预装 CANN、torch-npu 等依赖，免去手动配置环境：

```powershell
# Atlas A2 (910B)
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0

# Atlas A3
# export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0-a3

# 根据实际设备修改 /dev/davinci 编号
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

> 详细安装方式（pip / Docker / 源码构建）参见 [vllm-ascend 安装文档](https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/installation.html)。

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
