# 跨后端吞吐量对比

> 模型：Qwen3-0.6B · 工作负载：32 请求 × 256 输入 / 2048 输出 · 使用随机 token-id（无 tokenizer 开销）
>
> 使用 `examples/bench_compare.py` 统一基准，各后端共享相同随机数据集以保证公平。

---

## 测试环境

### nano-vllm-ascend / vllm-ascend

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B3 |
| HBM | 65536 MB |
| CANN | 24.1.rc1 |
| OS | Linux (localhost) |

### vllm / nano-vllm (GPU)

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 Laptop |
| VRAM | 24463 MB |
| Driver | 580.142 |
| CUDA | 13.0 |
| OS | Ubuntu (test-Legion-Y9000P) |

---

## 测试命令

```powershell
# nano-vllm-ascend (Ascend NPU, ACL Graph)
python examples/bench_compare.py --backend nano \
  --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
  --num-prompts 32 --input-len 256 --output-len 2048 \
  --max-num-seqs 32 --max-model-len 4096 --ignore-eos \
  --enable-decode-graph \
  --output-json bench_outputs/nano_32x256_2048.json

# vllm-ascend (Ascend NPU)
python examples/bench_compare.py --backend vllm \
  --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ \
  --num-prompts 32 --input-len 256 --output-len 2048 \
  --max-num-seqs 32 --max-model-len 4096 --dtype bfloat16 --ignore-eos \
  --output-json bench_outputs/vllm_ascend_32x256_2048.json

# vllm (NVIDIA GPU)
python examples/bench_compare.py --backend vllm \
  --model-path /home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/ \
  --num-prompts 32 --input-len 256 --output-len 2048 \
  --max-num-seqs 32 --max-model-len 4096 --ignore-eos \
  --output-json bench_outputs/vllm_32x256_2048.json

# nano-vllm (NVIDIA GPU)
python examples/bench_compare.py --backend nanovllm \
  --model-path /home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/ \
  --num-prompts 32 --input-len 256 --output-len 2048 \
  --max-num-seqs 32 --max-model-len 4096 --ignore-eos \
  --output-json bench_outputs/nanovllm_32x256_2048.json
```

---

## 结果汇总

| 指标 | nano-vllm-ascend | vllm-ascend | vllm (GPU) | nano-vllm (GPU) |
|------|-----------------|-------------|------------|-----------------|
| **设备** | **910B3** | **910B3** | **5090 Laptop** | **5090 Laptop** |
| elapsed_s | 50.57 | 52.20 | 28.96 | 29.03 |
| request_throughput (req/s) | 0.633 | 0.613 | 1.105 | 1.102 |
| **output_throughput (tok/s)** | **1296.05** | **1255.38** | **2262.88** | **2257.55** |
| **total_throughput (tok/s)** | **1458.05** | **1412.31** | **2545.73** | **2539.75** |
| avg_actual_output_len | 2048 | 2048 | 2048 | 2048 |
| HBM/VRAM peak (GB) | 50.28 | — | — | 17.17 |
| ignore_eos | ✓ | ✓ | ✓ | ✓ |
| ACL Graph / CUDA Graph | ✓ | ✓ | ✓ | ✓ |

---

## 分析

### 同平台对比（Ascend 910B3）

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| output_throughput | 1296.05 tok/s | 1255.38 tok/s | **+3.2%** |
| total_throughput | 1458.05 tok/s | 1412.31 tok/s | **+3.2%** |
| elapsed_s | 50.57 s | 52.20 s | **-3.1%** |

- nano-vllm-ascend 在 910B3 上比 vllm-ascend 快约 3.2%。
- vllm-ascend 启动耗时更长（torch.compile + ACL graph 捕获约 28s），但推理吞吐差距不大。

### 跨平台对比（910B3 vs 5090 Laptop）

| 指标 | nano-vllm-ascend (910B3) | vllm (5090) | 差距 |
|------|-------------------------|-------------|------|
| output_throughput | 1296.05 tok/s | 2262.88 tok/s | 5090 快 **74.6%** |
| total_throughput | 1458.05 tok/s | 2545.73 tok/s | 5090 快 **74.6%** |

- RTX 5090 Laptop 的吞吐量约为 910B3 的 1.75 倍。
- 需要注意：5090 Laptop 是消费级旗舰 GPU，910B3 是数据中心推理卡，定位不同。
- 5090 的 CUDA 生态更成熟（FlashAttention、torch.compile + Inductor），而 910B3 依赖 CANN + torch_npu，算子优化空间仍然较大。

### nano-vllm vs vllm 同平台对比

| 平台 | nano-vllm | vllm | 差异 |
|------|-----------|------|------|
| 5090 GPU | 2257.55 tok/s | 2262.88 tok/s | 基本持平（-0.2%） |

- 在 GPU 上 nano-vllm 与 vllm 吞吐几乎相同，说明 nano-vllm 的调度实现已经足够高效。

---

## 结论

1. **nano-vllm-ascend 在 910B3 上已超越 vllm-ascend**，吞吐提升约 3.2%。
2. **与 GPU 的差距主要来自硬件算力和软件生态**，而非推理引擎本身（nano-vllm 在 GPU 上与 vllm 持平）。
