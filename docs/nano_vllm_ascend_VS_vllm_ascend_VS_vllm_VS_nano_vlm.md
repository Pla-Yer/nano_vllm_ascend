# 跨后端吞吐量对比

> 模型：Qwen3-0.6B · 使用随机 token-id（无 tokenizer 开销）
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
  --output-json bench_outputs/vllm_32_256_2048.json

# nano-vllm (NVIDIA GPU)
python examples/bench_compare.py --backend nanovllm \
  --model-path /home/test/huggingface/Qwen3-0.6B/qwen/Qwen3-0.6B/ \
  --num-prompts 32 --input-len 256 --output-len 2048 \
  --max-num-seqs 32 --max-model-len 4096 --ignore-eos \
  --output-json bench_outputs/nanovllm_32_256_2048.json
```

---

## 结果汇总

### 1. 跨后端对比（32 请求 × 256 输入 / 2048 输出）

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

### 2. Ascend 同平台多工作负载对比

#### 2a. Prefill-heavy：32 请求 × 1024 输入 / 128 输出

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| elapsed_s | 3.34 | 3.12 | — |
| **output_throughput (tok/s)** | **1227.68** | **1313.09** | **-6.5%** |
| **total_throughput (tok/s)** | **11049.13** | **11817.85** | **-6.5%** |
| HBM peak (GB) | 50.85 | — | — |

> Prefill-heavy 场景下 vllm-ascend 的 `torch.compile` + FIA 融合更高效，nano 的逐层 prefill 路径有优化空间。

#### 2b. Decode-heavy：32 请求 × 128 输入 / 1024 输出

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| elapsed_s | 22.55 | 24.40 | **-7.6%** |
| **output_throughput (tok/s)** | **1453.11** | **1343.21** | **+8.2%** |
| **total_throughput (tok/s)** | **1634.74** | **1511.11** | **+8.2%** |
| HBM peak (GB) | 50.22 | — | — |

> Decode-heavy 场景下 ACL Graph replay 优势显著，nano 领先 8.2%。

#### 2c. 均衡负载：32 请求 × 2048 输入 / 2048 输出

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| elapsed_s | 63.42 | 67.22 | **-5.7%** |
| **output_throughput (tok/s)** | **1033.33** | **974.98** | **+6.0%** |
| **total_throughput (tok/s)** | **2066.66** | **1949.96** | **+6.0%** |
| HBM peak (GB) | 51.61 | — | — |
| ACL Graph replays | 4094 | — | — |

> 均衡负载下 nano 依靠 ACL Graph 在 decode 阶段的加速，整体领先 6.0%。

#### 2d. 大批量：100 请求 × 2048 输入 / 2048 输出

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| elapsed_s | 145.22 | 110.36 | +31.6% |
| **output_throughput (tok/s)** | **1410.23** | **1855.68** | **-24.0%** |
| **total_throughput (tok/s)** | **2820.45** | **3711.35** | **-24.0%** |
| HBM peak (GB) | 54.90 | — | — |
| ACL Graph replays | 4094 | — | — |

> 大批量下 vllm-ascend 的 `torch.compile` 动态 batch 适配更灵活，ACL Graph 在 batch=100 时的捕获/更新开销和 prefill 调度效率成为瓶颈。

#### 2e. 大批量：128 请求 × 2048 输入 / 2048 输出

| 指标 | nano-vllm-ascend | vllm-ascend | nano 优势 |
|------|-----------------|-------------|----------|
| elapsed_s | 204.81 | 173.41 | +18.1% |
| **output_throughput (tok/s)** | **1279.95** | **1511.69** | **-15.3%** |
| **total_throughput (tok/s)** | **2559.90** | **3023.37** | **-15.3%** |
| HBM peak (GB) | 58.01 | — | — |
| ACL Graph captures | 2 (bs=24,104) | — | — |
| ACL Graph replays | 8188 | — | — |

> 与 100 批次类似，大批量下 vllm-ascend 的连续批处理调度更高效。

### 3. ACL Graph 加速效果（E2E 微基准）

| 配置 | decode tok/s | total tok/s | 加速比 |
|------|-------------|-------------|--------|
| baseline (bs=1, 无 graph) | 18.95 | 23.53 | 1.0× |
| ACL Graph (bs=1) | 100.79 | 122.38 | **5.3× decode** |
| ACL Graph (bs=4) | 372.42 | 453.09 | **19.7× decode** |

> ACL Graph 对 decode 阶段加速极为显著，bs=1 时 decode 吞吐提升 5.3 倍，bs=4 时提升近 20 倍。

---

## 分析

### 同平台对比（Ascend 910B3）— 按工作负载特征

| 工作负载特征 | nano 优势 | 原因 |
|-------------|----------|------|
| Decode-heavy（短输入长输出） | **+8.2%** | ACL Graph replay 加速 decode |
| 均衡（等长输入输出） | **+3.2% ~ +6.0%** | ACL Graph 在 decode 占比高的场景有优势 |
| Prefill-heavy（长输入短输出） | **-6.5%** | 逐层 prefill 路径不如 torch.compile FIA 融合 |
| 大批量（100+ 请求） | **-15% ~ -24%** | ACL Graph 固定 batch 捕获 + 调度开销 |

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

1. **nano-vllm-ascend 在 decode-heavy 和均衡负载场景下超越 vllm-ascend**，吞吐提升 3.2%~8.2%，主要归功于 ACL Graph replay 对 decode 的加速。
2. **在 prefill-heavy 和大批量场景下 vllm-ascend 更快**，其 `torch.compile` + FIA 融合以及动态 batch 适配能力更强。
3. **ACL Graph 是 nano 的核心加速手段**：bs=1 decode 加速 5.3×，bs=4 加速 19.7×，但固定 batch 捕获限制了大批量灵活性。
4. **与 GPU 的差距主要来自硬件算力和软件生态**，而非推理引擎本身（nano-vllm 在 GPU 上与 vllm 持平）。
