# nano-vLLM-Ascend Decode ACL Graph 规划、分析与实现记录

> 适用项目：`nanovllm_ascend_m6`  
> 适用阶段：decode-only ACL graph / `_npu_paged_attention` graph task update  
> 记录日期：2026-06-09  
> 测试模型：Qwen3-0.6B  
> 核心结论：decode full graph 不能只捕获 Python model forward；`_npu_paged_attention` 的 `context_lens` 必须保持 CPU tensor，需要参考 vLLM-Ascend 在 full graph 下的 graph task group 捕获与 replay 前 task update。

---

## 1. 背景与目标

当前项目已经是一个最小 Ascend NPU 推理 runtime：

- prefill 使用 `torch_npu.npu_fused_infer_attention_score`
- decode 使用 `torch_npu._npu_paged_attention`
- KV cache 使用 paged block table
- 真实路径是 `LLM.generate()` / `LLM.step()` / `ModelRunner.prefill()` / `ModelRunner.decode()`

在现有性能记录中，小 batch decode 的主要瓶颈已经不只是算子计算本身，而是大量小算子和 Python/eager launch 开销。Graph 的目标是把 decode 阶段中稳定的模型执行图捕获下来，通过 replay 降低每个 token 的调度成本。

本阶段目标不是实现完整 vLLM-Ascend graph 系统，而是实现一个适合本项目边界的 v1：

- 只做 decode graph
- 默认关闭，通过 `enable_decode_graph=True` 显式启用
- prefill、prefix-cache prefill、chunked prefill、mixed prefill/decode 不进入 graph
- 不引入 `torch.compile`、Npugraph_EX、piecewise graph 或 vLLM 全套 forward context
- 保持当前 `_npu_paged_attention` decode 后端，不切换 attention backend

---

## 2. 为什么不能只用普通 ACL Graph

最初直觉可能是把 `Qwen3ForCausalLM.forward(..., is_prefill=False)` 放进 `torch.npu.NPUGraph`：

```text
capture:
  model(input_ids, position_ids, kv_cache, attn_metadata, is_prefill=False)

replay:
  copy new input ids / positions / metadata
  graph.replay()
```

但这个方案对 Ascend paged attention 不够。

decode attention 调用是：

```python
torch_npu._npu_paged_attention(
    query=q,
    key_cache=key_cache,
    value_cache=value_cache,
    num_kv_heads=...,
    num_heads=...,
    scale_value=...,
    block_table=block_tables,
    context_lens=context_lens,
    out=output,
)
```

其中 `context_lens` 在当前 torch_npu 接口中必须是 CPU tensor。它每个 decode step 都会变化：

```text
step 1: context_lens = [prompt_len + 1]
step 2: context_lens = [prompt_len + 2]
...
```

如果只做普通 graph replay，attention task 内部看到的 CPU 参数不会按新 step 正确更新。vLLM-Ascend 的 full graph 也专门处理了这个问题：capture 阶段把 paged attention 包在 graph task group 中保存 handle，replay 前重新用当前 `seq_lens/context_lens/workspace` 调一次 `_npu_paged_attention` 来 update 该 graph task。

因此本项目的正确方向是：

```text
model decode forward:      ACL graph capture/replay
_npu_paged_attention:      graph_task_group capture
attention CPU/context参数: replay 前 graph_task_update
```

---

## 3. vLLM-Ascend 参考点

本实现参考 vLLM-Ascend 的机制，而不是照搬其完整架构。

主要参考点：

- 文档：`ACL Graph` design
  - https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/ACL_Graph.html
- 文档：`Graph Mode Guide`
  - https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html
- 源码模式：`vllm_ascend/attention/attention_v1.py`
  - `full_graph_pa`
  - `update_graph_params`
  - `torch.npu.graph_task_group_begin/end`
  - `torch.npu.graph_task_update_begin/end`
  - `_npu_paged_attention_get_workspace`

vLLM-Ascend 的关键语义可以概括为：

1. full graph capture 阶段，paged attention 不是普通 eager op，而是 graph task group。
2. capture 时保存 attention task handle。
3. replay 前根据当前 decode metadata 更新 attention params。
4. `workspace` 也随 task params 一起维护。
5. 使用 stream / ExternalEvent 保证 task update 与 graph replay 的顺序。

本项目只取这条必要路径，不引入 vLLM 的平台层、配置层、worker 层和复杂 forward context。

---

## 4. 当前项目设计

### 4.1 Public API

新增两个 opt-in 参数：

```python
llm = LLM(
    model_path,
    max_num_seqs=4,
    enable_decode_graph=True,
    decode_graph_batch_sizes=None,
)
```

语义：

- `enable_decode_graph=False`：默认行为，完全走 eager decode
- `enable_decode_graph=True`：decode 尝试使用 ACL graph
- `decode_graph_batch_sizes=None`：默认捕获 `1..max_num_seqs` 的精确 batch size
- `decode_graph_batch_sizes=[1, 2]`：只捕获指定 decode batch size

v1 采用精确 batch size capture，不做 padding bucket。原因是当前 runtime 追求最小实现，padding dummy row 会引入额外 slot、KV 写入和采样边界问题。

### 4.2 模块划分

新增模块：

```text
src/nanovllm_ascend/npu/acl_graph.py
```

核心对象：

- `DecodeGraphRunner`
  - 管理每个 decode batch size 的 capture/replay
  - 捕获失败时只禁用该 batch size
  - 维护 captures / replays / updates / fallbacks / capture_failures 计数
- `DecodeGraphEntry`
  - 保存静态输入 tensor、静态 decode metadata、`NPUGraph`、update stream、logits 输出和 paged attention task 列表
- `PagedAttentionGraphTask`
  - 保存每层 `_npu_paged_attention` task update 所需参数
- `capture_decode_graph_tasks(...)`
  - 一个窄的 graph capture context，只在 decode graph capture 期间激活

改动模块：

- `engine.py`
  - `LLM.__init__` 增加 graph 参数并传给 runner
- `model_runner.py`
  - `ModelRunner.__init__` 创建 `DecodeGraphRunner`
  - `ModelRunner.decode()` 在 graph 可用时走 graph runner，否则保留 eager path
  - `decode_graph_stats()` 暴露 graph 计数
- `layers/npu_paged_attention.py`
  - eager 路径保持 `_npu_paged_attention`
  - graph capture context 激活时，改为 graph task group capture 并记录 task handle
- `examples/bench.py`
  - 增加 `--enable-decode-graph`
  - 打印 decode graph stats

### 4.3 Decode 路径

默认 eager decode 不变：

```text
ModelRunner.decode()
  -> append_next_token()
  -> prepare_decode_metadata()
  -> model(..., is_prefill=False)
  -> sampler.sample()
  -> seq.set_decode_result()
```

graph decode：

```text
ModelRunner.decode()
  -> append_next_token()
  -> prepare_decode_metadata()
  -> DecodeGraphRunner.forward()
       -> 首次 batch size: capture
       -> 后续 batch size: copy inputs + update attention tasks + replay
  -> sampler.sample()
  -> seq.set_decode_result()
```

sampling 不进入 graph。这样可以保持 sampling 参数、随机性和输出处理逻辑简单。

### 4.4 Capture 流程

首次遇到某个 decode batch size 时：

```text
1. clone input_ids / position_ids / block_tables / context_lens / slot_mapping
2. 创建 torch.npu.NPUGraph
3. 进入 capture_decode_graph_tasks(entry)
4. 执行 model(..., is_prefill=False)
5. 每层 NpuPagedAttention 捕获 graph task group:
   - 获取 workspace
   - graph_task_group_begin(stream)
   - _npu_paged_attention(..., workspace=workspace)
   - graph_task_group_end(stream)
   - 保存 handle / workspace / query / cache / block_table / context_lens / output
6. 保存 logits tensor
```

如果 capture 失败：

```text
disabled_batch_sizes.add(batch_size)
fallbacks += 1
capture_failures += 1
```

只禁用该 batch size，其他 batch size 仍可捕获。

### 4.5 Replay 流程

后续遇到相同 batch size：

```text
1. copy 当前 input_ids 到静态 input_ids
2. copy 当前 position_ids 到静态 position_ids
3. copy 当前 block_tables / context_lens / slot_mapping 到静态 metadata
4. 对每层 paged attention task 执行 graph_task_update:
   - 重新获取 workspace
   - graph_task_update_begin(update_stream, handle)
   - _npu_paged_attention(... 当前 context_lens/workspace ...)
   - graph_task_update_end(update_stream)
5. update stream 与 replay stream 做顺序同步
6. graph.replay()
7. 返回静态 logits
```

这里最重要的是第 4 步。它保证 CPU `context_lens` 和 attention workspace 在 replay 前变成当前 decode step 的值。

### 4.6 Block table 宽度

普通 eager metadata 中，`block_tables` 的列数等于当前 active seq 最大 block 数。decode 过程中序列跨 block 后，宽度可能增长。

graph tensor shape 不能随 replay 改变，所以 graph entry 使用固定宽度：

```text
max_blocks_per_seq = ceil(max_model_len / block_size)
```

capture/replay 时把当前较窄的 block table copy 到固定宽度 tensor 的前几列，其余列置 0。这样 block table 地址和 shape 稳定，同时仍保留当前 step 的真实 block id。

---

## 5. 实测结果

### 5.1 Generate

用户实测：

```text
generate throughput: 22 tok/s -> 42 tok/s
```

这说明 graph replay 对端到端 generate 有直接收益。generate 中仍包含 tokenization、prefill、sampling 和输出处理，因此端到端 throughput 不会等同于纯 decode graph 的收益，但已经能看到约 1.9x 提升。

### 5.2 Bench

测试命令：

```powershell
python examples/bench.py `
  --model-path /home/player/models/Qwen3/Qwen/Qwen3-0___6B/ `
  --batch-size 1 `
  --max-new-tokens 128 `
  --iters 5 `
  --output-json bench_outputs/test.json `
  --enable-decode-graph
```

运行配置：

```text
batch_size=1
max_num_seqs=1
max_new_tokens=128
max_model_len=2048
block_size=128
num_blocks=3359
enable_prefix_cache=False
enable_decode_graph=True
warmup_iters=1
iters=5
model_resident_hbm_gb=47.165
```

5 次结果：

| iter | prefill_s | decode_s | total_s | decode_tok/s | output_tok/s | total_tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0353 | 1.5389 | 1.5741 | 82.53 | 81.31 | 101.01 |
| 2 | 0.0350 | 1.5280 | 1.5631 | 83.11 | 81.89 | 101.72 |
| 3 | 0.0360 | 1.5336 | 1.5696 | 82.81 | 81.55 | 101.30 |
| 4 | 0.0354 | 1.5262 | 1.5616 | 83.21 | 81.97 | 101.82 |
| 5 | 0.0355 | 1.5404 | 1.5759 | 82.45 | 81.22 | 100.90 |

summary：

```text
prepare_s: avg=0.0009
prefill_s: avg=0.0354
decode_s: avg=1.5334
total_s: avg=1.5689
prefill_tok_s: avg=874.7061
decode_tok_s: avg=82.8222
output_tok_s: avg=81.5885
total_tok_s: avg=101.3482
baseline_allocated_gb: avg=49.4625
peak_allocated_gb: avg=49.5445
peak_incremental_gb: avg=0.0820
```

decode graph stats：

```text
captures: 1
replays: 762
updates: 762
fallbacks: 0
capture_failures: 0
```

这些计数很关键：

- `captures=1`：batch size 1 只捕获一次
- `replays=762`：后续 decode step 走 graph replay
- `updates=762`：每次 replay 前都执行了 paged attention task update
- `fallbacks=0`：实测路径没有退回 eager decode
- `capture_failures=0`：graph capture 成功

因此这次结果证明的不是“代码里有 graph 开关”，而是真实 decode 路径完成了 capture、attention task update 和 replay。

---

## 6. 性能解读

这次 bench 的核心变化是 decode：

```text
decode_s avg = 1.5334s
decode_tok/s avg = 82.8222
```

此前性能记录中 batch size 1 decode 大约在 28 tok/s 到 29 tok/s 附近。graph 后 decode 达到 82 tok/s 以上，说明瓶颈确实包含大量 eager launch / 小算子调度成本。

prefill 基本保持独立：

```text
prefill_s avg = 0.0354s
prefill_tok/s avg = 874.7
```

这符合设计预期：本阶段只优化 decode，不改变 prefill kernel、prefix cache 或 prefill metadata。

HBM 增量：

```text
peak_incremental_gb avg = 0.0820GB
```

decode graph 引入了静态 input/metadata/output tensor、graph task workspace 和 graph runtime 状态，但 batch size 1 下额外峰值较小，属于可接受范围。

---

## 7. 当前限制

当前实现仍是 v1，不应扩大解释范围：

- 只验证了 decode-only graph
- 只展示了 batch size 1 的 bench 结果
- prefill 没有 graph
- prefix-cache paged prefill 没有 graph
- mixed prefill/decode 没有 graph
- 暂未实现 vLLM 式 padding bucket
- 暂未实现 graph warmup policy、capture size 上限和运行时动态关闭策略
- 暂未把 graph stats 写入所有 example / server 输出

另外，`context_lens` 必须保持 CPU 的约束仍然存在。后续如果 torch_npu 接口变化，attention update 逻辑需要重新验证。

---

## 8. 后续计划

建议按以下顺序继续：

1. 扩展 correctness smoke：
   - graph off/on token ids 对比
   - batch size 1、2、4 对比
   - decode 跨 block 边界对比
2. 扩展 benchmark：
   - batch size 1/2/4
   - max_new_tokens 32/128/512
   - graph off/on 同脚本输出对比
3. 增加 graph stats 到 JSON：
   - captures
   - replays
   - updates
   - fallbacks
   - capture_failures
4. 再考虑 batch bucket：
   - 只有当 batch size 变化频繁、精确 capture 数量过多时再做
5. 最后再考虑 prefix-cache prefill graph：
   - 这条路径涉及 FIA paged prefill、workspace 和 `actual_seq_lengths_kv`
   - 复杂度高于 decode-only graph，不建议和当前 v1 混在一起

---

## 9. 结论

本项目的 graph v1 应明确定位为：

```text
decode-only full graph replay
+ paged attention graph task update
+ exact batch size capture
+ eager fallback
```

它不是通用 graph 框架，也不是 vLLM-Ascend 的完整复刻。当前实测已经显示该方向有效：generate throughput 从约 22 tok/s 提升到约 42 tok/s；bench 中 batch size 1 的 decode throughput 达到约 82.8 tok/s，并且 graph stats 显示 capture/replay/update 路径全部命中，没有 eager fallback。

