# NanoVLLM Ascend 架构设计

## 系统概览

NanoVLLM Ascend 是一个专为华为 Ascend NPU 设计的轻量级 LLM 推理引擎，支持 Qwen3 模型的连续批处理推理，具备 Paged KV Cache、Prefix Cache 复用和 Decode ACL Graph 加速等特性。

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User API)                         │
│              LLM / OpenAIChatService (/v1/chat/completions)      │
│                    (engine.py / openai_server.py)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    调度层 (MiniScheduler)                        │
│  - 连续批处理调度 (WAITING → RUNNING → FINISHED)                  │
│  - KV Block 预留与释放                                          │
│  - 序列完成检测                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       运行时层 (ModelRunner)                     │
│  - 模型加载 (HF → Qwen3ForCausalLM)                              │
│  - Tokenizer 管理                                                │
│  - Prefill/Decode 流程控制                                       │
│  - Prefix Cache 查找与序列准备                                    │
│  - Decode Graph 管理                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Qwen3      │  │ BlockManager │  │   Attention      │
│   Model      │  │ PagedKVCache │  │    Layers        │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌────────────────────────────────────────────────────────────────┐
│                      NPU 底层加速层                             │
│  - npu_fused_infer_attention_score (Prefill)                   │
│  - _npu_paged_attention (Decode)                               │
│  - npu_rms_norm (RMSNorm)                                      │
│  - npu_rotary_mul (RoPE)                                       │
└────────────────────────────────────────────────────────────────┘
```

## 模块结构

```
nanovllm_ascend/
├── __init__.py                  # 懒加载导出: LLM, EngineCore, SamplingParams
├── engine.py                    # 用户 API (LLM) + 推理核心 (EngineCore)
├── scheduler.py                 # 连续批处理调度器 (MiniScheduler)
├── model_runner.py              # 推理运行时 (ModelRunner)
├── sequence.py                  # 序列状态 (Sequence, SequenceStatus)
├── sampling_params.py           # 采样参数 (SamplingParams)
├── openai_server.py             # OpenAI 兼容服务 (FastAPI)
├── models/
│   ├── __init__.py
│   └── qwen3.py                 # Qwen3 模型实现
│       ├── Qwen3ForCausalLM     # 完整模型 (含 logits_indices 优化)
│       ├── Qwen3Model           # 编码器
│       ├── Qwen3DecoderLayer    # 解码器层
│       ├── Qwen3Attention       # 双路径注意力 (prefill/decode)
│       └── Qwen3MLP             # MLP 层 (SwiGLU)
├── layers/                      # 基础算子层
│   ├── __init__.py              # 懒加载导出
│   ├── activation.py            # SiLU 激活函数
│   ├── linear.py                # 线性层
│   ├── layernorm.py             # RMSNorm (npu_rms_norm)
│   ├── embed_head.py            # VocabParallelEmbedding / LMHead
│   ├── rotary_embedding.py      # RoPE 位置编码 (npu_rotary_mul)
│   ├── npu_prefill_attention.py # NPU Batch Prefill Attention (FIA)
│   ├── npu_paged_attention.py   # NPU Paged Attention (decode)
│   └── sampler.py               # 采样器 (greedy/top-k/top-p)
└── npu/
    ├── __init__.py              # 懒加载导出
    ├── block_manager.py         # Block 池管理 + Prefix Cache
    │   ├── BlockManager         # 统一块池管理器
    │   ├── PagedKVCacheMetadata # 注意力元数据
    │   └── BlockState           # 单 block 状态
    ├── paged_kv_cache.py        # 物理分页 KV Cache (TND 布局)
    └── acl_graph.py             # Decode ACL Graph 加速
        └── DecodeGraphRunner    # Graph 捕获/重放管理器
```

## 核心调用链

```
LLM.generate(prompts)
  → scheduler.reset()
  → runner.tokenize_prompts()
  → runner.prepare_sequences()          [prefix cache 查找]
  → scheduler.add_request()
  → LLM.step() 循环
    → EngineCore.step()
      → scheduler.plan_next_step()      [调度: prefill/decode/finish]
      → runner.prefill(seqs)            [FIA 内核, TND 布局]
        → model.forward(is_prefill=True)
          → Qwen3Attention: Q/K/V proj → q_norm/k_norm → RoPE
            → KV cache write → NpuBatchPrefillAttention
      → runner.decode(seqs)             [Paged Attention, 可选 ACL Graph]
        → DecodeGraphRunner.forward() 或 model.forward(is_prefill=False)
          → Qwen3Attention: Q/K/V proj → q_norm/k_norm → RoPE
            → KV cache write → NpuPagedAttention
      → sampler.sample()
```

## 核心组件详解

### 1. LLM (用户 API)

```
┌───────────────────────────────────────────────────┐
│                      LLM                          │
├───────────────────────────────────────────────────┤
│  __init__(model_path, max_model_len, block_size,  │
│           num_blocks, max_num_seqs, device_id,    │
│           npu_memory_utilization,                 │
│           enable_prefix_cache, enable_decode_graph)│
├───────────────────────────────────────────────────┤
│  generate(prompts, max_new_tokens, ...)           │
│    1. tokenize + prepare_sequences (prefix cache) │
│    2. step() 循环 → prefill → decode → ...        │
│    3. decode_token_ids() → 返回文本                │
├───────────────────────────────────────────────────┤
│  submit(prompt, ...)     # 异步提交单个请求        │
│  step()                  # 推进一步               │
│  has_unfinished()        # 是否有未完成序列       │
│  warm()                  # 预热                   │
│  clear_prefix_cache()    # 清空前缀缓存           │
└───────────────────────────────────────────────────┘
```

### 2. MiniScheduler (连续批处理调度器)

```
┌───────────────────────────────────────────────────┐
│                  MiniScheduler                     │
├───────────────────────────────────────────────────┤
│  状态：                                           │
│  - waiting: list[Sequence]    等待队列            │
│  - running: list[Sequence]    运行中序列          │
│  - block_manager: BlockManager                    │
├───────────────────────────────────────────────────┤
│  reset()                     # 清空所有状态       │
│  add_request(seq)            # 加入等待队列       │
│  plan_next_step(eos_token_id)                    │
│    → SchedulerStep(prefill_seqs, decode_seqs,    │
│                    finish_seqs)                   │
│    1. 检查运行中序列是否完成                       │
│    2. 从等待队列接纳新请求 (block 容量允许)        │
│    3. 收集 decode 序列                            │
│  finish_sequences()          # 释放 block 预留    │
│  has_unfinished()            # 是否还有未完成序列 │
└───────────────────────────────────────────────────┘
```

### 3. ModelRunner (推理运行时)

```
┌─────────────────────────────────────────────────────────────┐
│                      ModelRunner                             │
├─────────────────────────────────────────────────────────────┤
│  状态：                                                      │
│  - model: Qwen3ForCausalLM                                  │
│  - tokenizer: AutoTokenizer                                 │
│  - kv_cache: PagedKVCache                                   │
│  - block_manager: BlockManager                               │
│  - decode_graph_runner: DecodeGraphRunner (可选)             │
│  - config: 模型配置                                         │
├─────────────────────────────────────────────────────────────┤
│  _load_model()            # HF 权重 → 自定义 Qwen3           │
│  _num_blocks_from_memory()  # NPU 可用内存 → KV block 数     │
│  tokenize_prompts()       # chat template 分词               │
│  prepare_sequences()      # 序列准备 + prefix cache 查找     │
│  prefill(seqs)            # Prefill 推理                     │
│  decode(seqs)             # Decode 推理 (优先 decode graph)  │
│  free_seq(seq)            # 释放序列 block 资源              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Qwen3 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Qwen3ForCausalLM                           │
├─────────────────────────────────────────────────────────────┤
│  input_ids → [embed_tokens] → hidden_states                 │
│                           │                                 │
│                           ▼                                 │
│                    ┌─────────────┐                         │
│                    │  position   │                         │
│                    │  embeddings │  (RoPE)                 │
│                    └─────────────┘                         │
│                           │                                 │
│         ┌─────────────────┴─────────────────┐              │
│         ▼                                   ▼              │
│  ┌──────────────────────────────────────────────────┐     │
│  │           num_hidden_layers 次迭代               │     │
│  │  ┌────────────────────────────────────────────┐ │     │
│  │  │         Qwen3DecoderLayer                  │ │     │
│  │  │  ┌──────────────────────────────────────┐ │ │     │
│  │  │  │ input_layernorm → self_attn → residual│ │ │     │
│  │  │  │ post_attn_layernorm → mlp → residual  │ │ │     │
│  │  │  └──────────────────────────────────────┘ │ │     │
│  │  └────────────────────────────────────────────┘ │     │
│  └──────────────────────────────────────────────────┘     │
│                           │                                 │
│                           ▼                                 │
│                    [norm] → [lm_head] → logits            │
│                                                           │
│  优化: logits_indices 参数，prefill 时只计算 last token    │
└─────────────────────────────────────────────────────────────┘
```

### 5. Qwen3Attention (双路径注意力)

```
┌─────────────────────────────────────────────────────────────┐
│                     Qwen3Attention                           │
├─────────────────────────────────────────────────────────────┤
│  Projection:                                                 │
│  q_proj, k_proj, v_proj → Q, K, V                           │
│  q_norm, k_norm → QK 归一化 (Qwen3 特有)                    │
│  RoPE → 位置编码                                             │
├─────────────────────────────────────────────────────────────┤
│  Prefill 路径 (is_prefill=True):                             │
│    kv_cache.write() → TND 写入                               │
│    NpuBatchPrefillAttention(Q, K, V)                        │
│    → npu_fused_infer_attention_score (sparse_mode=3)        │
│    支持 dense prefill 和 paged prefill (cache hit)          │
├─────────────────────────────────────────────────────────────┤
│  Decode 路径 (is_prefill=False):                             │
│    kv_cache.write() → TND 写入                               │
│    NpuPagedAttention(Q, key_cache, value_cache)             │
│    → _npu_paged_attention                                   │
│    支持 ACL Graph 加速 (capture + replay)                   │
├─────────────────────────────────────────────────────────────┤
│  o_proj → 输出                                               │
└─────────────────────────────────────────────────────────────┘
```

### 6. BlockManager (块池管理 + Prefix Cache)

```
┌─────────────────────────────────────────────────────────────┐
│                     BlockManager                             │
├─────────────────────────────────────────────────────────────┤
│  Block 状态：                                                │
│  - BlockState: block_id, ref_count, block_hash              │
│  - BlockHash = (parent_hash, token_tuple)  递归哈希         │
│  - free_blocks: 空闲 block 池                                │
├─────────────────────────────────────────────────────────────┤
│  核心方法：                                                  │
│  - _alloc_block() / _release_block()  # 分配/释放           │
│  - find_longest_prefix_blocks()       # 前缀缓存查找       │
│  - cache_full_blocks()                # 缓存完整 block 哈希 │
│  - free_slot()                        # 释放序列所有 block  │
│  - clear_prefix_cache()               # 清空前缀缓存       │
├─────────────────────────────────────────────────────────────┤
│  元数据生成：                                                │
│  - prepare_prefill_metadata()  # Prefill: block_tables,    │
│  - prepare_decode_metadata()   #   context_lens, slot_map, │
│  │                               actual_seq_lengths, ...   │
│  - get_block_tables_tensor()                                 │
│  - get_context_lens_tensor()                                 │
│  - get_slot_mapping()                                        │
└─────────────────────────────────────────────────────────────┘
```

### 7. PagedKVCache (物理存储)

```
┌─────────────────────────────────────────────────────────────┐
│                     PagedKVCache                             │
├─────────────────────────────────────────────────────────────┤
│  物理存储 (TND 布局)：                                       │
│  key_cache:   [num_layers, num_blocks, block_size,          │
│                num_kv_heads, head_dim]                       │
│  value_cache: 同上                                          │
├─────────────────────────────────────────────────────────────┤
│  核心方法：                                                  │
│  - write(layer, new_kv, slot_mapping)  # index_copy_ 写入   │
│  - get_physical_cache(layer)           # 获取物理缓存引用   │
└─────────────────────────────────────────────────────────────┘
```

### 8. DecodeGraphRunner (ACL Graph 加速)

```
┌─────────────────────────────────────────────────────────────┐
│                  DecodeGraphRunner                            │
├─────────────────────────────────────────────────────────────┤
│  原理：捕获 decode 计算图，后续 replay 避免重复调度          │
│  支持的 batch_sizes: [1, 2, 4, 8, 16, 32] (默认)           │
├─────────────────────────────────────────────────────────────┤
│  forward():                                                 │
│    首次调用 → _capture(): torch.npu.NPUGraph() 捕获         │
│    后续调用 → graph replay + _update_paged_attention_tasks() │
│    capture 失败 → 禁用该 batch size，回退普通执行           │
├─────────────────────────────────────────────────────────────┤
│  Paged Attention Graph 集成：                                │
│  - capture 时: graph_task_group_begin/end 包裹 PA 调用      │
│  - replay 时: graph_task_update_begin/end 更新 PA 参数      │
│  - record_paged_attention_task() 记录 task 信息             │
└─────────────────────────────────────────────────────────────┘
```

### 9. Sampler (采样器)

```
┌─────────────────────────────────────────────────────────────┐
│                      Sampler                                 │
├─────────────────────────────────────────────────────────────┤
│  sample(logits, sampling_params):                            │
│    temperature ≤ 0 → 贪心 (argmax)                          │
│    temperature > 0 → 缩放 → top_k → top_p → multinomial    │
├─────────────────────────────────────────────────────────────┤
│  SamplingParams:                                             │
│    temperature=0.0, top_k=0 (禁用), top_p=1.0 (禁用)       │
│    is_greedy() / with_overrides()                            │
└─────────────────────────────────────────────────────────────┘
```

### 10. OpenAIChatService (OpenAI 兼容服务)

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenAIChatService                           │
├─────────────────────────────────────────────────────────────┤
│  FastAPI 端点: /v1/chat/completions                         │
│  支持模式：                                                  │
│  - 非流式 (create_chat_completion)                           │
│  - SSE 流式 (stream_chat_completion)                         │
├─────────────────────────────────────────────────────────────┤
│  架构：                                                      │
│  - 后台线程 run_engine() 驱动引擎 step() 循环               │
│  - 请求队列: QueuedRequest → PendingRequest                  │
│  - 并发请求共享同一引擎循环                                  │
└─────────────────────────────────────────────────────────────┘
```

## 数据流图

### Prefill 阶段

```
prompts (list[str])
    │
    ▼
┌──────────────────┐
│ tokenize_prompts  │  (chat template)
└──────────────────┘
    │
    ▼
┌──────────────────────┐
│ prepare_sequences    │  (prefix cache 查找, block 分配)
└──────────────────────┘
    │
    ▼
input_ids_flat [total_tokens]
position_ids_flat [total_tokens]
    │
    ▼
┌────────────────────────────┐
│ prepare_prefill_metadata   │
│ - 构建 slot_mapping        │
│ - 构建 actual_seq_lengths  │
│ - 构建 block_tables        │
└────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ model.forward(is_prefill=True)     │
│ ├─ embed_tokens                    │
│ ├─ for each layer:                 │
│ │   ├─ write(K, V) → TND 写入     │
│ │   └─ npu_fused_infer_attention   │
│ └─ lm_head (logits_indices 优化)   │
└────────────────────────────────────┘
    │
    ▼
┌──────────────────┐
│ sampler.sample() │
└──────────────────┘
    │
    ▼
next_tokens (list[int])
```

### Decode 阶段

```
active_seqs, token_ids, cache_positions
    │
    ▼
┌─────────────────────────┐
│ prepare_decode_metadata  │
│ - 分配新 block (如需)    │
│ - 更新 block_tables      │
│ - 构建 slot_mapping      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ DecodeGraphRunner.forward()         │
│ 或 model.forward(is_prefill=False)  │
│ ├─ embed_tokens                     │
│ ├─ for each layer:                  │
│ │   ├─ write(K, V) → TND 写入      │
│ │   └─ _npu_paged_attention         │
│ └─ lm_head                          │
└─────────────────────────────────────┘
    │
    ▼
┌──────────────────┐
│ sampler.sample() │
└──────────────────┘
    │
    ▼
next_tokens (list[int])
```

## Prefix Cache 流程

```
prepare_sequences()
    │
    ▼
┌────────────────────────────────────┐
│ block_manager.find_longest_prefix  │
│ _blocks(seq.token_ids)            │
│ - 逐 block 计算哈希               │
│ - 匹配已有缓存 block              │
│ - 返回匹配 block 列表 + 匹配长度  │
└────────────────────────────────────┘
    │
    ├── 有匹配 → 复用 block, 只 prefill 后缀
    │
    └── 无匹配 → 完整 prefill
    │
    ▼
prefill 后 → cache_full_blocks()  # 缓存完整 block 哈希
```

## NPU 内核汇总

| 内核 | 位置 | 用途 |
|------|------|------|
| `npu_fused_infer_attention_score` | `layers/npu_prefill_attention.py` | Prefill 批量注意力 (dense + paged cache-hit) |
| `_npu_paged_attention` | `layers/npu_paged_attention.py` | Decode 分页注意力 |
| `_npu_paged_attention_get_workspace` | `layers/npu_paged_attention.py` | 获取 PA workspace (graph capture) |
| `npu_rms_norm` | `layers/layernorm.py` | RMS 归一化 |
| `npu_rotary_mul` | `layers/rotary_embedding.py` | 旋转位置编码乘法 |

## 关键技术点

| 技术 | 用途 | 实现 |
|------|------|------|
| **PagedAttention** | 显存优化 | 虚拟内存分页，block 级分配 |
| **Continuous Batching** | 吞吐优化 | MiniScheduler 动态调度，容量释放后立即接纳新请求 |
| **Prefix Cache** | 前缀复用 | BlockHash 递归哈希，内容寻址 block 复用 |
| **Decode ACL Graph** | Decode 加速 | torch.npu.NPUGraph 捕获/重放，Paged Attention graph task 更新 |
| **Batch Prefill** | 批处理加速 | `npu_fused_infer_attention_score`，input_layout="TND" |
| **RoPE** | 位置编码 | `apply_rotary_pos_emb_tnd`，rotary_mode="half" |
| **QK Norm** | 训练稳定性 | Qwen3 特有，对 Q/K 归一化 |
| **SwiGLU** | 激活函数 | SiLU(gate) * up |
| **TND Layout** | 统一张量布局 | 模型内部唯一布局，KV Cache 物理存储 |

## 依赖关系

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Python    │     │   PyTorch   │     │  torch_npu  │
│   3.10+     │────▶│   2.x       │────▶│  (Ascend)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  CANN NPU Ops   │
                                      │ - fused_infer   │
                                      │ - paged_attn    │
                                      │ - rms_norm      │
                                      │ - rotary_mul    │
                                      └─────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  nanovllm   │────▶│ transformers│     │   FastAPI   │
│  _ascend    │     │  (HF)       │     │  + uvicorn  │
└─────────────┘     └─────────────┘     └─────────────┘
                        │                     │
                        ├─ AutoTokenizer      └─ /v1/chat/completions
                        ├─ AutoConfig
                        └─ AutoModelForCausalLM (权重加载)
```

## 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_block_manager.py` | Block 分配/释放、prefill/decode 元数据、prefix cache 复用 |
| `test_continuous_batching.py` | 连续批处理调度、容量释放后接纳、输出顺序保持 |
| `test_decode_graph.py` | Graph capture/replay、batch size 禁用、API 默认值 |
| `test_openai_server.py` | 消息转换、响应格式、流式 SSE、并发请求 |
| `test_rotary_embedding.py` | RoPE TND 形状和 dtype |
| `test_sampling.py` | 参数验证、贪心/top-k/top-p 采样、参数覆盖 |

## 示例脚本

| 脚本 | 用途 |
|------|------|
| `generate.py` | 基本文本生成 |
| `bench_e2e.py` | 端到端推理基准测试 (prefill/decode 分项计时) |
| `bench_layers.py` | 单层微基准测试 (RMSNorm/Linear/MLP/LMHead/RoPE) |
| `bench_compare.py` | 跨后端吞吐量对比 (nano/nanovllm/vllm) |
| `bench_prefix_cache_speedup.py` | Prefix cache 加速比基准测试 |
| `check_prefix_cache_equivalence.py` | Prefix cache 正确性验证 |
| `api/start_server.py` | OpenAI 兼容服务器启动 |
| `api/chat.py` | 交互式多轮聊天客户端 |
| `api/test_chat_completions.py` | 并发 HTTP 测试客户端 |
