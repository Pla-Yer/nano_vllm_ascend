# NanoVLLM Ascend 架构设计

## 系统概览

NanoVLLM Ascend 是一个专为华为 Ascend NPU 设计的轻量级 LLM 推理引擎，支持 Qwen3 模型的 batch prefill 和 paged decode 推理。

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (User API)                         │
│                              LLM                                 │
│                    (engine.py - 高层接口)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       运行时层 (ModelRunner)                     │
│  - 模型加载 (HF → Qwen3ForCausalLM)                              │
│  - Tokenizer 管理                                                │
│  - Prefill/Decode 流程控制                                       │
│  - KV Cache 生命周期管理                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Qwen3      │  │ PagedKVCache │  │   Attention      │
│   Model      │  │   Manager    │  │    Layers        │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌────────────────────────────────────────────────────────────────┐
│                      NPU 底层加速层                             │
│  - npu_fused_infer_attention_score (Prefill)                   │
│  - _npu_paged_attention (Decode)                               │
└────────────────────────────────────────────────────────────────┘
```

## 模块结构

```
nanovllm_ascend/
├── engine.py                    # 用户 API (LLM 类)
├── model_runner.py              # 推理运行时
├── models/
│   └── qwen3.py                 # Qwen3 模型实现
│       ├── Qwen3ForCausalLM     # 完整模型
│       ├── Qwen3Model           # 编码器
│       ├── Qwen3DecoderLayer    # 解码器层
│       ├── Qwen3Attention       # 注意力机制
│       └── Qwen3MLP             # MLP 层
├── layers/                      # 基础算子层
│   ├── activation.py            # SiLU 激活函数
│   ├── linear.py                # 线性层
│   ├── layernorm.py             # RMSNorm
│   ├── embed_head.py            # Embedding / LMHead
│   ├── rotary_embedding.py      # RoPE 位置编码
│   ├── npu_prefill_attention.py # NPU Batch Prefill Attention
│   └── npu_paged_attention.py   # NPU Paged Attention
└── npu/
    └── paged_kv_cache.py        # PagedAttention KV Cache 管理
```

## 核心组件详解

### 1. LLM (用户 API)

```
┌─────────────────────────────────────────┐
│                  LLM                    │
│  ┌─────────────────────────────────┐   │
│  │  __init__(model_path, ...)      │   │
│  │  - 初始化 ModelRunner           │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  generate(prompts, max_tokens)  │   │
│  │  1. prefill() - 初始提示处理     │   │
│  │  2. decode() 循环 - 逐词生成     │   │
│  │  3. decode_token_ids() - 解码    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2. ModelRunner (推理运行时)

```
┌─────────────────────────────────────────────────────────────┐
│                      ModelRunner                             │
├─────────────────────────────────────────────────────────────┤
│  状态：                                                      │
│  - model: Qwen3ForCausalLM                                  │
│  - tokenizer: AutoTokenizer                                 │
│  - kv_cache: PagedKVCache                                   │
│  - config: 模型配置                                         │
├─────────────────────────────────────────────────────────────┤
│  prefill(prompts):                                          │
│    1. _new_kv_cache() - 分配 KV Cache                       │
│    2. _tokenize_prompts() - 批量化 tokenize                  │
│    3. prepare_prefill_metadata() - 准备 attention 元数据     │
│    4. model.forward(is_prefill=True) - 前向传播             │
│    5. argmax → 返回下一个 token                             │
├─────────────────────────────────────────────────────────────┤
│  decode(active_slots, token_ids, cache_positions):          │
│    1. prepare_metadata() - 准备 decode 元数据                │
│    2. model.forward(is_prefill=False) - 前向传播            │
│    3. argmax → 返回下一个 token                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Qwen3 模型架构

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
└─────────────────────────────────────────────────────────────┘
```

### 4. Qwen3Attention (双路径注意力)

```
┌─────────────────────────────────────────────────────────────┐
│                     Qwen3Attention                           │
├─────────────────────────────────────────────────────────────┤
│  Projection:                                                 │
│  q_proj, k_proj, v_proj → Q, K, V                           │
│  q_norm, k_norm → 归一化                                     │
│  RoPE → 位置编码                                             │
├─────────────────────────────────────────────────────────────┤
│  Prefill 路径 (is_prefill=True):                             │
│    kv_cache.write_prefill()                                 │
│    NpuBatchPrefillAttention(Q, K, V)                        │
│    → npu_fused_infer_attention_score                        │
├─────────────────────────────────────────────────────────────┤
│  Decode 路径 (is_prefill=False):                             │
│    kv_cache.write_decode()                                  │
│    NpuPagedAttention(Q, key_cache, value_cache)             │
│    → _npu_paged_attention                                   │
├─────────────────────────────────────────────────────────────┤
│  o_proj → 输出                                               │
└─────────────────────────────────────────────────────────────┘
```

### 5. PagedKVCache (内存管理)

```
┌─────────────────────────────────────────────────────────────┐
│                     PagedKVCache                             │
├─────────────────────────────────────────────────────────────┤
│  物理存储：                                                  │
│  key_cache:   [num_layers, num_blocks, block_size,          │
│                num_kv_heads, head_dim]                       │
│  value_cache: 同上                                          │
├─────────────────────────────────────────────────────────────┤
│  逻辑映射：                                                  │
│  block_tables[seq_id] = [block_idx1, block_idx2, ...]       │
│  seq_lens[seq_id] = 当前序列长度                            │
│  free_blocks = 空闲 block 池                                  │
├─────────────────────────────────────────────────────────────┤
│  核心方法：                                                  │
│  - prepare_prefill_metadata() → 批处理元数据                │
│  - prepare_metadata() → Decode 元数据                       │
│  - write_prefill/decode() → KV 写入                         │
│  - get_physical_cache() → 获取物理缓存                      │
└─────────────────────────────────────────────────────────────┘
```

## 数据流图

### Prefill 阶段

```
prompts (list[str])
    │
    ▼
┌──────────────────┐
│ _tokenize_prompts│
└──────────────────┘
    │
    ▼
input_ids_flat [total_tokens]
position_ids_flat [total_tokens]
    │
    ▼
┌────────────────────────────┐
│ prepare_prefill_metadata   │
│ - 分配 block                │
│ - 构建 slot_mapping        │
│ - 构建 actual_seq_lengths  │
└────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ model.forward(is_prefill=True)     │
│ ├─ embed_tokens                    │
│ ├─ for each layer:                 │
│ │   ├─ write_prefill(K, V)         │
│ │   └─ npu_fused_infer_attention   │
│ └─ lm_head                         │
└────────────────────────────────────┘
    │
    ▼
next_tokens (list[int])
```

### Decode 阶段

```
active_slots, token_ids, cache_positions
    │
    ▼
┌─────────────────────────┐
│ prepare_metadata        │
│ - 分配新 block (如需)     │
│ - 更新 block_tables      │
│ - 构建 slot_mapping      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ model.forward(is_prefill=False) │
│ ├─ embed_tokens                 │
│ ├─ for each layer:              │
│ │   ├─ write_decode(K, V)       │
│ │   └─ _npu_paged_attention     │
│ └─ lm_head                      │
└─────────────────────────────────┘
    │
    ▼
next_tokens (list[int])
```

## 关键技术点

| 技术 | 用途 | 实现 |
|------|------|------|
| **PagedAttention** | 显存优化 | 虚拟内存分页，block 级分配 |
| **Batch Prefill** | 批处理加速 | `npu_fused_infer_attention_score` |
| **RoPE** | 位置编码 | `apply_rotary_pos_emb_tnd` |
| **QK Norm** | 训练稳定性 | Qwen3 特有，对 Q/K 归一化 |
| **SwiGLU** | 激活函数 | SiLU(gate) * up |

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
                                     └─────────────────┘

┌─────────────┐     ┌─────────────┐
│  nanovllm   │────▶│ transformers│
│  _ascend    │     │  (HF)       │
└─────────────┘     └─────────────┘
                       │
                       ├─ AutoTokenizer
                       ├─ AutoConfig
                       └─ AutoModelForCausalLM (权重加载)
```
