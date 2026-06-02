# Ascend Attention Kernels Metadata 说明

本文档整理 `nano_vllm_ascend` 项目中使用的两个 Ascend NPU attention kernel：

- `torch_npu.npu_fused_infer_attention_score`
- `torch_npu._npu_paged_attention`

这两个 kernel 分别对应 LLM 推理中的两个阶段：

```text
prefill 阶段:
  npu_fused_infer_attention_score

decode 阶段:
  _npu_paged_attention
```

本文重点说明：

- kernel 的作用
- 输入 tensor 的 shape
- metadata 的格式
- metadata 是如何生成的
- 在本项目中的具体调用路径
- 容易踩坑的地方

---

## 1. 总体关系

在本项目中，attention 执行路径如下：

```text
ModelRunner.prefill()
    ↓
Qwen3Attention.forward(is_prefill=True)
    ↓
npu_fused_infer_attention_score
    ↓
写入 PagedKVCache，供后续 decode 使用


ModelRunner.decode()
    ↓
Qwen3Attention.forward(is_prefill=False)
    ↓
先写入当前 token 的 K/V
    ↓
_npu_paged_attention
    ↓
从 PagedKVCache 中读取历史 K/V
```

两个 kernel 的核心区别：

| 阶段 | Kernel | 主要输入 | 主要 metadata | 作用 |
|---|---|---|---|---|
| Prefill | `npu_fused_infer_attention_score` | 当前 prompt 的连续 Q/K/V | `actual_seq_lengths`, `actual_seq_lengths_kv`, `atten_mask` | 对 batch 内多个 prompt 做 fused causal attention |
| Decode | `_npu_paged_attention` | 当前 token 的 Q + paged KV cache | `block_table`, `context_lens` | 根据 block table 从分页 KV cache 中读取历史 K/V 做 attention |

一句话理解：

```text
prefill kernel 解决：TND 拉平后的 token 如何按序列边界做 attention。
decode kernel 解决：分页 KV cache 中的历史 K/V 如何按 block table 读取。
```

---

# 2. `_npu_paged_attention`

## 2.1 作用

`_npu_paged_attention` 用于 decode 阶段。

decode 时，每条序列当前只输入一个 token 的 query，但是 attention 需要看到该序列历史所有 token 的 K/V。因此该 kernel 的核心工作是：

```text
当前 token 的 Q
    ↓
根据 block_table 找到该序列历史 K/V 所在的物理 block
    ↓
根据 context_lens 决定读取多少历史 token
    ↓
从 key_cache/value_cache 读取 K/V
    ↓
计算 attention(Q, K, V)
    ↓
输出当前 token 的 attention output
```

它本身不负责管理 KV cache，也不负责写 KV cache。

KV cache 的分配、写入、block table 维护，由本项目的 `PagedKVCache` 完成。

---

## 2.2 本项目中的调用位置

调用位置：

```text
src/nanovllm_ascend/layers/npu_paged_attention.py
```

核心调用：

```python
torch_npu._npu_paged_attention(
    query=query,
    key_cache=key_cache.contiguous(),
    value_cache=value_cache.contiguous(),
    num_kv_heads=self.num_key_value_heads,
    num_heads=self.num_heads,
    scale_value=self.scale,
    block_table=block_tables,
    context_lens=context_lens,
    out=output,
)
```

调用链：

```text
LLM.generate()
    ↓
ModelRunner.decode()
    ↓
Qwen3ForCausalLM.forward()
    ↓
Qwen3Attention.forward(is_prefill=False)
    ↓
NpuPagedAttention.forward()
    ↓
torch_npu._npu_paged_attention()
```

---

## 2.3 Kernel 输入格式

### 2.3.1 query

格式：

```text
query: [num_tokens, num_heads, head_dim]
```

含义：

- `num_tokens`：当前 decode 轮次参与计算的 token 数量
- 在普通 decode 中，每条 active sequence 一次只 decode 一个 token
- 因此 `num_tokens == active sequence 数量`

示例：

```text
active_slots = [0, 3, 5]

query[0] 对应 seq_slot=0
query[1] 对应 seq_slot=3
query[2] 对应 seq_slot=5

query.shape = [3, num_heads, head_dim]
```

注意：

```text
num_tokens 不是上下文长度。
context_lens 才表示每条序列的历史上下文长度。
```

---

### 2.3.2 key_cache

格式：

```text
key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

本项目完整 KV cache 的格式是：

```text
key_cache:   [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
value_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

每一层调用 attention 时，会取出单层 cache：

```text
key_cache_layer   = key_cache[layer_idx]
value_cache_layer = value_cache[layer_idx]
```

所以传给 `_npu_paged_attention` 的是：

```text
key_cache_layer:   [num_blocks, block_size, num_kv_heads, head_dim]
value_cache_layer: [num_blocks, block_size, num_kv_heads, head_dim]
```

---

### 2.3.3 value_cache

格式：

```text
value_cache: [num_blocks, block_size, num_kv_heads, head_dim_v]
```

在 Qwen3 中通常：

```text
head_dim_v == head_dim
```

所以本项目中可理解为：

```text
value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

---

### 2.3.4 output

格式：

```text
out: [num_tokens, num_heads, head_dim_v]
```

本项目中：

```python
output = torch.empty_like(query)
```

所以：

```text
out: [num_tokens, num_heads, head_dim]
```

---

## 2.4 Metadata：block_table

### 2.4.1 block_table 是什么？

`block_table` 是每条序列的逻辑 block 到物理 block 的映射。

格式：

```text
block_table: [num_tokens, max_blocks_per_seq]
dtype: int32
device: npu
```

其中：

```text
block_table[i, logical_block_idx] = physical_block_id
```

逻辑位置到物理 cache 位置的映射关系：

```text
logical_pos
    ↓
logical_block = logical_pos // block_size
block_offset  = logical_pos % block_size
physical_block = block_table[seq_index, logical_block]
physical_slot = physical_block * block_size + block_offset
```

---

### 2.4.2 block_table 的行顺序

`block_table` 的行顺序必须和 `query` 的行顺序一致。

例如：

```python
active_slots = [0, 3, 5]
```

那么：

```text
query[0]        ↔ seq_slot 0 ↔ block_table[0]
query[1]        ↔ seq_slot 3 ↔ block_table[1]
query[2]        ↔ seq_slot 5 ↔ block_table[2]
context_lens[0] ↔ seq_slot 0
context_lens[1] ↔ seq_slot 3
context_lens[2] ↔ seq_slot 5
```

不是全局 slot 顺序，而是当前 active batch 顺序。

---

### 2.4.3 本项目如何生成 block_table？

位置：

```text
src/nanovllm_ascend/npu/paged_kv_cache.py
```

核心函数：

```python
def get_block_tables_tensor(self, seq_slots) -> torch.Tensor:
    slots = self._normalize_seq_slots(seq_slots)
    max_num_blocks = max(len(self.block_tables[slot]) for slot in slots)

    block_tables = torch.zeros(
        (len(slots), max_num_blocks),
        dtype=torch.int32,
        device=self.device,
    )

    for i, slot in enumerate(slots):
        table = self.block_tables[slot]
        if table:
            block_tables[i, : len(table)] = torch.tensor(
                table,
                dtype=torch.int32,
                device=self.device,
            )

    return block_tables
```

生成结果：

```text
block_tables.shape = [len(active_slots), max_blocks_among_active_seqs]
```

未使用的 padding 位置填 0。

只要：

```text
context_lens[i] <= len(block_table[i]) * block_size
```

padding 部分不会被访问。

---

## 2.5 Metadata：context_lens

### 2.5.1 context_lens 是什么？

`context_lens[i]` 表示第 i 个 decode token 对应序列当前可见的 KV 长度。

它是：

```text
该序列当前已经写入 KV cache 的 token 总数
```

不是新增 token 数。

例如：

```text
prompt 长度 = 300
当前 decode token 写入 position = 300
写入当前 token 后 KV 长度 = 301

context_lens = 301
```

---

### 2.5.2 context_lens 格式

格式：

```text
context_lens: [num_tokens]
dtype: int32
device: cpu
```

本项目中显式转换为 CPU int32：

```python
context_lens = context_lens.to(
    device="cpu",
    dtype=torch.int32,
).contiguous()
```

这点很重要。

如果 decode batch 很小，频繁构造 CPU 侧 `context_lens` 可能成为额外开销来源之一。

---

### 2.5.3 本项目如何生成 context_lens？

位置：

```text
src/nanovllm_ascend/npu/paged_kv_cache.py
```

核心函数：

```python
def get_context_lens_tensor(self, seq_slots) -> torch.Tensor:
    slots = self._normalize_seq_slots(seq_slots)
    return torch.tensor(
        [self.seq_lens[slot] for slot in slots],
        dtype=torch.int32,
        device="cpu",
    )
```

其中 `self.seq_lens[slot]` 在 decode metadata 准备时更新。

---

## 2.6 Metadata：slot_mapping

`slot_mapping` 不传给 `_npu_paged_attention`。

它用于在调用 `_npu_paged_attention` 之前，把当前 token 的 K/V 写入 paged KV cache。

decode 阶段顺序：

```text
当前 token hidden_states
    ↓
q_proj/k_proj/v_proj
    ↓
得到当前 token 的 Q/K/V
    ↓
通过 slot_mapping 写入当前 token 的 K/V
    ↓
调用 _npu_paged_attention
```

本项目代码逻辑：

```python
kv_cache.write_decode(
    layer_idx=self.layer_idx,
    key_states=key_states,
    value_states=value_states,
    slot_mapping=attn_metadata.slot_mapping,
)

key_cache_layer, value_cache_layer = kv_cache.get_physical_cache(
    layer_idx=self.layer_idx
)

attn_output = self.paged_attn(
    q=query_states,
    key_cache=key_cache_layer,
    value_cache=value_cache_layer,
    block_tables=attn_metadata.block_tables,
    context_lens=attn_metadata.context_lens,
)
```

`slot_mapping` 的计算逻辑：

```python
logical_pos = start_pos + offset
block_idx = logical_pos // block_size
block_offset = logical_pos % block_size
slot = table[block_idx] * block_size + block_offset
```

---

## 2.7 Decode metadata 生成完整流程

调用位置：

```text
ModelRunner.decode()
```

核心输入：

```python
active_slots: list[int]
token_ids: list[int]
cache_positions: list[int]
```

其中：

```text
active_slots:
  当前仍在生成的序列 slot

token_ids:
  当前 decode 轮次输入的 token

cache_positions:
  当前 token 应该写入的逻辑位置
```

然后：

```python
attn_metadata = self.kv_cache.prepare_metadata(
    seq_slots=seq_slots,
    start_pos=cache_position,
    q_len=1,
)
```

`prepare_metadata()` 做三件事：

### 1. 确保 block 已分配

```python
end_pos = start + q_len
self._ensure_blocks(seq_slot=slot, end_pos=end_pos)
self.seq_lens[slot] = max(self.seq_lens[slot], end_pos)
```

### 2. 生成 slot_mapping

```python
slot_mapping = self.get_slot_mapping(slots, starts, q_len)
```

### 3. 生成 block_tables 和 context_lens

```python
PagedKVCacheMetadata(
    block_tables=self.get_block_tables_tensor(slots),
    context_lens=self.get_context_lens_tensor(slots),
    slot_mapping=self.get_slot_mapping(slots, starts, q_len),
)
```

---

## 2.8 Decode 示例

假设：

```text
block_size = 128
active_slots = [0, 2]
cache_positions = [300, 20]
```

已有 block table：

```text
seq_slot 0: [5, 9, 17]
seq_slot 2: [4]
```

则当前 token 写入后：

```text
seq_slot 0:
  当前 position = 300
  写完后 context_lens = 301

seq_slot 2:
  当前 position = 20
  写完后 context_lens = 21
```

生成：

```text
block_tables =
[
  [5, 9, 17],
  [4, 0, 0],
]

context_lens =
[
  301,
  21,
]
```

slot_mapping：

```text
seq_slot 0:
  logical_pos = 300
  block_idx = 300 // 128 = 2
  block_offset = 300 % 128 = 44
  physical_block = 17
  slot = 17 * 128 + 44

seq_slot 2:
  logical_pos = 20
  block_idx = 20 // 128 = 0
  block_offset = 20
  physical_block = 4
  slot = 4 * 128 + 20
```

所以：

```text
slot_mapping = [17 * 128 + 44, 4 * 128 + 20]
```

传给 `_npu_paged_attention` 的主要输入：

```text
query.shape        = [2, num_heads, head_dim]
key_cache.shape    = [num_blocks, 128, num_kv_heads, head_dim]
value_cache.shape  = [num_blocks, 128, num_kv_heads, head_dim]
block_table.shape  = [2, 3]
context_lens.shape = [2]
out.shape          = [2, num_heads, head_dim]
```

---

## 2.9 `_npu_paged_attention` 易踩坑点

### 坑 1：context_lens 应该包含当前 token

decode 中本项目是：

```text
先写当前 token 的 K/V
再调用 _npu_paged_attention
```

所以如果当前 token 写入 position 300：

```text
context_lens 应该是 301，而不是 300
```

---

### 坑 2：block_table 行顺序必须和 query 一致

必须满足：

```text
query[i] ↔ block_table[i] ↔ context_lens[i]
```

---

### 坑 3：block_table 必须覆盖 context_lens

必须满足：

```text
ceil(context_lens[i] / block_size) <= block_table.shape[1]
```

---

### 坑 4：context_lens 是 CPU int32 tensor

本项目中：

```text
context_lens.device = cpu
context_lens.dtype  = int32
```

不要误放到 NPU 上。

---

### 坑 5：num_tokens 不是上下文长度

```text
num_tokens = 当前 decode batch 中 token 数量
context_lens = 每个 token 对应序列的历史上下文长度
```

例如：

```text
batch = 8
每条上下文长度约 2048

query.shape = [8, num_heads, head_dim]
context_lens = [2049, 2049, ..., 2049]
```

---

# 3. `npu_fused_infer_attention_score`

## 3.1 作用

`npu_fused_infer_attention_score` 用于 prefill 阶段。

它可以理解为 Ascend NPU 上的推理版 fused attention / flash attention kernel。

在本项目中，它处理的是：

```text
多个 prompt 拼接后的连续 TND Q/K/V
```

通过 metadata 告诉 kernel：

```text
这些 token 分别属于哪些序列
每条序列的边界在哪里
是否需要 causal mask
```

---

## 3.2 本项目中的调用位置

调用位置：

```text
src/nanovllm_ascend/layers/npu_prefill_attention.py
```

核心调用：

```python
attn_out, _ = torch_npu.npu_fused_infer_attention_score(
    query=q.to(dtype=target_dtype).contiguous(),
    key=k.to(dtype=target_dtype).contiguous(),
    value=v.contiguous(),
    atten_mask=self.attn_mask,
    block_table=None,
    input_layout="TND",
    block_size=self.block_size,
    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
    actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_kv,
    num_key_value_heads=self.num_key_value_heads,
    num_heads=self.num_heads,
    scale=self.scale,
    sparse_mode=3,
)
```

调用链：

```text
LLM.generate()
    ↓
ModelRunner.prefill()
    ↓
Qwen3ForCausalLM.forward()
    ↓
Qwen3Attention.forward(is_prefill=True)
    ↓
NpuBatchPrefillAttention.forward()
    ↓
torch_npu.npu_fused_infer_attention_score()
```

---

## 3.3 输入 tensor 格式：TND

本项目使用：

```python
input_layout="TND"
```

TND 含义：

```text
T = total tokens
N = heads
D = head dim
```

所以输入 shape 是：

```text
query: [total_tokens, num_heads, head_dim]
key:   [total_tokens, num_kv_heads, head_dim]
value: [total_tokens, num_kv_heads, head_dim]
```

Q/K/V 的生成位置：

```text
src/nanovllm_ascend/models/qwen3.py
```

核心逻辑：

```python
query_states = self.q_proj(hidden_states).view(
    total_tokens,
    self.num_heads,
    self.head_dim,
)

key_states = self.k_proj(hidden_states).view(
    total_tokens,
    self.num_key_value_heads,
    self.head_dim,
)

value_states = self.v_proj(hidden_states).view(
    total_tokens,
    self.num_key_value_heads,
    self.head_dim,
)
```

---

## 3.4 Metadata：actual_seq_lengths

### 3.4.1 actual_seq_lengths 是什么？

`actual_seq_lengths` 是 TND 拉平之后，每条 query 序列的累计结束位置。

它不是每条序列的独立长度，而是 cumulative length。

错误理解：

```python
actual_seq_lengths = [5, 3, 7]
```

正确格式：

```python
actual_seq_lengths = [5, 8, 15]
```

含义：

```text
序列 0: token index [0, 5)
序列 1: token index [5, 8)
序列 2: token index [8, 15)
```

---

### 3.4.2 本项目如何生成 actual_seq_lengths？

位置：

```text
src/nanovllm_ascend/npu/paged_kv_cache.py
```

函数：

```text
prepare_prefill_metadata()
```

核心逻辑：

```python
actual_seq_lengths = []
total = 0
for seq_len in seq_lens:
    total += int(seq_len)
    actual_seq_lengths.append(total)
```

然后：

```python
actual_seq_lengths_q = actual_seq_lengths
```

---

### 3.4.3 格式要求

```text
actual_seq_lengths:
  type: list[int]
  length: batch_size
  最后一个元素: total_tokens
```

必须满足：

```text
actual_seq_lengths[-1] == query.shape[0]
```

---

## 3.5 Metadata：actual_seq_lengths_kv

### 3.5.1 普通 prefill

在本项目当前实现中：

```text
Q 长度 = K 长度 = V 长度
```

所以：

```python
actual_seq_lengths_kv = actual_seq_lengths_q.copy()
```

例如：

```text
seq_lens = [5, 3, 7]

actual_seq_lengths_q  = [5, 8, 15]
actual_seq_lengths_kv = [5, 8, 15]
```

---

### 3.5.2 更复杂场景

在完整 vLLM-Ascend 中，`npu_fused_infer_attention_score` 也可能用于：

```text
PrefillNoCache
PrefillCacheHit
ChunkedPrefill
DecodeOnly-like FIA
```

这些场景下：

```text
Q 长度不一定等于 KV 长度
```

例如 prefix cache hit 或 chunked prefill 时：

```text
query 只包含当前 chunk 的 token
key/value 可能来自已有 KV cache + 当前 chunk
```

这时：

```text
actual_seq_lengths_q:
  当前 query 的累计长度

actual_seq_lengths_kv:
  每条序列当前可见的完整 KV 长度
```

本项目当前没有实现这些复杂路径。

---

## 3.6 Metadata：atten_mask

### 3.6.1 本项目如何生成 atten_mask？

位置：

```text
src/nanovllm_ascend/layers/npu_prefill_attention.py
```

核心逻辑：

```python
attn_mask = torch.triu(
    torch.ones(max_mask_len, max_mask_len, dtype=torch.int8),
    diagonal=1,
)

self.register_buffer(
    "attn_mask",
    attn_mask,
    persistent=False,
)
```

这是一个 causal mask。

示意：

```text
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

其中上三角的 1 表示当前位置不能看未来 token。

---

### 3.6.2 atten_mask 尺寸

本项目中：

```text
atten_mask.shape = [max_model_len, max_model_len]
dtype = int8
device = 跟随 module buffer
```

构造 attention module 时：

```python
NpuBatchPrefillAttention(
    ...
    max_mask_len=max_model_len,
)
```

所以 mask 最大支持长度由 `max_model_len` 决定。

---

### 3.6.3 max_prompt_len 检查

本项目在 forward 中检查 batch 内最大 prompt 长度：

```python
max_prompt_len = 0
prev = 0
for end in attn_metadata.actual_seq_lengths_q:
    max_prompt_len = max(max_prompt_len, end - prev)
    prev = end

if max_prompt_len > self.max_mask_len:
    raise ValueError(...)
```

必须满足：

```text
max(seq_lens) <= max_mask_len
```

---

## 3.7 Metadata：block_table

在本项目的普通 prefill 中：

```python
block_table=None
```

含义：

```text
当前 prefill attention 不通过 paged KV cache 读取 K/V。
```

也就是说：

```text
npu_fused_infer_attention_score 使用当前传入的连续 q/k/v 做 attention。
```

但是注意：

```text
这不代表 prefill 阶段不写 KV cache。
```

本项目在调用 prefill kernel 前，会先把当前 prompt 的 K/V 写入 paged KV cache：

```python
kv_cache.write_prefill(
    layer_idx=self.layer_idx,
    key_states=key_states,
    value_states=value_states,
    slot_mapping=attn_metadata.slot_mapping,
)
```

因此 prefill 阶段有两条路径：

```text
attention 计算路径:
  q/k/v 连续 TND 张量
  ↓
  npu_fused_infer_attention_score

KV cache 写入路径:
  k/v 连续 TND 张量
  ↓
  slot_mapping
  ↓
  PagedKVCache
```

---

## 3.8 Metadata：slot_mapping

`slot_mapping` 不传给 `npu_fused_infer_attention_score`。

它用于 prefill 阶段写入 KV cache。

### 3.8.1 生成逻辑

位置：

```text
src/nanovllm_ascend/npu/paged_kv_cache.py
```

核心函数：

```text
get_prefill_slot_mapping()
```

逻辑：

```python
for slot, seq_len, start in zip(slots, seq_lens, starts):
    mapping.extend(self._slot_mapping_one(slot, start, seq_len))
```

单个 token 的物理位置计算：

```python
logical_pos = start_pos + offset
block_idx = logical_pos // block_size
block_offset = logical_pos % block_size
physical_slot = table[block_idx] * block_size + block_offset
```

---

### 3.8.2 示例

假设：

```text
block_size = 128
seq_len = 5
seq_slot = 0
seq_slot 0 分配到 physical block 7
```

那么 5 个 token 的 slot_mapping 是：

```text
7 * 128 + 0
7 * 128 + 1
7 * 128 + 2
7 * 128 + 3
7 * 128 + 4
```

---

## 3.9 sparse_mode

本项目使用：

```python
sparse_mode=3
```

工程语义可以这样理解：

| sparse_mode | 场景 |
|---|---|
| `0` | 非 causal / 普通 attention |
| `3` | causal attention，普通 LLM prefill |
| `4` | sliding window attention |

本项目是 Qwen3 causal LM，所以：

```text
sparse_mode=3
```

是合理的。

---

## 3.10 pre_tokens 和 next_tokens

本项目当前没有显式传：

```text
pre_tokens
next_tokens
```

官方测试和 vLLM-Ascend 中经常会传入这两个参数。

常见形式：

```python
pre_tokens=65535
next_tokens=65535
```

或者 sliding window 场景：

```python
pre_tokens=sliding_window
next_tokens=0
```

在本项目当前普通 causal prefill 中，如果默认值能正常工作，可以不显式传。

如果后续遇到 mask 行为不符合预期，可以考虑显式增加：

```python
pre_tokens=65535
next_tokens=65535
```

---

## 3.11 Prefill metadata 生成完整流程

调用位置：

```text
ModelRunner.prefill()
```

### 1. tokenize prompts

```python
ids = inputs["input_ids"][0]
input_ids_list.append(ids)
seq_lens.append(ids.numel())
```

得到：

```text
seq_lens: list[int]
```

---

### 2. 拉平 input_ids

```python
input_ids_flat = torch.cat(input_ids_list, dim=0)
```

得到：

```text
input_ids_flat: [total_tokens]
```

---

### 3. 构造 position_ids

```python
position_ids_flat = torch.cat(
    [torch.arange(seq_len, dtype=torch.long) for seq_len in seq_lens],
    dim=0,
)
```

得到：

```text
position_ids_flat: [total_tokens]
```

---

### 4. 创建 prefill metadata

```python
attn_metadata = self.kv_cache.prepare_prefill_metadata(
    seq_slots=seq_slots,
    seq_lens=seq_lens,
)
```

内部生成：

```text
actual_seq_lengths_q
actual_seq_lengths_kv
slot_mapping
block_tables
context_lens
```

其中：

```text
npu_fused_infer_attention_score 主要使用:
  actual_seq_lengths_q
  actual_seq_lengths_kv
  atten_mask

PagedKVCache.write_prefill 主要使用:
  slot_mapping
```

---

## 3.12 Prefill 示例

假设：

```text
batch = 3
seq_lens = [5, 3, 7]
block_size = 128
```

则：

```text
total_tokens = 15
```

输入张量：

```text
query.shape = [15, num_heads, head_dim]
key.shape   = [15, num_kv_heads, head_dim]
value.shape = [15, num_kv_heads, head_dim]
```

metadata：

```text
actual_seq_lengths_q  = [5, 8, 15]
actual_seq_lengths_kv = [5, 8, 15]
```

表示：

```text
seq 0: token index [0, 5)
seq 1: token index [5, 8)
seq 2: token index [8, 15)
```

如果物理 block 分配如下：

```text
seq 0 → block 10
seq 1 → block 11
seq 2 → block 12
```

则 prefill 写 KV cache 的 `slot_mapping` 类似：

```text
[
  10*128+0, 10*128+1, 10*128+2, 10*128+3, 10*128+4,
  11*128+0, 11*128+1, 11*128+2,
  12*128+0, 12*128+1, 12*128+2, 12*128+3, 12*128+4, 12*128+5, 12*128+6,
]
```

---

## 3.13 `npu_fused_infer_attention_score` 易踩坑点

### 坑 1：actual_seq_lengths 必须是累计长度

错误：

```python
actual_seq_lengths = [5, 3, 7]
```

正确：

```python
actual_seq_lengths = [5, 8, 15]
```

---

### 坑 2：actual_seq_lengths[-1] 必须等于 total_tokens

必须满足：

```text
actual_seq_lengths[-1] == query.shape[0]
```

---

### 坑 3：TND 下 Q/K/V 第一维必须一致

普通 prefill 中：

```text
query.shape[0] == key.shape[0] == value.shape[0]
```

---

### 坑 4：GQA 下 head 数要匹配

一般要求：

```text
num_heads % num_key_value_heads == 0
```

---

### 坑 5：atten_mask 尺寸要覆盖最大 prompt 长度

必须满足：

```text
max(seq_lens) <= max_mask_len
```

---

### 坑 6：block_table=None 只适合普通 prefill

如果后续实现：

```text
chunked prefill
prefix cache hit
prefill with existing KV cache
```

则不能简单使用：

```python
block_table=None
```

需要像 vLLM-Ascend 那样，根据不同 attention state 传入：

```text
block_table
actual_seq_lengths_kv
paged key/value cache view
```

---

# 4. 两个 kernel 的完整对比

## 4.1 输入对比

| 项目 | `npu_fused_infer_attention_score` | `_npu_paged_attention` |
|---|---|---|
| 阶段 | prefill | decode |
| query | `[total_tokens, num_heads, head_dim]` | `[num_decode_tokens, num_heads, head_dim]` |
| key | `[total_tokens, num_kv_heads, head_dim]` | paged `key_cache` |
| value | `[total_tokens, num_kv_heads, head_dim]` | paged `value_cache` |
| cache | 不从 cache 读 | 从 paged KV cache 读 |
| block_table | 普通 prefill 为 `None` | 必须提供 |
| seq metadata | `actual_seq_lengths` | `context_lens` |
| mask | causal mask / sparse_mode | 依赖 context_lens 和 block_table |
| 输出 | `[total_tokens, num_heads, head_dim]` | `[num_decode_tokens, num_heads, head_dim]` |

---

## 4.2 metadata 对比

| Metadata | 用于哪个 kernel | 格式 | 含义 |
|---|---|---|---|
| `actual_seq_lengths` | `npu_fused_infer_attention_score` | `list[int]` | TND 中每条 query 序列的累计结束位置 |
| `actual_seq_lengths_kv` | `npu_fused_infer_attention_score` | `list[int]` | 每条 KV 序列的累计结束位置或当前 KV 长度 |
| `atten_mask` | `npu_fused_infer_attention_score` | `[max_len, max_len]` | causal mask |
| `block_table` | `_npu_paged_attention` | `[num_tokens, max_blocks]`, int32, NPU | 逻辑 block 到物理 block 的映射 |
| `context_lens` | `_npu_paged_attention` | `[num_tokens]`, int32, CPU | 每条序列当前 KV 总长度 |
| `slot_mapping` | 写 KV cache | `[num_tokens]`, int64, NPU | 当前 token 的 K/V 应写入的物理 cache slot |

---

## 4.3 执行流程对比

### Prefill

```text
prompts
  ↓
tokenize
  ↓
input_ids_flat / position_ids_flat
  ↓
Q/K/V projection
  ↓
RoPE
  ↓
生成 actual_seq_lengths
  ↓
npu_fused_infer_attention_score
  ↓
attention output

同时:
  K/V
    ↓
  slot_mapping
    ↓
  写入 PagedKVCache
```

---

### Decode

```text
上一步生成的 token
  ↓
Q/K/V projection
  ↓
RoPE
  ↓
slot_mapping
  ↓
写入当前 token 的 K/V
  ↓
block_table + context_lens
  ↓
_npu_paged_attention
  ↓
attention output
```

---

# 5. 推荐在项目中保留的注释

可以在 `npu_prefill_attention.py` 中添加：

```python
# npu_fused_infer_attention_score uses flattened TND tensors.
# actual_seq_lengths / actual_seq_lengths_kv are cumulative sequence end
# positions, e.g. seq_lens=[5,3,7] -> actual_seq_lengths=[5,8,15].
# In normal prefill, q/k/v have the same total_tokens, so q lengths and
# kv lengths are identical. KV cache is written separately through
# slot_mapping before/around this attention call.
```

可以在 `npu_paged_attention.py` 中添加：

```python
# _npu_paged_attention is used for decode.
# query shape is [num_active_tokens, num_heads, head_dim].
# key_cache/value_cache are physical paged KV caches with shape
# [num_blocks, block_size, num_kv_heads, head_dim].
# block_table maps logical block index to physical block id for each
# active sequence. context_lens is CPU int32 and contains the total KV
# length after the current token has been written.
```

可以在 `paged_kv_cache.py` 中添加：

```python
# slot_mapping maps logical token positions to flattened physical KV cache
# slots:
#
#   logical_block = logical_pos // block_size
#   block_offset  = logical_pos % block_size
#   physical_block = block_table[logical_block]
#   slot = physical_block * block_size + block_offset
#
# It is used by index_copy_ to write K/V into paged cache.
```

---

# 6. 总结

`npu_fused_infer_attention_score` 和 `_npu_paged_attention` 分别服务于 LLM 推理中的两个关键阶段：

```text
prefill:
  计算量大，Q/K/V 是连续 token 张量。
  核心 metadata 是 actual_seq_lengths，用于描述 TND 中的序列边界。

decode:
  每步 token 少，但要读取大量历史 KV。
  核心 metadata 是 block_table + context_lens，用于从 paged KV cache 中找到历史上下文。
```

本项目当前采用的是一种清晰的最小实现：

```text
prefill:
  fused attention 计算当前 prompt 内 attention
  同时将 K/V 写入 paged KV cache

decode:
  将当前 token 的 K/V 写入 paged KV cache
  使用 paged attention 读取完整上下文并计算 attention
```

这正好对应 vLLM 的核心机制：

```text
prefill/decode 分离
Paged KV Cache
Block Table 间接寻址
Decode 增量生成
```

---

# 7. 参考源码

本文档主要参考以下源码与实现：

```text
本项目:
  src/nanovllm_ascend/layers/npu_prefill_attention.py
  src/nanovllm_ascend/layers/npu_paged_attention.py
  src/nanovllm_ascend/npu/paged_kv_cache.py
  src/nanovllm_ascend/models/qwen3.py
  src/nanovllm_ascend/model_runner.py

Ascend/pytorch:
  test/npu/test_compile_aclgraph_update.py
  test/npu/test_aclgraph_update.py
  torch_npu/npu/_npugraph_handlers/simple_handler.py
  torch_npu/npu/_npugraph_handlers/ifa_handler.py

vllm-project/vllm-ascend:
  vllm_ascend/attention/attention_v1.py
```
