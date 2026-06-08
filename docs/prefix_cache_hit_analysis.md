# nano-vLLM-Ascend Prefix Cache Hit 正确性与性能分析记录

> 适用项目：`nano-vllm-ascend`  
> 适用阶段：Paged KV Cache / Prefix Cache / NPU Prefill Attention 适配  
> 结论日期：2026-06  
> 测试模型：Qwen3-0.6B  
> 主要后端：`torch_npu.npu_fused_infer_attention_score`，TND layout，paged KV cache，`block_size=128`

---

## 1. 背景

本阶段的目标是在 nano-vLLM-Ascend 中实现 prefix cache，使多个请求共享相同长前缀时，不再重复计算前缀 token 的 KV，而是复用已经写入 paged KV cache 的物理 block。

实现后出现了一个现象：

- prefix cache 命中逻辑看起来是正确的；
- no-cache 与 cache-hit 生成结果可以做到 token 级一致；
- 但是性能上，cache-hit 请求的 latency 与 no-cache 很接近，甚至在某些端到端 bench 中略慢。

本文记录了从正确性验证、性能拆解、vLLM-Ascend 源码对照到最终结论的完整分析过程。

---

## 2. 当前实现的核心路径

当前 prefix cache hit 的 prefill attention 后端大致为：

```python
if attn_metadata.use_paged_prefill:
    key, value, cache_block_size = _paged_cache_for_fia(
        key_cache=key_cache,
        value_cache=value_cache,
    )
    block_table = attn_metadata.block_tables.to(
        device=query.device,
        dtype=torch.int32,
    ).contiguous()
    actual_seq_lengths_kv = (
        attn_metadata.context_lens.to(device="cpu", dtype=torch.int32)
        .contiguous()
        .tolist()
    )
    attn_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=self.attn_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=cache_block_size,
        actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=self.num_key_value_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=3,
    )
```

语义上是：

```text
Q     = runtime suffix tokens
K/V   = cached prefix KV + runtime suffix KV，均位于 paged KV cache
索引  = block_table
Q 长度 = actual_seq_lengths_q
KV 长度 = actual_seq_lengths_kv / context_lens，即完整上下文长度
```

也就是说，prefix cache hit 后并不是直接复用 attention 输出，而是让 suffix Q 对完整 KV cache 做 attention。

---

## 3. 正确性验证过程

### 3.1 构造长前缀，确保超过一个 block

为了避免短 prompt 导致 prefix cache 只命中很短的前缀，测试脚本改为自动构造长共享前缀。

典型测试条件：

```text
block_size=128
shared_prefix_tokens=529
expected_cached_len_floor=512
expected_cached_blocks_floor=4
warm_prompt_tokens=545
hit_prompt_tokens=561
hit_prompt_tokens % block_size = 49
tokens_until_next_block = 79
```

含义：

- 共享前缀大于 128 token；
- 至少可以复用 4 个完整 prefix blocks；
- hit prompt 在生成 80 个 token 左右时会跨过当前 decode block 边界。

---

### 3.2 单请求等价性验证

先使用单请求模式验证：

```text
no-cache:      直接运行 hit_prompt
cache-hit:     warm_prompt -> hit_prompt
```

测试 `max_new_tokens=8/16/24/32/64`，结果全部 PASS：

```text
max_new_tokens=8:  PASS
max_new_tokens=16: PASS
max_new_tokens=24: PASS
max_new_tokens=32: PASS
max_new_tokens=64: PASS
```

这说明：

- cached prefill 首 token 正确；
- suffix KV 写入正确；
- decode 延续正确；
- position ids、context lengths、block table 在单请求场景下基本正确。

---

### 3.3 跨 decode 新 block 验证

由于 `hit_prompt_tokens=561`，`561 % 128 = 49`，当前 block 还剩：

```text
128 - 49 = 79 tokens
```

因此进一步测试：

```text
max_new_tokens=64   # 未跨新 block
max_new_tokens=80   # 已跨新 block
max_new_tokens=96   # 已跨新 block
max_new_tokens=128  # 已跨新 block
```

结果：

```text
max_new_tokens=64:  PASS
max_new_tokens=80:  PASS
max_new_tokens=96:  PASS
max_new_tokens=128: PASS
```

这说明：

- decode 跨 block 分配正确；
- 新 block 追加到 block table 的逻辑正确；
- decode KV 写入 offset 正确；
- cached prefix blocks 没有被错误覆盖。

---

### 3.4 batch hit/miss 混合验证

最开始曾出现过 `no-cache 单请求` 与 `cache-hit batch 请求` 结果不一致的问题。后续发现这个对比方式不公平，因为两边 batch 路径不同。

改为公平对比：

```text
no-cache:      [hit_prompt, miss_prompt]
cache-hit:     warm_prompt -> [hit_prompt, miss_prompt]
```

最终结果：

```text
seq0 hit_prompt:  PASS
seq1 miss_prompt: PASS

PASS: batch cache-hit path is token-equivalent to batch no-cache path.
```

这说明：

- batch 下 hit/miss 混合请求正确；
- batch block table 没有错位；
- logits indexing / sampler 没有错位；
- miss prompt 不会误命中；
- hit prompt 复用 cached blocks 后仍与 no-cache token 级等价。

---

## 4. 正确性结论

当前 prefix cache 的功能正确性可以认为已经通过：

```text
1. 多 block prefix cache 命中正确；
2. cached prefill 正确；
3. suffix KV 写入正确；
4. 单请求 decode 延续正确；
5. decode 跨新 block 正确；
6. batch hit/miss 混合正确；
7. greedy 解码下 no-cache 与 cache-hit token_ids 完全一致。
```

因此，后续问题不再是 correctness bug，而是 performance 问题。

---

## 5. 性能实验结果

### 5.1 端到端在线 cache-hit bench：`max_new_tokens=8`

测试参数：

```text
block_size=128
batch_size=4
iters=5
shared_prefix_min_tokens=1024
max_new_tokens=8
```

结果：

```text
avg_no_cache_time         = 3.0128s
avg_cache_hit_online_time = 3.1526s
online_speedup            = 0.956x
online_latency_reduction  = -4.64%
```

token 账面统计：

```text
avg_no_cache_prompt_tokens       = 4844
avg_hit_prompt_tokens            = 4844
avg_expected_cached_tokens       = 4608
avg_expected_runtime_prompt_tokens = 236
avg_saved_prompt_ratio           = 95.13%
```

现象：虽然理论上省掉了约 95% 的 prompt tokens，但 cache-hit 在线请求没有变快，反而略慢。

---

### 5.2 端到端在线 cache-hit bench：`max_new_tokens=1`

为了排除 decode 稀释收益，将 `max_new_tokens` 降到 1。

结果：

```text
avg_no_cache_time         = 0.7187s
avg_cache_hit_online_time = 0.7550s
online_speedup            = 0.952x
online_latency_reduction  = -5.04%
```

结论：

```text
不是 decode 稀释导致 cache-hit 不明显。
```

因为几乎只测 TTFT / prefill 时，cache-hit 仍然没有显著加速。

---

### 5.3 prefill-only 分段计时

进一步对 prefill 做单独计时，结果：

```text
no_cache_prefill_time:  avg=0.1041s
cache_hit_prefill_time: avg=0.0964s

no_cache_tokenize_prepare_prefill_time:  avg=0.1097s
cache_hit_tokenize_prepare_prefill_time: avg=0.1024s

prefill_compute_speedup=1.080x
tokenize_prepare_prefill_speedup=1.071x
```

真实命中信息：

```text
prompt_tokens=1223
cached_tokens=1152
runtime_prompt_tokens=71
cached_blocks=9
```

含义：

- prefix cache 确实命中了；
- input 确实从 1223 token 裁剪到了 71 token；
- 但 prefill compute 只从 0.1041s 降到 0.0964s；
- 实际只提升约 8%。

这解释了为什么端到端 bench 中几乎没有加速：prefix cache 在 prefill 中只省了约 7.7ms，容易被 metadata、paged attention、Python eager 调度等开销抵消。

---

### 5.4 attention 后端同步计时

在 `npu_fused_infer_attention_score` 前后添加 `torch.npu.synchronize()` 后，得到：

```text
without cache hit attention time ≈ 0.00036s ~ 0.00043s
with cache hit attention time    ≈ 0.00031s ~ 0.00035s
```

注意：最开始的 `0.77s` 级别调用属于首次初始化、编译或懒加载开销，不应作为稳定层耗时对比。

结论：

```text
cache-hit attention 确实略快，但仍然与 no-cache dense prefill attention 同量级。
```

也就是说，Q token 数从 1223 降到 71，并没有让 attention 耗时按 `71/1223` 线性下降。

---

## 6. vLLM-Ascend 最新源码对照

### 6.1 Attention 状态定义

vLLM-Ascend 中定义了以下 attention 状态：

```python
class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4
```

也就是说，prefix cache hit 在 backend 中是独立状态，不是普通 dense prefill，也不是 decode-only。

---

### 6.2 KV cache shape 与 block size

vLLM-Ascend 的 KV cache shape：

```python
return (2, num_blocks, block_size, num_kv_heads, head_size)
```

支持的 kernel block size：

```python
return [128]
```

这说明 vLLM-Ascend 当前 Ascend attention 主路径也使用 `block_size=128`。

当前 nano-vLLM-Ascend 的 `block_size=128` 与 vLLM-Ascend 对齐。

---

### 6.3 metadata 构建流程

vLLM-Ascend 在 metadata builder 中准备：

```python
block_table = common_attn_metadata.block_table_tensor
seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

attn_metadata = AscendMetadata(
    block_tables=block_table,
    seq_lens=seq_lens,
    seq_lens_cpu=seq_lens,
    seq_lens_list=seq_lens.tolist(),
    actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
    slot_mapping=slot_mapping,
    ...
)
```

也就是说，它在进入每层 attention 前就准备好了：

```text
actual_seq_lengths_q = 本轮实际计算的 query 长度
seq_lens_list        = 每个请求完整上下文长度
block_tables         = 每个请求的物理 KV block 表
slot_mapping         = 本轮新 token 写入 KV cache 的位置
```

这与当前实现的语义一致，但 vLLM-Ascend 避免了每层重复 `.to("cpu").tolist()`。

---

### 6.4 `PrefillNoCache` 与 `PrefillCacheHit` 的参数差异

vLLM-Ascend 的 `_get_fia_params()` 中：

#### PrefillNoCache

```python
if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
    block_size = 128
    block_table = None
    actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
```

语义：

```text
Q     = full prompt
K/V   = full prompt dense K/V
KV len = Q len
block_table = None
```

#### PrefillCacheHit

```python
elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
    batch_size = attn_metadata.seq_lens.shape[0]
    block_table = attn_metadata.block_tables[:batch_size, :]
    num_block, block_size, _, _ = self.key_cache.shape
    key = self.key_cache.view(num_block, block_size, -1)
    value = self.value_cache.view(num_block, block_size, -1)
    actual_seq_lengths_kv = attn_metadata.seq_lens_list
```

语义：

```text
Q     = runtime suffix
K/V   = paged KV cache
KV len = full context length
block_table = cached prefix blocks + suffix blocks
```

这与当前 nano 实现的整体后端路径一致。

---

### 6.5 prefix cache hit prefill 走 FIA，不是 `_npu_paged_attention`

vLLM-Ascend 的 `forward_impl()` 中：

```python
if (
    attn_metadata.attn_state == AscendAttentionState.DecodeOnly
    and using_paged_attention(num_tokens, self.vllm_config)
    and self.sliding_window is None
):
    output = self.forward_paged_attention(query, attn_metadata, output)
else:
    output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
```

因此：

```text
DecodeOnly 满足条件时：
    走 _npu_paged_attention

PrefillCacheHit：
    走 npu_fused_infer_attention_score + block_table

ChunkedPrefill：
    走 npu_fused_infer_attention_score + block_table
```

这点很重要：vLLM-Ascend 的 prefix cache hit prefill 不是 decode 的 `_npu_paged_attention` 快路径，而是 FIA 的 paged-KV prefill 路径。

---

### 6.6 vLLM-Ascend 的 latency 优化点

vLLM-Ascend 并不是没有 latency 优化，只是优化点主要在工程路径上，而不是换成了一个完全不同的 prefix-cache attention 数学核。

主要优化包括：

#### 1. 输入裁剪

prefix cache hit 后，scheduler 只提交 suffix tokens，模型前向只处理 runtime suffix。

#### 2. metadata 预处理

`seq_lens_list`、`actual_seq_lengths_q` 在 metadata 构建阶段生成，避免每层重复 CPU/NPU 转换。

#### 3. KV cache 直接 view

vLLM-Ascend 使用：

```python
key = self.key_cache.view(num_block, block_size, -1)
value = self.value_cache.view(num_block, block_size, -1)
```

而不是每层 `.contiguous().view(...)`。

#### 4. workspace 复用

full graph FIA 路径中会缓存 workspace：

```python
workspace = graph_params.workspaces.get(num_tokens)

if workspace is None:
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)
    update_graph_params_workspaces(num_tokens, workspace)
```

#### 5. `.out(...)` 版本复用输出 buffer

vLLM-Ascend 使用：

```python
torch_npu.npu_fused_infer_attention_score.out(
    ...,
    workspace=workspace,
    out=[output, softmax_lse],
)
```

而不是每次返回新的 `attn_out` tensor。

#### 6. ACL graph / graph task replay

full graph 路径会 capture graph task，并在后续通过 graph task update/replay 降低 eager 调度开销。

#### 7. Decode-only 专用 `_npu_paged_attention`

decode-only 满足条件时走：

```python
torch_npu._npu_paged_attention(
    query=query,
    key_cache=self.key_cache,
    value_cache=self.value_cache,
    block_table=attn_metadata.block_tables,
    context_lens=attn_metadata.seq_lens,
    out=output,
)
```

这才是 decode latency 的专用快路径。

---

## 7. 为什么 prefix cache hit 但耗时接近？

### 7.1 cache hit 不是复用 attention 输出

prefix cache 复用的是前缀 token 的 KV，不是前缀的 attention 输出。

cache hit 后：

```text
Q len  = runtime suffix length
KV len = full context length
```

例如当前测试：

```text
Q len  = 71
KV len = 1223
```

attention 不是：

```text
71 × 71
```

而是：

```text
71 × 1223
```

所以 attention 计算和访存不会按 `71/1223` 线性下降。

---

### 7.2 PageAttention / paged KV 有 block table 间接寻址开销

cache-hit prefill 走的是：

```text
npu_fused_infer_attention_score + block_table + paged KV cache
```

这比 dense prefill 多了：

```text
block_table 查询
paged KV 物理 block 间接访问
actual_seq_lengths_kv 处理
paged prefill tiling
```

因此它更适合 KV 管理和 serving 吞吐场景，不一定在小 batch、小 suffix 场景下带来明显单请求 latency 降低。

---

### 7.3 当前模型和 shape 太小，固定开销占主导

当前测试条件：

```text
model = Qwen3-0.6B
batch_size = 1 或 4
runtime_suffix_tokens = 71 或每批合计 236
```

这种 shape 对 NPU 来说较小，很多耗时来自：

```text
kernel launch
算子调度
tiling 开销
workspace 准备
block_table metadata
Python eager 调用
小矩阵低利用率
```

而不是纯 token 数相关的 FLOPs。

因此，即使 prefix cache 命中 95% 以上 prompt tokens，也可能只带来约 1.08x 的 prefill compute speedup。

---

### 7.4 当前 nano 实现尚未对齐 vLLM-Ascend 工程快路径

当前实现与 vLLM-Ascend 在算法路径上一致，但工程快路径还缺少：

```text
1. metadata 预处理，避免每层 context_lens.to(cpu).tolist()
2. block_table 预先转为 NPU int32 contiguous
3. key_cache/value_cache 直接 view，避免每层 contiguous
4. npu_fused_infer_attention_score.out
5. workspace cache
6. output buffer 复用
7. ACL graph / graph task replay
8. decode-only _npu_paged_attention 快路径
```

因此当前 nano 的性能更接近 vLLM-Ascend 的 eager FIA 路径，而不是完整 production 优化路径。

---

## 8. 最终结论

### 8.1 correctness 结论

prefix cache 正确性已经通过：

```text
- 多 block 命中正确；
- cached prefill 正确；
- suffix KV 写入正确；
- 单请求 decode 正确；
- decode 跨 block 正确；
- batch hit/miss 混合正确；
- greedy 解码下 no-cache 与 cache-hit token_ids 完全一致。
```

### 8.2 performance 结论

当前性能提升不明显的根因不是 cache 没命中，而是：

```text
1. cache-hit prefill 仍然需要 suffix Q attend 到完整 KV cache；
2. paged KV + block_table 路径有额外固定开销；
3. Qwen3-0.6B、小 batch、短 suffix 场景下固定调度开销占主导；
4. 当前 nano 实现尚未对齐 vLLM-Ascend 的 workspace/out/graph replay 等工程优化；
5. attention 后端同步计时显示 with-cache 与 without-cache 单层耗时同量级。
```

因此更准确的表述是：

```text
prefix cache hit 功能正确，但当前 eager paged-prefill FIA 后端未能把 token 裁剪收益充分转化为 latency 收益。
```

---

## 9. 后续优化建议

### 9.1 低风险优化：对齐 vLLM-Ascend metadata 处理

将这些每层操作前移到 metadata 构建阶段：

```python
# 当前不建议每层做
attn_metadata.context_lens.to(device="cpu", dtype=torch.int32).tolist()
attn_metadata.block_tables.to(device=query.device, dtype=torch.int32).contiguous()
```

改为：

```python
# metadata 构建阶段一次性准备
attn_metadata.seq_lens_list = context_lens_cpu.tolist()
attn_metadata.block_tables = block_tables.to(device=device, dtype=torch.int32).contiguous()
```

attention forward 中直接使用：

```python
block_table = attn_metadata.block_tables
actual_seq_lengths_kv = attn_metadata.seq_lens_list
```

---

### 9.2 去掉每层 KV cache `.contiguous()`

当前：

```python
key = key_cache.contiguous().view(num_blocks, cache_block_size, -1)
value = value_cache.contiguous().view(num_blocks, cache_block_size, -1)
```

建议：

```python
assert key_cache.is_contiguous()
assert value_cache.is_contiguous()

key = key_cache.view(num_blocks, cache_block_size, -1)
value = value_cache.view(num_blocks, cache_block_size, -1)
```

如果 assert 失败，应从 KV cache 分配布局修，而不是每层复制。

---

### 9.3 使用 `.out(...)` 与 output buffer 复用

参考 vLLM-Ascend：

```python
softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)

torch_npu.npu_fused_infer_attention_score.out(
    query=query,
    key=key,
    value=value,
    atten_mask=attn_mask,
    block_table=block_table,
    input_layout="TND",
    block_size=block_size,
    actual_seq_lengths=actual_seq_lengths_q,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    num_key_value_heads=num_kv_heads,
    num_heads=num_heads,
    scale=scale,
    sparse_mode=3,
    workspace=workspace,
    out=[output, softmax_lse],
)
```

目标是减少临时 tensor 分配与 Python eager 返回开销。

---

### 9.4 实现 workspace cache

按 `num_tokens` 缓存 FIA workspace：

```python
workspace = workspaces.get(num_tokens)
if workspace is None:
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)
    workspaces[num_tokens] = workspace
```

这可以减少每次相同 shape 的 workspace 查询或分配开销。

---

### 9.5 decode-only 切换到 `_npu_paged_attention`

当前若 decode 仍走 FIA，可以对齐 vLLM-Ascend：

```python
torch_npu._npu_paged_attention(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    num_kv_heads=num_kv_heads,
    num_heads=num_heads,
    scale_value=scale,
    block_table=block_table,
    context_lens=context_lens,
    out=output,
)
```

decode 是长期生成阶段，专用 paged attention 快路径可能比 FIA 更重要。

---

### 9.6 对比另一条 prefix-cache-hit 后端：gather dense K/V

当前路径：

```text
suffix Q + paged KV + block_table + FIA
```

可以额外实现实验路径：

```text
cached blocks + suffix KV -> gather 成连续 dense K/V
suffix Q + dense full K/V -> 普通 dense FIA
```

对于 Qwen3-0.6B、小 batch、小 suffix 场景，`gather + dense FIA` 可能反而比 `paged FIA` 更低 latency。需要实际 benchmark 比较。

---

## 10. 建议保留的回归测试

建议在项目中保留以下测试脚本：

```text
check_prefix_cache_equivalence.py
    单请求 no-cache vs cache-hit token 级等价

check_prefix_cache_boundary.py
    decode 跨 block 后 token 级等价

check_prefix_cache_batch_equivalence.py
    batch hit/miss 混合 token 级等价

bench_prefix_cache_speedup.py
    cache-hit online speedup / amortized speedup / prefill-only speedup
```

建议回归矩阵：

```text
block_size: 128
prefix length: 1 block, 2 blocks, 4 blocks, 8 blocks, 16 blocks
batch size: 1, 4, 8
max_new_tokens: 1, 8, 32, 128
batch pattern:
  [hit]
  [hit, miss]
  [miss, hit]
  [hit, hit]
  [hit, hit, miss]
```

---

## 11. 一句话总结

> 当前 nano-vLLM-Ascend 的 prefix cache 已经实现了正确的 token 级复用与 batch hit/miss 混合支持；性能未明显提升的原因不是 cache miss，而是 cache-hit prefill 仍走 `npu_fused_infer_attention_score + paged KV + block_table` 路径，suffix Q 仍需访问完整 KV，同时当前 eager 实现缺少 vLLM-Ascend 的 metadata 预处理、workspace/out 复用和 graph replay 等工程优化。因此现阶段应将问题定位为 backend/engineering latency optimization，而不是 prefix cache correctness。

