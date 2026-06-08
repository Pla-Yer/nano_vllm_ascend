# nano-vLLM-Ascend Runtime Design Record

> 本文以当前仓库代码为准，描述已经落地的 runtime 设计。
> 旧记录里提到但当前代码中不存在的抽象，例如 `ModelInputBatch`，不再作为现状描述。

---

## 1. 当前运行时目标

当前项目不是完整 serving engine，而是一个尽量简洁的 mini runtime，目标是稳定跑通：

- `Qwen3`
- `Ascend NPU`
- `bf16`
- batch prefill
- paged decode
- 可配置采样
- 带最小 budget 判断的连续 admission

当前设计重点不是 benchmark，而是先把请求状态、调度、KV block 生命周期、模型执行边界固定下来。

---

## 2. 当前实际分层

当前代码中的核心分层如下：

```text
LLM (engine.py)
  对外同步接口 generate()
  负责一次性 tokenize、创建 Sequence、汇总输出

EngineCore (engine.py)
  消费 scheduler step
  驱动 prefill / decode / free_seq
  处理最小运行时异常收尾

MiniScheduler (scheduler.py)
  管理 waiting / running / finished
  管理 max_num_seqs admission
  管理 block budget admission
  统一判断 eos / max_new_tokens / runtime abort
  产出 SchedulerStep

Sequence (sequence.py)
  单条请求的唯一状态载体
  保存 prompt、prompt_token_ids、sampling_params、生成状态、budget 信息

ModelRunner (model_runner.py)
  负责模型加载、张量构造、prefill/decode 执行
  只接收 list[Sequence]
  从 Sequence 读取 prompt_token_ids 和 sampling_params
  把 next_token_id / cache_position / generated_token_ids 写回 Sequence

BlockManager (npu/block_manager.py)
  管理逻辑 block 分配与 metadata 生成

PagedKVCache (npu/paged_kv_cache.py)
  管理物理 key/value cache tensor

Sampler + SamplingParams
  负责 logits -> token_id
```

当前核心原则：

- 请求状态归 `Sequence`
- 运行时推进归 `EngineCore`
- 调度与 budget 归 `MiniScheduler`
- 张量与模型执行归 `ModelRunner`
- 逻辑 block 生命周期归 `BlockManager`
- 物理 KV tensor 归 `PagedKVCache`

---

## 3. Sequence 设计

### 3.1 角色

`Sequence` 是单条请求的唯一状态对象。当前 runtime 不再用多组并行数组维护请求状态。

这意味着以下信息都绑定在同一个对象上：

- 请求文本
- tokenize 结果
- 采样参数
- 当前生成位置
- 当前待写入 decode 的 token
- 已生成 token
- budget 信息
- finish reason

### 3.2 当前字段

当前 `Sequence` 的关键字段包括：

```python
seq_id: int
prompt: str
max_new_tokens: int
prompt_token_ids: torch.Tensor
sampling_params: SamplingParams

status: SequenceStatus
estimated_prompt_len: int
reserved_blocks: int
finish_reason: str | None

prompt_len: int
cache_position: int
next_token_id: int | None
generated_token_ids: list[int]
```

### 3.3 当前设计含义

- `prompt_token_ids` 在进入队列前就写入 `Sequence`
  - 因此 tokenize 只做一次
  - 后续 prefill 不再重复 tokenizer
- `sampling_params` 直接跟随 `Sequence`
  - 后续 prefill / decode 都从 `seq.sampling_params` 读
  - 不再在 runtime 链路里额外透传一份 sampling 配置
- `estimated_prompt_len` 当前直接由 `prompt_token_ids.numel()` 得到
- `reserved_blocks` 是 admission 时为整条请求预留的 block 数
- `finish_reason` 当前仅用于内部调度和测试，不暴露到 `generate()` 返回

### 3.4 Sequence 不负责什么

`Sequence` 不负责：

- tokenizer 调用
- scheduler 决策
- block 分配
- 模型 forward
- attention metadata 生成

它只保存请求状态，并提供少量状态更新方法：

- `set_prefill_result(...)`
- `append_next_token()`
- `set_decode_result(...)`
- `reach_max_tokens()`
- `finish(...)`

---

## 4. Scheduler 设计

### 4.1 当前职责

`MiniScheduler` 当前不是简单的 waiting/running 容器，而是最小可用的 runtime scheduler。

它负责：

- 管理 `waiting / running / finished`
- 管理 `max_num_seqs`
- 管理 block budget
- 判断序列是否该结束
- 产出一步执行计划 `SchedulerStep`

### 4.2 SchedulerStep

当前 step 抽象已经落地为：

```python
@dataclass
class SchedulerStep:
    prefill_seqs: list[Sequence]
    decode_seqs: list[Sequence]
    finish_seqs: list[Sequence]
```

`EngineCore` 每轮只消费一个 `SchedulerStep`。

### 4.3 当前 step 顺序

`plan_next_step(eos_token_id)` 的顺序固定为：

1. 先扫描 `running`，找出该结束的序列
2. 释放这些序列占用的并发位和 reserved budget
3. 从 `waiting` 按 FIFO 吸纳可入场请求进入 `prefill_seqs`
4. 对剩余可运行序列生成 `decode_seqs`

这是当前最小连续 admission 逻辑：waiting 请求可以在后续 step 中继续入场。

### 4.4 当前 budget 策略

budget 判断当前由 scheduler 负责，且采用最简单的整条请求预留策略。

每条请求的预算固定为：

```python
required_blocks = ceil((prompt_len + max_new_tokens) / block_size)
```

对应代码中的：

- `estimated_prompt_len`
- `reserved_blocks`
- `reserved_blocks_total`

admission 条件为：

```python
len(running) < max_num_seqs
and reserved_blocks_total + seq.reserved_blocks <= total_num_blocks
```

当前含义是：

- 不是只判断 prefill 够不够
- 而是判断整条请求最坏情况下的 prefill + decode block 上限是否能被容纳

### 4.5 finish reason

当前 scheduler 统一写 finish reason，至少包括：

- `eos`
- `max_new_tokens`
- `aborted_no_capacity`
- `aborted_runtime_error`

其中：

- `aborted_no_capacity` 表示单条请求自身预算超过总 block 上限，且系统当前没有任何 running 请求可释放
- `aborted_runtime_error` 表示执行期抛出异常后由 `EngineCore` 回写

### 4.6 当前刻意不做的事

为了保持简洁，scheduler 当前不做：

- 抢占
- 优先级调度
- block 碎片预测
- decode 前逐 token 重新预算
- prefix cache admission
- chunked prefill admission

---

## 5. Engine 设计

### 5.1 当前角色

当前 `EngineCore` 已经独立出来，`LLM.generate()` 只保留入口和收尾。

`LLM.generate()` 负责：

1. 解析本次调用的 `SamplingParams`
2. 一次性对全部 prompt 做 tokenize
3. 用 `prompt_token_ids + sampling_params` 创建 `Sequence`
4. 调用 `EngineCore.run(...)`
5. 汇总输出

`EngineCore` 负责：

1. 调用 `scheduler.plan_next_step(...)`
2. 对 `finish_seqs` 调 `runner.free_seq(...)`
3. 执行 `runner.prefill(...) / runner.decode(...)`
4. 在 `RuntimeError` 时调用 `scheduler.abort_sequences(...)` 并做最佳努力释放

### 5.2 当前数据流

当前 `generate()` 的关键数据流是：

```text
prompts
  -> runner.tokenize_prompts(prompts)
  -> scheduler.add_request(prompt, max_new_tokens, prompt_token_ids, sampling_params)
  -> engine_core.run(eos_token_id)
  -> decode generated_token_ids to texts
```

### 5.3 当前异常处理

当前 `EngineCore` 对 `prefill` / `decode` 的 `RuntimeError` 做最小处理：

- 捕获异常
- 调 `scheduler.abort_sequences(...)`
- 最佳努力 `runner.free_seq(seq)`

这属于非正常路径，不是常态调度逻辑的一部分。

---

## 6. ModelRunner 设计

### 6.1 当前职责

`ModelRunner` 当前负责：

- 加载 tokenizer / config / model
- 构造 prefill / decode 张量
- 调用 `BlockManager` 生成 metadata
- 调用模型 forward
- 调用 `Sampler`
- 把结果写回 `Sequence`

### 6.2 当前接口

当前接口已经统一为：

```python
tokenize_prompts(prompts: list[str]) -> list[torch.Tensor]
prefill(seqs: list[Sequence]) -> None
decode(seqs: list[Sequence]) -> None
free_seq(seq: Sequence) -> None
```

注意：

- 不再有 `estimate_prompt_lens(...)`
- 不再在 prefill 中重新 tokenizer
- 不再在 `prefill/decode` 签名里单独传 `sampling_params`

### 6.3 prefill

当前 prefill 路径：

1. 从 `seq.prompt_token_ids` 直接组 batch
2. 根据 `seq.estimated_prompt_len` 构造 `position_ids_flat`
3. 通过 `BlockManager.prepare_prefill_metadata(...)` 申请真实 metadata
4. 模型 forward
5. 取每条请求最后一个位置的 logits
6. 用各自的 `seq.sampling_params` 做采样
7. 写回：
   - `next_token_id`
   - `prompt_len`
   - `cache_position`

### 6.4 decode

当前 decode 路径：

1. 调 `seq.append_next_token()`，把当前 `next_token_id` 视为正式输出 token
2. 用 `seq.cache_position` 作为当前 decode 的 `position_id`
3. 通过 `BlockManager.prepare_decode_metadata(...)` 构造 decode metadata
4. 模型 forward
5. 对每条序列分别用 `seq.sampling_params` 采样
6. 写回新的 `next_token_id`
7. `cache_position += 1`

### 6.5 当前刻意保持简单的点

- prefill / decode 都只接收 `list[Sequence]`
- 不额外引入 `Batch` 或 `ModelInputBatch` 抽象
- 采样逐条从 `Sequence` 读取，不再维护单独并行 sampling 数组

---

## 7. Sampling 设计

### 7.1 SamplingParams

当前 `SamplingParams` 已经是独立对象，默认值表达 greedy：

```python
temperature: float = 0.0
top_k: int = 0
top_p: float = 1.0
```

语义为：

- `temperature <= 0` -> greedy
- `top_k <= 0` -> 不启用 top-k
- `top_p >= 1.0` -> 不启用 top-p

### 7.2 Sampler

当前 `Sampler` 是纯 logits 后处理组件，负责：

```text
logits [batch, vocab] -> token_ids [batch]
```

当前支持：

- greedy
- temperature
- top-k
- top-p

### 7.3 Sampling 配置绑定方式

当前设计中，采样参数跟着 `Sequence` 走，而不是跟着某一轮 runtime 调用的局部变量走。

这带来两个直接好处：

- 后续如果不同 sequence 要支持不同采样参数，不需要再改 `ModelRunner` 接口
- runtime 链路更短，避免 `engine -> scheduler -> runner` 反复透传一份采样配置

---

## 8. BlockManager 与 PagedKVCache

### 8.1 BlockManager

`BlockManager` 当前只负责逻辑 block 生命周期和 metadata：

- `ensure_blocks(...)`
- `prepare_prefill_metadata(...)`
- `prepare_decode_metadata(...)`
- `free_slot(...)`
- `reset()`
- `get_block_tables_tensor(...)`
- `get_context_lens_tensor(...)`
- `get_slot_mapping(...)`

它不负责 admission policy。budget 决策已经在 scheduler。

### 8.2 PagedKVCache

`PagedKVCache` 当前只负责物理 KV tensor：

- `key_cache`
- `value_cache`
- `write_prefill(...)`
- `write_decode(...)`
- `get_physical_cache(...)`

它不负责：

- block 分配
- seq_lens
- request lifecycle

---

## 9. 当前对外接口

### 9.1 Python API

当前公开入口仍是：

```python
from nanovllm_ascend import EngineCore, LLM, SamplingParams

llm = LLM(model_path)
outputs = llm.generate(
    prompts,
    max_new_tokens=128,
    sampling_params=SamplingParams(temperature=0.8, top_k=20, top_p=0.9),
)
```

`EngineCore` 已可独立导入，但当前主要仍作为 `LLM` 内部运行时组件使用。

### 9.2 返回格式

当前 `generate()` 返回：

```python
list[dict[str, str | list[int]]]
```

每项为：

```python
{
    "texts": "...",
    "token_ids": [...],
}
```

当前不返回：

- `finish_reason`
- runtime metrics
- scheduler trace

---

## 10. 当前仍然未做的抽象

以下内容在旧记录里出现过，但当前代码并没有落地，不应再被视为现状：

- `ModelInputBatch`
- prefix cache
- chunked prefill
- preemption
- swap
- serving API
- streaming output

如果后续要继续扩展，应以现在这套实际边界为起点，而不是回到旧文档中的规划状态。

---

## 11. 一句话总结

当前 runtime 的实际设计可以概括为：

```text
LLM 只 tokenize 一次并创建 Sequence；
EngineCore 负责逐 step 推进 runtime；
Sequence 同时携带 prompt_token_ids 与 sampling_params；
scheduler 负责 step、finish 和 block budget admission；
runner 只消费 Sequence 并执行 prefill/decode；
block manager 管逻辑 block，paged kv cache 管物理 tensor。
```
