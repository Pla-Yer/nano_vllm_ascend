# 连续批处理与 OpenAI API 分析

日期：2026-06-08

## 范围

本文档记录了本仓库中 OpenAI 风格 API 连续批处理路径的实现、调试和验证过程。

当前运行时保持预填充（prefill）和解码（decode）分离。本工作不引入分块预填充（chunked prefill），也不改变模型执行路径。HTTP 请求被路由到现有的 `LLM.submit()` 和 `LLM.step()` 接口，使得一个引擎循环拥有 NPU 执行权。

## 设计

API 服务实现于 `src/nanovllm_ascend/openai_server.py`。

该服务提供：

- `POST /v1/chat/completions`
- 非流式 OpenAI 风格 `chat.completion` 响应
- 当 `stream=true` 时，提供流式 `chat.completion.chunk` SSE 响应

重要的运行时规则是：HTTP 处理器不调用 `LLM.generate()`，也不直接调用 `LLM.step()`。每个处理器构建提示词（prompt），创建待处理请求对象，并将其推入 incoming 队列。

单个后台引擎工作线程随后重复执行：

1. 清空 incoming HTTP 请求
2. 调用 `LLM.submit()` 将它们添加到调度器等待队列
3. 当有未完成的工作时调用 `LLM.step()`
4. 将完成的输出或流式增量分派回对应的 HTTP 请求

这保持了 FastAPI 请求处理的并发能力，同时保留单一的 NPU 引擎执行所有者。

## 调度器行为

调度器保持预填充/解码分离。

每次 `LLM.step()` 调用 `MiniScheduler.plan_next_step()`，执行以下序列：

1. 完成已达到 EOS 或 `max_new_tokens` 的运行中序列
2. 如果容量允许，将等待中的序列接纳为运行中
3. 为新接纳的序列调度预填充
4. 为已有下一个 token 的运行中序列调度解码

本实现中的连续批处理意味着：当早期请求已在运行时，后续请求可以进入调度器等待队列，然后在稍后的步骤边界处被接纳（如果容量可用）。

这并不意味着请求可以插入到正在执行的 NPU kernel 中间。也不意味着晚到的请求可以避免其自身的预填充开销。

## 容量条件

等待中的请求只有在两个条件都满足时才能进入运行状态：

```python
len(running) < max_num_seqs
reserved_blocks_total + seq.reserved_blocks <= total_num_blocks
```

`seq.reserved_blocks` 的计算方式为：

```python
runtime_prompt_len + max_new_tokens
```

这对 API 测试很重要。如果客户端请求较大的 `max_tokens`，调度器会为该完整生成预算预留足够的 KV 块。API 测试脚本默认使用 `--max-tokens 1024`，单个请求可能占用大部分默认 KV 块池。

在这种情况下，第二个 HTTP 请求已被 API 接受并提交到调度器，但它保持在 `waiting` 状态，因为调度器无法接纳它，直到第一个请求释放 KV 块。观察到的延迟看起来接近串行执行，尽管 API 队列路径是正确的。

## 测试与 API 差异

fake 和 real 连续批处理测试使用较小的生成预算和足够的调度器容量。在这些条件下，在几个步骤后提交的第二个请求可以迅速从 `waiting` 移动到 `running`。

API 最初表现较差，因为实际运行使用了 API 默认值，其 `max_tokens` 预算大得多。这增加了每个请求的块预留，在默认 `num_blocks` 下阻止了并发接纳。

这不是由 FastAPI 序列化、Python 锁争用或缺失调度器提交引起的。这是由调度器容量引起的。

## API 验证结果

使用容量兼容的设置后，两个并发 API 请求（使用相同提示词）：

```bash
--prompt "explain kv cache"
```

以几乎相同的耗时完成：

```text
total_elapsed=14.198s
total_elapsed=14.211s
```

这符合重叠请求的预期连续批处理行为：两个请求共享同一个引擎循环，一起推进，而不是一个请求等待另一个完成。

## FIA Mask 问题

在 API 测试期间，设置：

```python
parser.add_argument("--max-model-len", type=int, default=512)
```

导致 Ascend NPU 在 `npu_fused_infer_attention_score` 中报错：

```text
when sparseMode is 3, the input mask has 2 dims in total, maskDim 1 shall be 2048
```

即使没有 `--enable-prefix-cache` 也发生了此问题。

原因是 `max_model_len` 被用于两个不同的概念：

1. 最大接受提示词长度
2. 传递给 Ascend FIA 的物理因果 mask 形状

当 `max_model_len=512` 时，预填充注意力模块创建了 `512 x 512` 的 mask。当前 `sparse_mode=3` 调用使用的 Ascend FIA split-fuse 路径期望 mask 宽度为 `2048`。

修复方案保留 `max_model_len` 作为请求长度限制，但以最小物理尺寸 2048 创建 FIA mask：

```python
self.fia_mask_len = max(max_mask_len, 2048)
```

这保持了短上下文 API 运行的有效性，同时满足 NPU kernel 的 mask 形状要求。

## 与 vLLM 的比较

vLLM 的 OpenAI 兼容聊天服务器遵循与本项目相同的高级服务理念：HTTP 层负责请求解析和响应格式化，而生成的工作被提交到异步引擎路径。在 vLLM 的聊天服务路径中，`create_chat_completion` 构建引擎请求并调用 `engine_client.generate(...)`；返回的生成器随后被流式响应路径或完整响应路径消费。

主要相似之处在于 API 并发与引擎执行的分离。HTTP 请求允许并发到达，但实际的模型执行集中在引擎调度器后面。这也是本项目使用围绕 `LLM.submit()` 和 `LLM.step()` 的单个后台工作线程，而不是让每个 FastAPI 处理器独立调用 `LLM.generate()` 的原因。

主要区别在于调度器粒度。

在本项目中，当序列容量和预预留 KV 块容量都允许时，调度器将整个请求接纳为运行状态：

```python
len(running) < max_num_seqs
reserved_blocks_total + seq.reserved_blocks <= total_num_blocks
```

一旦接纳，新请求会为其运行时提示词接收完整的预填充步骤。这很简单，符合当前的 PD 分离实现，但意味着大的 `max_tokens` 请求可以预留足够的 KV 块来阻止后续请求进入运行状态。

vLLM 的 V1 调度器使用每次调度迭代的 token 预算。它从 `max_num_batched_tokens` 初始化 `token_budget`，先调度运行中的请求，然后在 token 预算和序列容量允许的情况下调度等待中的请求。调度器还将调度的 token 限制在请求限制和 `max_model_len` 内。这意味着接纳和执行由每步调度的 token 驱动，而不仅仅是整个请求的预留。

vLLM 还有更完整的 KV 缓存策略。当 KV 缓存空间不足时，vLLM 可以抢占请求并在稍后重新计算它们。这是一种生产环境的鲁棒性机制；本项目目前保持更简单的行为：仅在块足够时接纳，否则将请求留在 `waiting` 状态。

分块预填充是另一个故意的差异。vLLM V1 在可能时启用分块预填充：大的预填充可以分成较小的块，并在 `max_num_batched_tokens` 预算下与解码请求批处理。这改善了解码的 token 间延迟和 GPU 利用率，因为内存受限的解码工作可以与计算受限的预填充工作共享批处理。本项目明确不实现分块预填充，因此请求的预填充在接纳后仍作为一个预填充单元调度。

由此产生的权衡是：

- 本项目更容易理解：HTTP 队列、调度器等待队列、完整预填充接纳，然后解码
- vLLM 在混合工作负载下更灵活：token 预算调度、分块预填充、抢占、更丰富的 KV 缓存计算
- 本项目需要一起调整 `max_tokens`、`num_blocks` 和 `max_num_seqs`，因为 `max_tokens` 直接影响整个请求的块预留
- vLLM 可以使用 `max_num_batched_tokens` 调整每次迭代调度的工作量，特别是当启用分块预填充时

对于当前项目目标，vLLM 比较支持当前设计而不是替换它。API 修复应保持单工作线程的 `submit/step` 循环。如果需要后续的调度器改进，应该是围绕 token 预算计算的有意识的调度器重新设计。它不应隐藏在 HTTP API 层内部。

## 当前边界

本实现有意保持当前架构精简：

- 无分块预填充
- 无新的推理回退路径
- 无多引擎执行
- 无遗留 completions 端点
- 无认证、模型列表、指标、取消、超时或重试层

API 连续批处理路径现在是：

```text
HTTP 请求
  -> incoming 队列
  -> 后台引擎工作线程
  -> LLM.submit()
  -> 调度器等待队列
  -> LLM.step()
  -> 预填充/解码
  -> HTTP 响应或 SSE 块
```

## 实际运行指南

对于连续批处理实验，一起选择 `max_tokens`、`max_num_seqs` 和 `num_blocks`。

示例：

```bash
python examples/api/start_server.py \
  --model-path /path/to/Qwen3-0.6B \
  --port 7000 \
  --max-model-len 512 \
  --max-num-seqs 2 \
  --num-blocks 32
```

然后测试并发请求：

```bash
python examples/api/test_chat_completions.py \
  --url http://127.0.0.1:7000/v1/chat/completions \
  --concurrent 2 \
  --max-tokens 128 \
  --prompt "explain kv cache" \
  --prompt "explain kv cache"
```

如果显著提高 `max_tokens`，也必须相应提高 `num_blocks`，以便多个请求可以并发接纳。

## 参考资料

- vLLM OpenAI API 服务器：https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/api_server/
- vLLM 聊天服务路径：https://docs.vllm.ai/en/v0.14.0/api/vllm/entrypoints/openai/serving_chat/
- vLLM V1 调度器源码文档：https://docs.vllm.ai/en/v0.11.1/api/vllm/v1/core/sched/scheduler/
- vLLM 调度器配置：https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM 优化指南（包括分块预填充）：https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
