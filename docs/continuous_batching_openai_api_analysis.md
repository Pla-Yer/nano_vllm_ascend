# Continuous Batching and OpenAI API Analysis

Date: 2026-06-08

## Scope

This note records the implementation, debugging, and validation of the OpenAI-style API continuous batching path in this repository.

The current runtime keeps prefill and decode separated. This work does not introduce chunked prefill and does not change the model execution path. HTTP requests are routed into the existing `LLM.submit()` and `LLM.step()` interface so that one engine loop owns NPU execution.

## Design

The API service is implemented in `src/nanovllm_ascend/openai_server.py`.

The service exposes:

- `POST /v1/chat/completions`
- non-streaming OpenAI-style `chat.completion` responses
- streaming `chat.completion.chunk` SSE responses when `stream=true`

The important runtime rule is that HTTP handlers do not call `LLM.generate()` and do not call `LLM.step()` directly. Each handler builds a prompt, creates a pending request object, and pushes it into an incoming queue.

A single background engine worker then repeatedly:

1. drains incoming HTTP requests
2. calls `LLM.submit()` to append them to the scheduler waiting queue
3. calls `LLM.step()` when there is unfinished work
4. dispatches finished outputs or streaming deltas back to the corresponding HTTP request

This keeps FastAPI request handling concurrent while preserving a single NPU engine execution owner.

## Scheduler Behavior

The scheduler remains prefill/decode separated.

Each `LLM.step()` calls `MiniScheduler.plan_next_step()`, which performs the following sequence:

1. finish running sequences that have reached EOS or `max_new_tokens`
2. admit waiting sequences into running if capacity allows
3. schedule prefill for newly admitted sequences
4. schedule decode for running sequences that already have a next token

Continuous batching in this implementation means that a later request can enter the scheduler waiting queue while earlier requests are already running, then be admitted at a later step boundary if capacity is available.

It does not mean a request can be inserted into the middle of an already executing NPU kernel. It also does not mean a late request avoids its own prefill cost.

## Capacity Conditions

A waiting request can move into running only when both conditions are true:

```python
len(running) < max_num_seqs
reserved_blocks_total + seq.reserved_blocks <= total_num_blocks
```

`seq.reserved_blocks` is computed from:

```python
runtime_prompt_len + max_new_tokens
```

This is important for API testing. If the client requests a large `max_tokens`, the scheduler reserves enough KV blocks for that full generation budget. With the API test script defaulting to `--max-tokens 1024`, a single request can reserve most of the default KV block pool.

In that case, the second HTTP request is already accepted by the API and submitted to the scheduler, but it remains in `waiting` because the scheduler cannot admit it until the first request frees KV blocks. The observed latency then looks close to serial execution even though the API queueing path is correct.

## Test vs API Difference

The fake and real continuous batching tests use small generation budgets and enough scheduler capacity. Under those conditions, a second request submitted after several steps can quickly move from `waiting` to `running`.

The API initially looked worse because the real run used API defaults with a much larger `max_tokens` budget. That increased per-request block reservation and prevented concurrent admission under the default `num_blocks`.

This was not caused by FastAPI serialization, Python lock contention, or a missing scheduler submit. It was caused by scheduler capacity.

## API Validation Result

After using capacity-compatible settings, two concurrent API requests with the same prompt:

```bash
--prompt "explain kv cache"
```

completed with nearly identical elapsed time:

```text
total_elapsed=14.198s
total_elapsed=14.211s
```

This matches the expected continuous batching behavior for overlapping requests: both requests share the same engine loop and progress together rather than one request waiting for the other to finish.

## FIA Mask Issue

During API testing, setting:

```python
parser.add_argument("--max-model-len", type=int, default=512)
```

caused an Ascend NPU error in `npu_fused_infer_attention_score`:

```text
when sparseMode is 3, the input mask has 2 dims in total, maskDim 1 shall be 2048
```

This happened even without `--enable-prefix-cache`.

The cause was that `max_model_len` was used for two different concepts:

1. maximum accepted prompt length
2. physical causal mask shape passed to Ascend FIA

When `max_model_len=512`, the prefill attention module created a `512 x 512` mask. The Ascend FIA split-fuse path used by the current `sparse_mode=3` call expects the mask width to be `2048`.

The fix keeps `max_model_len` as the request length limit but creates the FIA mask with a minimum physical size of 2048:

```python
self.fia_mask_len = max(max_mask_len, 2048)
```

This keeps short-context API runs valid while satisfying the NPU kernel's mask shape requirement.

## Comparison With vLLM

vLLM's OpenAI-compatible chat server follows the same high-level serving idea as this project: the HTTP layer does request parsing and response formatting, while generation work is submitted to an asynchronous engine path. In vLLM's chat serving path, `create_chat_completion` builds an engine request and calls `engine_client.generate(...)`; the returned generator is then consumed either by the streaming response path or by the full-response path.

The main similarity is the separation between API concurrency and engine execution. HTTP requests are allowed to arrive concurrently, but actual model execution is centralized behind the engine scheduler. This is the same reason this project uses a single background worker around `LLM.submit()` and `LLM.step()` instead of letting each FastAPI handler call `LLM.generate()` independently.

The main difference is scheduler granularity.

In this project, the scheduler admits a whole request into running when both sequence capacity and pre-reserved KV block capacity allow it:

```python
len(running) < max_num_seqs
reserved_blocks_total + seq.reserved_blocks <= total_num_blocks
```

Once admitted, a new request receives a full prefill step for its runtime prompt. This is simple and matches the current PD-separated implementation, but it means a large `max_tokens` request can reserve enough KV blocks to block later requests from entering running.

vLLM's V1 scheduler uses a token budget per scheduling iteration. It initializes a `token_budget` from `max_num_batched_tokens`, schedules running requests first, and then schedules waiting requests while token budget and sequence capacity remain. The scheduler also caps scheduled tokens by request limits and `max_model_len`. This means admission and execution are driven by per-step scheduled tokens rather than only by whole-request reservation.

vLLM also has a more complete KV-cache policy. When KV cache space is insufficient, vLLM can preempt requests and later recompute them. This is a production robustness mechanism; this project currently keeps the simpler behavior of admitting only when enough blocks are available and otherwise leaving requests in `waiting`.

Chunked prefill is another deliberate difference. vLLM V1 enables chunked prefill when possible: large prefills can be split into smaller chunks and batched with decode requests under the `max_num_batched_tokens` budget. This improves decode inter-token latency and GPU utilization because memory-bound decode work can share a batch with compute-bound prefill work. This project explicitly does not implement chunked prefill now, so a request's prefill is still scheduled as one prefill unit after admission.

The resulting tradeoff is:

- this project is easier to reason about: HTTP queue, scheduler waiting queue, whole-prefill admission, then decode
- vLLM is more flexible under mixed workloads: token-budget scheduling, chunked prefill, preemption, richer KV-cache accounting
- this project requires `max_tokens`, `num_blocks`, and `max_num_seqs` to be tuned together because `max_tokens` directly affects whole-request block reservation
- vLLM can use `max_num_batched_tokens` to tune the amount of work scheduled per iteration, especially when chunked prefill is enabled

For the current project goal, the vLLM comparison supports the current design rather than replacing it. The API fix should remain the single-worker `submit/step` loop. The next scheduler improvement, if needed later, should be a conscious scheduler redesign around token-budget accounting. It should not be hidden inside the HTTP API layer.

## Current Boundaries

This implementation intentionally keeps the current architecture small:

- no chunked prefill
- no new inference fallback path
- no multi-engine execution
- no legacy completions endpoint
- no auth, model listing, metrics, cancellation, timeout, or retry layer

The API continuous batching path is now:

```text
HTTP request
  -> incoming queue
  -> background engine worker
  -> LLM.submit()
  -> scheduler waiting queue
  -> LLM.step()
  -> prefill/decode
  -> HTTP response or SSE chunks
```

## Practical Run Guidance

For continuous batching experiments, choose `max_tokens`, `max_num_seqs`, and `num_blocks` together.

Example:

```bash
python examples/api/start_server.py \
  --model-path /path/to/Qwen3-0.6B \
  --port 7000 \
  --max-model-len 512 \
  --max-num-seqs 2 \
  --num-blocks 32
```

Then test concurrent requests:

```bash
python examples/api/test_chat_completions.py \
  --url http://127.0.0.1:7000/v1/chat/completions \
  --concurrent 2 \
  --max-tokens 128 \
  --prompt "explain kv cache" \
  --prompt "explain kv cache"
```

If `max_tokens` is raised significantly, `num_blocks` must also be raised enough for multiple requests to be admitted concurrently.

## References

- vLLM OpenAI API server: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/api_server/
- vLLM chat serving path: https://docs.vllm.ai/en/v0.14.0/api/vllm/entrypoints/openai/serving_chat/
- vLLM V1 scheduler source docs: https://docs.vllm.ai/en/v0.11.1/api/vllm/v1/core/sched/scheduler/
- vLLM scheduler config: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM optimization guide, including chunked prefill: https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
