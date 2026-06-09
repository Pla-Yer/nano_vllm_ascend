from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .engine import LLM
from .sampling_params import SamplingParams


@dataclass
class PendingRequest:
    model: str
    created: int
    stream: bool
    request_id: int = -1
    prompt_tokens: int = 0
    completion_id: str = ""
    emitted_chars: int = 0
    output: dict[str, object] | None = None
    submitted: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    stream_queue: queue.Queue[str | None] = field(default_factory=queue.Queue)


@dataclass
class QueuedRequest:
    prompt: str
    max_new_tokens: int
    sampling_params: SamplingParams
    pending: PendingRequest


def message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content)


def messages_to_prompt(messages: list[dict[str, object]]) -> str:
    lines = [
        f"{message['role']}: {message_content_to_text(message['content'])}"
        for message in messages
    ]
    lines.append("assistant:")
    return "\n".join(lines)


class OpenAIChatService:
    def __init__(self, llm: LLM, model_name: str):
        self.llm = llm
        self.model_name = model_name
        self.incoming: queue.Queue[QueuedRequest] = queue.Queue()
        self.completed: queue.Queue[int] = queue.Queue()
        self.pending: dict[int, PendingRequest] = {}
        self.worker = threading.Thread(target=self.run_engine, daemon=True)
        self.worker.start()

    def run_engine(self) -> None:
        while True:
            self.drop_completed()
            submitted = self.submit_incoming()
            active = self.llm.has_unfinished()

            if active:
                outputs = self.llm.step()
                self.emit_stream_deltas()
                self.finish_outputs(outputs)

            if not active and not submitted:
                time.sleep(0.001)

    def submit_incoming(self) -> bool:
        submitted = False
        while True:
            try:
                item = self.incoming.get_nowait()
            except queue.Empty:
                return submitted

            request_id = self.llm.submit(
                item.prompt,
                max_new_tokens=item.max_new_tokens,
                sampling_params=item.sampling_params,
            )
            seq = self.llm.scheduler.seqs[request_id]
            item.pending.request_id = request_id
            item.pending.prompt_tokens = seq.estimated_prompt_len
            item.pending.completion_id = f"chatcmpl-{request_id}"
            self.pending[request_id] = item.pending
            item.pending.submitted.set()
            submitted = True

    def drop_completed(self) -> None:
        while True:
            try:
                request_id = self.completed.get_nowait()
            except queue.Empty:
                return
            self.pending.pop(request_id, None)

    def finish_outputs(self, outputs: list[dict[str, object]]) -> None:
        for output in outputs:
            request_id = int(output["request_id"])
            pending = self.pending[request_id]
            pending.output = output
            if pending.stream:
                pending.stream_queue.put(self.make_stream_finish_chunk(pending))
                pending.stream_queue.put(None)
            pending.finished.set()

    def emit_stream_deltas(self) -> None:
        for request_id, pending in list(self.pending.items()):
            if not pending.stream:
                continue
            seq = self.llm.scheduler.seqs[request_id]
            text = self.llm.runner.tokenizer.decode(
                seq.generated_token_ids,
                skip_special_tokens=True,
            )
            delta = text[pending.emitted_chars :]
            if delta:
                pending.emitted_chars = len(text)
                pending.stream_queue.put(self.make_stream_delta_chunk(pending, delta))

    def queue_chat_request(
        self,
        request: dict[str, object],
        stream: bool,
    ) -> PendingRequest:
        pending = PendingRequest(
            model=str(request.get("model", self.model_name)),
            created=int(time.time()),
            stream=stream,
        )
        self.incoming.put(
            QueuedRequest(
                prompt=messages_to_prompt(request["messages"]),
                max_new_tokens=int(request.get("max_tokens", 128)),
                sampling_params=SamplingParams(
                    temperature=float(request.get("temperature", 0.0)),
                    top_k=0,
                    top_p=float(request.get("top_p", 1.0)),
                ),
                pending=pending,
            )
        )
        return pending

    def create_chat_completion(self, request: dict[str, object]) -> dict[str, object]:
        pending = self.queue_chat_request(request, stream=False)
        pending.submitted.wait()
        pending.finished.wait()
        output = pending.output
        completion_tokens = len(output["token_ids"])
        self.completed.put(pending.request_id)

        return {
            "id": pending.completion_id,
            "object": "chat.completion",
            "created": pending.created,
            "model": pending.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["texts"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": pending.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": pending.prompt_tokens + completion_tokens,
            },
        }

    def stream_chat_completion(self, request: dict[str, object]) -> Iterator[str]:
        pending = self.queue_chat_request(request, stream=True)
        pending.submitted.wait()
        yield self.make_stream_role_chunk(pending)

        while True:
            item = pending.stream_queue.get()
            if item is None:
                break
            yield item

        self.completed.put(pending.request_id)
        yield "data: [DONE]\n\n"

    def make_stream_role_chunk(self, pending: PendingRequest) -> str:
        return self.make_sse_chunk(
            pending,
            delta={"role": "assistant"},
            finish_reason=None,
        )

    def make_stream_delta_chunk(self, pending: PendingRequest, content: str) -> str:
        return self.make_sse_chunk(
            pending,
            delta={"content": content},
            finish_reason=None,
        )

    def make_stream_finish_chunk(self, pending: PendingRequest) -> str:
        return self.make_sse_chunk(
            pending,
            delta={},
            finish_reason="stop",
        )

    def make_sse_chunk(
        self,
        pending: PendingRequest,
        delta: dict[str, str],
        finish_reason: str | None,
    ) -> str:
        chunk = {
            "id": pending.completion_id,
            "object": "chat.completion.chunk",
            "created": pending.created,
            "model": pending.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def create_app(service: OpenAIChatService):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat_completions(request: dict[str, object]):
        if request.get("stream"):
            return StreamingResponse(
                service.stream_chat_completion(request),
                media_type="text/event-stream",
            )
        return service.create_chat_completion(request)

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="nanovllm-ascend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--npu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--enable-prefix-cache", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--warm-prompt", default="warm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        device_id=args.device_id,
        npu_memory_utilization=args.npu_memory_utilization,
        enable_prefix_cache=args.enable_prefix_cache,
    )
    print(f"KV cache blocks: {llm.runner.num_blocks}")
    if args.warm:
        llm.warm(args.warm_prompt)
    service = OpenAIChatService(llm=llm, model_name=args.model_name)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
