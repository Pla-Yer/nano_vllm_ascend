from __future__ import annotations

import json
import threading

from nanovllm_ascend.openai_server import OpenAIChatService, messages_to_prompt

from tests.test_continuous_batching import make_llm


def chat_request(content: str, max_tokens: int = 1):
    return {
        "model": "qwen3-test",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def sse_payloads(chunks):
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: "):
                payloads.append(line.removeprefix("data: "))
    return payloads


def test_messages_to_prompt_keeps_roles_in_order():
    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Explain KV cache."},
        ]
    )

    assert prompt == "\n".join(
        [
            "system: You are concise.",
            "user: Hello",
            "assistant: Hi",
            "user: Explain KV cache.",
            "assistant:",
        ]
    )


def test_chat_completion_response_shape():
    service = OpenAIChatService(make_llm(), model_name="qwen3-test")

    response = service.create_chat_completion(chat_request("hello", max_tokens=1))

    assert response["object"] == "chat.completion"
    assert response["model"] == "qwen3-test"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["message"]["content"] == "0"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["completion_tokens"] == 1
    assert response["usage"]["total_tokens"] == (
        response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"]
    )


def test_stream_chat_completion_returns_delta_chunks():
    service = OpenAIChatService(make_llm(), model_name="qwen3-test")

    chunks = list(
        service.stream_chat_completion(
            {
                **chat_request("hello", max_tokens=2),
                "stream": True,
            }
        )
    )
    payloads = sse_payloads(chunks)
    json_payloads = [json.loads(payload) for payload in payloads[:-1]]

    assert payloads[-1] == "[DONE]"
    assert json_payloads[0]["object"] == "chat.completion.chunk"
    assert json_payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [
        payload["choices"][0]["delta"]
        for payload in json_payloads
        if payload["choices"][0]["delta"].get("content")
    ] == [{"content": "0"}, {"content": ",1"}]
    assert json_payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_concurrent_chat_requests_share_one_engine_loop():
    llm = make_llm(max_num_seqs=1)
    service = OpenAIChatService(llm, model_name="qwen3-test")
    responses = []

    threads = [
        threading.Thread(
            target=lambda text=text: responses.append(
                service.create_chat_completion(chat_request(text, max_tokens=2))
            )
        )
        for text in ("first", "second")
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(response["id"] for response in responses) == [
        "chatcmpl-0",
        "chatcmpl-1",
    ]
    assert llm.runner.prefilled == [0, 1]
