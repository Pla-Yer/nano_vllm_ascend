from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7000/v1/chat/completions")
    parser.add_argument("--model", default="nanovllm-ascend")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def build_payload(args, prompt: str):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": args.stream,
    }


def post_json(url: str, payload: dict[str, object]):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream(url: str, payload: dict[str, object]):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    content_parts = []
    with urllib.request.urlopen(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            chunks.append(data)
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content_parts.append(delta["content"])
    return {
        "chunks": chunks,
        "content": "".join(content_parts),
    }


def run_request(args, index: int, prompt: str, results: list[dict[str, object] | None]):
    started = time.perf_counter()
    payload = build_payload(args, prompt)
    response = post_stream(args.url, payload) if args.stream else post_json(args.url, payload)
    elapsed = time.perf_counter() - started
    results[index] = {
        "elapsed": elapsed,
        "response": response,
    }


def print_result(index: int, result: dict[str, object]):
    response = result["response"]
    print(f"\n===== request {index} =====")
    print(f"elapsed={result['elapsed']:.3f}s")
    if "chunks" in response:
        print(f"chunks={len(response['chunks'])}")
        print(response["content"])
        return

    choice = response["choices"][0]
    print(f"id={response['id']}")
    print(f"usage={response['usage']}")
    print(choice["message"]["content"])


def main():
    args = parse_args()
    prompts = args.prompt or ["Hello, briefly introduce yourself."]
    while len(prompts) < args.concurrent:
        prompts.append(f"Test request {len(prompts)}. Answer in one short sentence.")

    results: list[dict[str, object] | None] = [None] * args.concurrent
    threads = [
        threading.Thread(
            target=run_request,
            args=(args, index, prompts[index], results),
        )
        for index in range(args.concurrent)
    ]

    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for index, result in enumerate(results):
        print_result(index, result)

    print(f"\ntotal_elapsed={time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
