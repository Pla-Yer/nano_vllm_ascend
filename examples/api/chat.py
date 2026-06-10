"""Interactive multi-turn chat client for the OpenAI-compatible server.

Usage:
    1. Start the server:
       python examples/api/start_server.py --model-path <model> --warm

    2. Run this script:
       python examples/api/chat.py
       python examples/api/chat.py --url http://127.0.0.1:8000/v1/chat/completions
       python examples/api/chat.py --no-stream
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-turn chat client")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="Chat completions endpoint URL",
    )
    parser.add_argument("--model", default="nanovllm-ascend")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default=None, help="System prompt (optional)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    return parser.parse_args()


def build_payload(
    args,
    messages: list[dict[str, str]],
    stream: bool,
) -> dict:
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": stream,
    }


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream(url: str, payload: dict) -> str:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    with urllib.request.urlopen(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            chunk_data = line.removeprefix("data: ")
            if chunk_data == "[DONE]":
                break
            chunk = json.loads(chunk_data)
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                content_parts.append(text)
                print(text, end="", flush=True)
    print()
    return "".join(content_parts)


def chat_round(
    url: str,
    args,
    messages: list[dict[str, str]],
    stream: bool,
) -> str:
    payload = build_payload(args, messages, stream)
    if stream:
        return post_stream(url, payload)

    response = post_json(url, payload)
    content = response["choices"][0]["message"]["content"]
    usage = response["usage"]
    print(content)
    print(
        f"  [tokens: prompt={usage['prompt_tokens']}, "
        f"completion={usage['completion_tokens']}, "
        f"total={usage['total_tokens']}]"
    )
    return content


def main():
    args = parse_args()
    stream = not args.no_stream

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"Chat client connected to {args.url}")
    print(f"Model: {args.model}  Stream: {stream}")
    print("Type your message. Commands: /clear  /quit  /history")
    print("-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break

        if user_input == "/clear":
            messages.clear()
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("[conversation cleared]")
            continue

        if user_input == "/history":
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if len(content) > 120:
                    content = content[:120] + "..."
                print(f"  [{role}] {content}")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        assistant_reply = chat_round(args.url, args, messages, stream)

        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
