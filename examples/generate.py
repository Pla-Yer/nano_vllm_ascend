from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import LLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        device_id=args.device_id,
    )
    t = time.time()
    texts = llm.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    for i, text in enumerate(texts):
        print(f"\n===== output {i} =====")
        print(text)
    print(f"Generation time: {time.time() - t:.2f} s")
    throughput = sum(len(text) for text in texts) / (time.time() - t)
    print(f"Throughput: {throughput:.2f} chars/s")

if __name__ == "__main__":
    main()

