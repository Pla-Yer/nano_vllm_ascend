from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import SamplingParams
from nanovllm_ascend.engine import LLM


def scheduler_state(llm):
    return {
        "waiting": [seq.seq_id for seq in llm.scheduler.waiting],
        "running": list(llm.scheduler.running),
        "finished": list(llm.scheduler.finished),
    }


def step_once(llm, step_id: int, final_outputs: list, title: str | None = None):
    start = time.perf_counter()
    outputs = llm.step()
    step_latency = time.perf_counter() - start

    final_outputs.extend(outputs)

    if title:
        print(f"\nstep {step_id} {title}")
    else:
        print(f"\nstep {step_id}")

    print(f"step latency={step_latency:.3f}s")
    print(f"outputs={outputs}")
    print(f"state={scheduler_state(llm)}")

    return outputs


def run_real_demo(args):
    if args.prompt is None:
        args.prompt = [
            "Explain continuous batching",
            "Explain kv cache",
        ]

    if len(args.prompt) < 2:
        raise ValueError("real demo needs at least two prompts. Please pass --prompt twice.")

    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        device_id=args.device_id,
        enable_prefix_cache=args.enable_prefix_cache,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    first_id = llm.submit(
        args.prompt[0],
        max_new_tokens=args.first_max_new_tokens,
        sampling_params=sampling_params,
    )

    print(f"submit first request_id={first_id}")
    print(f"first_reserved_blocks={llm.scheduler.seqs[first_id].reserved_blocks}")
    print(f"state={scheduler_state(llm)}")

    t0 = time.perf_counter()
    final_outputs = []

    for step_id in range(1, args.second_submit_after_steps + 1):
        step_once(
            llm=llm,
            step_id=step_id,
            final_outputs=final_outputs,
            title="before second submit",
        )

    second_id = llm.submit(
        args.prompt[1],
        max_new_tokens=args.second_max_new_tokens,
        sampling_params=sampling_params,
    )

    print(f"\nsubmit second request_id={second_id} while first is running")
    print(f"second_reserved_blocks={llm.scheduler.seqs[second_id].reserved_blocks}")
    print(f"max_num_seqs={args.max_num_seqs}")
    print(f"num_blocks={args.num_blocks}")
    print(f"state={scheduler_state(llm)}")

    step_id = args.second_submit_after_steps + 1

    while llm.has_unfinished():
        step_once(
            llm=llm,
            step_id=step_id,
            final_outputs=final_outputs,
        )
        step_id += 1

    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(output["token_ids"]) for output in final_outputs)

    print("\n===== final outputs =====")

    for output in sorted(final_outputs, key=lambda item: item["request_id"]):
        print(f"\nrequest_id={output['request_id']}")
        print(f"token_ids={output['token_ids']}")
        print(output["texts"])

    print(f"\nelapsed={elapsed:.3f}s")
    print(f"tokens={total_tokens}")

    if elapsed > 0:
        print(f"throughput={total_tokens / elapsed:.2f} tokens/s")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=os.environ.get("NANOVLLM_ASCEND_MODEL_PATH"),
        required=os.environ.get("NANOVLLM_ASCEND_MODEL_PATH") is None,
    )

    parser.add_argument("--prompt", action="append")
    parser.add_argument("--first-max-new-tokens", type=int, default=128)
    parser.add_argument("--second-max-new-tokens", type=int, default=128)

    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--max-num-seqs", type=int, default=2)

    parser.add_argument("--second-submit-after-steps", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--enable-prefix-cache", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    run_real_demo(args)


if __name__ == "__main__":
    main()