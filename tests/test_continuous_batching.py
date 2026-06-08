from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanovllm_ascend import SamplingParams
from nanovllm_ascend.engine import EngineCore, LLM
from nanovllm_ascend.scheduler import MiniScheduler


class FakeTokenIds:
    def __init__(self, values):
        self.values = values

    def numel(self):
        return len(self.values)


class FakeTokenizer:
    eos_token_id = -1

    def decode(self, token_ids, skip_special_tokens=True):
        return ",".join(str(token_id) for token_id in token_ids)


class FakeRunner:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.prefilled = []
        self.decoded = []
        self.freed = []

    def tokenize_prompts(self, prompts):
        return [FakeTokenIds(list(range(1, len(prompt) + 1))) for prompt in prompts]

    def prepare_sequences(self, seqs):
        for seq in seqs:
            seq.estimated_prompt_len = seq.prompt_token_ids.numel()
            seq.runtime_prompt_len = seq.estimated_prompt_len

    def prefill(self, seqs):
        for seq in seqs:
            self.prefilled.append(seq.seq_id)
            seq.set_prefill_result(seq.seq_id * 10, seq.estimated_prompt_len)

    def decode(self, seqs):
        for seq in seqs:
            self.decoded.append(seq.seq_id)
            token_id = seq.append_next_token()
            seq.set_decode_result(token_id + 1)

    def free_seq(self, seq):
        self.freed.append(seq.seq_id)


def make_llm(max_num_seqs=2):
    llm = object.__new__(LLM)
    llm.runner = FakeRunner()
    llm.scheduler = MiniScheduler(
        max_num_seqs=max_num_seqs,
        block_size=4,
        total_num_blocks=16,
    )
    llm.engine_core = EngineCore(llm.runner, llm.scheduler)
    return llm


def test_submit_adds_request_to_waiting_queue():
    llm = make_llm()

    request_id = llm.submit("hello", max_new_tokens=1)

    assert request_id == 0
    assert [seq.seq_id for seq in llm.scheduler.waiting] == [0]
    assert llm.has_unfinished()


def test_step_driven_continuous_batching_admits_later_request_after_capacity_frees():
    llm = make_llm(max_num_seqs=1)

    first_id = llm.submit("first", max_new_tokens=2)
    assert llm.step() == []
    second_id = llm.submit("second", max_new_tokens=1)

    assert llm.runner.prefilled == [first_id]
    assert [seq.seq_id for seq in llm.scheduler.waiting] == [second_id]

    assert llm.step() == []
    first_outputs = llm.step()
    assert [output["request_id"] for output in first_outputs] == [first_id]
    assert llm.runner.prefilled == [first_id]

    assert llm.step() == []
    second_outputs = llm.step()
    assert [output["request_id"] for output in second_outputs] == [second_id]
    assert llm.runner.prefilled == [first_id, second_id]


def test_generate_keeps_original_output_shape_and_order():
    llm = make_llm()

    outputs = llm.generate(["a", "bb"], max_new_tokens=1)

    assert outputs == [
        {"texts": "0", "token_ids": [0]},
        {"texts": "10", "token_ids": [10]},
    ]


def scheduler_state(llm):
    return {
        "waiting": [seq.seq_id for seq in llm.scheduler.waiting],
        "running": list(llm.scheduler.running),
        "finished": list(llm.scheduler.finished),
    }


def run_fake_demo():
    llm = make_llm(max_num_seqs=1)

    first_id = llm.submit("first", max_new_tokens=3)
    print(f"submit first request_id={first_id}")
    print(f"state={scheduler_state(llm)}")

    print("\nstep 1: prefill first")
    outputs = llm.step()
    print(f"outputs={outputs}")
    print(f"state={scheduler_state(llm)}")

    second_id = llm.submit("second", max_new_tokens=2)
    print(f"\nsubmit second request_id={second_id} while first is running")
    print(f"state={scheduler_state(llm)}")

    step_id = 2
    while llm.has_unfinished():
        outputs = llm.step()
        print(f"\nstep {step_id}")
        print(f"outputs={outputs}")
        print(f"state={scheduler_state(llm)}")
        step_id += 1


def run_real_demo(args):
    llm = LLM(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=1,
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
    print(f"state={scheduler_state(llm)}")

    t0 = time.perf_counter()
    outputs = llm.step()
    print("\nstep 1: prefill first")
    print(f"outputs={outputs}")
    print(f"state={scheduler_state(llm)}")

    second_id = llm.submit(
        args.prompt[1],
        max_new_tokens=args.second_max_new_tokens,
        sampling_params=sampling_params,
    )
    print(f"\nsubmit second request_id={second_id} while first is running")
    print(f"state={scheduler_state(llm)}")

    step_id = 2
    final_outputs = []
    while llm.has_unfinished():
        outputs = llm.step()
        final_outputs.extend(outputs)
        print(f"\nstep {step_id}")
        print(f"outputs={outputs}")
        print(f"state={scheduler_state(llm)}")
        step_id += 1

    total_tokens = sum(len(output["token_ids"]) for output in final_outputs)
    elapsed = time.perf_counter() - t0
    print("\n===== final outputs =====")
    for output in sorted(final_outputs, key=lambda item: item["request_id"]):
        print(f"\nrequest_id={output['request_id']}")
        print(f"token_ids={output['token_ids']}")
        print(output["texts"])
    print(f"\nelapsed={elapsed:.3f}s")
    print(f"tokens={total_tokens}")
    print(f"throughput={total_tokens / elapsed:.2f} tokens/s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("NANOVLLM_ASCEND_MODEL_PATH"))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--first-max-new-tokens", type=int, default=4)
    parser.add_argument("--second-max-new-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-prefix-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path is None:
        print("No --model-path provided; running fake continuous batching demo.")
        run_fake_demo()
        return

    if args.prompt is None:
        args.prompt = [
            "Explain continuous batching in one short paragraph.",
            "Give one sentence about KV cache reuse.",
        ]
    run_real_demo(args)


if __name__ == "__main__":
    main()
