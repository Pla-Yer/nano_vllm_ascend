from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch_npu
from transformers import AutoConfig

from nanovllm_ascend.layers import (
    LMHead,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb_tnd,
)
from nanovllm_ascend.models.qwen3 import Qwen3MLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tokens", default="1,16,128,512")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def sync_npu() -> None:
    torch.npu.synchronize()


def init_module(module: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(mean=0.0, std=0.02)
    return module


@torch.inference_mode()
def bench_ms(fn, warmup_iters: int, iters: int) -> float:
    for _ in range(warmup_iters):
        fn()
    sync_npu()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_npu()

    return (time.perf_counter() - t0) * 1000.0 / iters


def stat(values: list[float]) -> dict[str, float]:
    xs = sorted(values)
    return {
        "avg": statistics.mean(xs),
        "min": xs[0],
        "max": xs[-1],
        "p50": xs[len(xs) // 2],
    }


def bench_layer(
    name: str,
    shape: str,
    fn,
    warmup_iters: int,
    iters: int,
    repeats: int = 3,
) -> dict:
    values = [
        bench_ms(fn, warmup_iters=warmup_iters, iters=iters)
        for _ in range(repeats)
    ]
    s = stat(values)
    row = {
        "name": name,
        "shape": shape,
        "avg_ms": s["avg"],
        "p50_ms": s["p50"],
        "min_ms": s["min"],
        "max_ms": s["max"],
    }
    print(
        f"{name:<22} {shape:<30} "
        f"avg={row['avg_ms']:.4f} ms "
        f"p50={row['p50_ms']:.4f} ms "
        f"min={row['min_ms']:.4f} ms "
        f"max={row['max_ms']:.4f} ms"
    )
    return row


def main() -> None:
    args = parse_args()

    torch.npu.set_device(args.device_id)
    device = "npu"
    dtype = get_dtype(args.dtype)

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokens_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    print("nanovllm_ascend layer benchmark")
    print(f"model_path={args.model_path}")
    print(f"device=npu:{args.device_id}")
    print(f"dtype={dtype}")
    print(f"hidden_size={hidden_size}")
    print(f"intermediate_size={intermediate_size}")
    print(f"num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}")
    print(f"head_dim={head_dim}")
    print(f"vocab_size={vocab_size}")
    print(f"tokens={tokens_list}")
    print(f"warmup_iters={args.warmup_iters}")
    print(f"iters={args.iters}")
    print()

    rows: list[dict] = []

    # 1. hidden RMSNorm: decoder input_layernorm / post_attention_layernorm / final norm
    hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps).to(
        device=device,
        dtype=dtype,
    )
    init_module(hidden_norm)

    # 2. head RMSNorm: q_norm / k_norm
    q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps).to(
        device=device,
        dtype=dtype,
    )
    init_module(q_norm)

    # 3. common Linear
    linear_hidden = Linear(hidden_size, hidden_size, bias=False).to(
        device=device,
        dtype=dtype,
    )
    init_module(linear_hidden)

    # 4. MLP
    mlp = Qwen3MLP(config).to(device=device, dtype=dtype)
    init_module(mlp)

    # 5. LMHead
    lm_head = LMHead(hidden_size, vocab_size, bias=False).to(
        device=device,
        dtype=dtype,
    )
    init_module(lm_head)

    # 6. RoPE
    rotary = RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=getattr(config, "max_position_embeddings", args.max_model_len),
        rope_theta=config.rope_theta,
    ).to(device=device)

    for total_tokens in tokens_list:
        x = torch.randn(
            total_tokens,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        rows.append(
            bench_layer(
                name="rms_norm_hidden",
                shape=f"[{total_tokens}, {hidden_size}]",
                fn=lambda x=x: hidden_norm(x),
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        )

        q = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

        rows.append(
            bench_layer(
                name="rms_norm_head",
                shape=f"[{total_tokens}, {num_heads}, {head_dim}]",
                fn=lambda q=q: q_norm(q),
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        )

        rows.append(
            bench_layer(
                name="linear_hidden",
                shape=f"[{total_tokens}, {hidden_size}] -> [{total_tokens}, {hidden_size}]",
                fn=lambda x=x: linear_hidden(x),
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        )

        rows.append(
            bench_layer(
                name="mlp",
                shape=f"[{total_tokens}, {hidden_size}] -> [{total_tokens}, {hidden_size}]",
                fn=lambda x=x: mlp(x),
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        )

        # prefill 当前实现会算所有 token 的 logits；
        # 后面 P2 会把它改成只算 last-token logits。
        rows.append(
            bench_layer(
                name="lm_head_all_tokens",
                shape=f"[{total_tokens}, {hidden_size}] -> [{total_tokens}, {vocab_size}]",
                fn=lambda x=x: lm_head(x),
                warmup_iters=args.warmup_iters,
                iters=max(10, args.iters // 5),
            )
        )

        x_last = torch.randn(
            1,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        rows.append(
            bench_layer(
                name="lm_head_one_token",
                shape=f"[1, {hidden_size}] -> [1, {vocab_size}]",
                fn=lambda x_last=x_last: lm_head(x_last),
                warmup_iters=args.warmup_iters,
                iters=max(10, args.iters // 5),
            )
        )

        position_ids = torch.arange(
            total_tokens,
            device=device,
            dtype=torch.long,
        )

        q_rope = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        k_rope = torch.randn(
            total_tokens,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

        def rope_full():
            cos, sin = rotary(position_ids.unsqueeze(0))
            return apply_rotary_pos_emb_tnd(
                q_rope,
                k_rope,
                cos.squeeze(0),
                sin.squeeze(0),
            )

        rows.append(
            bench_layer(
                name="rope_full",
                shape=f"q=[{total_tokens}, {num_heads}, {head_dim}], k=[{total_tokens}, {num_kv_heads}, {head_dim}]",
                fn=rope_full,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
        )

    result = {
        "config": {
            "model_path": args.model_path,
            "device_id": args.device_id,
            "dtype": args.dtype,
            "tokens": tokens_list,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
        },
        "rows": rows,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nwrote_json={args.output_json}")


if __name__ == "__main__":
    main()