"""Single-image Qwen3-VL compatibility and latency smoke test on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers the Ascend backend with PyTorch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _gib(value: int) -> float:
    return round(value / 1024**3, 3)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="请简洁描述画面中的人物、物体、场景、动作和可读文字。",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs cannot be negative")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable inside the container")

    device = torch.device("npu:0")
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    torch.npu.synchronize()
    load_seconds = time.perf_counter() - load_started

    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.prompt},
        ],
    }]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[rendered], images=[image], padding=True, return_tensors="pt"
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    def infer(max_new_tokens: int) -> tuple[str, int, float]:
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started
        input_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = int(generated.shape[1] - input_tokens)
        answer = processor.batch_decode(
            generated[:, input_tokens:], skip_special_tokens=True
        )[0]
        return answer, new_tokens, elapsed

    for _ in range(args.warmup_runs):
        infer(min(8, args.max_new_tokens))

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    latencies: list[float] = []
    throughputs: list[float] = []
    answer = ""
    new_tokens = 0
    for _ in range(args.runs):
        answer, new_tokens, elapsed = infer(args.max_new_tokens)
        latencies.append(elapsed)
        throughputs.append(new_tokens / elapsed if elapsed else 0.0)

    result = {
        "model": str(args.model),
        "image": str(args.image),
        "device": str(device),
        "dtype": "bfloat16",
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "load_seconds": round(load_seconds, 3),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "samples": [round(value, 3) for value in latencies],
        },
        "tokens_per_second_mean": round(statistics.mean(throughputs), 3),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "new_tokens_last_run": new_tokens,
        "peak_allocated_gib": _gib(torch.npu.max_memory_allocated()),
        "peak_reserved_gib": _gib(torch.npu.max_memory_reserved()),
        "answer": answer,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
