from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import torch


def enable_trusted_official_checkpoints() -> None:
    """Restore the torch<2.6 default expected by DiariZen's official checkpoints."""
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = compatible_load


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / source.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export unmodified DiariZen diarization turns.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="BUT-FIT/diarizen-wavlm-base-s80-md")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    enable_trusted_official_checkpoints()
    from diarizen.pipelines.inference import DiariZenPipeline

    started = time.perf_counter()
    pipeline = DiariZenPipeline.from_pretrained(args.model)
    pipeline.segmentation_batch_size = args.batch_size
    pipeline.embedding_batch_size = args.batch_size
    loaded = time.perf_counter() - started
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = pipeline(str(args.audio), sess_name=args.audio.stem)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    turns = [
        {
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": str(speaker),
        }
        for turn, _, speaker in result.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    duration = duration_seconds(args.audio)
    payload = {
        "model": args.model,
        "audio_seconds": round(duration, 3),
        "load_seconds": round(loaded, 3),
        "infer_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4),
        "peak_cuda_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
        "batch_size": args.batch_size,
        "speakers": sorted({turn["speaker"] for turn in turns}),
        "num_turns": len(turns),
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "turns"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
