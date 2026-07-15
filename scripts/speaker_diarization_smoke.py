from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline


def load_pcm16_mono(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sample width {sample_width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(samples).unsqueeze(0), sample_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "--exclusive", action="store_true",
        help="Export the non-overlapping exclusive diarization timeline.",
    )
    args = parser.parse_args()

    waveform, sample_rate = load_pcm16_mono(args.audio)
    if args.max_seconds is not None:
        waveform = waveform[:, : int(args.max_seconds * sample_rate)]
    device = torch.device(args.device)
    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(args.model)
    pipeline.to(device)
    loaded_seconds = time.perf_counter() - started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    run_seconds = []
    output = None
    for _ in range(args.repeat):
        infer_started = time.perf_counter()
        output = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        run_seconds.append(time.perf_counter() - infer_started)
    assert output is not None
    infer_seconds = run_seconds[-1]

    diarization = (
        output.exclusive_speaker_diarization
        if args.exclusive
        else output.speaker_diarization
    )
    turns = [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, speaker in diarization
    ]
    audio_seconds = waveform.shape[1] / sample_rate
    result = {
        "audio": str(args.audio),
        "audio_seconds": round(audio_seconds, 3),
        "load_seconds": round(loaded_seconds, 3),
        "infer_seconds": round(infer_seconds, 3),
        "run_seconds": [round(value, 3) for value in run_seconds],
        "rtf": round(infer_seconds / audio_seconds, 4),
        "speakers": sorted({turn["speaker"] for turn in turns}),
        "num_turns": len(turns),
        "speaker_count_mode": "oracle" if args.num_speakers is not None else "automatic",
        "requested_speaker_count": args.num_speakers,
        "timeline": "exclusive" if args.exclusive else "regular",
        "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None,
        "turns": turns,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
