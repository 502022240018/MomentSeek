from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert 3D-Speaker JSON to the speaker review schema.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--audio-seconds", type=float, required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    turns = [
        {
            "start": round(float(item["start"]), 3),
            "end": round(float(item["stop"]), 3),
            "speaker": f"SPEAKER_{int(item['speaker']):02d}",
        }
        for item in source.values()
    ]
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    payload = {
        "model": "modelscope/3D-Speaker",
        "audio_seconds": args.audio_seconds,
        "speakers": sorted({item["speaker"] for item in turns}),
        "num_turns": len(turns),
        "turns": turns,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
