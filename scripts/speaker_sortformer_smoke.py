from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemo.collections.asr.models import SortformerEncLabelModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sortformer and emit the speaker review schema.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="nvidia/diar_sortformer_4spk-v1")
    parser.add_argument("--audio-seconds", type=float, required=True)
    args = parser.parse_args()

    model = SortformerEncLabelModel.from_pretrained(args.model, map_location="cuda")
    model.eval()
    prediction = model.diarize(audio=str(args.audio), batch_size=1)[0]
    turns = []
    for segment in prediction:
        start, end, raw_speaker = segment.split()
        speaker_index = int(raw_speaker.rsplit("_", 1)[1])
        turns.append(
            {
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "speaker": f"SPEAKER_{speaker_index:02d}",
            }
        )
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    payload = {
        "model": args.model,
        "model_max_speakers": 4,
        "audio_seconds": args.audio_seconds,
        "speakers": sorted({item["speaker"] for item in turns}),
        "num_turns": len(turns),
        "turns": turns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
