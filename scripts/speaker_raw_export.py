from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export raw diarization turns for manual review.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    turns = payload["turns"]
    stem = args.json_path.with_suffix("")
    uri = stem.name

    rttm_lines = []
    label_lines = []
    for turn in turns:
        start = float(turn["start"])
        end = float(turn["end"])
        speaker = str(turn["speaker"])
        rttm_lines.append(
            f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
        label_lines.append(f"{start:.3f}\t{end:.3f}\t{speaker}")

    stem.with_suffix(".rttm").write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    stem.with_suffix(".labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    review = {
        "audio": str(args.audio.resolve()),
        "raw_json": str(args.json_path.resolve()),
        "rttm": str(stem.with_suffix(".rttm").resolve()),
        "audacity_labels": str(stem.with_suffix(".labels.txt").resolve()),
        "note": "Raw Community-1 output; no filtering, merging, or ASR alignment applied.",
    }
    stem.with_suffix(".review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
