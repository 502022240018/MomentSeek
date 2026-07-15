from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def load_cases(path: str | Path) -> list[dict]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("speaker cases must use schema_version=1 and contain a cases list")
    seen: set[str] = set()
    cases = []
    for index, item in enumerate(payload["cases"]):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = str(item.get("id") or "").strip()
        media_path = Path(str(item.get("media_path") or ""))
        start = float(item.get("start_seconds", 0))
        end = float(item.get("end_seconds", 0))
        if not case_id or case_id in seen:
            raise ValueError(f"cases[{index}] has an empty or duplicate id")
        if not media_path.is_file():
            raise FileNotFoundError(f"case media is missing: {media_path}")
        if start < 0 or end <= start:
            raise ValueError(f"case {case_id} has an invalid time range")
        seen.add(case_id)
        cases.append({**item, "id": case_id, "media_path": str(media_path), "start_seconds": start, "end_seconds": end})
    if not cases:
        raise ValueError("speaker cases list is empty")
    return cases


def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("ffmpeg is required to prepare speaker evaluation audio") from exc


def prepare_case(case: dict, output_dir: Path, overwrite: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{case['id']}.wav"
    duration = float(case["end_seconds"]) - float(case["start_seconds"])
    if overwrite or not audio_path.is_file() or audio_path.stat().st_size == 0:
        command = [
            ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(case['start_seconds']):.3f}", "-i", str(case["media_path"]),
            "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        subprocess.run(command, check=True)
    return {
        "id": case["id"],
        "audio_path": str(audio_path.resolve()),
        "audio_seconds": round(duration, 3),
        "language": case.get("language"),
        "scenario": case.get("scenario", []),
        "truth_rttm": case.get("truth_rttm"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate speaker evaluation cases.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--cases", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cases", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.command == "validate":
        print(json.dumps({"validated": len(cases)}, ensure_ascii=False))
        return 0

    output_dir = Path(args.output_dir)
    prepared = [prepare_case(case, output_dir, args.overwrite) for case in cases]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "cases": prepared}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"prepared": len(prepared), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
