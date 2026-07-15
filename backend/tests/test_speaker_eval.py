import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "speaker_eval.py"
SPEC = importlib.util.spec_from_file_location("speaker_eval", SCRIPT)
speaker_eval = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(speaker_eval)


def test_load_cases_validates_local_media(tmp_path):
    media = tmp_path / "demo.mp4"
    media.write_bytes(b"video")
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "cases": [{
            "id": "demo",
            "media_path": str(media),
            "start_seconds": 1,
            "end_seconds": 3,
            "language": "zh",
            "scenario": ["dialogue"],
        }],
    }), encoding="utf-8")

    cases = speaker_eval.load_cases(manifest)

    assert cases[0]["id"] == "demo"
    assert cases[0]["start_seconds"] == 1.0
    assert cases[0]["end_seconds"] == 3.0


def test_load_cases_rejects_invalid_time_range(tmp_path):
    media = tmp_path / "demo.mp4"
    media.write_bytes(b"video")
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "cases": [{"id": "demo", "media_path": str(media), "start_seconds": 3, "end_seconds": 3}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid time range"):
        speaker_eval.load_cases(manifest)
