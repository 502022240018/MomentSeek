from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk, SpeechUnit


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_asr_debug_artifacts(
    *,
    debug_dir: str | Path,
    enabled: bool,
    save_raw_transcript: bool,
    raw_items: Sequence[RawTranscriptItem],
    retrieval_chunks: Sequence[RetrievalChunk],
    repair_stats: Mapping[str, object],
    speech_units: Sequence[SpeechUnit] | None = None,
) -> None:
    if not enabled:
        return
    target = Path(debug_dir)
    target.mkdir(parents=True, exist_ok=True)
    if speech_units is not None:
        _write_json(target / "asr_speech_units.json", [unit.to_dict() for unit in speech_units])
    if save_raw_transcript:
        _write_json(target / "asr_raw_transcript.json", [item.to_dict() for item in raw_items])
    _write_json(target / "asr_retrieval_chunks.json", [chunk.to_dict() for chunk in retrieval_chunks])
    _write_json(target / "asr_repair_report.json", dict(repair_stats))
