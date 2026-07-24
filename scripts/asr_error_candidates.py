from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.indexing.asr_text import normalize_asr_text, normalize_search_text


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _video_names(runtime_dir: Path) -> dict[str, str]:
    db_path = runtime_dir / "catalog.sqlite3"
    if not db_path.exists():
        return {}
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return {str(row["id"]): str(row["name"]) for row in connection.execute("select id, name from videos")}
    finally:
        connection.close()


def _load_asr_chunks(asr_path: Path) -> list[dict[str, Any]]:
    with np.load(asr_path, allow_pickle=True) as data:
        times = data["chunk_times_ms"].astype(np.int64)
        texts = data["texts"].astype(str)
    return [
        {
            "chunk_id": index,
            "start_ms": int(times[index][0]),
            "end_ms": int(times[index][1]),
            "text": str(texts[index]),
        }
        for index in range(len(texts))
    ]


def _token_repetition(text: str) -> bool:
    tokens = [token for token in re.split(r"[\s,，。.!?！？、]+", text) if token]
    if len(tokens) < 3:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values(), default=0) >= 3


def suspect_reasons(text: str, duration_ms: int) -> list[str]:
    normalized = normalize_asr_text(text)
    compact = normalize_search_text(normalized)
    cjk_count = len(_CJK_RE.findall(normalized))
    latin_count = len(_LATIN_RE.findall(normalized))
    reasons: list[str] = []
    if cjk_count and cjk_count <= 4 and latin_count == 0:
        reasons.append("short_cjk_token")
    if _token_repetition(normalized):
        reasons.append("repeated_phrase")
    if duration_ms >= 7000 and cjk_count <= 8 and latin_count == 0:
        reasons.append("long_sparse_text")
    if cjk_count and latin_count:
        reasons.append("mixed_script")
    if compact and len(compact) <= 2:
        reasons.append("very_short_normalized")
    return reasons


def iter_candidate_records(runtime_dir: str | Path, limit_per_video: int = 200) -> Iterable[dict[str, Any]]:
    runtime_path = Path(runtime_dir)
    names = _video_names(runtime_path)
    for asr_path in sorted(runtime_path.glob("indexes/*/asr.npz")):
        video_id = asr_path.parent.name
        emitted = 0
        for chunk in _load_asr_chunks(asr_path):
            duration_ms = max(0, int(chunk["end_ms"]) - int(chunk["start_ms"]))
            text = str(chunk["text"] or "").strip()
            reasons = suspect_reasons(text, duration_ms)
            if not reasons:
                continue
            yield {
                "video_id": video_id,
                "video_name": names.get(video_id, video_id),
                "chunk_id": int(chunk["chunk_id"]),
                "start_ms": int(chunk["start_ms"]),
                "end_ms": int(chunk["end_ms"]),
                "duration_ms": duration_ms,
                "asr_text": text,
                "normalized_text": normalize_asr_text(text),
                "suspect_reasons": reasons,
                "manual_label": "",
                "correct_text": "",
                "query_should_hit": "",
                "notes": "",
            }
            emitted += 1
            if emitted >= limit_per_video:
                break


def _record_to_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _write_html(records: list[dict[str, Any]], html_path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['video_name'])}</td>"
            f"<td>{record['start_ms']}-{record['end_ms']}</td>"
            f"<td>{html.escape(record['asr_text'])}</td>"
            f"<td>{html.escape(', '.join(record['suspect_reasons']))}</td>"
            f"<td>{html.escape(record.get('correct_text') or '')}</td>"
            f"<td>{html.escape(record.get('query_should_hit') or '')}</td>"
            "</tr>"
        )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ASR Error Candidates</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;word-break:break-word}"
        "th{background:#f4f4f4}</style></head><body>"
        "<h1>ASR Error Candidates</h1>"
        f"<p>Total candidates: {len(records)}</p>"
        "<table><tr><th>video</th><th>ms</th><th>ASR text</th><th>reasons</th><th>correct text</th><th>query</th></tr>"
        + "".join(rows)
        + "</table></body></html>",
        encoding="utf-8",
    )


def export_candidates(
    runtime_dir: str | Path,
    output_dir: str | Path,
    *,
    limit_per_video: int = 200,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = output_path / f"asr_error_candidates_{stamp}.jsonl"
    html_path = output_path / f"asr_error_candidates_{stamp}.html"
    records = list(iter_candidate_records(runtime_dir, limit_per_video=limit_per_video))
    jsonl_path.write_text("\n".join(_record_to_json(record) for record in records) + ("\n" if records else ""), encoding="utf-8")
    _write_html(records, html_path)
    return jsonl_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export suspicious ASR chunks for manual retrieval evaluation.")
    parser.add_argument("--runtime", default="runtime-server")
    parser.add_argument("--out", default="runtime/analysis")
    parser.add_argument("--limit-per-video", type=int, default=200)
    args = parser.parse_args()
    jsonl_path, html_path = export_candidates(args.runtime, args.out, limit_per_video=args.limit_per_video)
    print(jsonl_path)
    print(html_path)


if __name__ == "__main__":
    main()
