from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.indexing.asr_postprocess import default_strategy_names, postprocess_asr_chunks, strategy_config


def _load_asr_chunks(path: Path) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        times = data["chunk_times_ms"].astype(np.int64)
        texts = data["texts"].astype(str)
    chunks: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        chunks.append({
            "start_ms": int(times[index][0]),
            "end_ms": int(times[index][1]),
            "text": str(text),
        })
    return chunks


def _manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _video_label(index_dir: Path) -> str:
    manifest = _manifest(index_dir)
    return str(
        manifest.get("filename")
        or manifest.get("name")
        or manifest.get("video_name")
        or manifest.get("video_id")
        or index_dir.name
    )


def _duration_stats(chunks: list[dict[str, Any]]) -> dict[str, float]:
    if not chunks:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    durations = np.asarray(
        [max(0, int(item.get("end_ms", 0)) - int(item.get("start_ms", 0))) for item in chunks],
        dtype=np.float32,
    )
    return {
        "mean": float(np.mean(durations) / 1000.0),
        "p50": float(np.percentile(durations, 50) / 1000.0),
        "p90": float(np.percentile(durations, 90) / 1000.0),
        "max": float(np.max(durations) / 1000.0),
    }


def _source_text(raw_chunks: list[dict[str, Any]], source_ids: list[int]) -> str:
    values = []
    for source_id in source_ids:
        if 0 <= int(source_id) < len(raw_chunks):
            values.append(str(raw_chunks[int(source_id)].get("text") or ""))
    return " | ".join(values)


def _example_rows(raw_chunks: list[dict[str, Any]], processed_chunks: list[dict[str, Any]], limit: int = 10) -> str:
    rows: list[str] = []
    merged_first = [
        item for item in processed_chunks
        if len(item.get("source_chunk_ids") or []) > 1
    ]
    samples = (merged_first + processed_chunks)[:limit]
    for index, item in enumerate(samples):
        source_ids = [int(value) for value in (item.get("source_chunk_ids") or [])]
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(_source_text(raw_chunks, source_ids))}</td>"
            f"<td>{html.escape(str(item.get('text') or ''))}</td>"
            f"<td>{int(item.get('start_ms', 0))}-{int(item.get('end_ms', 0))}</td>"
            f"<td>{html.escape(str(item.get('semantic_reason') or ''))}</td>"
            f"<td>{','.join(str(value) for value in source_ids)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _strategy_section(strategy: str, raw_chunks: list[dict[str, Any]], segment_ids: list[int]) -> tuple[str, str]:
    processed, stats = postprocess_asr_chunks(
        raw_chunks,
        segment_ids=segment_ids,
        config=strategy_config(strategy),
    )
    durations = _duration_stats(processed)
    short_chunks = sum(1 for item in processed if len(str(item.get("text") or "").replace(" ", "")) <= 8)
    semantic_chunks = sum(1 for item in processed if item.get("semantic_eligible", True))
    summary_row = (
        "<tr>"
        f"<td>{html.escape(strategy)}</td>"
        f"<td>{stats['raw_chunks']}</td>"
        f"<td>{stats['processed_chunks']}</td>"
        f"<td>{stats['merged_chunks']}</td>"
        f"<td>{short_chunks}</td>"
        f"<td>{semantic_chunks}</td>"
        f"<td>{stats['semantic_ineligible_chunks']}</td>"
        f"<td>{durations['mean']:.2f}</td>"
        f"<td>{durations['p90']:.2f}</td>"
        f"<td>{durations['max']:.2f}</td>"
        "</tr>"
    )
    example_block = (
        f"<h4>{html.escape(strategy)}</h4>"
        "<table>"
        "<tr><th>#</th><th>raw chunks</th><th>processed chunk</th><th>ms</th><th>semantic</th><th>source ids</th></tr>"
        + _example_rows(raw_chunks, processed)
        + "</table>"
    )
    return summary_row, example_block


def _index_section(asr_path: Path) -> str:
    index_dir = asr_path.parent
    raw_chunks = _load_asr_chunks(asr_path)
    segment_ids = [max(0, int(chunk["start_ms"]) // 5000) for chunk in raw_chunks]
    summary_rows: list[str] = []
    examples: list[str] = []
    for strategy in default_strategy_names():
        summary_row, example_block = _strategy_section(strategy, raw_chunks, segment_ids)
        summary_rows.append(summary_row)
        examples.append(example_block)
    return (
        f"<section><h2>{html.escape(_video_label(index_dir))}</h2>"
        "<table>"
        "<tr><th>strategy</th><th>raw</th><th>processed</th><th>merged</th><th>short</th>"
        "<th>semantic</th><th>semantic skipped</th><th>mean s</th><th>p90 s</th><th>max s</th></tr>"
        + "".join(summary_rows)
        + "</table>"
        + "".join(examples)
        + "</section>"
    )


def build_report(runtime_dir: str | Path, output_dir: str | Path) -> Path:
    runtime_path = Path(runtime_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_path / f"asr_postprocess_report_{stamp}.html"
    asr_paths = sorted(runtime_path.glob("indexes/*/asr.npz"))
    if asr_paths:
        sections = "".join(_index_section(path) for path in asr_paths)
    else:
        sections = "<p>No ASR indexes found.</p>"
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ASR Postprocess Strategy Report</title>"
        "<style>"
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;line-height:1.45;color:#222}"
        "section{margin:0 0 40px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 24px;table-layout:fixed}"
        "td,th{border:1px solid #cfcfcf;padding:6px;vertical-align:top;word-break:break-word}"
        "th{background:#f3f5f7}"
        "h1{margin-bottom:4px}"
        "p.meta{color:#666;margin-top:0}"
        "</style></head><body>"
        "<h1>ASR Postprocess Strategy Report</h1>"
        f"<p class='meta'>Generated at {html.escape(stamp)}. Runtime: {html.escape(str(runtime_path))}</p>"
        + sections
        + "</body></html>",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ASR postprocess merge strategies on existing indexes.")
    parser.add_argument("--runtime", default="runtime-server", help="Runtime directory containing indexes/*/asr.npz")
    parser.add_argument("--out", default="runtime/analysis", help="Output directory for the HTML report")
    args = parser.parse_args()
    print(build_report(args.runtime, args.out))


if __name__ == "__main__":
    main()
