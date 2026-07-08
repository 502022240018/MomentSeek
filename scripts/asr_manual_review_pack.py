from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


LABELS = [
    "",
    "correct",
    "minor_error",
    "wrong_word",
    "wrong_entity",
    "hallucination",
    "language_issue",
    "unclear",
]

REASON_PRIORITY = [
    "mixed_script",
    "repeated_phrase",
    "long_sparse_text",
    "short_cjk_token",
    "very_short_normalized",
]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record.get("video_id") or ""),
        int(record.get("chunk_id") or -1),
        int(record.get("start_ms") or -1),
        int(record.get("end_ms") or -1),
    )


def _primary_reason(record: dict[str, Any]) -> str:
    reasons = [str(value) for value in record.get("suspect_reasons") or []]
    for reason in REASON_PRIORITY:
        if reason in reasons:
            return reason
    return reasons[0] if reasons else "unknown"


def _ordered_video_ids(records: list[dict[str, Any]], focus_video_ids: list[str] | None) -> list[str]:
    seen = []
    for video_id in focus_video_ids or []:
        if video_id and video_id not in seen:
            seen.append(video_id)
    for record in records:
        video_id = str(record.get("video_id") or "")
        if video_id and video_id not in seen:
            seen.append(video_id)
    return seen


def select_review_records(
    records: Iterable[dict[str, Any]],
    *,
    sample_size: int = 40,
    focus_video_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for record in records:
        unique.setdefault(_record_key(record), dict(record))
    ordered = sorted(unique.values(), key=lambda row: (str(row.get("video_id") or ""), int(row.get("start_ms") or 0)))
    video_ids = _ordered_video_ids(ordered, focus_video_ids)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in ordered:
        buckets.setdefault((str(record.get("video_id") or ""), _primary_reason(record)), []).append(record)

    bucket_order: list[tuple[str, str]] = []
    focus_set = set(focus_video_ids or [])
    for video_id in video_ids:
        weight = 2 if video_id in focus_set else 1
        for reason in REASON_PRIORITY + ["unknown"]:
            if (video_id, reason) not in buckets:
                continue
            for _ in range(weight):
                bucket_order.append((video_id, reason))

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int, int, int]] = set()
    cursor = {bucket: 0 for bucket in bucket_order}
    while len(selected) < sample_size:
        made_progress = False
        for bucket in bucket_order:
            values = buckets[bucket]
            index = cursor[bucket]
            if index >= len(values):
                continue
            record = values[index]
            cursor[bucket] = index + 1
            key = _record_key(record)
            if key in selected_keys:
                continue
            selected.append(record)
            selected_keys.add(key)
            made_progress = True
            if len(selected) >= sample_size:
                break
        if not made_progress:
            break
    return selected


def _find_video_file(upload_dir: Path, video_id: str) -> Path | None:
    matches = sorted(upload_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def _relative_video_url(video_path: Path, output_path: Path) -> str:
    relative = os.path.relpath(video_path, start=output_path.parent)
    return Path(relative).as_posix()


def _video_fragment(record: dict[str, Any]) -> str:
    offset_ms = int(record.get("media_start_offset_ms") or 0)
    start_ms = max(0, int(record.get("start_ms") or 0) + offset_ms - 500)
    end_ms = max(start_ms + 500, int(record.get("end_ms") or 0) + offset_ms + 500)
    record["video_seek_start_ms"] = start_ms
    record["video_seek_end_ms"] = end_ms
    start = start_ms / 1000.0
    end = end_ms / 1000.0
    return f"#t={start:.3f},{end:.3f}"


def _probe_media_start_seconds(video_path: Path) -> float:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return 0.0
    try:
        result = subprocess.run(
            [get_ffmpeg_exe(), "-hide_banner", "-i", str(video_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return 0.0
    text = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Duration:\s*[^,]+,\s*start:\s*([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    return max(0.0, float(match.group(1)))


def _media_start_offsets(upload_dir: Path, records: list[dict[str, Any]]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for record in records:
        video_id = str(record.get("video_id") or "")
        if not video_id or video_id in offsets:
            continue
        video_path = _find_video_file(upload_dir, video_id)
        offsets[video_id] = _probe_media_start_seconds(video_path) if video_path else 0.0
    return offsets


def _attach_video_urls(
    records: list[dict[str, Any]],
    upload_dir: Path,
    output_path: Path,
    *,
    media_start_offsets: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    offsets = media_start_offsets if media_start_offsets is not None else _media_start_offsets(upload_dir, records)
    hydrated = []
    for record in records:
        row = dict(record)
        video_path = _find_video_file(upload_dir, str(row.get("video_id") or ""))
        offset_seconds = float(offsets.get(str(row.get("video_id") or ""), 0.0))
        row["media_start_offset_ms"] = int(round(offset_seconds * 1000))
        if video_path:
            row["video_url"] = _relative_video_url(video_path, output_path) + _video_fragment(row)
            row["video_file"] = video_path.name
        else:
            row["video_url"] = ""
            row["video_file"] = ""
        hydrated.append(row)
    return hydrated


def _json_for_script(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2).replace("</", "<\\/")


def _label_options(current: str) -> str:
    values = []
    for label in LABELS:
        selected = " selected" if label == current else ""
        text = label or "unlabeled"
        values.append(f"<option value=\"{html.escape(label)}\"{selected}>{html.escape(text)}</option>")
    return "".join(values)


def _write_review_html(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        video_html = (
            f"<video controls preload=\"metadata\" src=\"{html.escape(record['video_url'])}\"></video>"
            if record.get("video_url")
            else "<span class=\"missing\">missing video file</span>"
        )
        rows.append(
            f"<tr data-index=\"{index}\">"
            f"<td class=\"num\">{index + 1}</td>"
            f"<td><div class=\"video-name\">{html.escape(str(record.get('video_name') or record.get('video_id') or ''))}</div>"
            f"<div class=\"meta\">{html.escape(str(record.get('video_file') or ''))}</div>{video_html}</td>"
            f"<td><div class=\"meta\">chunk {int(record.get('chunk_id') or 0)}</div>"
            f"<div>ASR {int(record.get('start_ms') or 0)}-{int(record.get('end_ms') or 0)} ms</div>"
            f"<div class=\"meta\">player {int(record.get('video_seek_start_ms') or 0)}-{int(record.get('video_seek_end_ms') or 0)} ms</div>"
            f"<div class=\"meta\">media offset +{int(record.get('media_start_offset_ms') or 0) / 1000.0:.3f}s</div>"
            f"<div class=\"meta\">{html.escape(', '.join(record.get('suspect_reasons') or []))}</div></td>"
            f"<td class=\"asr-text\">{html.escape(str(record.get('asr_text') or ''))}</td>"
            f"<td><select data-field=\"manual_label\">{_label_options(str(record.get('manual_label') or ''))}</select></td>"
            f"<td><textarea data-field=\"correct_text\" rows=\"3\">{html.escape(str(record.get('correct_text') or ''))}</textarea></td>"
            f"<td><textarea data-field=\"query_should_hit\" rows=\"3\">{html.escape(str(record.get('query_should_hit') or ''))}</textarea></td>"
            f"<td><textarea data-field=\"notes\" rows=\"3\">{html.escape(str(record.get('notes') or ''))}</textarea></td>"
            "</tr>"
        )
    output_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>ASR Manual Review</title>"
        "<style>"
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#202124;background:#fafafa}"
        "header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:14px}"
        "h1{font-size:22px;margin:0}button{border:1px solid #1a73e8;background:#1a73e8;color:white;padding:8px 12px;border-radius:4px;cursor:pointer}"
        ".toolbar{display:flex;gap:8px;align-items:center}.hint{font-size:13px;color:#5f6368;margin:8px 0 16px}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed;background:white}"
        "td,th{border:1px solid #dadce0;padding:6px;vertical-align:top;word-break:break-word}"
        "th{background:#f1f3f4;font-size:13px;text-align:left}.num{width:38px;text-align:right;color:#5f6368}"
        "video{width:100%;max-height:140px;background:#111}.video-name{font-weight:600}.meta{font-size:12px;color:#5f6368;margin-top:4px}"
        ".asr-text{font-size:15px;line-height:1.4}textarea,select{width:100%;box-sizing:border-box;font:inherit}"
        "textarea{resize:vertical}.missing{color:#b3261e;font-size:13px}"
        "</style></head><body>"
        "<header><div><h1>ASR Manual Review</h1>"
        f"<div class=\"hint\">{len(records)} records. 填写 correct_text、manual_label、query_should_hit 后导出 JSONL。</div></div>"
        "<div class=\"toolbar\"><button onclick=\"exportJsonl()\">导出 JSONL</button><button onclick=\"copyJsonl()\">复制 JSONL</button></div></header>"
        "<table><thead><tr><th>#</th><th>video</th><th>time/reasons</th><th>ASR text</th><th>manual_label</th><th>correct_text</th><th>query_should_hit</th><th>notes</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f"<script type=\"application/json\" id=\"review-data\">{_json_for_script(records)}</script>"
        "<script>"
        "const records=JSON.parse(document.getElementById('review-data').textContent);"
        "function collectRecords(){document.querySelectorAll('tr[data-index]').forEach(row=>{const i=Number(row.dataset.index);row.querySelectorAll('[data-field]').forEach(input=>{records[i][input.dataset.field]=input.value;});});return records;}"
        "function toJsonl(){return collectRecords().map(row=>JSON.stringify(row)).join('\\n')+'\\n';}"
        "function exportJsonl(){const blob=new Blob([toJsonl()],{type:'application/x-ndjson;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='asr_error_review_20260707.filled.jsonl';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}"
        "async function copyJsonl(){await navigator.clipboard.writeText(toJsonl());}"
        "</script></body></html>",
        encoding="utf-8",
    )


def create_review_pack(
    candidates_path: str | Path,
    output_path: str | Path,
    *,
    upload_dir: str | Path = "runtime-server/uploads",
    sample_size: int = 40,
    focus_video_ids: list[str] | None = None,
    media_start_offsets: dict[str, float] | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    records = select_review_records(
        _load_jsonl(candidates_path),
        sample_size=sample_size,
        focus_video_ids=focus_video_ids,
    )
    hydrated = _attach_video_urls(records, Path(upload_dir), output, media_start_offsets=media_start_offsets)
    _write_review_html(hydrated, output)
    return {"html_path": str(output), "records": len(hydrated)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a single-file ASR manual review HTML that can export JSONL.")
    parser.add_argument("--candidates", required=True, help="Candidate JSONL from scripts/asr_error_candidates.py")
    parser.add_argument("--out", default="runtime/analysis/asr_error_review_20260707.html")
    parser.add_argument("--upload-dir", default="runtime-server/uploads")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--focus-video-id", action="append", default=[])
    args = parser.parse_args()
    result = create_review_pack(
        args.candidates,
        args.out,
        upload_dir=args.upload_dir,
        sample_size=args.sample_size,
        focus_video_ids=args.focus_video_id,
    )
    print(result["html_path"])
    print(json.dumps({"records": result["records"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
