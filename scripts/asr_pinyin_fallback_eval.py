from __future__ import annotations

import argparse
import html
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from app.search import lexical_score


PinyinConverter = Callable[[str], list[str]]


def _default_pinyin_converter(text: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise RuntimeError(
            "pypinyin is required for this experiment. Install backend requirements or run `pip install pypinyin`."
        ) from exc
    return [value for value in lazy_pinyin(text, style=Style.NORMAL, errors="ignore") if value]


def _normalize_syllable(value: str) -> str:
    folded = value.casefold()
    for old, new in (
        ("zh", "z"),
        ("ch", "c"),
        ("sh", "s"),
        ("ang", "an"),
        ("eng", "en"),
        ("ing", "in"),
    ):
        folded = folded.replace(old, new)
    return folded


def pinyin_similarity(query: str, text: str, *, converter: PinyinConverter | None = None) -> float:
    converter = converter or _default_pinyin_converter
    query_values = [_normalize_syllable(value) for value in converter(query)]
    text_values = [_normalize_syllable(value) for value in converter(text)]
    if len(query_values) < 2 or not text_values:
        return 0.0
    query_joined = " ".join(query_values)
    text_joined = " ".join(text_values)
    if query_joined in text_joined:
        return 1.0
    return float(SequenceMatcher(None, query_joined, text_joined).ratio())


def _load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _load_chunks(runtime_dir: str | Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for asr_path in sorted(Path(runtime_dir).glob("indexes/*/asr.npz")):
        video_id = asr_path.parent.name
        with np.load(asr_path, allow_pickle=True) as data:
            times = data["chunk_times_ms"].astype(np.int64)
            texts = data["texts"].astype(str)
        for index in range(len(texts)):
            chunks.append({
                "video_id": video_id,
                "chunk_id": index,
                "start_ms": int(times[index][0]),
                "end_ms": int(times[index][1]),
                "text": str(texts[index]),
            })
    return chunks


def _target_matches(case: dict[str, Any], chunk: dict[str, Any]) -> bool:
    if str(case.get("video_id") or "") != str(chunk["video_id"]):
        return False
    target_start = int(case.get("target_start_ms", case.get("start_ms", -1)))
    target_end = int(case.get("target_end_ms", case.get("end_ms", -1)))
    return int(chunk["start_ms"]) == target_start and int(chunk["end_ms"]) == target_end


def _rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    converter: PinyinConverter | None,
    top_k: int,
    use_pinyin: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for chunk in chunks:
        lexical = lexical_score(query, str(chunk["text"]))
        pinyin = pinyin_similarity(query, str(chunk["text"]), converter=converter) if use_pinyin else 0.0
        score = max(lexical, min(0.72, pinyin))
        if score <= 0:
            continue
        ranked.append({**chunk, "score": float(score), "lexical_score": float(lexical), "pinyin_score": float(pinyin)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _evaluate_cases(
    cases: Iterable[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    converter: PinyinConverter | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    summary = {"cases": 0, "baseline_hits": 0, "pinyin_hits": 0, "rescued": 0}
    for case in cases:
        if case.get("should_hit", True) is False:
            continue
        summary["cases"] += 1
        query = str(case["query"])
        baseline = _rank_chunks(query, chunks, converter=converter, top_k=top_k, use_pinyin=False)
        pinyin = _rank_chunks(query, chunks, converter=converter, top_k=top_k, use_pinyin=True)
        baseline_hit = any(_target_matches(case, chunk) for chunk in baseline)
        pinyin_hit = any(_target_matches(case, chunk) for chunk in pinyin)
        summary["baseline_hits"] += int(baseline_hit)
        summary["pinyin_hits"] += int(pinyin_hit)
        summary["rescued"] += int(pinyin_hit and not baseline_hit)
        rows.append({
            "case": case,
            "baseline_hit": baseline_hit,
            "pinyin_hit": pinyin_hit,
            "baseline_top": baseline[:3],
            "pinyin_top": pinyin[:3],
        })
    return rows, summary


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, int], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "asr_pinyin_fallback_eval.json"
    html_path = output_dir / "asr_pinyin_fallback_eval.html"
    payload = {"summary": summary, "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    table_rows = []
    for row in rows:
        case = row["case"]
        pinyin_top = row["pinyin_top"][0] if row["pinyin_top"] else {}
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('id') or ''))}</td>"
            f"<td>{html.escape(str(case.get('query') or ''))}</td>"
            f"<td>{row['baseline_hit']}</td>"
            f"<td>{row['pinyin_hit']}</td>"
            f"<td>{html.escape(str(pinyin_top.get('text') or ''))}</td>"
            f"<td>{float(pinyin_top.get('pinyin_score') or 0):.3f}</td>"
            "</tr>"
        )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ASR Pinyin Fallback Eval</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;word-break:break-word}"
        "th{background:#f4f4f4}</style></head><body>"
        "<h1>ASR Pinyin Fallback Eval</h1>"
        f"<p>{html.escape(json.dumps(summary, ensure_ascii=False))}</p>"
        "<table><tr><th>case</th><th>query</th><th>baseline hit</th><th>pinyin hit</th><th>pinyin top</th><th>pinyin score</th></tr>"
        + "".join(table_rows)
        + "</table></body></html>",
        encoding="utf-8",
    )
    return json_path, html_path


def evaluate_file(
    runtime_dir: str | Path,
    eval_path: str | Path,
    output_dir: str | Path,
    *,
    converter: PinyinConverter | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    cases = _load_eval_cases(eval_path)
    chunks = _load_chunks(runtime_dir)
    rows, summary_counts = _evaluate_cases(cases, chunks, converter=converter, top_k=top_k)
    cases_count = max(1, summary_counts["cases"])
    summary = {
        **summary_counts,
        "top_k": top_k,
        "baseline_recall_at_k": summary_counts["baseline_hits"] / cases_count,
        "pinyin_recall_at_k": summary_counts["pinyin_hits"] / cases_count,
    }
    json_path, html_path = _write_outputs(rows, summary, Path(output_dir))
    return {"summary": summary, "json_path": str(json_path), "html_path": str(html_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR pinyin fallback on a jsonl retrieval eval set.")
    parser.add_argument("--runtime", default="runtime-server")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", default="runtime/analysis")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = evaluate_file(args.runtime, args.eval, args.out, top_k=args.top_k)
    print(result["json_path"])
    print(result["html_path"])
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
