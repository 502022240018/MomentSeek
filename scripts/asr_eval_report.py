from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt(value: Any, digits: int) -> str:
    return f"{_as_float(value):.{digits}f}"


def _load_summary(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ASR eval summary: {summary_path}")
    return _read_json(summary_path)


def overall_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in summary.get("runs") or []:
        if not isinstance(run, dict):
            continue
        aggregate = run.get("aggregate") if isinstance(run.get("aggregate"), dict) else {}
        timing = run.get("timing") if isinstance(run.get("timing"), dict) else {}
        speed = run.get("speed") if isinstance(run.get("speed"), dict) else {}
        audio_seconds = _as_float(aggregate.get("audio_seconds") or speed.get("audio_seconds"))
        rows.append(
            {
                "run": str(run.get("name") or ""),
                "samples": _as_int(aggregate.get("sample_count")),
                "audio_h": audio_seconds / 3600.0,
                "total_s": _as_float(timing.get("total_seconds")),
                "x_total": _as_float(speed.get("x_total")),
                "global_cer": _as_float(_dig(aggregate, "global_cer", "cer")),
                "win30_cer": _as_float(_dig(aggregate, "window30_cer", "cer")),
                "recall2": _as_float(_dig(aggregate, "local", "2000", "avg_recall")),
                "f1_2s": _as_float(_dig(aggregate, "local", "2000", "avg_f1")),
                "failed": _as_int(aggregate.get("failed_samples")),
            }
        )
    return rows


def _sample_ids(summary: dict[str, Any]) -> list[str]:
    setup = summary.get("setup") if isinstance(summary.get("setup"), dict) else {}
    samples = setup.get("samples") if isinstance(setup.get("samples"), list) else []
    ids: list[str] = []
    for sample in samples:
        if isinstance(sample, dict) and sample.get("sample_id"):
            ids.append(str(sample["sample_id"]))
    return ids


def _run_names(summary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for run in summary.get("runs") or []:
        if isinstance(run, dict) and run.get("name"):
            names.append(str(run["name"]))
    return names


def _sample_summary(eval_dir: Path, run_name: str, sample_id: str) -> dict[str, Any] | None:
    path = eval_dir / "runs" / run_name / "samples" / sample_id / "summary.json"
    if not path.exists():
        return None
    return _read_json(path)


def sample_runtime_rows(
    eval_dir: Path,
    summary: dict[str, Any],
    baseline_dir: Path | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = sample_ids or _sample_ids(summary)
    for sample_id in ids:
        for run_name in _run_names(summary):
            current = _sample_summary(eval_dir, run_name, sample_id)
            if current is None:
                continue
            baseline = _sample_summary(baseline_dir, run_name, sample_id) if baseline_dir else None
            current_s = _as_float(current.get("elapsed_seconds"))
            baseline_s = _as_float(baseline.get("elapsed_seconds")) if baseline else None
            delta_pct = None
            if baseline_s:
                delta_pct = ((current_s - baseline_s) / baseline_s) * 100.0
            rows.append(
                {
                    "sample": sample_id,
                    "run": run_name,
                    "current_s": current_s,
                    "baseline_s": baseline_s,
                    "delta_pct": delta_pct,
                    "current_cer": _as_float(_dig(current, "metrics", "global_cer", "cer")),
                    "current_recall2": _as_float(_dig(current, "metrics", "local", "2000", "avg_recall")),
                }
            )
    return rows


def _overall_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| run | samples | audio_h | total_s | x_total | global_cer | win30_cer | recall@2s | f1@2s | failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {samples} | {audio_h} | {total_s} | {x_total} | {global_cer} | {win30_cer} | {recall2} | {f1_2s} | {failed} |".format(
                run=row["run"],
                samples=row["samples"],
                audio_h=_fmt(row["audio_h"], 3),
                total_s=_fmt(row["total_s"], 1),
                x_total=_fmt(row["x_total"], 2),
                global_cer=_fmt(row["global_cer"], 3),
                win30_cer=_fmt(row["win30_cer"], 3),
                recall2=_fmt(row["recall2"], 3),
                f1_2s=_fmt(row["f1_2s"], 3),
                failed=row["failed"],
            )
        )
    return lines


def _sample_table(rows: list[dict[str, Any]], include_baseline: bool) -> list[str]:
    if include_baseline:
        lines = [
            "| sample | run | current_s | baseline_s | delta_pct | current_cer | current_recall@2s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            baseline_s = "" if row["baseline_s"] is None else _fmt(row["baseline_s"], 1)
            delta_pct = "" if row["delta_pct"] is None else _fmt(row["delta_pct"], 1)
            lines.append(
                "| {sample} | {run} | {current_s} | {baseline_s} | {delta_pct} | {current_cer} | {current_recall2} |".format(
                    sample=row["sample"],
                    run=row["run"],
                    current_s=_fmt(row["current_s"], 1),
                    baseline_s=baseline_s,
                    delta_pct=delta_pct,
                    current_cer=_fmt(row["current_cer"], 3),
                    current_recall2=_fmt(row["current_recall2"], 3),
                )
            )
        return lines

    lines = [
        "| sample | run | current_s | current_cer | current_recall@2s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {run} | {current_s} | {current_cer} | {current_recall2} |".format(
                sample=row["sample"],
                run=row["run"],
                current_s=_fmt(row["current_s"], 1),
                current_cer=_fmt(row["current_cer"], 3),
                current_recall2=_fmt(row["current_recall2"], 3),
            )
        )
    return lines


def build_markdown(
    eval_dir: Path | str,
    baseline_dir: Path | str | None = None,
    sample_ids: list[str] | None = None,
) -> str:
    eval_path = Path(eval_dir)
    baseline_path = Path(baseline_dir) if baseline_dir else None
    summary = _load_summary(eval_path)
    lines = [
        "# ASR Evaluation Report",
        "",
        f"eval_dir: `{eval_path}`",
    ]
    if baseline_path:
        lines.append(f"baseline_dir: `{baseline_path}`")
    lines.extend(["", "## Overall"])
    lines.extend(_overall_table(overall_rows(summary)))

    runtime_rows = sample_runtime_rows(eval_path, summary, baseline_dir=baseline_path, sample_ids=sample_ids)
    if runtime_rows:
        section = "Sample Runtime Comparison" if baseline_path else "Sample Runtime"
        lines.extend(["", f"## {section}"])
        lines.extend(_sample_table(runtime_rows, include_baseline=baseline_path is not None))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stable Markdown report from ASR evaluation summary.json files.")
    parser.add_argument("--eval-dir", required=True, type=Path, help="Evaluation directory containing summary.json")
    parser.add_argument("--baseline-dir", type=Path, help="Optional baseline evaluation directory")
    parser.add_argument("--sample-id", action="append", dest="sample_ids", help="Restrict sample runtime table to one sample id; repeatable")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file instead of stdout")
    args = parser.parse_args()

    markdown = build_markdown(args.eval_dir, baseline_dir=args.baseline_dir, sample_ids=args.sample_ids)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
