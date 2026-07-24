from __future__ import annotations

import csv
import glob
import html
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("eval/visual/outputs")
CHART_DIR = OUT_DIR / "charts_910b"


def pct(value: float | str) -> float:
    return float(value) * 100.0


def fmt(value: float | str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float | str, digits: int = 1) -> str:
    return f"{pct(value):.{digits}f}%"


def model_label(name: str) -> str:
    if "B-32" in name:
        return "ViT-B-32"
    if "B-16" in name:
        return "ViT-B-16"
    if "L-14" in name:
        return "ViT-L-14"
    return name


def load_effect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(OUT_DIR / "clip_eval_all_910b_*.local.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = model_label(data["model"]["model_name"])
        for task in ("image", "sequence"):
            task_data = data.get(task) or {}
            for strategy, result in (task_data.get("strategies") or {}).items():
                overall = result["overall"]
                rows.append({
                    "model": model,
                    "task": task,
                    "strategy": strategy,
                    "queries": overall["queries"],
                    "recall_at_1": overall["recall_at_1"],
                    "recall_at_5": overall["recall_at_5"],
                    "recall_at_10": overall["recall_at_10"],
                    "recall_at_20": overall["recall_at_20"],
                    "mrr": overall["mrr"],
                    "median_rank": overall["median_rank"],
                    "mean_rank": overall["mean_rank"],
                    "elapsed_seconds": data.get("elapsed_seconds"),
                    "source_json": Path(path).name,
                })
    return rows


def load_speed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(OUT_DIR / "clip_speed_benchmark_*_910b_warm.server.csv"))):
        with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                model = model_label(row.get("model_name") or row.get("model") or "")
                parsed = {"source_csv": Path(path).name}
                parsed.update(row)
                parsed["model"] = model
                for key in [
                    "items",
                    "source_prepare_seconds",
                    "preprocess_cpu_seconds",
                    "transfer_to_device_seconds",
                    "encode_device_seconds",
                    "to_cpu_seconds",
                    "encode_pipeline_seconds",
                    "views",
                    "views_per_second_total",
                    "views_per_second_encoder_only",
                    "score_cpu_seconds",
                    "score_queries_per_second",
                    "total_without_text_seconds",
                    "model_load_seconds",
                    "warmup_seconds",
                    "run_elapsed_seconds",
                ]:
                    if key in parsed and parsed[key] != "":
                        parsed[key] = float(parsed[key])
                rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(effect_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in sorted({row["model"] for row in effect_rows}):
        for task in ("image", "sequence"):
            candidates = [row for row in effect_rows if row["model"] == model and row["task"] == task]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: (row["mrr"], row["recall_at_10"], row["recall_at_1"]))
            rows.append(best)
    return rows


def find_speed(speed_rows: list[dict[str, Any]], model: str, task: str, scenario: str) -> dict[str, Any] | None:
    for row in speed_rows:
        if row["model"] == model and row.get("task") == task and row.get("scenario") == scenario:
            return row
    return None


def speed_scenario_for_strategy(strategy: str) -> str:
    if strategy.startswith("sliding_") and strategy.endswith("_center_crop"):
        return strategy[: -len("_center_crop")]
    if strategy.startswith("cells_sliding_") and strategy.endswith("_center_crop"):
        return strategy[: -len("_center_crop")]
    return strategy


def write_bar_svg(
    path: Path,
    title: str,
    values: list[tuple[str, float]],
    *,
    unit: str = "",
    width: int = 920,
    bar_height: int = 28,
    gap: int = 12,
    max_value: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_value = max_value or max((value for _, value in values), default=1.0)
    left = 260
    right = 150
    top = 58
    height = top + len(values) * (bar_height + gap) + 30
    chart_width = width - left - right
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:14px;fill:#222}.title{font-size:20px;font-weight:700}.axis{fill:#666;font-size:12px}.bar{fill:#7c3aed}.bar2{fill:#06b6d4}</style>',
        f'<text class="title" x="20" y="30">{html.escape(title)}</text>',
    ]
    for idx, (label, value) in enumerate(values):
        y = top + idx * (bar_height + gap)
        w = 0 if max_value <= 0 else chart_width * value / max_value
        css = "bar" if idx % 2 == 0 else "bar2"
        lines.append(f'<text x="20" y="{y + 19}">{html.escape(label)}</text>')
        lines.append(f'<rect class="{css}" x="{left}" y="{y}" width="{w:.1f}" height="{bar_height}" rx="5"/>')
        lines.append(f'<text x="{left + w + 8:.1f}" y="{y + 19}">{value:.2f}{html.escape(unit)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stacked_svg(
    path: Path,
    title: str,
    rows: list[tuple[str, dict[str, float]]],
    *,
    width: int = 980,
    bar_height: int = 30,
    gap: int = 16,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        ("source_prepare_seconds", "#94a3b8", "source/crop"),
        ("preprocess_cpu_seconds", "#f97316", "CPU preprocess"),
        ("transfer_to_device_seconds", "#64748b", "transfer"),
        ("encode_device_seconds", "#7c3aed", "NPU encode"),
        ("to_cpu_seconds", "#22c55e", "to CPU"),
        ("score_cpu_seconds", "#06b6d4", "CPU score"),
    ]
    totals = [sum(row.get(key, 0.0) for key, _, _ in parts) for _, row in rows]
    max_total = max(totals or [1.0])
    left = 250
    right = 140
    top = 76
    chart_width = width - left - right
    height = top + len(rows) * (bar_height + gap) + 72
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#222}.title{font-size:20px;font-weight:700}.legend{font-size:12px}</style>',
        f'<text class="title" x="20" y="30">{html.escape(title)}</text>',
    ]
    lx = 20
    for key, color, label in parts:
        lines.append(f'<rect x="{lx}" y="48" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{lx + 18}" y="59">{html.escape(label)}</text>')
        lx += 132
    for idx, (label, row) in enumerate(rows):
        y = top + idx * (bar_height + gap)
        x = left
        total = totals[idx]
        lines.append(f'<text x="20" y="{y + 20}">{html.escape(label)}</text>')
        for key, color, _ in parts:
            value = row.get(key, 0.0)
            w = 0 if max_total <= 0 else chart_width * value / max_total
            if w > 0:
                lines.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_height}" fill="{color}"/>')
            x += w
        lines.append(f'<text x="{left + chart_width + 12}" y="{y + 20}">{total:.2f}s</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[list[str]], header: list[str]) -> str:
    all_rows = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def main() -> None:
    effect = load_effect_rows()
    speed = load_speed_rows()
    best = best_rows(effect)
    write_csv(OUT_DIR / "visual_clip_910b_effect_summary.csv", effect)
    write_csv(OUT_DIR / "visual_clip_910b_speed_summary.csv", speed)
    write_csv(OUT_DIR / "visual_clip_910b_best_summary.csv", best)

    best_chart_values = []
    for row in best:
        best_chart_values.append((f"{row['model']} {row['task']} best MRR", float(row["mrr"])))
    write_bar_svg(CHART_DIR / "best_mrr.svg", "Best MRR by model/task on MomentSeek eval", best_chart_values, max_value=1.0)

    r10_values = []
    for row in best:
        r10_values.append((f"{row['model']} {row['task']} R@10", pct(row["recall_at_10"])))
    write_bar_svg(CHART_DIR / "best_recall_at_10.svg", "Best Recall@10 by model/task", r10_values, unit="%", max_value=100.0)

    selected_scenarios = [
        ("image", "center_crop"),
        ("image", "sliding_mvp_mix"),
        ("sequence", "sheet_whole_center_crop"),
        ("sequence", "cells_mean_center_crop"),
        ("sequence", "cells_sliding_top3"),
    ]
    speed_values = []
    for model in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        for task, scenario in selected_scenarios:
            row = find_speed(speed, model, task, scenario)
            if row:
                speed_values.append((f"{model} {task}/{scenario}", float(row["total_without_text_seconds"])))
    write_bar_svg(CHART_DIR / "selected_total_seconds.svg", "910B total seconds for selected scenarios", speed_values, unit="s")

    stacked_rows = []
    for model in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        row = find_speed(speed, model, "sequence", "cells_sliding_top3")
        if row:
            stacked_rows.append((model, {key: float(row.get(key, 0.0) or 0.0) for key in [
                "source_prepare_seconds",
                "preprocess_cpu_seconds",
                "transfer_to_device_seconds",
                "encode_device_seconds",
                "to_cpu_seconds",
                "score_cpu_seconds",
            ]}))
    write_stacked_svg(CHART_DIR / "sequence_cells_sliding_top3_breakdown.svg", "910B timing breakdown: sequence/cells_sliding_top3", stacked_rows)

    best_table = []
    for row in best:
        scenario = row["strategy"]
        speed_row = find_speed(speed, row["model"], row["task"], speed_scenario_for_strategy(scenario))
        speed_text = fmt(speed_row["total_without_text_seconds"]) + "s" if speed_row else "-"
        best_table.append([
            row["model"],
            row["task"],
            scenario,
            fmt_pct(row["recall_at_1"]),
            fmt_pct(row["recall_at_5"]),
            fmt_pct(row["recall_at_10"]),
            fmt(row["mrr"], 3),
            fmt(row["median_rank"], 1),
            speed_text,
        ])

    speed_table = []
    for model in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        for task, scenario in selected_scenarios:
            row = find_speed(speed, model, task, scenario)
            if not row:
                continue
            speed_table.append([
                model,
                task,
                scenario,
                str(int(row["views"])),
                fmt(row["source_prepare_seconds"]),
                fmt(row["preprocess_cpu_seconds"]),
                fmt(row["transfer_to_device_seconds"]),
                fmt(row["encode_device_seconds"]),
                fmt(row["score_cpu_seconds"]),
                fmt(row["total_without_text_seconds"]),
                fmt(row["views_per_second_encoder_only"], 1),
            ])

    top_strategy_lines = []
    for model in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        for task in ("image", "sequence"):
            candidates = sorted(
                [row for row in effect if row["model"] == model and row["task"] == task],
                key=lambda row: (row["mrr"], row["recall_at_10"], row["recall_at_1"]),
                reverse=True,
            )[:5]
            top_strategy_lines.append(f"### {model} / {task}")
            top_strategy_lines.append("")
            top_strategy_lines.append(markdown_table(
                [[
                    row["strategy"],
                    fmt_pct(row["recall_at_1"]),
                    fmt_pct(row["recall_at_5"]),
                    fmt_pct(row["recall_at_10"]),
                    fmt(row["mrr"], 3),
                    fmt(row["median_rank"], 1),
                ] for row in candidates],
                ["strategy", "R@1", "R@5", "R@10", "MRR", "Median rank"],
            ))
            top_strategy_lines.append("")

    md = [
        "# Visual CLIP 910B Evaluation Report",
        "",
        "This report summarizes MomentSeek visual retrieval evaluation on the drama-server Ascend 910B environment. Speed numbers are server-side 910B measurements only; local RTX 3060 measurements are not used here.",
        "",
        "## Environment",
        "",
        "- Server: `drama-server` / `cluster-worker-poeub`",
        "- Container: `momentseek-current-app`",
        "- Device mapping: container `npu:0` = host physical NPU 2 via `ASCEND_VISIBLE_DEVICES=2`",
        "- Batch size: image/video embedding batch `128`, text batch `256`",
        "- Eval set: 300 image items, 200 sequence/contact-sheet items; max 3 generated queries per item",
        "- Speed methodology: benchmark does a warmup image+text batch before per-scenario timing. Per-scenario totals exclude model load, warmup, and text embedding time; rows include those fields separately in CSV.",
        "",
        "## Headline best strategy per model/task",
        "",
        markdown_table(best_table, ["Model", "Task", "Best strategy", "R@1", "R@5", "R@10", "MRR", "Median rank", "910B seconds"]),
        "",
        "![Best MRR](charts_910b/best_mrr.svg)",
        "",
        "![Best Recall@10](charts_910b/best_recall_at_10.svg)",
        "",
        "## Selected 910B speed breakdown",
        "",
        markdown_table(speed_table, [
            "Model",
            "Task",
            "Scenario",
            "Views",
            "Source/crop s",
            "CPU preprocess s",
            "Transfer s",
            "NPU encode s",
            "CPU score s",
            "Total s",
            "Encoder views/s",
        ]),
        "",
        "![Selected total seconds](charts_910b/selected_total_seconds.svg)",
        "",
        "![Sequence cells sliding breakdown](charts_910b/sequence_cells_sliding_top3_breakdown.svg)",
        "",
        "## Top strategies by model/task",
        "",
        *top_strategy_lines,
        "## Output files",
        "",
        "- `visual_clip_910b_effect_summary.csv`: all strategy effectiveness metrics",
        "- `visual_clip_910b_speed_summary.csv`: all 910B timing rows and timing components",
        "- `visual_clip_910b_best_summary.csv`: best strategy per model/task",
        "- `charts_910b/*.svg`: report figures",
        "",
        "## Notes",
        "",
        "- `center_crop` is the vanilla CLIP preprocessing path.",
        "- `letterbox` keeps aspect ratio and pads to CLIP square input; it avoids geometric distortion but may shrink small objects.",
        "- `sliding_*` crops multiple square windows along the long edge, embeds each view, and aggregates per original frame/segment.",
        "- `sheet_whole_*` embeds the whole 5s contact sheet as one image; `cells_*` splits the contact sheet back into sampled frames/cells before aggregation.",
    ]
    (OUT_DIR / "visual_clip_910b_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "effect_rows": len(effect),
        "speed_rows": len(speed),
        "report": str(OUT_DIR / "visual_clip_910b_report.md"),
        "charts": str(CHART_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
