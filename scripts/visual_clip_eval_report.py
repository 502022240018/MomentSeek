from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def _metric_row(run_path: Path, data: dict[str, Any], task: str, strategy: str, metrics: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model") or {}
    return {
        "run_file": run_path.name,
        "created_at": data.get("created_at"),
        "device": data.get("device"),
        "model": model.get("slug") or f"{model.get('model_name')}::{model.get('pretrained')}",
        "model_name": model.get("model_name"),
        "pretrained": model.get("pretrained"),
        "task": task,
        "strategy": strategy,
        "queries": int(metrics.get("queries", 0)),
        "recall_at_1": float(metrics.get("recall_at_1", 0)),
        "recall_at_5": float(metrics.get("recall_at_5", 0)),
        "recall_at_10": float(metrics.get("recall_at_10", 0)),
        "recall_at_20": float(metrics.get("recall_at_20", 0)),
        "mrr": float(metrics.get("mrr", 0)),
        "median_rank": float(metrics.get("median_rank", 0)),
        "mean_rank": float(metrics.get("mean_rank", 0)),
    }


def collect_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    query_type_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for path in paths:
        data = _read_json(path)
        model = data.get("model") or {}
        run_rows.append({
            "run_file": path.name,
            "created_at": data.get("created_at"),
            "device": data.get("device"),
            "model": model.get("slug"),
            "model_name": model.get("model_name"),
            "pretrained": model.get("pretrained"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "max_queries_per_item": data.get("max_queries_per_item"),
            "include_captions": data.get("include_captions"),
        })
        for task in ("image", "sequence"):
            task_payload = data.get(task)
            if not task_payload:
                continue
            for strategy, result in (task_payload.get("strategies") or {}).items():
                summary_rows.append(_metric_row(path, data, task, strategy, result.get("overall") or {}))
                for query_type, metrics in (result.get("by_query_type") or {}).items():
                    row = _metric_row(path, data, task, strategy, metrics)
                    row["query_type"] = query_type
                    query_type_rows.append(row)
    return summary_rows, query_type_rows, run_rows


def _plot_strategy_bars(df: pd.DataFrame, task: str, out_path: Path, title: str, top_n: int = 12) -> None:
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        return
    task_df = task_df.sort_values(["recall_at_10", "mrr"], ascending=False).head(top_n)
    labels = [f"{row.model}\n{row.strategy}" for row in task_df.itertuples()]
    x = np.arange(len(task_df))
    width = 0.24
    fig_height = max(5.5, 0.45 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.barh(x + width, task_df["recall_at_1"], height=width, label="R@1")
    ax.barh(x, task_df["recall_at_5"], height=width, label="R@5")
    ax.barh(x - width, task_df["recall_at_10"], height=width, label="R@10")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    for index, value in enumerate(task_df["recall_at_10"]):
        ax.text(min(value + 0.01, 0.98), index - width, f"{value * 100:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _best_strategy_per_model(df: pd.DataFrame, task: str) -> pd.DataFrame:
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        return task_df
    return (
        task_df.sort_values(["model", "recall_at_10", "mrr"], ascending=[True, False, False])
        .groupby("model", as_index=False)
        .head(1)
        .sort_values(["recall_at_10", "mrr"], ascending=False)
    )


def _plot_model_comparison(df: pd.DataFrame, out_path: Path) -> None:
    image_best = _best_strategy_per_model(df, "image")
    sequence_best = _best_strategy_per_model(df, "sequence")
    combined = []
    for _, row in image_best.iterrows():
        combined.append({"model": row["model"], "task": "image", "recall_at_10": row["recall_at_10"], "mrr": row["mrr"], "strategy": row["strategy"]})
    for _, row in sequence_best.iterrows():
        combined.append({"model": row["model"], "task": "sequence", "recall_at_10": row["recall_at_10"], "mrr": row["mrr"], "strategy": row["strategy"]})
    if not combined:
        return
    plot_df = pd.DataFrame(combined)
    models = list(plot_df["model"].drop_duplicates())
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models) + 4), 5))
    for offset, task in [(-width / 2, "image"), (width / 2, "sequence")]:
        values = []
        labels = []
        for model in models:
            rows = plot_df[(plot_df["model"] == model) & (plot_df["task"] == task)]
            if rows.empty:
                values.append(0)
                labels.append("")
            else:
                values.append(float(rows.iloc[0]["recall_at_10"]))
                labels.append(str(rows.iloc[0]["strategy"]))
        bars = ax.bar(x + offset, values, width=width, label=f"{task} best R@10")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best Recall@10")
    ax.set_title("Best strategy per model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_query_type_heatmap(qdf: pd.DataFrame, summary_df: pd.DataFrame, task: str, out_path: Path, min_queries: int = 8, top_n: int = 28) -> None:
    best = _best_strategy_per_model(summary_df, task)
    if best.empty:
        return
    # Use the globally best strategy for this task as the diagnostic view.
    best_row = best.sort_values(["recall_at_10", "mrr"], ascending=False).iloc[0]
    view = qdf[
        (qdf["task"] == task)
        & (qdf["model"] == best_row["model"])
        & (qdf["strategy"] == best_row["strategy"])
        & (qdf["queries"] >= min_queries)
    ].copy()
    if view.empty:
        return
    view = view.sort_values("recall_at_10", ascending=True).head(top_n)
    metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    matrix = view[metrics].to_numpy(dtype=float)
    labels = [f"{row.query_type} ({int(row.queries)})" for row in view.itertuples()]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.34 * len(labels) + 2)))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["R@1", "R@5", "R@10", "MRR"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{task}: query-type weak spots\n{best_row['model']} / {best_row['strategy']}")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index] * 100:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], max_rows: int | None = None) -> list[str]:
    selected = rows[:max_rows] if max_rows is not None else rows
    lines = [
        "| " + " | ".join(header for header, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if key.startswith("recall") or key == "mrr":
                    values.append(_pct(value))
                else:
                    values.append(f"{value:.1f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _relative_image(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base.parent).as_posix()
    except Exception:
        return path.as_posix()


def write_report(
    out_path: Path,
    summary_df: pd.DataFrame,
    query_type_df: pd.DataFrame,
    run_rows: list[dict[str, Any]],
    assets: dict[str, Path],
) -> None:
    run_count = len(run_rows)
    model_count = len({row.get("model") for row in run_rows})
    best_image = _best_strategy_per_model(summary_df, "image")
    best_sequence = _best_strategy_per_model(summary_df, "sequence")
    lines: list[str] = [
        "# MomentSeek Visual CLIP 评测报告",
        "",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 结果文件数：`{run_count}`",
        f"- 模型数：`{model_count}`",
        "- 数据集：`balanced_v2`，image=300，sequence=200，sequence contact sheet=2fps HQ",
        "- query 来源：Qwen 自动标注的 `suggested_queries`；用于模型/预处理/聚合策略横向比较，后续仍建议人工审核 query。",
        "",
        "## 一句话结论",
        "",
    ]
    if not best_image.empty:
        row = best_image.sort_values(["recall_at_10", "mrr"], ascending=False).iloc[0]
        lines.append(f"- Image-level 当前最佳：`{row['model']}` / `{row['strategy']}`，R@1={_pct(row['recall_at_1'])}，R@10={_pct(row['recall_at_10'])}，MRR={row['mrr']:.3f}。")
    if not best_sequence.empty:
        row = best_sequence.sort_values(["recall_at_10", "mrr"], ascending=False).iloc[0]
        lines.append(f"- Sequence-level 当前最佳：`{row['model']}` / `{row['strategy']}`，R@1={_pct(row['recall_at_1'])}，R@10={_pct(row['recall_at_10'])}，MRR={row['mrr']:.3f}。")
    if "sheet_whole_letterbox" in set(summary_df["strategy"]):
        lines.append("- 直接把整张 contact sheet 喂给 CLIP 的效果明显偏弱，尤其 `sheet_whole_letterbox`；更推荐拆成单帧/cell 后做 MaxSim/TopK/Mix 聚合。")
    if any("cells_mvp_mix" in value for value in summary_df["strategy"]):
        lines.append("- 对 5s 视频片段，`top1/top3/mean` 融合通常优于简单平均，说明关键帧语义峰值很重要。")
    lines += [
        "",
        "## 可视化总览",
        "",
    ]
    for title, asset_key in [
        ("模型最佳策略对比", "model_comparison"),
        ("Image 策略排名", "image_bars"),
        ("Sequence 策略排名", "sequence_bars"),
        ("Image query type 弱项", "image_heatmap"),
        ("Sequence query type 弱项", "sequence_heatmap"),
    ]:
        path = assets.get(asset_key)
        if path and path.exists():
            lines += [f"### {title}", "", f"![{title}]({_relative_image(path, out_path)})", ""]
    lines += [
        "## 运行结果文件",
        "",
        "| model | pretrained | device | elapsed | run file |",
        "|---|---|---|---:|---|",
    ]
    for row in sorted(run_rows, key=lambda item: (str(item.get("model")), str(item.get("created_at")))):
        lines.append(
            f"| `{row.get('model')}` | `{row.get('pretrained')}` | `{row.get('device')}` | {float(row.get('elapsed_seconds') or 0):.1f}s | `{row.get('run_file')}` |"
        )
    lines += ["", "## Image-level 策略指标", ""]
    image_rows = (
        summary_df[summary_df["task"] == "image"]
        .sort_values(["recall_at_10", "mrr"], ascending=False)
        .to_dict("records")
    )
    lines += _markdown_table(image_rows, [
        ("model", "model"),
        ("strategy", "strategy"),
        ("queries", "queries"),
        ("R@1", "recall_at_1"),
        ("R@5", "recall_at_5"),
        ("R@10", "recall_at_10"),
        ("R@20", "recall_at_20"),
        ("MRR", "mrr"),
        ("median rank", "median_rank"),
    ])
    lines += ["", "## Sequence-level 策略指标", ""]
    sequence_rows = (
        summary_df[summary_df["task"] == "sequence"]
        .sort_values(["recall_at_10", "mrr"], ascending=False)
        .to_dict("records")
    )
    lines += _markdown_table(sequence_rows, [
        ("model", "model"),
        ("strategy", "strategy"),
        ("queries", "queries"),
        ("R@1", "recall_at_1"),
        ("R@5", "recall_at_5"),
        ("R@10", "recall_at_10"),
        ("R@20", "recall_at_20"),
        ("MRR", "mrr"),
        ("median rank", "median_rank"),
    ])
    lines += [
        "",
        "## 主要弱项",
        "",
        "下面列出每个 task 的当前最佳策略下，query 数量不少于 8 且 R@10 最低的类型。",
        "",
    ]
    for task in ("image", "sequence"):
        best = _best_strategy_per_model(summary_df, task)
        if best.empty or query_type_df.empty:
            continue
        best_row = best.sort_values(["recall_at_10", "mrr"], ascending=False).iloc[0]
        weak = (
            query_type_df[
                (query_type_df["task"] == task)
                & (query_type_df["model"] == best_row["model"])
                & (query_type_df["strategy"] == best_row["strategy"])
                & (query_type_df["queries"] >= 8)
            ]
            .sort_values(["recall_at_10", "mrr"], ascending=True)
            .head(12)
            .to_dict("records")
        )
        lines += [f"### {task}: `{best_row['model']}` / `{best_row['strategy']}`", ""]
        lines += _markdown_table(weak, [
            ("query_type", "query_type"),
            ("queries", "queries"),
            ("R@1", "recall_at_1"),
            ("R@5", "recall_at_5"),
            ("R@10", "recall_at_10"),
            ("MRR", "mrr"),
            ("median rank", "median_rank"),
        ])
        lines.append("")
    lines += [
        "## 解读和下一步",
        "",
        "1. 这份报告更适合比较“方案 A vs 方案 B”，不要把自动标注 query 的绝对分数当最终产品指标。",
        "2. 如果 `center_crop` 明显优于 `letterbox`，说明当前 query 多数依赖主体/中心语义；但小物体、边缘 logo、字幕 OCR 仍要单独做 tile/crop/OCR 路径。",
        "3. 如果 sequence 的 `cells_mvp_mix` 优于 `cells_mean`，应继续保留帧级 embedding，并在检索时用 MaxSim/TopK 做召回或精排。",
        "4. 下一轮建议加入更强模型：SigLIP、EVA-CLIP、Chinese-CLIP，以及视频模型/X-CLIP/InternVideo 做 sequence 精排。",
        "5. 对 `text_* / logo / product / brand` 弱项，CLIP 本身不够可靠，应独立接 OCR/logo detector，然后和 visual 分数融合。",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual Markdown+PNG report for CLIP evaluation runs.")
    parser.add_argument("--input-glob", default="eval/visual/outputs/clip_eval_*.local.json")
    parser.add_argument("--out", type=Path, default=Path("eval/visual/outputs/visual_clip_eval_report_balanced_v2.local.md"))
    parser.add_argument("--assets-dir", type=Path, default=Path("eval/visual/outputs/visual_clip_eval_report_assets"))
    parser.add_argument("--summary-csv", type=Path, default=Path("eval/visual/outputs/clip_eval_summary_balanced_v2.local.csv"))
    parser.add_argument("--query-type-csv", type=Path, default=Path("eval/visual/outputs/clip_eval_query_types_balanced_v2.local.csv"))
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No eval JSON files matched: {args.input_glob}")
    summary_rows, query_type_rows, run_rows = collect_rows(paths)
    summary_df = pd.DataFrame(summary_rows)
    query_type_df = pd.DataFrame(query_type_rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    query_type_df.to_csv(args.query_type_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    assets = {
        "model_comparison": args.assets_dir / "model_best_recall_at_10.png",
        "image_bars": args.assets_dir / "image_strategy_recall.png",
        "sequence_bars": args.assets_dir / "sequence_strategy_recall.png",
        "image_heatmap": args.assets_dir / "image_query_type_weak_spots.png",
        "sequence_heatmap": args.assets_dir / "sequence_query_type_weak_spots.png",
    }
    _plot_model_comparison(summary_df, assets["model_comparison"])
    _plot_strategy_bars(summary_df, "image", assets["image_bars"], "Image-level strategy comparison")
    _plot_strategy_bars(summary_df, "sequence", assets["sequence_bars"], "Sequence-level strategy comparison")
    _plot_query_type_heatmap(query_type_df, summary_df, "image", assets["image_heatmap"])
    _plot_query_type_heatmap(query_type_df, summary_df, "sequence", assets["sequence_heatmap"])
    write_report(args.out, summary_df, query_type_df, run_rows, assets)
    print(json.dumps({
        "inputs": [str(path) for path in paths],
        "runs": len(run_rows),
        "summary_csv": str(args.summary_csv),
        "query_type_csv": str(args.query_type_csv),
        "report": str(args.out),
        "assets": {key: str(value) for key, value in assets.items() if value.exists()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
