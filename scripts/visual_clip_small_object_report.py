from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from visual_clip_eval_examples import _card, _escape, _load_items, _score


BUCKET_LABELS = {
    "object_product_detail": "物体 / 产品 / 细节",
    "brand_logo": "品牌 / logo",
    "text_ocr": "文字 / OCR / 小字",
}

BRAND_LOGO_TERMS = {
    "brand",
    "branding",
    "logo",
    "sponsor",
    "sponsorship",
    "watermark",
    "show identification",
    "social media end card",
}

AD_BRAND_TERMS = {
    "advertisement",
    "commercial",
    "product placement",
}

TEXT_TERMS = {
    "text",
    "ocr",
    "subtitle",
    "caption",
    "title",
    "credits",
    "credit",
    "word",
    "words",
    "sign",
    "overlay",
    "proverb",
    "reading",
}

OBJECT_TERMS = {
    "object",
    "product",
    "detail",
    "attribute",
    "bottle",
    "cup",
    "ball",
    "football",
    "book",
    "books",
    "plate",
    "dish",
    "food",
    "milk",
    "drink",
    "package",
    "packaging",
    "phone",
    "bag",
    "shoe",
    "scoreboard",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def _normalize(value: str) -> str:
    return re.sub(r"[_/\\-]+", " ", value.casefold())


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def classify_small_target(query_type: str, query: str) -> str | None:
    type_text = _normalize(query_type)
    query_text = _normalize(query)
    combined = f"{type_text} {query_text}"
    if _contains_any(combined, BRAND_LOGO_TERMS):
        return "brand_logo"
    if _contains_any(combined, TEXT_TERMS):
        return "text_ocr"
    if _contains_any(combined, AD_BRAND_TERMS):
        return "brand_logo"
    if _contains_any(combined, OBJECT_TERMS):
        return "object_product_detail"
    return None


def _metric_summary(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {"queries": 0}
    return {
        "queries": len(ranks),
        "recall_at_1": mean(1 if rank <= 1 else 0 for rank in ranks),
        "recall_at_5": mean(1 if rank <= 5 else 0 for rank in ranks),
        "recall_at_10": mean(1 if rank <= 10 else 0 for rank in ranks),
        "recall_at_20": mean(1 if rank <= 20 else 0 for rank in ranks),
        "mrr": mean(1.0 / rank for rank in ranks),
        "median_rank": float(median(ranks)),
        "mean_rank": mean(ranks),
    }


def _strategy_small_metrics(strategy_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ranks_by_bucket: dict[str, list[int]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for row in strategy_payload.get("details") or []:
        bucket = classify_small_target(str(row.get("query_type") or ""), str(row.get("query") or ""))
        if not bucket:
            continue
        rank = int(row.get("rank", 0))
        ranks_by_bucket[bucket].append(rank)
        enriched = dict(row)
        enriched["small_target_bucket"] = bucket
        selected.append(enriched)
    all_ranks = [int(row["rank"]) for row in selected]
    by_bucket = {bucket: _metric_summary(ranks) for bucket, ranks in sorted(ranks_by_bucket.items())}
    return _metric_summary(all_ranks), by_bucket, selected


def collect_small_rows(run_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    query_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    for run_path in run_paths:
        data = _read_json(run_path)
        model = data.get("model") or {}
        model_slug = str(model.get("slug") or f"{model.get('model_name')}_{model.get('pretrained')}")
        payloads[model_slug] = data
        for task in ("image", "sequence"):
            task_payload = data.get(task) or {}
            for strategy, strategy_payload in (task_payload.get("strategies") or {}).items():
                overall, by_bucket, selected = _strategy_small_metrics(strategy_payload)
                summary_rows.append({
                    "run_file": run_path.name,
                    "model": model_slug,
                    "model_name": model.get("model_name"),
                    "pretrained": model.get("pretrained"),
                    "device": data.get("device"),
                    "task": task,
                    "strategy": strategy,
                    "bucket": "all_small_targets",
                    "bucket_label": "全部小目标",
                    **overall,
                })
                for bucket, metrics in by_bucket.items():
                    summary_rows.append({
                        "run_file": run_path.name,
                        "model": model_slug,
                        "model_name": model.get("model_name"),
                        "pretrained": model.get("pretrained"),
                        "device": data.get("device"),
                        "task": task,
                        "strategy": strategy,
                        "bucket": bucket,
                        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
                        **metrics,
                    })
                for row in selected:
                    key = (task, str(row.get("query_id")))
                    query_rows_by_key.setdefault(key, {
                        "task": task,
                        "bucket": row.get("small_target_bucket"),
                        "bucket_label": BUCKET_LABELS.get(str(row.get("small_target_bucket")), str(row.get("small_target_bucket"))),
                        "query_id": row.get("query_id"),
                        "query_type": row.get("query_type"),
                        "query": row.get("query"),
                        "positive_item_id": row.get("positive_item_id"),
                    })
    return summary_rows, list(query_rows_by_key.values()), payloads


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _title, key in columns:
            value = row.get(key, "")
            if key.startswith("recall_at"):
                value = _pct(value)
            elif key == "mrr":
                value = _pct(value)
            elif key in {"median_rank", "mean_rank"} and value != "":
                value = f"{float(value):.1f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _ranked(rows: list[dict[str, Any]], task: str, bucket: str, min_queries: int = 1) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if row.get("task") == task
            and row.get("bucket") == bucket
            and int(row.get("queries", 0)) >= min_queries
        ],
        key=lambda row: (float(row.get("recall_at_10", 0)), float(row.get("mrr", 0)), -float(row.get("mean_rank", 0))),
        reverse=True,
    )


def write_markdown_report(path: Path, summary_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]) -> None:
    image_best = _ranked(summary_rows, "image", "all_small_targets")[:1]
    sequence_best = _ranked(summary_rows, "sequence", "all_small_targets")[:1]
    lines = [
        "# 小目标检索专项测验 - balanced_v2",
        "",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 自动筛选 query 数：image=`{sum(1 for row in query_rows if row['task'] == 'image')}`，sequence=`{sum(1 for row in query_rows if row['task'] == 'sequence')}`",
        "- 筛选口径：基于 `query_type + query` 的启发式规则，覆盖物体/产品/细节、品牌/logo、文字/OCR/小字。",
        "- 注意：这仍然是 Qwen 自动标注 query 的离线压力测验，适合比较配置，不等同于人工 gold set 最终指标。",
        "",
        "## 一句话结论",
        "",
    ]
    if image_best:
        row = image_best[0]
        lines.append(f"- 单帧小目标当前最好：`{row['model']}` / `{row['strategy']}`，R@1={_pct(row['recall_at_1'])}，R@10={_pct(row['recall_at_10'])}，MRR={_pct(row['mrr'])}。")
    if sequence_best:
        row = sequence_best[0]
        lines.append(f"- 片段小目标当前最好：`{row['model']}` / `{row['strategy']}`，R@1={_pct(row['recall_at_1'])}，R@10={_pct(row['recall_at_10'])}，MRR={_pct(row['mrr'])}。")
    lines += [
        "",
        "## 小目标总榜",
        "",
        "### 单帧检索",
        "",
    ]
    columns = [
        ("model", "model"),
        ("strategy", "strategy"),
        ("queries", "queries"),
        ("R@1", "recall_at_1"),
        ("R@5", "recall_at_5"),
        ("R@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("中位排名", "median_rank"),
        ("平均排名", "mean_rank"),
    ]
    lines += _markdown_table(_ranked(summary_rows, "image", "all_small_targets"), columns)
    lines += ["", "### 片段检索", ""]
    lines += _markdown_table(_ranked(summary_rows, "sequence", "all_small_targets"), columns)
    lines += ["", "## 分桶表现", ""]
    for bucket, label in BUCKET_LABELS.items():
        lines += [f"### {label}", "", "#### 单帧检索", ""]
        lines += _markdown_table(_ranked(summary_rows, "image", bucket), columns)
        lines += ["", "#### 片段检索", ""]
        lines += _markdown_table(_ranked(summary_rows, "sequence", bucket), columns)
        lines.append("")
    lines += [
        "## 口径说明",
        "",
        "- `物体 / 产品 / 细节`：命中 query_type 或 query 中的 object/product/detail/bottle/cup/ball/book/food/milk 等词。",
        "- `品牌 / logo`：优先命中 brand/logo/branding/sponsor/watermark；若没有文字/OCR 词，再把 advertisement/commercial/product placement 归入该桶。",
        "- `文字 / OCR / 小字`：命中 text/ocr/subtitle/title/credits/sign/overlay 等词。",
        "- 如果要升级成正式测验，下一步应人工审核 `clip_small_object_queries_balanced_v2.local.csv`，剔除不是真小目标的 query，并补充明确的小目标位置 notes。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _select_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidates: list[dict[str, Any]]) -> None:
        for row in candidates:
            key = str(row.get("query_id"))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= limit:
                return

    misses = sorted([row for row in rows if int(row.get("rank", 0)) > 10], key=lambda row: int(row.get("rank", 0)), reverse=True)
    hits = sorted([row for row in rows if int(row.get("rank", 0)) <= 10], key=lambda row: int(row.get("rank", 0)))
    add(misses[:1])
    add(hits[:1])
    if len(selected) < limit:
        add(sorted(rows, key=lambda row: (int(row.get("rank", 0)) <= 10, -int(row.get("rank", 0)))))
    return selected[:limit]


def _details_index(data: dict[str, Any], task: str, strategy: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("query_id")): row
        for row in ((data.get(task) or {}).get("strategies") or {}).get(strategy, {}).get("details") or []
    }


def _default_example_configs(task: str) -> list[tuple[str, str, str]]:
    if task == "image":
        return [
            ("B-16 推荐 sliding_max", "ViT-B-16_openai", "sliding_max_center_crop"),
            ("B-16 默认 center_crop", "ViT-B-16_openai", "center_crop"),
            ("B-16 全图 letterbox", "ViT-B-16_openai", "letterbox"),
            ("B-32 基础版 center_crop", "ViT-B-32_openai", "center_crop"),
        ]
    return [
        ("B-16 推荐 cells_sliding_top3", "ViT-B-16_openai", "cells_sliding_top3_center_crop"),
        ("B-16 cell 均值", "ViT-B-16_openai", "cells_mean_center_crop"),
        ("B-16 整张 sheet", "ViT-B-16_openai", "sheet_whole_center_crop"),
        ("B-32 基础版整张 sheet", "ViT-B-32_openai", "sheet_whole_center_crop"),
    ]


def _render_result_panel(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    items: dict[str, dict[str, Any]],
    label: str,
    strategy: str,
    row: dict[str, Any] | None,
    positive_id: str,
) -> str:
    if not row:
        return f"<section class=\"result-panel\"><h4>{_escape(label)} <code>{_escape(strategy)}</code></h4><p>该配置没有这条 query。</p></section>"
    cards = []
    for index, result in enumerate(row.get("top10") or [], start=1):
        item_id = str(result.get("item_id"))
        cards.append(_card(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            item_id=item_id,
            item=items.get(item_id),
            title=f"返回 Top {index}",
            score=result.get("score"),
            positive=item_id == positive_id,
        ))
    return (
        "<section class=\"result-panel\">"
        f"<h4>{_escape(label)} <code>{_escape(strategy)}</code></h4>"
        "<div class=\"top10-grid\">"
        + "".join(cards)
        + "</div></section>"
    )


def _render_example(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    items: dict[str, dict[str, Any]],
    payloads: dict[str, Any],
    configs: list[tuple[str, str, str]],
    row: dict[str, Any],
) -> str:
    positive_id = str(row.get("positive_item_id"))
    positive = items.get(positive_id)
    query_id = str(row.get("query_id"))
    summary_rows = []
    panels = []
    for label, model_slug, strategy in configs:
        data = payloads.get(model_slug)
        config_row = _details_index(data, task, strategy).get(query_id) if data else None
        if config_row:
            top10 = config_row.get("top10") or []
            top1 = str(top10[0].get("item_id")) if top10 else ""
            rank = int(config_row.get("rank", 0))
            summary_rows.append(
                "<tr>"
                f"<td>{_escape(label)}</td>"
                f"<td><code>{_escape(strategy)}</code></td>"
                f"<td>{rank}</td>"
                f"<td>{'是' if rank <= 10 else '否'}</td>"
                f"<td>{_score(config_row.get('positive_score'))}</td>"
                f"<td><code>{_escape(top1)}</code></td>"
                "</tr>"
            )
        else:
            summary_rows.append(f"<tr><td>{_escape(label)}</td><td><code>{_escape(strategy)}</code></td><td colspan=\"4\">缺失</td></tr>")
        panels.append(_render_result_panel(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            items=items,
            label=label,
            strategy=strategy,
            row=config_row,
            positive_id=positive_id,
        ))
    bucket = str(row.get("small_target_bucket"))
    rank = int(row.get("rank", 0))
    return (
        f"<section class=\"example {'hit' if rank <= 10 else 'miss'}\">"
        "<div class=\"example-header\">"
        f"<div><span class=\"badge {'hit' if rank <= 10 else 'miss'}\">{'命中@10' if rank <= 10 else '未命中@10'}</span>"
        f"<span class=\"rank\">样例桶：{_escape(BUCKET_LABELS.get(bucket, bucket))}</span>"
        f"<span class=\"rank\">参考排名：{rank}</span></div>"
        f"<code>{_escape(query_id)}</code></div>"
        f"<p class=\"query\">{_escape(row.get('query'))}</p>"
        "<div class=\"positive-row\">"
        + _card(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            item_id=positive_id,
            item=positive,
            title="正样本 / 原图",
            score=row.get("positive_score"),
            positive=True,
        )
        + "</div>"
        "<table class=\"metrics example-summary\"><tr><th>配置</th><th>策略</th><th>正样本排名</th><th>命中@10</th><th>正样本分数</th><th>top1</th></tr>"
        + "".join(summary_rows)
        + "</table>"
        "<div class=\"compare-results\">"
        + "".join(panels)
        + "</div></section>"
    )


def _html_css() -> str:
    return """
body { margin: 0; background: #f6f7f9; color: #18202a; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1560px; margin: 0 auto; padding: 28px 24px 56px; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin-top: 30px; font-size: 22px; }
h3 { margin: 0; font-size: 17px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; font-size: 12px; word-break: break-all; }
.summary, details { background: #fff; border: 1px solid #dfe4ea; border-radius: 8px; padding: 16px; margin: 14px 0; }
summary { cursor: pointer; font-weight: 700; font-size: 17px; }
.metrics { border-collapse: collapse; margin: 12px 0; }
.metrics th, .metrics td { border: 1px solid #dfe4ea; padding: 6px 9px; text-align: right; }
.metrics th:first-child, .metrics td:first-child, .metrics th:nth-child(2), .metrics td:nth-child(2), .metrics th:nth-child(6), .metrics td:nth-child(6) { text-align: left; }
.example { border-top: 1px solid #e4e8ee; margin-top: 16px; padding-top: 16px; }
.example-header { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.badge, .rank { display: inline-flex; align-items: center; min-height: 24px; padding: 0 8px; border-radius: 999px; margin-right: 6px; font-size: 12px; font-weight: 700; }
.badge.hit { background: #dff4e5; color: #11632f; }
.badge.miss { background: #fee2de; color: #9f2418; }
.rank { background: #edf1f6; color: #3c4654; }
.query { margin: 10px 0; font-size: 16px; font-weight: 650; }
.positive-row { display: grid; grid-template-columns: minmax(220px, 360px); margin-bottom: 12px; }
.compare-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; align-items: start; }
.result-panel { min-width: 0; }
.result-panel h4 { margin: 10px 0 8px; font-size: 15px; }
.top10-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
.card { margin: 0; border: 1px solid #dfe4ea; border-radius: 8px; overflow: hidden; background: #fbfcfe; }
.positive-hit { border-color: #29a05b; box-shadow: inset 0 0 0 1px #29a05b; }
.thumb { background: #e9edf3; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; }
.thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
figcaption { display: grid; gap: 4px; padding: 8px; min-height: 92px; }
figcaption span { color: #596575; font-size: 12px; }
@media (max-width: 720px) { main { padding: 18px 12px 36px; } .example-header { align-items: flex-start; flex-direction: column; } }
"""


def write_examples_html(
    path: Path,
    repo_root: Path,
    payloads: dict[str, Any],
    examples_per_bucket: int,
) -> None:
    primary_model = "ViT-B-16_openai"
    primary_data = payloads.get(primary_model)
    if not primary_data:
        return
    sections = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>小目标检索专项测验样例</title>",
        f"<style>{_html_css()}</style></head><body><main>",
        "<h1>小目标检索专项测验样例</h1>",
        "<section class=\"summary\"><p>每个样例展示正样本原图，以及多个配置的实际 Top10 返回。query 文本保留评估时的原文。</p></section>",
    ]
    for task, primary_strategy in [
        ("image", "sliding_max_center_crop"),
        ("sequence", "cells_sliding_top3_center_crop"),
    ]:
        task_payload = primary_data.get(task) or {}
        strategy_payload = (task_payload.get("strategies") or {}).get(primary_strategy) or {}
        items = _load_items(repo_root, str(task_payload.get("manifest")), task)
        rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in strategy_payload.get("details") or []:
            bucket = classify_small_target(str(row.get("query_type") or ""), str(row.get("query") or ""))
            if bucket:
                enriched = dict(row)
                enriched["small_target_bucket"] = bucket
                rows_by_bucket[bucket].append(enriched)
        sections.append(f"<h2>{'单帧检索' if task == 'image' else '片段检索'}：小目标样例</h2>")
        configs = _default_example_configs(task)
        for bucket, label in BUCKET_LABELS.items():
            examples = _select_examples(rows_by_bucket.get(bucket, []), examples_per_bucket)
            sections.append(f"<details open><summary>{_escape(label)} ({len(rows_by_bucket.get(bucket, []))} 条 query)</summary>")
            for row in examples:
                sections.append(_render_example(
                    repo_root=repo_root,
                    out_path=path,
                    task=task,
                    items=items,
                    payloads=payloads,
                    configs=configs,
                    row=row,
                ))
            sections.append("</details>")
    sections.append("</main></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成小目标检索专项测验报告。")
    parser.add_argument("--runs", nargs="+", type=Path, default=[
        Path("eval/visual/outputs/clip_eval_baseline_vit_b16_openai_balanced_v2.local.json"),
        Path("eval/visual/outputs/clip_eval_baseline_vit_b32_openai_balanced_v2.local.json"),
    ])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-md", type=Path, default=Path("eval/visual/outputs/visual_clip_small_object_report_balanced_v2.local.md"))
    parser.add_argument("--out-html", type=Path, default=Path("eval/visual/outputs/visual_clip_small_object_examples_balanced_v2.local.html"))
    parser.add_argument("--summary-csv", type=Path, default=Path("eval/visual/outputs/clip_small_object_summary_balanced_v2.local.csv"))
    parser.add_argument("--queries-csv", type=Path, default=Path("eval/visual/outputs/clip_small_object_queries_balanced_v2.local.csv"))
    parser.add_argument("--examples-per-bucket", type=int, default=2)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_paths = [(repo_root / path).resolve() if not path.is_absolute() else path.resolve() for path in args.runs]
    summary_rows, query_rows, payloads = collect_small_rows(run_paths)
    _write_csv(
        (repo_root / args.summary_csv).resolve() if not args.summary_csv.is_absolute() else args.summary_csv.resolve(),
        summary_rows,
        [
            "run_file",
            "model",
            "model_name",
            "pretrained",
            "device",
            "task",
            "strategy",
            "bucket",
            "bucket_label",
            "queries",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "mrr",
            "median_rank",
            "mean_rank",
        ],
    )
    _write_csv(
        (repo_root / args.queries_csv).resolve() if not args.queries_csv.is_absolute() else args.queries_csv.resolve(),
        sorted(query_rows, key=lambda row: (row["task"], row["bucket"], row["query_id"])),
        ["task", "bucket", "bucket_label", "query_id", "query_type", "query", "positive_item_id"],
    )
    out_md = (repo_root / args.out_md).resolve() if not args.out_md.is_absolute() else args.out_md.resolve()
    out_html = (repo_root / args.out_html).resolve() if not args.out_html.is_absolute() else args.out_html.resolve()
    write_markdown_report(out_md, summary_rows, query_rows)
    write_examples_html(out_html, repo_root, payloads, max(1, args.examples_per_bucket))
    print(out_md)
    print(out_html)


if __name__ == "__main__":
    main()
