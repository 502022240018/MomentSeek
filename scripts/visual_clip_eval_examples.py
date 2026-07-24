from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def _score(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def _image_src(out_path: Path, target: Path) -> str:
    rel = os.path.relpath(target, out_path.parent)
    return rel.replace(os.sep, "/")


def _item_path(repo_root: Path, item: dict[str, Any]) -> Path:
    return _resolve_repo_path(repo_root, str(item.get("path") or ""))


def _load_items(repo_root: Path, manifest_path: str, task: str) -> dict[str, dict[str, Any]]:
    manifest = _read_json(_resolve_repo_path(repo_root, manifest_path))
    if task == "image":
        return {str(item["image_id"]): item for item in manifest.get("frames") or []}
    if task == "sequence":
        return {str(item["sheet_id"]): item for item in manifest.get("sheets") or []}
    raise ValueError(f"Unknown task: {task}")


def _meta_text(task: str, item: dict[str, Any] | None) -> str:
    if not item:
        return "manifest 中缺少该 item"
    parts = []
    group_id = item.get("group_id")
    if group_id:
        parts.append(f"分组={group_id}")
    resolution = item.get("resolution_label")
    if resolution:
        parts.append(str(resolution))
    if task == "image":
        if item.get("time") is not None:
            parts.append(f"t={float(item['time']):.1f}s")
        width = item.get("width")
        height = item.get("height")
        if width and height:
            parts.append(f"{width}x{height}")
    else:
        if item.get("start") is not None and item.get("end") is not None:
            parts.append(f"{float(item['start']):.1f}-{float(item['end']):.1f}s")
        sample_times = item.get("sample_times") or []
        if sample_times:
            parts.append(f"{len(sample_times)} 帧")
    return " | ".join(parts)


def _task_label(task: str) -> str:
    return {
        "image": "单帧检索",
        "sequence": "片段检索",
    }.get(task, task)


def _metric_table(metrics: dict[str, Any]) -> str:
    return (
        "<table class=\"metrics\"><tr>"
        "<th>query 数</th><th>R@1</th><th>R@5</th><th>R@10</th>"
        "<th>R@20</th><th>MRR</th><th>中位排名</th>"
        "</tr><tr>"
        f"<td>{int(metrics.get('queries', 0))}</td>"
        f"<td>{_pct(metrics.get('recall_at_1'))}</td>"
        f"<td>{_pct(metrics.get('recall_at_5'))}</td>"
        f"<td>{_pct(metrics.get('recall_at_10'))}</td>"
        f"<td>{_pct(metrics.get('recall_at_20'))}</td>"
        f"<td>{_pct(metrics.get('mrr'))}</td>"
        f"<td>{_escape(metrics.get('median_rank', ''))}</td>"
        "</tr></table>"
    )


def _comparison_metric_table(rows: list[tuple[str, str, dict[str, Any] | None]]) -> str:
    lines = [
        "<table class=\"metrics compare-metrics\"><tr>",
        "<th>版本</th><th>策略</th><th>query 数</th><th>R@1</th><th>R@5</th><th>R@10</th>",
        "<th>R@20</th><th>MRR</th><th>中位排名</th>",
        "</tr>",
    ]
    for label, strategy, metrics in rows:
        metrics = metrics or {}
        lines.append(
            "<tr>"
            f"<td>{_escape(label)}</td>"
            f"<td><code>{_escape(strategy)}</code></td>"
            f"<td>{int(metrics.get('queries', 0)) if metrics else ''}</td>"
            f"<td>{_pct(metrics.get('recall_at_1')) if metrics else ''}</td>"
            f"<td>{_pct(metrics.get('recall_at_5')) if metrics else ''}</td>"
            f"<td>{_pct(metrics.get('recall_at_10')) if metrics else ''}</td>"
            f"<td>{_pct(metrics.get('recall_at_20')) if metrics else ''}</td>"
            f"<td>{_pct(metrics.get('mrr')) if metrics else ''}</td>"
            f"<td>{_escape(metrics.get('median_rank', '')) if metrics else ''}</td>"
            "</tr>"
        )
    lines.append("</table>")
    return "".join(lines)


def _select_examples(details: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            key = str(row.get("query_id"))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            if len(selected) >= limit:
                return

    misses = sorted(
        [row for row in details if int(row.get("rank", 10**9)) > 10],
        key=lambda row: int(row.get("rank", 0)),
        reverse=True,
    )
    hits = sorted(
        [row for row in details if int(row.get("rank", 10**9)) <= 10],
        key=lambda row: int(row.get("rank", 10**9)),
    )
    add(misses[:1])
    add(hits[:1])
    if len(selected) < limit:
        remaining = sorted(
            details,
            key=lambda row: (
                int(row.get("rank", 10**9)) <= 10,
                -int(row.get("rank", 0)),
                str(row.get("query_id")),
            ),
        )
        add(remaining)
    return selected[:limit]


def _card(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    item_id: str,
    item: dict[str, Any] | None,
    title: str,
    score: Any = None,
    positive: bool = False,
) -> str:
    if item:
        src = _image_src(out_path, _item_path(repo_root, item))
        img = f"<img loading=\"lazy\" src=\"{_escape(src)}\" alt=\"{_escape(item_id)}\">"
    else:
        img = "<div class=\"missing\">图片缺失</div>"
    classes = "card positive-hit" if positive else "card"
    score_text = f"<span>分数={_score(score)}</span>" if score is not None else ""
    return (
        f"<figure class=\"{classes}\">"
        f"<div class=\"thumb\">{img}</div>"
        "<figcaption>"
        f"<strong>{_escape(title)}</strong>"
        f"<code>{_escape(item_id)}</code>"
        f"<span>{_escape(_meta_text(task, item))}</span>"
        f"{score_text}"
        "</figcaption>"
        "</figure>"
    )


def _result_summary(row: dict[str, Any] | None, label: str, strategy: str) -> str:
    if not row:
        return (
            "<tr>"
            f"<td>{_escape(label)}</td><td><code>{_escape(strategy)}</code></td>"
            "<td colspan=\"4\">该对比运行中没有这条 query</td>"
            "</tr>"
        )
    rank = int(row.get("rank", 0))
    top10 = row.get("top10") or []
    top1 = str(top10[0].get("item_id")) if top10 else ""
    return (
        "<tr>"
        f"<td>{_escape(label)}</td>"
        f"<td><code>{_escape(strategy)}</code></td>"
        f"<td>{rank}</td>"
        f"<td>{'是' if rank <= 10 else '否'}</td>"
        f"<td>{_score(row.get('positive_score'))}</td>"
        f"<td><code>{_escape(top1)}</code></td>"
        "</tr>"
    )


def _top10_panel(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    items: dict[str, dict[str, Any]],
    row: dict[str, Any] | None,
    title: str,
    strategy: str,
) -> str:
    if not row:
        return (
            "<section class=\"result-panel\">"
            f"<h4>{_escape(title)} <code>{_escape(strategy)}</code></h4>"
            "<p class=\"missing-note\">该对比运行中没有这条 query。</p>"
            "</section>"
        )
    positive_id = str(row.get("positive_item_id"))
    top_cards = []
    for index, result in enumerate(row.get("top10") or [], start=1):
        item_id = str(result.get("item_id"))
        top_cards.append(
            _card(
                repo_root=repo_root,
                out_path=out_path,
                task=task,
                item_id=item_id,
                item=items.get(item_id),
                title=f"返回 Top {index}",
                score=result.get("score"),
                positive=item_id == positive_id,
            )
        )
    return (
        "<section class=\"result-panel\">"
        f"<h4>{_escape(title)} <code>{_escape(strategy)}</code></h4>"
        "<div class=\"top10-grid\">"
        + "".join(top_cards)
        + "</div>"
        "</section>"
    )


def _comparison_example_block(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    items: dict[str, dict[str, Any]],
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
    primary_label: str,
    primary_strategy: str,
    baseline_label: str,
    baseline_strategy: str,
) -> str:
    positive_id = str(row.get("positive_item_id"))
    positive_item = items.get(positive_id)
    rank = int(row.get("rank", 0))
    baseline_rank = int(baseline_row.get("rank", 0)) if baseline_row else None
    hit_class = "hit" if rank <= 10 else "miss"
    baseline_hit = baseline_rank is not None and baseline_rank <= 10
    query = row.get("query") or ""
    summary = (
        "<table class=\"metrics example-summary\"><tr>"
        "<th>版本</th><th>策略</th><th>正样本排名</th><th>命中@10</th><th>正样本分数</th><th>top1</th>"
        "</tr>"
        + _result_summary(row, primary_label, primary_strategy)
        + _result_summary(baseline_row, baseline_label, baseline_strategy)
        + "</table>"
    )
    return (
        f"<section class=\"example {hit_class}\">"
        "<div class=\"example-header\">"
        f"<div><span class=\"badge {hit_class}\">{'命中@10' if rank <= 10 else '未命中@10'}</span>"
        f"<span class=\"rank\">当前排名：{rank}</span>"
        f"<span class=\"rank\">基础版排名：{baseline_rank if baseline_rank is not None else '缺失'}</span>"
        f"<span class=\"badge {'hit' if baseline_hit else 'miss'}\">基础版{'命中@10' if baseline_hit else '未命中@10'}</span></div>"
        f"<code>{_escape(row.get('query_id', ''))}</code>"
        "</div>"
        f"<p class=\"query\">{_escape(query)}</p>"
        "<div class=\"positive-row\">"
        + _card(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            item_id=positive_id,
            item=positive_item,
            title="正样本 / 原图",
            score=row.get("positive_score"),
            positive=True,
        )
        + "</div>"
        + summary
        + "<div class=\"compare-results\">"
        + _top10_panel(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            items=items,
            row=row,
            title=primary_label,
            strategy=primary_strategy,
        )
        + _top10_panel(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            items=items,
            row=baseline_row,
            title=baseline_label,
            strategy=baseline_strategy,
        )
        + "</div>"
        "</section>"
    )


def _example_block(
    *,
    repo_root: Path,
    out_path: Path,
    task: str,
    items: dict[str, dict[str, Any]],
    row: dict[str, Any],
) -> str:
    positive_id = str(row.get("positive_item_id"))
    positive_item = items.get(positive_id)
    rank = int(row.get("rank", 0))
    hit_class = "hit" if rank <= 10 else "miss"
    top10 = row.get("top10") or []
    top_cards = []
    for index, result in enumerate(top10, start=1):
        item_id = str(result.get("item_id"))
        top_cards.append(
            _card(
                repo_root=repo_root,
                out_path=out_path,
                task=task,
                item_id=item_id,
                item=items.get(item_id),
                title=f"返回 Top {index}",
                score=result.get("score"),
                positive=item_id == positive_id,
            )
        )
    query = row.get("query") or ""
    return (
        f"<section class=\"example {hit_class}\">"
        "<div class=\"example-header\">"
        f"<div><span class=\"badge {hit_class}\">{'命中@10' if rank <= 10 else '未命中@10'}</span>"
        f"<span class=\"rank\">正样本排名：{rank}</span>"
        f"<span class=\"rank\">正样本分数：{_score(row.get('positive_score'))}</span></div>"
        f"<code>{_escape(row.get('query_id', ''))}</code>"
        "</div>"
        f"<p class=\"query\">{_escape(query)}</p>"
        "<div class=\"positive-row\">"
        + _card(
            repo_root=repo_root,
            out_path=out_path,
            task=task,
            item_id=positive_id,
            item=positive_item,
            title="正样本 / 原图",
            score=row.get("positive_score"),
            positive=True,
        )
        + "</div>"
        "<div class=\"top10-grid\">"
        + "".join(top_cards)
        + "</div>"
        "</section>"
    )


def _strategy_glossary() -> str:
    return """
<section class="glossary">
  <h2>策略说明</h2>
  <dl>
    <dt>center_crop</dt>
    <dd>CLIP 默认预处理：先缩放，再取中心方形裁剪。本报告里它是单帧检索的“不做额外处理”基础版。</dd>
    <dt>letterbox</dt>
    <dd>保留完整画面比例，再 padding 成 CLIP 需要的方图。优点是边缘不丢，缺点是主体会变小。</dd>
    <dt>sliding_mean / sliding_max / sliding_top3</dt>
    <dd>沿画面的长边生成多个方形滑窗 crop，每个 crop 单独算相似度，再用平均值、最大值或 top3 平均值聚合。</dd>
    <dt>sliding_mvp_mix</dt>
    <dd>滑窗分数的加权融合：0.65 * max + 0.25 * top3 + 0.10 * mean。</dd>
    <dt>sheet_whole_*</dt>
    <dd>仅用于片段检索：把整张 contact sheet 当作一张图编码。本报告里 <code>sheet_whole_center_crop</code> 是片段检索的“不拆 cell、不做聚合”基础版。</dd>
    <dt>cells_*</dt>
    <dd>仅用于片段检索：把 contact sheet 拆成单帧 cell，每个 cell 单独编码，再跨 cell 聚合分数。</dd>
    <dt>cells_sliding_*</dt>
    <dd>仅用于片段检索：先拆成 cell，再对每个 cell 做滑窗 crop，最后把该片段内所有视图分数聚合。</dd>
  </dl>
</section>
"""


def _css() -> str:
    return """
body {
  margin: 0;
  background: #f6f7f9;
  color: #18202a;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1440px;
  margin: 0 auto;
  padding: 28px 24px 56px;
}
h1, h2, h3 {
  line-height: 1.2;
}
h1 {
  margin: 0 0 8px;
  font-size: 28px;
}
h2 {
  margin-top: 30px;
  font-size: 22px;
}
h3 {
  margin: 0;
  font-size: 17px;
}
p {
  max-width: 980px;
}
code {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  word-break: break-all;
}
.summary, .glossary, details {
  background: #fff;
  border: 1px solid #dfe4ea;
  border-radius: 8px;
  padding: 16px;
  margin: 14px 0;
}
.glossary dl {
  display: grid;
  grid-template-columns: minmax(170px, 240px) 1fr;
  gap: 8px 18px;
  margin: 0;
}
.glossary dt {
  font-weight: 700;
}
.glossary dd {
  margin: 0;
}
summary {
  cursor: pointer;
  font-weight: 700;
  font-size: 17px;
}
.metrics {
  border-collapse: collapse;
  margin: 12px 0;
}
.metrics th, .metrics td {
  border: 1px solid #dfe4ea;
  padding: 6px 9px;
  text-align: right;
}
.metrics th:first-child, .metrics td:first-child {
  text-align: left;
}
.compare-metrics td:nth-child(2), .compare-metrics th:nth-child(2),
.example-summary td:nth-child(2), .example-summary th:nth-child(2),
.example-summary td:nth-child(6), .example-summary th:nth-child(6) {
  text-align: left;
}
.example {
  border-top: 1px solid #e4e8ee;
  margin-top: 16px;
  padding-top: 16px;
}
.example-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.badge, .rank {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  margin-right: 6px;
  font-size: 12px;
  font-weight: 700;
}
.badge.hit {
  background: #dff4e5;
  color: #11632f;
}
.badge.miss {
  background: #fee2de;
  color: #9f2418;
}
.rank {
  background: #edf1f6;
  color: #3c4654;
}
.query {
  margin: 10px 0;
  font-size: 16px;
  font-weight: 650;
}
.positive-row {
  display: grid;
  grid-template-columns: minmax(220px, 360px);
  margin-bottom: 12px;
}
.top10-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.compare-results {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  align-items: start;
}
.result-panel {
  min-width: 0;
}
.result-panel h4 {
  margin: 10px 0 8px;
  font-size: 15px;
}
.card {
  margin: 0;
  border: 1px solid #dfe4ea;
  border-radius: 8px;
  overflow: hidden;
  background: #fbfcfe;
}
.positive-hit {
  border-color: #29a05b;
  box-shadow: inset 0 0 0 1px #29a05b;
}
.thumb {
  background: #e9edf3;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
figcaption {
  display: grid;
  gap: 4px;
  padding: 8px;
  min-height: 92px;
}
figcaption span {
  color: #596575;
  font-size: 12px;
}
.missing {
  color: #8b95a3;
}
.missing-note {
  color: #8b95a3;
  margin: 8px 0;
}
@media (max-width: 720px) {
  main {
    padding: 18px 12px 36px;
  }
  .glossary dl {
    display: block;
  }
  .glossary dt {
    margin-top: 10px;
  }
  .example-header {
    align-items: flex-start;
    flex-direction: column;
  }
  .top10-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
  .compare-results {
    grid-template-columns: 1fr;
  }
}
"""


def _render_task(
    *,
    repo_root: Path,
    out_path: Path,
    data: dict[str, Any],
    baseline_data: dict[str, Any] | None,
    task: str,
    strategy: str,
    baseline_strategy: str | None,
    primary_label: str,
    baseline_label: str,
    examples_per_type: int,
    min_queries: int,
    open_first: int,
) -> str:
    task_payload = data.get(task) or {}
    strategy_payload = (task_payload.get("strategies") or {}).get(strategy)
    if not strategy_payload:
        raise KeyError(f"Strategy not found for {task}: {strategy}")
    baseline_payload = None
    baseline_details_by_id: dict[str, dict[str, Any]] = {}
    if baseline_data and baseline_strategy:
        baseline_task_payload = baseline_data.get(task) or {}
        baseline_payload = (baseline_task_payload.get("strategies") or {}).get(baseline_strategy)
        if not baseline_payload:
            raise KeyError(f"Baseline strategy not found for {task}: {baseline_strategy}")
        baseline_details_by_id = {
            str(row.get("query_id")): row
            for row in baseline_payload.get("details") or []
        }
    items = _load_items(repo_root, str(task_payload.get("manifest")), task)
    details_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strategy_payload.get("details") or []:
        details_by_type[str(row.get("query_type") or "unknown")].append(row)

    query_type_metrics = strategy_payload.get("by_query_type") or {}
    types = [
        (query_type, metrics)
        for query_type, metrics in query_type_metrics.items()
        if int(metrics.get("queries", 0)) >= min_queries
    ]
    types.sort(key=lambda pair: (float(pair[1].get("recall_at_10", 0)), pair[0]))

    overall = strategy_payload.get("overall") or {}
    baseline_overall = (baseline_payload or {}).get("overall") if baseline_payload else None
    sections = [
        f"<h2>{_escape(_task_label(task))}: {_escape(strategy)}</h2>",
        _comparison_metric_table([
            (primary_label, strategy, overall),
            (baseline_label, baseline_strategy or "", baseline_overall),
        ]) if baseline_payload else _metric_table(overall),
        (
            f"<p>仅展示 query 数不少于 {min_queries} 的类型。"
            f"每类最多抽 {examples_per_type} 条样例；如果条件允许，优先放一条未命中@10 和一条命中@10，方便对照。</p>"
        ),
    ]
    for index, (query_type, metrics) in enumerate(types):
        rows = _select_examples(details_by_type.get(query_type, []), examples_per_type)
        open_attr = " open" if index < open_first else ""
        blocks = [
            f"<details{open_attr}>",
            (
                f"<summary>{_escape(query_type)} "
                f"({int(metrics.get('queries', 0))} 条 query，R@10={_pct(metrics.get('recall_at_10'))})</summary>"
            ),
            _comparison_metric_table([
                (primary_label, strategy, metrics),
                (
                    baseline_label,
                    baseline_strategy or "",
                    ((baseline_payload or {}).get("by_query_type") or {}).get(query_type) if baseline_payload else None,
                ),
            ]) if baseline_payload else _metric_table(metrics),
        ]
        for row in rows:
            if baseline_payload:
                blocks.append(
                    _comparison_example_block(
                        repo_root=repo_root,
                        out_path=out_path,
                        task=task,
                        items=items,
                        row=row,
                        baseline_row=baseline_details_by_id.get(str(row.get("query_id"))),
                        primary_label=primary_label,
                        primary_strategy=strategy,
                        baseline_label=baseline_label,
                        baseline_strategy=baseline_strategy or "",
                    )
                )
            else:
                blocks.append(
                    _example_block(
                        repo_root=repo_root,
                        out_path=out_path,
                        task=task,
                        items=items,
                        row=row,
                    )
                )
        blocks.append("</details>")
        sections.append("".join(blocks))
    return "\n".join(sections)


def write_report(
    *,
    run_path: Path,
    out_path: Path,
    repo_root: Path,
    baseline_run_path: Path | None,
    image_strategy: str,
    sequence_strategy: str,
    baseline_image_strategy: str,
    baseline_sequence_strategy: str,
    examples_per_type: int,
    min_queries: int,
    open_first: int,
) -> None:
    data = _read_json(run_path)
    baseline_data = _read_json(baseline_run_path) if baseline_run_path else None
    model = data.get("model") or {}
    baseline_model = (baseline_data or {}).get("model") or {}
    created_at = datetime.now().isoformat(timespec="seconds")
    primary_label = str(model.get("slug") or "current")
    baseline_label = str(baseline_model.get("slug") or "baseline")
    body = [
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Visual CLIP 查询样例与 Top10 对比</title>",
        f"<style>{_css()}</style>",
        "</head>",
        "<body><main>",
        "<h1>Visual CLIP 查询样例与 Top10 对比</h1>",
        "<section class=\"summary\">",
        f"<p>生成时间：<code>{_escape(created_at)}</code>；来源结果：<code>{_escape(run_path.name)}</code>。</p>",
        (
            f"<p>当前推荐配置：<code>{_escape(primary_label)}</code>；"
            f"单帧=<code>{_escape(image_strategy)}</code>，片段=<code>{_escape(sequence_strategy)}</code>。</p>"
        ),
        (
            f"<p>基础版对比：<code>{_escape(baseline_label)}</code>；"
            f"单帧=<code>{_escape(baseline_image_strategy)}</code>，"
            f"片段=<code>{_escape(baseline_sequence_strategy)}</code>。"
            "这里的基础版表示使用 CLIP 默认输入路径，不做 sliding crops，也不做 cell 级拆分聚合。</p>"
            if baseline_data
            else ""
        ),
        "</section>",
        _strategy_glossary(),
        _render_task(
            repo_root=repo_root,
            out_path=out_path,
            data=data,
            baseline_data=baseline_data,
            task="image",
            strategy=image_strategy,
            baseline_strategy=baseline_image_strategy if baseline_data else None,
            primary_label=primary_label,
            baseline_label=baseline_label,
            examples_per_type=examples_per_type,
            min_queries=min_queries,
            open_first=open_first,
        ),
        _render_task(
            repo_root=repo_root,
            out_path=out_path,
            data=data,
            baseline_data=baseline_data,
            task="sequence",
            strategy=sequence_strategy,
            baseline_strategy=baseline_sequence_strategy if baseline_data else None,
            primary_label=primary_label,
            baseline_label=baseline_label,
            examples_per_type=examples_per_type,
            min_queries=min_queries,
            open_first=open_first,
        ),
        "</main></body></html>",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成包含 query 样例和 Top10 检索结果的 HTML 报告。")
    parser.add_argument("--run", type=Path, default=Path("eval/visual/outputs/clip_eval_baseline_vit_b16_openai_balanced_v2.local.json"))
    parser.add_argument("--baseline-run", type=Path, default=Path("eval/visual/outputs/clip_eval_baseline_vit_b32_openai_balanced_v2.local.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/visual/outputs/visual_clip_eval_examples_balanced_v2.local.html"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--image-strategy", default="sliding_max_center_crop")
    parser.add_argument("--sequence-strategy", default="cells_sliding_top3_center_crop")
    parser.add_argument("--baseline-image-strategy", default="center_crop")
    parser.add_argument("--baseline-sequence-strategy", default="sheet_whole_center_crop")
    parser.add_argument("--examples-per-type", type=int, default=2)
    parser.add_argument("--min-queries", type=int, default=8)
    parser.add_argument("--open-first", type=int, default=2)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_path = _resolve_repo_path(repo_root, args.run).resolve()
    out_path = _resolve_repo_path(repo_root, args.out).resolve()
    write_report(
        run_path=run_path,
        out_path=out_path,
        repo_root=repo_root,
        baseline_run_path=_resolve_repo_path(repo_root, args.baseline_run).resolve() if args.baseline_run else None,
        image_strategy=args.image_strategy,
        sequence_strategy=args.sequence_strategy,
        baseline_image_strategy=args.baseline_image_strategy,
        baseline_sequence_strategy=args.baseline_sequence_strategy,
        examples_per_type=max(0, args.examples_per_type),
        min_queries=max(1, args.min_queries),
        open_first=max(0, args.open_first),
    )
    print(out_path)


if __name__ == "__main__":
    main()
