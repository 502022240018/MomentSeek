from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from visual_clip_eval import (
    ModelSpec,
    OpenClipRunner,
    _cache_path,
    _load_embeddings_cache,
    _save_embeddings_cache,
    _spatial_view_embeddings_for_paths,
    _view_score_strategies,
)
from visual_clip_eval_examples import _card, _escape


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _eval_name(result: dict[str, Any]) -> str:
    queries_path = Path(str(result.get("queries_path") or ""))
    stem = queries_path.stem
    if "public_small_object_eval_v1_hd" in stem:
        return "public_small_object_eval_v1_hd"
    if "public_image_eval_v1_hd" in stem:
        return "public_image_eval_v1_hd"
    if "public_image_eval_v1" in stem:
        return "public_image_eval_v1"
    return stem or "public_image_eval"


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def _metric_summary(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {"queries": 0}
    values = np.asarray(ranks, dtype=np.float32)
    return {
        "queries": int(len(values)),
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "recall_at_20": float(np.mean(values <= 20)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def _load_items(path: Path) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    payload = _read_json(path)
    frames = payload.get("frames") or []
    item_ids = [str(item["image_id"]) for item in frames]
    paths = [str(item["path"]) for item in frames]
    return item_ids, paths, {str(item["image_id"]): item for item in frames}


def _load_queries(path: Path) -> list[dict[str, Any]]:
    queries = []
    for row in _read_jsonl(path):
        positives = [str(value) for value in (row.get("positive_image_ids") or [])]
        if not positives:
            continue
        enriched = dict(row)
        enriched["positive_image_ids"] = positives
        queries.append(enriched)
    return queries


def _rank_multi_positive(scores: np.ndarray, positive_indices: list[int]) -> tuple[int, float]:
    positive_scores = scores[np.asarray(positive_indices, dtype=np.int32)]
    best_positive_score = float(np.max(positive_scores))
    return int(1 + np.sum(scores > best_positive_score)), best_positive_score


def _evaluate_scores(
    queries: list[dict[str, Any]],
    item_ids: list[str],
    score_fn,
) -> dict[str, Any]:
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    ranks: list[int] = []
    by_type: dict[str, list[int]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for query in queries:
        positive_indices = [item_to_index[item_id] for item_id in query["positive_image_ids"] if item_id in item_to_index]
        if not positive_indices:
            continue
        scores = score_fn()
        rank, positive_score = _rank_multi_positive(scores, positive_indices)
        ranks.append(rank)
        query_type = str(query.get("query_type") or "unknown")
        by_type[query_type].append(rank)
        top_indices = np.argsort(scores)[::-1][:10]
        details.append({
            "query_id": query.get("query_id"),
            "query": query.get("query"),
            "query_type": query_type,
            "positive_image_ids": query["positive_image_ids"],
            "rank": rank,
            "positive_score": positive_score,
            "top10": [
                {
                    "item_id": item_ids[int(index)],
                    "score": float(scores[int(index)]),
                    "is_positive": item_ids[int(index)] in set(query["positive_image_ids"]),
                }
                for index in top_indices
            ],
        })
    return {
        "overall": _metric_summary(ranks),
        "by_query_type": {query_type: _metric_summary(type_ranks) for query_type, type_ranks in sorted(by_type.items())},
        "details": details,
    }


def evaluate_public_image(
    *,
    runner: OpenClipRunner,
    manifest_path: Path,
    queries_path: Path,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    item_ids, image_paths, item_map = _load_items(manifest_path)
    queries = _load_queries(queries_path)
    query_embeddings = runner.encode_texts([str(query["query"]) for query in queries], batch_size)
    strategies: dict[str, Any] = {}
    for mode in ("center_crop", "letterbox"):
        cache_path = _cache_path(cache_dir, "public_image_items", runner.spec, mode)
        embeddings = _load_embeddings_cache(cache_path, item_ids) if use_cache else None
        if embeddings is None:
            embeddings = runner.encode_images(image_paths, mode, batch_size)
            if use_cache:
                _save_embeddings_cache(cache_path, item_ids, embeddings)
        scores_matrix = query_embeddings @ embeddings.T
        counter = [0]
        strategies[mode] = _evaluate_scores(
            queries,
            item_ids,
            lambda matrix=scores_matrix, counter=counter: _score_from_matrix(matrix, counter),
        )
    view_embeddings, _view_item_indices, view_offsets = _spatial_view_embeddings_for_paths(
        runner,
        item_ids,
        image_paths,
        "public_image_spatial_views",
        batch_size,
        cache_dir,
        use_cache,
    )
    for strategy_name in ("sliding_mean", "sliding_max", "sliding_top3", "sliding_mvp_mix"):
        counter = [0]
        strategies[f"{strategy_name}_center_crop"] = _evaluate_scores(
            queries,
            item_ids,
            lambda name=strategy_name, counter=counter: _score_public_view(
                query_embeddings,
                counter,
                name,
                view_embeddings,
                view_offsets,
                len(item_ids),
            ),
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": runner.device,
        "model": {
            "model_name": runner.spec.model_name,
            "pretrained": runner.spec.pretrained,
            "slug": runner.spec.slug,
        },
        "manifest": str(manifest_path),
        "queries_path": str(queries_path),
        "items": len(item_ids),
        "queries": len(queries),
        "query_type_counts": dict(Counter(str(query.get("query_type") or "unknown") for query in queries)),
        "strategies": strategies,
        "item_sample": list(item_map.values())[:3],
    }


def _score_from_matrix(matrix: np.ndarray, counter: list[int]) -> np.ndarray:
    index = counter[0]
    counter[0] += 1
    return matrix[index]


def _score_public_view(
    query_embeddings: np.ndarray,
    counter: list[int],
    strategy_name: str,
    view_embeddings: np.ndarray,
    view_offsets: np.ndarray,
    item_count: int,
) -> np.ndarray:
    index = counter[0]
    counter[0] += 1
    return _view_score_strategies(
        query_embeddings[index],
        view_embeddings,
        view_offsets,
        item_count,
        prefix="sliding",
    )[strategy_name]


def _strategy_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for strategy, payload in (result.get("strategies") or {}).items():
        rows.append({"strategy": strategy, **(payload.get("overall") or {})})
    return sorted(rows, key=lambda row: (row.get("recall_at_10", 0), row.get("mrr", 0)), reverse=True)


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    eval_name = _eval_name(result)
    lines = [
        f"# {eval_name} 图片检索评测报告",
        "",
        f"- 生成时间：`{result['created_at']}`",
        f"- 模型：`{result['model']['slug']}`",
        f"- 图片数：`{result['items']}`",
        f"- query 数：`{result['queries']}`",
        f"- query 类型：`{result['query_type_counts']}`",
        "- 评估口径：multi-positive，TopK 中命中任意正样本即算命中。",
        "",
        "## 策略总榜",
        "",
        "| strategy | queries | R@1 | R@5 | R@10 | R@20 | MRR | 中位排名 | 平均排名 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _strategy_rows(result):
        lines.append(
            "| {strategy} | {queries} | {r1} | {r5} | {r10} | {r20} | {mrr} | {median:.1f} | {mean:.1f} |".format(
                strategy=row["strategy"],
                queries=row.get("queries", 0),
                r1=_pct(row.get("recall_at_1", 0)),
                r5=_pct(row.get("recall_at_5", 0)),
                r10=_pct(row.get("recall_at_10", 0)),
                r20=_pct(row.get("recall_at_20", 0)),
                mrr=_pct(row.get("mrr", 0)),
                median=float(row.get("median_rank", 0)),
                mean=float(row.get("mean_rank", 0)),
            )
        )
    lines += ["", "## 按 query_type", ""]
    for strategy, payload in (result.get("strategies") or {}).items():
        lines += [
            f"### {strategy}",
            "",
            "| query_type | queries | R@1 | R@5 | R@10 | MRR | 中位排名 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for query_type, metrics in (payload.get("by_query_type") or {}).items():
            lines.append(
                "| {qt} | {queries} | {r1} | {r5} | {r10} | {mrr} | {median:.1f} |".format(
                    qt=query_type,
                    queries=metrics.get("queries", 0),
                    r1=_pct(metrics.get("recall_at_1", 0)),
                    r5=_pct(metrics.get("recall_at_5", 0)),
                    r10=_pct(metrics.get("recall_at_10", 0)),
                    mrr=_pct(metrics.get("mrr", 0)),
                    median=float(metrics.get("median_rank", 0)),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["model", "strategy", "query_type", "queries", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "mrr", "median_rank", "mean_rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        model = result["model"]["slug"]
        for strategy, payload in (result.get("strategies") or {}).items():
            writer.writerow({"model": model, "strategy": strategy, "query_type": "all", **(payload.get("overall") or {})})
            for query_type, metrics in (payload.get("by_query_type") or {}).items():
                writer.writerow({"model": model, "strategy": strategy, "query_type": query_type, **metrics})


def write_examples_html(result: dict[str, Any], manifest_path: Path, path: Path, strategy: str) -> None:
    eval_name = _eval_name(result)
    _item_ids, _paths, items = _load_items(manifest_path)
    details = ((result.get("strategies") or {}).get(strategy) or {}).get("details") or []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_type[str(row.get("query_type") or "unknown")].append(row)
    css = """
body { margin: 0; background: #f6f7f9; color: #18202a; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1440px; margin: 0 auto; padding: 28px 24px 56px; }
h1 { margin: 0 0 8px; font-size: 28px; }
details { background: #fff; border: 1px solid #dfe4ea; border-radius: 8px; padding: 16px; margin: 14px 0; }
summary { cursor: pointer; font-size: 18px; font-weight: 700; }
.example { border-top: 1px solid #e4e8ee; margin-top: 16px; padding-top: 16px; }
.query { font-size: 16px; font-weight: 700; }
.top10-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
.card { margin: 0; border: 1px solid #dfe4ea; border-radius: 8px; overflow: hidden; background: #fbfcfe; }
.positive-hit { border-color: #29a05b; box-shadow: inset 0 0 0 1px #29a05b; }
.thumb { background: #e9edf3; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; }
.thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
figcaption { display: grid; gap: 4px; padding: 8px; min-height: 92px; }
figcaption span { color: #596575; font-size: 12px; }
code { font-size: 12px; word-break: break-all; }
"""
    lines = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{_escape(eval_name)} CLIP Top10 样例</title>",
        f"<style>{css}</style></head><body><main>",
        f"<h1>{_escape(eval_name)} CLIP Top10 样例：{_escape(strategy)}</h1>",
    ]
    for query_type, rows in sorted(by_type.items()):
        selected = _select_examples(rows, 2)
        lines.append(f"<details open><summary>{_escape(query_type)} ({len(rows)} 条 query)</summary>")
        for row in selected:
            lines.append(f"<section class=\"example\"><p class=\"query\">{_escape(row.get('query'))}</p><p><code>{_escape(row.get('query_id'))}</code> | rank={row.get('rank')} | positive_score={row.get('positive_score'):.4f}</p>")
            lines.append("<div class=\"top10-grid\">")
            positives = set(row.get("positive_image_ids") or [])
            for index, result in enumerate(row.get("top10") or [], start=1):
                item_id = str(result.get("item_id"))
                lines.append(_card(
                    repo_root=Path.cwd(),
                    out_path=path,
                    task="image",
                    item_id=item_id,
                    item=items.get(item_id),
                    title=f"返回 Top {index}",
                    score=result.get("score"),
                    positive=item_id in positives,
                ))
            lines.append("</div></section>")
        lines.append("</details>")
    lines.append("</main></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _select_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    misses = sorted([row for row in rows if int(row.get("rank", 0)) > 10], key=lambda row: int(row.get("rank", 0)), reverse=True)
    hits = sorted([row for row in rows if int(row.get("rank", 0)) <= 10], key=lambda row: int(row.get("rank", 0)))
    selected: list[dict[str, Any]] = []
    for pool in (misses[:1], hits[:1], rows):
        for row in pool:
            if row not in selected:
                selected.append(row)
            if len(selected) >= limit:
                return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public image eval v1 with CLIP multi-positive metrics.")
    parser.add_argument("--image-manifest", type=Path, default=Path("eval/visual/image_retrieval/frames.public_image_eval_v1.local.json"))
    parser.add_argument("--queries", type=Path, default=Path("eval/visual/image_retrieval/queries.public_image_eval_v1.local.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("eval/visual/outputs/clip_eval_public_image_v1_b32_openai.local.json"))
    parser.add_argument("--out-md", type=Path, default=Path("eval/visual/outputs/visual_public_image_eval_v1_report.local.md"))
    parser.add_argument("--out-csv", type=Path, default=Path("eval/visual/outputs/clip_eval_public_image_v1_summary.local.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("eval/visual/outputs/visual_public_image_eval_v1_top10.local.html"))
    parser.add_argument("--examples-strategy", default="sliding_max_center_crop")
    parser.add_argument("--cache-dir", type=Path, default=Path("runtime/eval/visual/public_image_eval_v1/clip_cache"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    runner = OpenClipRunner(ModelSpec(args.model, args.pretrained), args.device)
    result = evaluate_public_image(
        runner=runner,
        manifest_path=args.image_manifest,
        queries_path=args.queries,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(result, args.out_md)
    write_summary_csv(result, args.out_csv)
    write_examples_html(result, args.image_manifest, args.out_html, args.examples_strategy)
    print(args.out_json)
    print(args.out_md)
    print(args.out_html)


if __name__ == "__main__":
    main()
