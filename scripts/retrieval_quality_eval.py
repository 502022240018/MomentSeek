from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_K = (1, 5, 10)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(value)
    return rows


def _query_id(row: dict[str, Any]) -> str:
    value = str(row.get("query_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("query row is missing query_id/id")
    return value


def _targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("targets", row.get("positives", []))
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(f"query {_query_id(row)} targets/positives must be a list of objects")
    return raw


def _seconds(item: dict[str, Any], name: str) -> float | None:
    for key, scale in ((name, 1.0), (f"{name}_time", 1.0), (f"{name}_ms", 0.001)):
        value = item.get(key)
        if value is not None:
            return float(value) * scale
    return None


def temporal_iou(result: dict[str, Any], target: dict[str, Any]) -> float | None:
    result_start, result_end = _seconds(result, "start"), _seconds(result, "end")
    target_start, target_end = _seconds(target, "start"), _seconds(target, "end")
    if None in (result_start, result_end, target_start, target_end):
        return None
    intersection = max(0.0, min(result_end, target_end) - max(result_start, target_start))
    union = max(result_end, target_end) - min(result_start, target_start)
    return intersection / union if union > 0 else 0.0


def overlap_seconds(result: dict[str, Any], target: dict[str, Any]) -> float | None:
    result_start, result_end = _seconds(result, "start"), _seconds(result, "end")
    target_start, target_end = _seconds(target, "start"), _seconds(target, "end")
    if None in (result_start, result_end, target_start, target_end):
        return None
    return max(0.0, min(result_end, target_end) - max(result_start, target_start))


def matches(
    result: dict[str, Any],
    target: dict[str, Any],
    min_overlap_seconds: float,
    min_tiou: float,
) -> tuple[bool, float | None]:
    target_video = str(target.get("video_id") or target.get("group_id") or "")
    result_video = str(result.get("video_id") or result.get("group_id") or "")
    if target_video and target_video != result_video:
        return False, None
    tiou = temporal_iou(result, target)
    overlap = overlap_seconds(result, target)
    if tiou is None or overlap is None:
        return bool(target_video and result_video), None
    return overlap >= min_overlap_seconds and tiou >= min_tiou, tiou


def _dcg(values: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(values))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(answer_rows: list[dict[str, Any]], no_answer_rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "answer_queries": len(answer_rows),
        "no_answer_queries": len(no_answer_rows),
    }
    ranks = [row["rank"] for row in answer_rows if row["rank"] is not None]
    summary["mrr"] = _mean([1.0 / rank for rank in ranks]) or 0.0
    for k in ks:
        summary[f"recall_at_{k}"] = (
            sum(row["rank"] is not None and row["rank"] <= k for row in answer_rows) / len(answer_rows)
            if answer_rows else 0.0
        )
        summary[f"ndcg_at_{k}"] = _mean([row["ndcg"][str(k)] for row in answer_rows]) or 0.0
        summary[f"false_positive_rate_at_{k}"] = _mean(
            [row["false_positive_rate"][str(k)] for row in answer_rows]
        ) or 0.0
    summary["mean_first_hit_tiou"] = _mean(
        [row["first_hit_tiou"] for row in answer_rows if row["first_hit_tiou"] is not None]
    )
    summary["no_answer_false_accept_rate"] = (
        sum(row["accepted_results"] > 0 for row in no_answer_rows) / len(no_answer_rows)
        if no_answer_rows else None
    )
    return summary


def evaluate(
    queries: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = DEFAULT_K,
    min_overlap_seconds: float = 1.0,
    min_tiou: float = 0.0,
) -> dict[str, Any]:
    result_map: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        query_id = _query_id(row)
        if query_id in result_map:
            raise ValueError(f"duplicate result row for query {query_id}")
        results = row.get("results", [])
        if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
            raise ValueError(f"result row {query_id} results must be a list of objects")
        result_map[query_id] = results

    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries:
        query_id = _query_id(query)
        if query_id in seen:
            raise ValueError(f"duplicate query {query_id}")
        seen.add(query_id)
        targets = _targets(query)
        results = result_map.get(query_id, [])
        accepted_results = [item for item in results if item.get("above_threshold") is not False]
        labels: list[int] = []
        hit_tious: list[float | None] = []
        for result in accepted_results:
            target_matches = [matches(result, target, min_overlap_seconds, min_tiou) for target in targets]
            matched = any(value[0] for value in target_matches)
            labels.append(int(matched))
            tious = [value[1] for value in target_matches if value[0] and value[1] is not None]
            hit_tious.append(max(tious) if tious else None)

        rank = next((index for index, label in enumerate(labels, start=1) if label), None)
        ndcg: dict[str, float] = {}
        false_positive_rate: dict[str, float] = {}
        for k in ks:
            observed = labels[:k]
            ideal_hits = min(len(targets), k)
            ideal = _dcg([1] * ideal_hits)
            ndcg[str(k)] = _dcg(observed) / ideal if ideal else 0.0
            false_positive_rate[str(k)] = (
                sum(not value for value in observed) / len(observed) if observed else 0.0
            )
        details.append({
            "query_id": query_id,
            "query": query.get("query", ""),
            "split": query.get("split", "unspecified"),
            "category": query.get("category", query.get("query_type", "unspecified")),
            "has_answer": bool(targets),
            "target_count": len(targets),
            "accepted_results": len(accepted_results),
            "rank": rank,
            "first_hit_tiou": hit_tious[rank - 1] if rank is not None else None,
            "ndcg": ndcg,
            "false_positive_rate": false_positive_rate,
        })

    answer_rows = [row for row in details if row["has_answer"]]
    no_answer_rows = [row for row in details if not row["has_answer"]]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"answer": [], "no_answer": []})
    for row in details:
        grouped[row["split"]]["answer" if row["has_answer"] else "no_answer"].append(row)
    return {
        "schema_version": 1,
        "config": {
            "ks": list(ks),
            "min_overlap_seconds": min_overlap_seconds,
            "min_tiou": min_tiou,
            "below_threshold_results_ignored": True,
        },
        "overall": _summary(answer_rows, no_answer_rows, ks),
        "by_split": {
            split: _summary(rows["answer"], rows["no_answer"], ks)
            for split, rows in sorted(grouped.items())
        },
        "details": details,
        "missing_result_queries": sorted(seen - set(result_map)),
        "unknown_result_queries": sorted(set(result_map) - seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal retrieval results with one shared protocol.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--min-tiou", type=float, default=0.0)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    args = parser.parse_args()
    if args.min_overlap_seconds < 0 or not 0 <= args.min_tiou <= 1:
        parser.error("overlap must be >= 0 and tIoU must be within [0, 1]")
    ks = tuple(sorted(set(args.k)))
    if not ks or ks[0] <= 0:
        parser.error("all k values must be positive")
    payload = evaluate(
        read_jsonl(args.queries),
        read_jsonl(args.results),
        ks=ks,
        min_overlap_seconds=args.min_overlap_seconds,
        min_tiou=args.min_tiou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
