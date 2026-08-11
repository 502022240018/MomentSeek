#!/usr/bin/env python3
"""Run a reproducible, resumable Planner Lab sample evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


KS = (1, 3, 5, 10, 20)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_result_name(semantic_query_id: str) -> str:
    return hashlib.sha1(semantic_query_id.encode("utf-8")).hexdigest()[:20] + ".json"


def ground_truth_segment_count(group: dict[str, Any]) -> int:
    return sum(
        len(record.get("segments") or [])
        for record in group.get("records") or []
    )


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def select_stratified_sample(
    groups: list[dict[str, Any]],
    per_category: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select multi-answer, Chinese and non-Chinese cases before hashed fill."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        buckets[str(group.get("category") or "uncategorized")].append(group)

    selected: list[dict[str, Any]] = []
    for category in sorted(buckets):
        candidates = list(buckets[category])
        random.Random(f"{seed}:{category}").shuffle(candidates)
        chosen: list[dict[str, Any]] = []
        predicates = (
            lambda row: ground_truth_segment_count(row) > 1,
            lambda row: _has_cjk(str(row.get("query") or "")),
            lambda row: not _has_cjk(str(row.get("query") or "")),
        )
        for predicate in predicates:
            match = next(
                (row for row in candidates if row not in chosen and predicate(row)),
                None,
            )
            if match is not None and len(chosen) < per_category:
                chosen.append(match)
        chosen.extend(
            row
            for row in candidates
            if row not in chosen
            and len(chosen) < per_category
        )
        selected.extend(chosen[:per_category])
    return selected


def build_sample_catalog(
    source: dict[str, Any],
    per_category: int,
    seed: int,
) -> dict[str, Any]:
    groups = select_stratified_sample(
        list(source.get("groups") or []),
        per_category,
        seed,
    )
    return {
        "schema_version": "planner-lab-sample-v1",
        "created_at": utcnow(),
        "source_schema_version": source.get("schema_version"),
        "source_catalog_sha256": source.get("catalog_sha256"),
        "selection": {
            "strategy": "per_category_multi_answer_language_then_seeded_fill",
            "per_category": per_category,
            "seed": seed,
        },
        "query_count": len(groups),
        "groups": groups,
    }


def select_run_groups(
    groups: list[dict[str, Any]],
    semantic_query_ids: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = groups
    if semantic_query_ids:
        by_id = {str(group["semantic_query_id"]): group for group in groups}
        missing = [query_id for query_id in semantic_query_ids if query_id not in by_id]
        if missing:
            raise ValueError(f"Unknown semantic query IDs: {', '.join(missing)}")
        selected = [by_id[query_id] for query_id in semantic_query_ids]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def _post_form(
    session: requests.Session,
    url: str,
    data: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    response = session.post(url, data=data, timeout=(15, timeout_seconds))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def run_query(
    session: requests.Session,
    base_url: str,
    group: dict[str, Any],
    folder_id: str,
    plan_id: str,
    mode: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    query = str(group["query"])
    scope = json.dumps([folder_id], ensure_ascii=False)
    planning_started = time.perf_counter()
    plan_set = _post_form(
        session,
        f"{base_url.rstrip('/')}/api/planner-lab/plans",
        {"query_text": query, "folder_ids": scope, "mode": mode},
        timeout_seconds,
    )
    planning_seconds = time.perf_counter() - planning_started
    plan = next(
        (row for row in plan_set.get("plans") or [] if row.get("plan_id") == plan_id),
        None,
    )
    if plan is None:
        raise RuntimeError(f"Planner did not return plan_id={plan_id!r}")

    execution_started = time.perf_counter()
    execution = _post_form(
        session,
        f"{base_url.rstrip('/')}/api/planner-lab/execute",
        {
            "query_text": query,
            "folder_ids": scope,
            "plan": json.dumps(plan, ensure_ascii=False),
        },
        timeout_seconds,
    )
    execution_seconds = time.perf_counter() - execution_started
    return {
        "benchmark": {
            "semantic_query_id": group["semantic_query_id"],
            "query": query,
            "plan_id": plan_id,
            "mode": mode,
            "planning_seconds": round(planning_seconds, 3),
            "execution_seconds": round(execution_seconds, 3),
            "total_seconds": round(planning_seconds + execution_seconds, 3),
            "completed_at": utcnow(),
        },
        "plan_set": plan_set,
        "selected_plan": plan,
        "execution": execution,
    }


def positive_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[1], right[1]) > max(left[0], right[0])


def _truth_segments(group: dict[str, Any]) -> list[tuple[str, float, float]]:
    return [
        (str(record["video_id"]), float(segment["start_sec"]), float(segment["end_sec"]))
        for record in group.get("records") or []
        for segment in record.get("segments") or []
    ]


def maximum_segment_matches(
    ranked: list[dict[str, Any]],
    truth: list[tuple[str, float, float]],
    limit: int,
) -> int:
    adjacency: list[list[int]] = []
    for result in ranked[:limit]:
        span = (float(result["start_sec"]), float(result["end_sec"]))
        adjacency.append([
            index
            for index, (video_id, start, end) in enumerate(truth)
            if video_id == result["source_video_id"]
            and positive_overlap(span, (start, end))
        ])
    assigned: dict[int, int] = {}

    def augment(result_index: int, seen: set[int]) -> bool:
        for truth_index in adjacency[result_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            previous = assigned.get(truth_index)
            if previous is None or augment(previous, seen):
                assigned[truth_index] = result_index
                return True
        return False

    for result_index in range(len(adjacency)):
        augment(result_index, set())
    return len(assigned)


def score_response(
    group: dict[str, Any],
    response: dict[str, Any],
    platform_to_source: dict[str, str],
) -> dict[str, Any]:
    truth = _truth_segments(group)
    execution = response.get("execution") or {}
    ranked: list[dict[str, Any]] = []
    for rank, result in enumerate(execution.get("results") or [], start=1):
        source_video_id = platform_to_source.get(str(result.get("video_id")), "")
        span = (
            float(result.get("start_time") or 0),
            float(result.get("end_time") or 0),
        )
        matched = any(
            source_video_id == video_id and positive_overlap(span, (start, end))
            for video_id, start, end in truth
        )
        ranked.append({
            "rank": rank,
            "platform_video_id": result.get("video_id"),
            "source_video_id": source_video_id,
            "start_sec": span[0],
            "end_sec": span[1],
            "score": result.get("score"),
            "modalities": result.get("modalities") or [],
            "segment_relevant": matched,
        })
    first_hit = next((row["rank"] for row in ranked if row["segment_relevant"]), None)
    trace = list(execution.get("trace") or [])
    tool_ids = [str((row.get("step") or {}).get("tool_id") or "unknown") for row in trace]
    matched_by_k = {str(k): maximum_segment_matches(ranked, truth, k) for k in KS}
    benchmark = response.get("benchmark") or {}
    return {
        "semantic_query_id": group["semantic_query_id"],
        "query": group["query"],
        "category": group.get("category"),
        "subcategory": group.get("subcategory"),
        "ground_truth_segment_count": len(truth),
        "first_hit_rank": first_hit,
        "matched_segments_at_k": matched_by_k,
        "planning_seconds": float(benchmark.get("planning_seconds") or 0),
        "execution_seconds": float(benchmark.get("execution_seconds") or 0),
        "total_seconds": float(benchmark.get("total_seconds") or 0),
        "tool_calls": len(trace),
        "tool_ids": tool_ids,
        "vlm_calls": sum(tool_id == "vlm.rerank" for tool_id in tool_ids),
        "stop_reason": execution.get("stop_reason"),
        "wrong_but_stable": execution.get("stop_reason") == "ranking_stable" and first_hit is None,
        "ranked": ranked,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    summary: dict[str, Any] = {
        "query_count": len(rows),
        "successful_query_count": len(valid),
        "failed_query_count": len(rows) - len(valid),
    }
    if not valid:
        return summary
    truth_total = sum(row["ground_truth_segment_count"] for row in valid)
    summary.update({
        "hit_at_k": {
            str(k): round(sum((row["first_hit_rank"] or 10**9) <= k for row in valid) / len(valid), 4)
            for k in KS
        },
        "macro_segment_recall_at_k": {
            str(k): round(statistics.mean(
                row["matched_segments_at_k"][str(k)] / row["ground_truth_segment_count"]
                for row in valid
            ), 4)
            for k in KS
        },
        "micro_segment_recall_at_k": {
            str(k): round(
                sum(row["matched_segments_at_k"][str(k)] for row in valid) / truth_total,
                4,
            )
            for k in KS
        },
        "latency_seconds": {
            name: {
                "p50": round(_percentile([row[name] for row in valid], 0.5), 3),
                "p95": round(_percentile([row[name] for row in valid], 0.95), 3),
                "mean": round(statistics.mean(row[name] for row in valid), 3),
            }
            for name in ("planning_seconds", "execution_seconds", "total_seconds")
        },
        "mean_tool_calls": round(statistics.mean(row["tool_calls"] for row in valid), 3),
        "vlm_call_rate": round(sum(row["vlm_calls"] > 0 for row in valid) / len(valid), 4),
        "wrong_but_stable_rate": round(sum(row["wrong_but_stable"] for row in valid) / len(valid), 4),
        "stop_reasons": dict(Counter(str(row["stop_reason"]) for row in valid)),
        "tool_usage": dict(Counter(tool for row in valid for tool in row["tool_ids"])),
    })
    return summary


def build_report(
    sample: dict[str, Any],
    responses_dir: Path,
    platform_to_source: dict[str, str],
    groups: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    report_groups = list(groups if groups is not None else sample.get("groups") or [])
    for group in report_groups:
        path = responses_dir / safe_result_name(str(group["semantic_query_id"]))
        if not path.is_file():
            rows.append({
                "semantic_query_id": group["semantic_query_id"],
                "query": group["query"],
                "category": group.get("category"),
                "error": "missing response",
            })
            continue
        response = read_json(path)
        if response.get("error"):
            rows.append({
                "semantic_query_id": group["semantic_query_id"],
                "query": group["query"],
                "category": group.get("category"),
                "error": response["error"],
            })
            continue
        rows.append(score_response(group, response, platform_to_source))
    report = {
        "schema_version": "planner-lab-sample-report-v1",
        "generated_at": utcnow(),
        "selection": sample.get("selection"),
        "sample_query_count": len(sample.get("groups") or []),
        "pending_query_count": max(
            0,
            len(sample.get("groups") or []) - len(report_groups),
        ),
        "overall": aggregate(rows),
        "by_category": {
            category: aggregate([row for row in rows if row.get("category") == category])
            for category in sorted({str(row.get("category")) for row in rows})
        },
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--plan-id", choices=("fast", "balanced", "deep"), default="balanced")
    parser.add_argument("--mode", choices=("guide", "assist", "auto"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Run one semantic_query_id; repeat for multiple exact cases",
    )
    parser.add_argument("--timeout-seconds", type=float, default=360)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.per_category < 1:
        parser.error("--per-category must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.run_dir / "sample_catalog.json"
    source_catalog_sha256 = file_sha256(args.catalog)
    if sample_path.is_file():
        sample = read_json(sample_path)
        recorded_sha256 = sample.get("source_catalog_sha256")
        if recorded_sha256 and recorded_sha256 != source_catalog_sha256:
            raise RuntimeError(
                "Existing sample_catalog.json was created from a different catalog"
            )
        if not recorded_sha256:
            sample["source_catalog_sha256"] = source_catalog_sha256
            write_json(sample_path, sample)
    else:
        sample = build_sample_catalog(read_json(args.catalog), args.per_category, args.seed)
        sample["source_catalog_sha256"] = source_catalog_sha256
        write_json(sample_path, sample)

    state = read_json(args.run_state)
    folder_id = str(state["folder"]["id"])
    platform_to_source = {
        str(row["platform_video_id"]): str(source_id)
        for source_id, row in (state.get("videos") or {}).items()
        if row.get("platform_video_id")
    }
    if args.prepare_only:
        print(json.dumps({"sample": str(sample_path), "query_count": sample["query_count"]}, ensure_ascii=False))
        return 0

    responses_dir = args.run_dir / "query_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    try:
        groups = select_run_groups(
            list(sample.get("groups") or []),
            args.query_id,
            args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    with requests.Session() as session:
        for index, group in enumerate(groups, start=1):
            output_path = responses_dir / safe_result_name(str(group["semantic_query_id"]))
            if output_path.is_file():
                continue
            print(f"[{index}/{len(groups)}] {group['query']}", flush=True)
            try:
                payload = run_query(
                    session,
                    args.base_url,
                    group,
                    folder_id,
                    args.plan_id,
                    args.mode,
                    args.timeout_seconds,
                )
            except Exception as exc:
                payload = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": utcnow(),
                }
            write_json(output_path, payload)

    report, rows = build_report(sample, responses_dir, platform_to_source, groups=groups)
    report["configuration"] = {
        "base_url": args.base_url,
        "folder_id": folder_id,
        "plan_id": args.plan_id,
        "mode": args.mode,
        "result_limit": 24,
    }
    write_json(args.run_dir / "query_scores.json", rows)
    write_json(args.run_dir / "summary.json", report)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
