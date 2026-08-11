from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_planner_lab_sample.py"
SPEC = importlib.util.spec_from_file_location("evaluate_planner_lab_sample", SCRIPT_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def _group(group_id: str, category: str, query: str, segments: int = 1) -> dict:
    return {
        "semantic_query_id": group_id,
        "query": query,
        "category": category,
        "records": [{
            "video_id": "source-1",
            "segments": [
                {"start_sec": index * 10, "end_sec": index * 10 + 5}
                for index in range(segments)
            ],
        }],
    }


def test_stratified_sample_is_deterministic_and_keeps_coverage():
    groups = [
        _group("a1", "A", "中文单答案"),
        _group("a2", "A", "English answer"),
        _group("a3", "A", "中文多答案", 2),
        _group("a4", "A", "Another answer"),
        _group("b1", "B", "中文答案"),
        _group("b2", "B", "English answer"),
        _group("b3", "B", "Many answers", 3),
    ]

    first = evaluation.select_stratified_sample(groups, per_category=3, seed=7)
    second = evaluation.select_stratified_sample(groups, per_category=3, seed=7)

    assert [row["semantic_query_id"] for row in first] == [
        row["semantic_query_id"] for row in second
    ]
    assert len(first) == 6
    for category in ("A", "B"):
        selected = [row for row in first if row["category"] == category]
        assert any(evaluation.ground_truth_segment_count(row) > 1 for row in selected)
        assert any(evaluation._has_cjk(row["query"]) for row in selected)
        assert any(not evaluation._has_cjk(row["query"]) for row in selected)


def test_score_uses_positive_overlap_and_one_to_one_segment_matching():
    group = _group("q1", "A", "query", segments=2)
    response = {
        "benchmark": {
            "planning_seconds": 2,
            "execution_seconds": 3,
            "total_seconds": 5,
        },
        "execution": {
            "stop_reason": "ranking_stable",
            "trace": [{"step": {"tool_id": "visual.search"}}],
            "results": [
                {"video_id": "platform-1", "start_time": 1, "end_time": 4, "score": 0.9},
                {"video_id": "platform-1", "start_time": 2, "end_time": 5, "score": 0.8},
                {"video_id": "platform-1", "start_time": 10, "end_time": 15, "score": 0.7},
            ],
        },
    }

    row = evaluation.score_response(group, response, {"platform-1": "source-1"})

    assert row["first_hit_rank"] == 1
    assert row["matched_segments_at_k"]["1"] == 1
    assert row["matched_segments_at_k"]["3"] == 2
    assert row["wrong_but_stable"] is False
    assert row["tool_ids"] == ["visual.search"]
