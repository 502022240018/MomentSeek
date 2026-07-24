from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_vlm_rerank_phase1.py"
SPEC = importlib.util.spec_from_file_location("build_vlm_rerank_phase1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frame_times_are_ordered_and_inside_interval() -> None:
    values = MODULE._frame_times(10.0, 20.0, 4)
    assert values == [12.0, 14.0, 16.0, 18.0]


def test_seed_queries_are_complex_and_balanced() -> None:
    path = SCRIPT.parents[1] / "eval" / "vlm_rerank_phase1" / "queries.seed.jsonl"
    rows = MODULE.read_jsonl(path)
    MODULE.validate_queries(rows)
    assert len(rows) == 20
    assert sum(row["mode"] == "visual_only" for row in rows) == 10
    assert sum(row["mode"] == "evidence_fusion" for row in rows) == 10
    assert all(len(row["constraints"]) >= 3 for row in rows)


def test_model_evidence_removes_retrieval_score_leakage() -> None:
    rows = MODULE._model_evidence([{
        "modality": "asr", "text": "预算增长", "best_time": 12.5,
        "score": 0.91, "raw_score": 0.82, "decision": "strong", "detail": "命中",
    }])
    assert rows == [{"modality": "asr", "text": "预算增长", "best_time": 12.5}]


def test_channel_union_round_robins_and_merges_overlaps() -> None:
    visual = [
        {"video_id": "v1", "start_time": 0, "end_time": 5, "modalities": ["visual"], "evidence": []},
        {"video_id": "v2", "start_time": 0, "end_time": 5, "modalities": ["visual"], "evidence": []},
    ]
    asr = [
        {"video_id": "v1", "start_time": 4, "end_time": 7, "modalities": ["asr"], "evidence": [{"text": "x"}]},
        {"video_id": "v3", "start_time": 0, "end_time": 5, "modalities": ["asr"], "evidence": []},
    ]
    rows = MODULE._merge_channel_results({"visual": visual, "asr": asr})
    assert [row["video_id"] for row in rows] == ["v1", "v2", "v3"]
    assert rows[0]["end_time"] == 7
    assert [source["channel"] for source in rows[0]["retrieval_sources"]] == ["visual", "asr"]


def test_validate_dataset_accepts_null_and_graded_labels(tmp_path: Path) -> None:
    MODULE.write_jsonl(tmp_path / "candidates" / "candidate_sets.jsonl", [{
        "query_id": "q1",
        "candidates": [{"candidate_id": "q1__r01", "frame_paths": ["frames/a.jpg"]}],
    }])
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "a.jpg").write_bytes(b"image")
    MODULE.write_jsonl(tmp_path / "annotations.jsonl", [{
        "query_id": "q1",
        "candidate_id": "q1__r01",
        "relevance": 3,
        "constraint_labels": {"action": True, "temporal": None},
    }])
    MODULE.validate_dataset(tmp_path)


def test_validate_dataset_rejects_invalid_relevance(tmp_path: Path) -> None:
    MODULE.write_jsonl(tmp_path / "candidates" / "candidate_sets.jsonl", [{
        "query_id": "q1", "candidates": [{"candidate_id": "c1", "frame_paths": []}],
    }])
    MODULE.write_jsonl(tmp_path / "annotations.jsonl", [{
        "query_id": "q1", "candidate_id": "c1", "relevance": 4, "constraint_labels": {},
    }])
    with pytest.raises(ValueError, match="relevance"):
        MODULE.validate_dataset(tmp_path)
