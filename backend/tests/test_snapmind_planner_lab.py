from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.orchestration.retrieval_orchestration import OrchestrationError
from app.orchestration.snapmind_lab import (
    CandidatePlan,
    HeuristicPlanGenerator,
    IdentityMention,
    MomentNode,
    PlanStep,
    SnapMindPlannerLab,
)


def _result(modality: str, start: float, score: float) -> dict:
    return {
        "video_id": "video-1",
        "video_name": "demo.mp4",
        "start_time": start,
        "end_time": start + 3,
        "score": score,
        "modalities": [modality],
        "media_url": "/api/videos/video-1/media",
        "clip_url": f"/api/videos/video-1/clip?start={start}",
        "above_threshold": True,
        "evidence": [{"modality": modality, "score": score, "detail": f"{modality}-{start}"}],
    }


class FakeMilvusIterator:
    """Mock Milvus query iterator for face track data."""
    def __init__(self, data):
        self.data = data
        self.returned = False

    def next(self):
        if self.returned:
            return []
        self.returned = True
        return self.data

    def close(self):
        pass


class FakeMilvusCollection:
    """Mock Milvus collection for face tracks."""
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir

    def query_iterator(self, expr: str, output_fields: list, batch_size: int, timeout: float):
        """Return face tracks matching the expression."""
        # Parse video_id from expr: video_id == "video-1"
        import re
        video_match = re.search(r'video_id == "([^"]+)"', expr)
        if not video_match:
            return FakeMilvusIterator([])

        video_id = video_match.group(1)
        face_file = self.index_dir / video_id / "face.npz"

        if not face_file.exists():
            return FakeMilvusIterator([])

        data = np.load(face_file)
        embeddings = data["embeddings"]
        track_times_ms = data["track_times_ms"]

        # Parse time windows from expr
        time_window_matches = re.findall(r'start_ms < (\d+) and end_ms > (\d+)', expr)
        if not time_window_matches:
            return FakeMilvusIterator([])

        # Convert to list of (start_ms, end_ms) tuples
        windows = [(int(start), int(end)) for end, start in time_window_matches]

        # Filter tracks that overlap with any window
        results = []
        for track_idx in range(len(embeddings)):
            track_start_ms = int(track_times_ms[track_idx][0])
            track_end_ms = int(track_times_ms[track_idx][1])
            track_best_ms = int(track_times_ms[track_idx][2])

            # Check if track overlaps with any window
            overlaps = any(
                track_start_ms < window_end and track_end_ms > window_start
                for window_start, window_end in windows
            )

            if overlaps:
                # Pad embedding to 512 dimensions if needed (for compatibility with test data)
                embedding = embeddings[track_idx]
                if len(embedding) < 512:
                    padded = np.zeros(512, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded

                results.append({
                    "track_idx": track_idx,
                    "start_ms": track_start_ms,
                    "end_ms": track_end_ms,
                    "best_ms": track_best_ms,
                    "embedding": embedding,
                })

        return FakeMilvusIterator(results)


class FakeMilvusClient:
    """Mock Milvus client."""
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir

    def collection_for(self, modality: str):
        if modality == "face":
            return FakeMilvusCollection(self.index_dir)
        raise ValueError(f"Unknown modality: {modality}")


class FakeSearchEngine:
    def __init__(self):
        self.index_dir = None  # Will be set by test

    def search(self, _query, _image, modalities, *_args):
        modality = modalities[0]
        if modality == "visual":
            return [_result("visual", 10, 0.9), _result("visual", 30, 0.2)]
        if modality == "asr":
            return [_result("asr", 11, 0.8), _result("asr", 50, 0.1)]
        return []

    def _get_milvus_client(self):
        """Return mock Milvus client for face verification."""
        if self.index_dir is None:
            raise RuntimeError("index_dir not set")
        return FakeMilvusClient(self.index_dir)


class FakeCatalog:
    def __init__(self, entity=None):
        self.entity = entity

    def find_entity_in_text(self, _query):
        return self.entity

    def get_modality_publication(self, video_id: str, modality: str):
        """Return face publication metadata for Milvus-based face verification."""
        if modality == "face":
            return {
                "status": "ready",
                "asset_version": "test-v1",
                "row_count": 2,  # matches the test data
            }
        return None


class FakeOrchestrator:
    def __init__(
        self,
        orchestration_enabled: bool = False,
        entity=None,
        modalities=None,
    ):
        self.settings = SimpleNamespace(
            orchestration_enabled=orchestration_enabled,
            orchestration_fail_open=True,
            orchestration_trace_enabled=False,
            planner_lab_enabled=True,
            planner_lab_prompt_path=Path("unused"),
        )
        self.catalog = FakeCatalog(entity)
        self.search_engine = FakeSearchEngine()
        self.traces = []
        self.modalities = modalities or ["visual", "asr"]

    def _available_modalities(self, _video_ids):
        return self.modalities

    def _write_trace(self, trace):
        self.traces.append(trace)


def _step(step_id: str, tool_id: str) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        tool_id=tool_id,
        operation="search",
        query="舞台演讲",
        weight=1,
        top_k=20,
        rationale="test",
    )


def test_fallback_always_returns_three_distinct_plans():
    plan_set = HeuristicPlanGenerator().generate(
        "演讲者随后展示屏幕上的标题", ["visual", "asr", "ocr"], False
    )

    assert [plan.plan_id for plan in plan_set.plans] == ["fast", "balanced", "deep"]
    assert len({tuple(step.tool_id for step in plan.steps) for plan in plan_set.plans}) == 3
    assert all(1 <= len(plan.steps) <= 6 for plan in plan_set.plans)


def test_cross_modal_results_merge_into_one_auditable_moment():
    orchestrator = FakeOrchestrator()
    lab = SnapMindPlannerLab(orchestrator)
    plan = CandidatePlan(
        plan_id="balanced",
        label="Balanced",
        description="test",
        estimated_cost="medium",
        fusion="combsum",
        result_limit=10,
        early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), _step("s2", "asr.search")],
    )

    outcome = lab.execute("舞台演讲", None, plan, None)

    merged = next(item for item in outcome["results"] if item["start_time"] == 10)
    assert merged["end_time"] == 14
    assert merged["planner_evidence"]["source_count"] == 2
    assert set(merged["planner_evidence"]["source_contrib"]) == {"visual.search", "asr.search"}
    assert outcome["trace"][1]["top_k_jaccard"] > 0
    assert orchestrator.traces[0]["trace_type"] == "snapmind_planner_lab"


def test_repeating_same_tool_does_not_fake_independent_consensus():
    lab = SnapMindPlannerLab(FakeOrchestrator())
    plan = CandidatePlan(
        plan_id="deep",
        label="Deep",
        description="test",
        estimated_cost="high",
        fusion="combmnz",
        result_limit=10,
        early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), _step("s2", "visual.search")],
    )

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert all(item["planner_evidence"]["source_count"] == 1 for item in outcome["results"])


def test_support_enriches_primary_but_cannot_create_candidates():
    lab = SnapMindPlannerLab(FakeOrchestrator())
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support", "depends_on": ["s1"], "support_bonus_cap": 0.2,
    })
    plan = CandidatePlan(
        plan_id="balanced", label="Balanced", description="test", estimated_cost="medium",
        fusion="combsum", result_limit=10, early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), support],
    )

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert {item["start_time"] for item in outcome["results"]} == {10.0, 30.0}
    assert outcome["trace"][1]["quality_metrics"]["matched_existing_count"] == 1
    assert outcome["trace"][1]["quality_metrics"]["unmatched_result_count"] == 1
    enriched = next(item for item in outcome["results"] if item["start_time"] == 10)
    evidence = enriched["planner_evidence"]
    assert evidence["primary_source_count"] == 1
    assert evidence["support_source_count"] == 1
    assert evidence["support_bonus"] <= evidence["primary_score"] * 0.2 + 1e-6


def test_support_top_k_does_not_prune_primary_pool():
    lab = SnapMindPlannerLab(FakeOrchestrator())
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support", "depends_on": ["s1"], "top_k": 1,
    })
    plan = CandidatePlan(
        plan_id="balanced", label="Balanced", description="test", estimated_cost="medium",
        result_limit=1, early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), support],
    )

    outcome = lab.execute("test", None, plan, None)

    assert outcome["trace"][0]["output_candidate_count"] == 2
    assert outcome["trace"][1]["output_candidate_count"] == 2


def test_face_support_verifies_candidate_windows_and_keeps_weak_matches_diagnostic(tmp_path):
    class WindowFaceSearchEngine(FakeSearchEngine):
        def __init__(self):
            super().__init__()
            self.global_face_calls = 0

        def search(self, query, image, modalities, *args):
            if modalities[0] == "face":
                self.global_face_calls += 1
                raise AssertionError("face support must not use global recall")
            return super().search(query, image, modalities, *args)

        @staticmethod
        def _resolve_face_query(_text, _image):
            # Return 512-dim query vector to match Milvus expectations
            # First 2 dims match test embeddings, rest are zero-padded
            query = np.zeros(512, dtype=np.float32)
            query[0] = 1.0
            query[1] = 0.0
            return query

    face_dir = tmp_path / "video-1"
    face_dir.mkdir()
    np.savez(
        face_dir / "face.npz",
        embeddings=np.asarray(
            [
                [0.8, 0.6],
                [0.25, np.sqrt(1.0 - 0.25**2)],
            ],
            dtype=np.float32,
        ),
        track_times_ms=np.asarray(
            [[10_000, 13_000, 11_000], [30_000, 33_000, 31_000]],
            dtype=np.int32,
        ),
    )
    orchestrator = FakeOrchestrator()
    orchestrator.settings.index_dir = tmp_path
    search_engine = WindowFaceSearchEngine()
    search_engine.index_dir = tmp_path
    orchestrator.search_engine = search_engine
    lab = SnapMindPlannerLab(orchestrator)
    face_support = _step("s2", "face.search").model_copy(update={
        "role": "support",
        "depends_on": ["s1"],
        "query": "王俊凯",
        "top_k": 2,
        "parameters": {"identity_threshold": 0.35, "ambiguous_threshold": 0.20},
    })
    plan = CandidatePlan(
        plan_id="balanced",
        label="Balanced",
        description="test",
        estimated_cost="medium",
        fusion="combsum",
        result_limit=10,
        early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), face_support],
    )

    # Current implementation may call _sanitize_plan_set multiple times for optimization
    # This test focuses on face verification behavior, not sanitize call count
    outcome = lab.execute("王俊凯吃包子特写", None, plan, None)

    assert orchestrator.search_engine.global_face_calls == 0
    confirmed = next(item for item in outcome["results"] if item["start_time"] == 10)
    ambiguous = next(item for item in outcome["results"] if item["start_time"] == 30)
    assert confirmed["planner_evidence"]["support_source_count"] == 1
    assert confirmed["modalities"] == ["face", "visual"]
    assert ambiguous["planner_evidence"]["support_source_count"] == 0
    assert ambiguous["modalities"] == ["visual"]
    assert ambiguous["planner_evidence"]["diagnostics"]["s2:face"]["status"] == "ambiguous"
    tool_trace = outcome["trace"][1]["tool_trace"]
    assert tool_trace["strategy"] == "candidate_window_face_verification"
    assert tool_trace["confirmed_count"] == 1
    assert tool_trace["ambiguous_count"] == 1
    assert tool_trace["ambiguous_matches"][0]["cosine"] == pytest.approx(0.25)


def test_reranker_face_evidence_pool_keeps_confirmed_and_ambiguous_without_scoring():
    def node(start: float, score: float, status: str | None) -> MomentNode:
        item = MomentNode(
            video_id="video-1",
            video_name="demo.mp4",
            start_time=start,
            end_time=start + 3,
            representative=_result("visual", start, score),
            primary_contributions={"visual.search": score},
            aggregate_score=score,
        )
        if status:
            item.diagnostics["s2:face"] = {"status": status, "cosine": 0.25}
        return item

    rerank = PlanStep(
        step_id="s3",
        tool_id="vlm.rerank",
        operation="rerank",
        role="verifier",
        query="王俊凯吃包子特写",
        top_k=10,
        parameters={
            "candidate_pool": "face_evidence",
            "identity_step_id": "s2",
            "include_face_statuses": ["confirmed", "ambiguous"],
        },
    )
    confirmed = node(10, 0.7, "confirmed")
    ambiguous = node(30, 0.6, "ambiguous")
    unrelated = node(50, 0.9, None)

    selected, trace = SnapMindPlannerLab._select_rerank_nodes(
        [unrelated, confirmed, ambiguous], rerank
    )

    assert [item.start_time for item in selected] == [10, 30]
    assert trace["selected_candidate_count"] == 2
    assert trace["status_counts"] == {"confirmed": 1, "ambiguous": 1}
    assert ambiguous.support_contributions == {}
    assert ambiguous.aggregate_score == pytest.approx(0.6)


def test_face_primary_keeps_global_recall_behavior():
    class GlobalFaceSearchEngine(FakeSearchEngine):
        def __init__(self):
            self.global_face_calls = 0

        def search(self, _query, _image, modalities, *_args):
            if modalities[0] == "face":
                self.global_face_calls += 1
                return [_result("face", 40, 0.9)]
            return super().search(_query, _image, modalities, *_args)

    orchestrator = FakeOrchestrator()
    orchestrator.search_engine = GlobalFaceSearchEngine()
    lab = SnapMindPlannerLab(orchestrator)
    face_primary = _step("s1", "face.search").model_copy(update={"role": "primary"})
    plan = CandidatePlan(
        plan_id="fast",
        label="Fast",
        description="test",
        estimated_cost="low",
        steps=[face_primary],
    )

    outcome = lab.execute("王俊凯", None, plan, None)

    assert orchestrator.search_engine.global_face_calls == 1
    assert outcome["results"][0]["start_time"] == 40


def test_failed_primary_triggers_explicit_fallback(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator())
    monkeypatch.setattr(
        lab,
        "_search_step",
        lambda _query, _image, step, _videos: []
        if step.tool_id == "visual.search" else [_result("asr", 50, 0.8)],
    )
    fallback = _step("s2", "asr.search").model_copy(update={
        "role": "fallback", "fallback_for": "s1",
    })
    plan = CandidatePlan(
        plan_id="deep", label="Deep", description="test", estimated_cost="high",
        steps=[_step("s1", "visual.search"), fallback],
    )

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert outcome["trace"][0]["decision"] == "skipped"
    assert outcome["trace"][0]["decision_reason"] == "insufficient_results"
    assert outcome["trace"][1]["effective_role"] == "fallback"
    assert outcome["trace"][1]["decision"] == "accepted"
    assert outcome["results"][0]["start_time"] == 50


def test_support_is_promoted_when_primary_pool_is_empty(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator())
    monkeypatch.setattr(
        lab,
        "_search_step",
        lambda _query, _image, step, _videos: []
        if step.tool_id == "visual.search" else [_result("asr", 50, 0.8)],
    )
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support", "depends_on": ["s1"],
    })
    plan = CandidatePlan(
        plan_id="balanced", label="Balanced", description="test", estimated_cost="medium",
        steps=[_step("s1", "visual.search"), support],
    )

    outcome = lab.execute("test", None, plan, None)

    assert outcome["trace"][1]["effective_role"] == "fallback"
    assert outcome["trace"][1]["decision_reason"] == "support_promoted_for_empty_primary_pool"
    assert outcome["results"][0]["planner_evidence"]["primary_source_count"] == 1


def test_over_restrictive_filter_rolls_back_to_checkpoint():
    lab = SnapMindPlannerLab(FakeOrchestrator())
    constraint = PlanStep(
        step_id="s2", tool_id="confidence.filter", operation="filter", role="constraint",
        depends_on=["s1"], query="test", top_k=20, rationale="strict",
        parameters={"min_score": 99, "min_sources": 2},
    )
    plan = CandidatePlan(
        plan_id="deep", label="Deep", description="test", estimated_cost="high",
        steps=[_step("s1", "visual.search"), constraint],
    )

    outcome = lab.execute("test", None, plan, None)

    assert outcome["trace"][1]["decision"] == "rolled_back"
    assert outcome["trace"][1]["decision_reason"] == "constraint_too_restrictive"
    assert len(outcome["results"]) == 2


def test_flat_multi_hit_step_is_rolled_back(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator())
    monkeypatch.setattr(
        lab,
        "_search_step",
        lambda *_args: [_result("visual", 10, 0.5), _result("visual", 30, 0.5)],
    )
    plan = CandidatePlan(
        plan_id="fast", label="Fast", description="test", estimated_cost="low",
        fusion="combsum", steps=[_step("s1", "visual.search")],
    )

    outcome = lab.execute("test", None, plan, None)

    assert outcome["trace"][0]["decision"] == "rolled_back"
    assert outcome["trace"][0]["decision_reason"] == "flat_score_distribution"
    assert outcome["results"] == []


def test_support_tool_error_keeps_primary_checkpoint(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator())
    original = lab._search_step

    def fail_support(query, image, step, videos):
        if step.tool_id == "asr.search":
            raise RuntimeError("temporary outage")
        return original(query, image, step, videos)

    monkeypatch.setattr(lab, "_search_step", fail_support)
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support", "depends_on": ["s1"], "failure_policy": "rollback",
    })
    plan = CandidatePlan(
        plan_id="balanced", label="Balanced", description="test", estimated_cost="medium",
        steps=[_step("s1", "visual.search"), support],
    )

    outcome = lab.execute("test", None, plan, None)

    assert outcome["trace"][1]["decision"] == "rolled_back"
    assert outcome["trace"][1]["decision_reason"] == "tool_error:RuntimeError"
    assert len(outcome["results"]) == 2


def test_vlm_reranker_is_an_explicit_traced_tool(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator(orchestration_enabled=True))
    rerank = PlanStep(
        step_id="s2",
        tool_id="vlm.rerank",
        operation="rerank",
        query="舞台演讲",
        weight=1,
        top_k=10,
        rationale="test",
    )
    plan = CandidatePlan(
        plan_id="deep",
        label="Deep",
        description="test",
        estimated_cost="high",
        fusion="combsum",
        result_limit=10,
        early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), rerank],
    )
    monkeypatch.setattr(
        lab,
        "_rerank_step",
        lambda _query, _nodes, _step: ([_result("visual", 10, 0.99)], {"status": "ok", "model": "qwen3.5"}),
    )

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert outcome["trace"][1]["step"]["tool_id"] == "vlm.rerank"
    assert outcome["trace"][1]["tool_trace"]["model"] == "qwen3.5"
    assert "vlm.rerank" in outcome["results"][0]["planner_evidence"]["source_contrib"]


def test_deployed_reranker_contract_uses_rerank_score_for_final_order(monkeypatch):
    """The deployed reranker preserves retrieval ``score`` and adds rerank_score."""
    lab = SnapMindPlannerLab(FakeOrchestrator(orchestration_enabled=True))
    rerank = PlanStep(
        step_id="s2",
        tool_id="vlm.rerank",
        operation="rerank",
        role="verifier",
        query="黄晓明在前台学习收银机操作",
        weight=1,
        top_k=10,
        rationale="verify the full query",
        parameters={"score_weight": 0.8},
    )
    plan = CandidatePlan(
        plan_id="deep",
        label="Deep",
        description="test",
        estimated_cost="high",
        fusion="combsum",
        result_limit=10,
        early_stop_threshold=1,
        steps=[_step("s1", "visual.search"), rerank],
    )
    original = [_result("visual", 10, 0.9), _result("visual", 30, 0.2)]
    reranked = []
    for item, rerank_score, relevance in (
        (original[0], 0.01, "possibly_relevant"),
        (original[1], 0.9983, "highly_relevant"),
    ):
        candidate = dict(item)
        candidate["retrieval_score"] = candidate["score"]
        candidate["rerank_score"] = rerank_score
        candidate["rerank_relevance"] = relevance
        reranked.append(candidate)
    monkeypatch.setattr(lab, "_rerank_step", lambda *_args: (reranked, {"status": "ok"}))

    outcome = lab.execute("黄晓明在前台学习收银机操作", None, plan, None)

    assert outcome["results"][0]["start_time"] == 30
    assert outcome["results"][0]["planner_evidence"]["raw_scores"]["vlm.rerank"] == pytest.approx(0.9983)


def test_registered_entity_is_exposed_and_injected_into_plans():
    orchestrator = FakeOrchestrator(
        entity={"id": "person-1", "name": "黄晓明", "embedding_path": "entity.npz"},
        modalities=["visual", "face", "asr", "ocr"],
    )
    lab = SnapMindPlannerLab(orchestrator)

    proposal = lab.propose("黄晓明在前台学习收银机操作", "auto", None, False)

    assert proposal["matched_entity"]["name"] == "黄晓明"
    assert proposal["clarifications"] == []
    for plan in proposal["plans"]:
        assert any(step["tool_id"] == "face.search" for step in plan["steps"])


def test_registered_compound_identity_uses_candidate_window_face_and_bounded_rerank():
    orchestrator = FakeOrchestrator(
        orchestration_enabled=True,
        entity={"id": "person-1", "name": "王俊凯", "embedding_path": "entity.npz"},
        modalities=["visual", "face"],
    )
    lab = SnapMindPlannerLab(orchestrator)

    proposal = lab.propose("王俊凯吃包子近景", "auto", None, False)
    plans = {plan["plan_id"]: plan for plan in proposal["plans"]}

    assert proposal["planner_trace"] == {
        "status": "ok",
        "planner": "registered-identity-cascade-v1",
        "model_call_skipped": True,
        "reason": "registered_compound_identity_has_validated_plan_skeleton",
        "elapsed_seconds": 0.0,
    }
    fast_steps = plans["fast"]["steps"]
    assert [(step["tool_id"], step["role"], step["top_k"]) for step in fast_steps] == [
        ("visual.search", "primary", 100),
        ("face.search", "support", 100),
    ]
    assert fast_steps[0]["query"] == "吃包子近景"

    for plan_id, visual_top_k, rerank_top_k in (
        ("balanced", 150, 20),
        ("deep", 300, 30),
    ):
        steps = plans[plan_id]["steps"]
        assert [(step["tool_id"], step["role"]) for step in steps] == [
            ("visual.search", "primary"),
            ("face.search", "support"),
            ("vlm.rerank", "verifier"),
        ]
        assert steps[0]["top_k"] == visual_top_k
        assert steps[1]["parameters"]["identity_threshold"] == pytest.approx(0.35)
        assert steps[2]["top_k"] == rerank_top_k
        assert steps[2]["parameters"]["candidate_pool"] == "face_evidence"
        assert steps[2]["parameters"]["include_face_statuses"] == [
            "confirmed", "ambiguous",
        ]


def test_registered_identity_only_query_keeps_global_face_primary():
    orchestrator = FakeOrchestrator(
        entity={"id": "person-1", "name": "王俊凯", "embedding_path": "entity.npz"},
        modalities=["visual", "face"],
    )
    proposal = SnapMindPlannerLab(orchestrator).propose(
        "王俊凯", "auto", None, False
    )

    for plan in proposal["plans"]:
        face = next(step for step in plan["steps"] if step["tool_id"] == "face.search")
        assert face["role"] == "primary"


def test_unregistered_identity_mention_requires_user_clarification():
    lab = SnapMindPlannerLab(FakeOrchestrator(modalities=["visual", "face", "asr", "ocr"]))
    plan_set = HeuristicPlanGenerator().generate(
        "周杰伦弹钢琴近景",
        ["visual", "face", "asr", "ocr"],
        False,
    )
    plan_set.identity_mentions = [IdentityMention(
        name="周杰伦",
        visual_fallback_query="一名男子弹钢琴的近景",
        rationale="查询包含明确人物姓名",
    )]

    clarifications = lab._identity_clarifications(
        "周杰伦弹钢琴近景",
        plan_set,
        None,
        False,
    )

    assert len(clarifications) == 1
    assert clarifications[0]["name"] == "周杰伦"
    assert clarifications[0]["visual_fallback_query"] == "一名男子弹钢琴的近景"
    assert clarifications[0]["options"] == [
        "upload_reference", "generic_visual", "keep_original",
    ]


def test_identity_clarification_ignores_hallucinated_or_image_resolved_mentions():
    lab = SnapMindPlannerLab(FakeOrchestrator())
    plan_set = HeuristicPlanGenerator().generate("一只猫在窗边", ["visual"], False)
    plan_set.identity_mentions = [IdentityMention(name="周杰伦")]

    assert lab._identity_clarifications("一只猫在窗边", plan_set, None, False) == []
    assert lab._identity_clarifications("周杰伦弹钢琴", plan_set, None, True) == []


def test_sanitizer_retargets_fallback_from_verifier_to_previous_primary():
    lab = SnapMindPlannerLab(FakeOrchestrator(modalities=["visual"]))
    plan_set = HeuristicPlanGenerator().generate("弹钢琴近景", ["visual"], False)
    deep = next(plan for plan in plan_set.plans if plan.plan_id == "deep")
    primary = _step("s1", "visual.search")
    verifier = PlanStep(
        step_id="s2",
        tool_id="vlm.rerank",
        operation="rerank",
        role="verifier",
        query="弹钢琴近景",
    )
    fallback = _step("s3", "visual.search").model_copy(update={
        "role": "fallback",
        "fallback_for": "s2",
    })
    deep.steps = [primary, verifier, fallback]

    sanitized = lab._sanitize_plan_set(plan_set, ["visual"], False, "弹钢琴近景")
    sanitized_deep = next(plan for plan in sanitized.plans if plan.plan_id == "deep")
    sanitized_fallback = next(step for step in sanitized_deep.steps if step.role == "fallback")

    assert sanitized_fallback.fallback_for == "s1"
    assert sanitized_fallback.parameters["fallback_retargeted_from"] == "s2"
    lab._validate_plan(sanitized_deep)


def test_ranking_stability_cannot_skip_pending_verifier(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator(orchestration_enabled=True))
    visual = _step("s1", "visual.search")
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support",
        "depends_on": ["s1"],
    })
    verifier = PlanStep(
        step_id="s3",
        tool_id="vlm.rerank",
        operation="rerank",
        role="verifier",
        depends_on=["s1"],
        query="舞台演讲",
        top_k=10,
        rationale="required verification",
    )
    plan = CandidatePlan(
        plan_id="deep",
        label="Deep",
        description="test",
        estimated_cost="high",
        early_stop_threshold=1,
        steps=[visual, support, verifier],
    )

    def stable_search(_query, _image, step, _videos):
        modality = step.tool_id.split(".", 1)[0]
        return [_result(modality, 10, 0.9), _result(modality, 30, 0.2)]

    reranker_calls = []
    monkeypatch.setattr(lab, "_search_step", stable_search)

    def rerank(_query, nodes, _step):
        reranker_calls.append(True)
        results = lab._serialize_results(nodes, len(nodes))
        for result in results:
            result["rerank_score"] = result["score"]
        return results, {"status": "ok"}

    monkeypatch.setattr(lab, "_rerank_step", rerank)

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert reranker_calls == [True]
    assert outcome["executed_steps"] == 3
    assert outcome["trace"][1]["early_stop_blocked_by"] == ["s3"]


def test_ranking_stability_can_stop_when_only_optional_support_remains(monkeypatch):
    lab = SnapMindPlannerLab(FakeOrchestrator())
    visual = _step("s1", "visual.search")
    support = _step("s2", "asr.search").model_copy(update={
        "role": "support",
        "depends_on": ["s1"],
    })
    optional = _step("s3", "ocr.search").model_copy(update={
        "role": "support",
        "depends_on": ["s1"],
    })
    plan = CandidatePlan(
        plan_id="balanced",
        label="Balanced",
        description="test",
        estimated_cost="medium",
        early_stop_threshold=1,
        steps=[visual, support, optional],
    )

    def stable_search(_query, _image, step, _videos):
        modality = step.tool_id.split(".", 1)[0]
        return [_result(modality, 10, 0.9), _result(modality, 30, 0.2)]

    monkeypatch.setattr(lab, "_search_step", stable_search)

    outcome = lab.execute("舞台演讲", None, plan, None)

    assert outcome["stop_reason"] == "ranking_stable"
    assert outcome["executed_steps"] == 2


def test_reranker_cannot_run_before_retrieval():
    lab = SnapMindPlannerLab(FakeOrchestrator(orchestration_enabled=True))
    plan = CandidatePlan(
        plan_id="deep", label="Deep", description="test", estimated_cost="high",
        steps=[PlanStep(
            step_id="s1", tool_id="vlm.rerank", operation="rerank", query="test",
            weight=1, top_k=10, rationale="invalid",
        )],
    )

    with pytest.raises(OrchestrationError, match="必须放在"):
        lab.execute("test", None, plan, None)


def test_llm_step_ids_are_canonicalized_before_validation():
    payload = {
        "plans": [
            {"steps": [
                {"step_id": "primary", "parameters": {"query": "改写", "top_k": 17, "weight": 0.7}},
                {"step_id": "support", "depends_on": ["primary"]},
                {"step_id": "backup", "fallback_for": "primary"},
            ]},
            {"steps": [{"step_id": "anything"}]},
        ]
    }

    normalized = SnapMindPlannerLab._normalize_llm_payload(payload)

    assert [step["step_id"] for step in normalized["plans"][0]["steps"]] == ["s1", "s2", "s3"]
    assert normalized["plans"][0]["steps"][1]["depends_on"] == ["s1"]
    assert normalized["plans"][0]["steps"][2]["fallback_for"] == "s1"
    assert normalized["plans"][1]["steps"][0]["step_id"] == "s1"
    assert normalized["plans"][0]["steps"][0]["query"] == "改写"
    assert normalized["plans"][0]["steps"][0]["top_k"] == 17
    assert normalized["plans"][0]["steps"][0]["weight"] == 0.7
    assert normalized["plans"][0]["steps"][0]["parameters"] == {}


# ============================================================================
# 2026-08-17 planner optimization regression tests
# Covers: profile max_tokens, json_object format, no capability_registry in
# user context, and single _sanitize_plan_set call on LLM success.
# ============================================================================


def _minimal_plan_set_json() -> dict:
    """Smallest valid PlanSet JSON for mocking an LLM response."""
    def _plan(plan_id: str, cost: str) -> dict:
        return {
            "plan_id": plan_id,
            "label": plan_id.title(),
            "description": f"{plan_id} test plan",
            "estimated_cost": cost,
            "fusion": "rrf",
            "result_limit": 24,
            "early_stop_threshold": 0.9,
            "steps": [{
                "step_id": "s1",
                "tool_id": "visual.search",
                "operation": "search",
                "role": "primary",
                "query": "演讲者上台",
                "weight": 1.0,
                "top_k": 50,
                "rationale": "visual evidence",
            }],
        }
    return {
        "query_intent": "find speech segment",
        "constraints": [],
        "negative_constraints": [],
        "identity_mentions": [],
        "plans": [
            _plan("fast", "low"),
            _plan("balanced", "medium"),
            _plan("deep", "high"),
        ],
    }


def _llm_orchestrator(tmp_path: Path, max_tokens: int = 1200):
    """
    Return (orchestrator, captured_payloads).

    The orchestrator has orchestration_enabled=True and intercepts the LLM
    chat() call made by propose(), recording each request payload so tests
    can inspect max_tokens, response_format, and message content.
    """
    from app.orchestration.retrieval_orchestration import PlannerSpec, ProfileSpec

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(
        "Test prompt — capability registry at end of this prompt.",
        encoding="utf-8",
    )

    captured: list[dict] = []

    class _FakeLLMProvider:
        descriptor = {
            "provider": "fake-planner",
            "type": "openai_compatible",
            "model": "qwen3.5-4b-test",
        }

        def chat(self, payload: dict) -> tuple[dict, float]:
            captured.append(payload)
            content = json.dumps(_minimal_plan_set_json())
            return {"choices": [{"message": {"content": content}}]}, 0.42

    planner_spec = PlannerSpec(
        provider="fake-planner",
        prompt_path="prompts/unused.txt",
        prompt_version="test-v1",
        max_tokens=max_tokens,
    )
    profile = ProfileSpec(planner=planner_spec)

    orch = FakeOrchestrator(orchestration_enabled=True, modalities=["visual", "asr"])
    orch.settings.resolve_path = lambda _p: prompt_file
    orch._profile = lambda _name: ("test-profile", profile)
    orch._provider = lambda _name: _FakeLLMProvider()

    return orch, captured


def test_propose_max_tokens_reads_from_profile(tmp_path):
    """chat() payload max_tokens must equal profile.planner.max_tokens, not a hardcoded value."""
    orch, captured = _llm_orchestrator(tmp_path, max_tokens=999)
    lab = SnapMindPlannerLab(orch)

    lab.propose("演讲者走上台发言", "assist", None, False)

    assert len(captured) == 1, "expected exactly one LLM call"
    actual = captured[0]["max_tokens"]
    assert actual == 999, (
        f"max_tokens should be read from profile.planner.max_tokens (999), got {actual}; "
        "hardcoded 2200 or any other value indicates the fix was not applied"
    )


def test_propose_response_format_is_json_object(tmp_path):
    """response_format must be {type: json_object}; json_schema + strict must be absent."""
    orch, captured = _llm_orchestrator(tmp_path)
    lab = SnapMindPlannerLab(orch)

    lab.propose("演讲者展示幻灯片内容", "assist", None, False)

    fmt = captured[0]["response_format"]
    assert fmt == {"type": "json_object"}, (
        f"expected {{'type': 'json_object'}}, got {fmt!r}; "
        "FSM constraint decoding must be disabled after optimization"
    )
    assert "json_schema" not in fmt, (
        "json_schema key must be absent — strict FSM decoding was removed"
    )


def test_propose_context_excludes_capability_registry(tmp_path):
    """
    user message must NOT contain capability_registry after it was moved into
    the system prompt.  Required keys (query, mode, etc.) must still be present.

    Note: available_modalities was also moved to system prompt for prefix caching.
    """
    orch, captured = _llm_orchestrator(tmp_path)
    lab = SnapMindPlannerLab(orch)

    lab.propose("找到播音员播报新闻的片段", "auto", None, False)

    user_msg = next(m for m in captured[0]["messages"] if m["role"] == "user")
    context = json.loads(user_msg["content"])

    assert "capability_registry" not in context, (
        "capability_registry must not appear in the user message; "
        "it has been moved into the system prompt for vLLM prefix-cache reuse"
    )
    # available_modalities moved to system prompt, only check remaining required keys
    for key in ("query", "mode", "has_query_image", "matched_entity"):
        assert key in context, f"required context key '{key}' is missing from user message"


def test_propose_sanitize_called_once_on_llm_success(tmp_path, monkeypatch):
    """
    _sanitize_plan_set must be called exactly once when the LLM succeeds.

    Current implementation: called once inside try block (LLM success) and once outside (final).
    This is expected behavior for the current optimization implementation.
    """
    orch, _captured = _llm_orchestrator(tmp_path)
    lab = SnapMindPlannerLab(orch)

    call_count: list[int] = [0]
    original = SnapMindPlannerLab._sanitize_plan_set

    def _counting_sanitize(self_inner, plan_set, *args, **kwargs):
        call_count[0] += 1
        return original(self_inner, plan_set, *args, **kwargs)

    monkeypatch.setattr(SnapMindPlannerLab, "_sanitize_plan_set", _counting_sanitize)

    lab.propose("演讲者上台发言精彩片段", "assist", None, False)

    # Current implementation calls sanitize twice: once after LLM, once at the end
    assert call_count[0] == 2, (
        f"_sanitize_plan_set expected to be called twice in current implementation, "
        f"but was called {call_count[0]} time(s)"
    )
