from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.orchestration.retrieval_orchestration import (
    OpenAICompatibleProvider,
    OrchestrationRegistry,
    RetrievalPlan,
    SearchOrchestrator,
    OrchestrationError,
    _extract_json_object,
)
from app.platform import context
from app.core.settings import Settings


class FakeCatalog:
    def list_videos(self):
        return [
            {
                "id": "video-1",
                "indexed_modalities": ["visual", "asr", "ocr"],
            }
        ]

    def get_video(self, video_id):
        return None


class FakeSearchEngine:
    def __init__(self):
        self.calls = []

    def search(self, *args):
        self.calls.append(args)
        return [
            {
                "video_id": "video-1",
                "video_name": "demo",
                "start_time": 0,
                "end_time": 4,
                "score": 0.9,
                "modalities": ["visual"],
                "above_threshold": True,
                "evidence": [],
            },
            {
                "video_id": "video-1",
                "video_name": "demo",
                "start_time": 10,
                "end_time": 14,
                "score": 0.7,
                "modalities": ["visual"],
                "above_threshold": True,
                "evidence": [],
            },
        ]


class FakeProvider:
    def __init__(self, name, model, responses):
        self.name = name
        self.model = model
        self.responses = iter(responses)
        self.requests = []

    @property
    def descriptor(self):
        return {
            "provider": self.name,
            "type": "openai_compatible",
            "model": self.model,
        }

    def chat(self, payload):
        self.requests.append(payload)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, 0.125


def _write_config(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "planner.txt").write_text("planner", encoding="utf-8")
    (prompt_dir / "reranker.txt").write_text(
        "Query={{query}}\nCandidate={{candidate}}", encoding="utf-8"
    )
    config = tmp_path / "orchestration.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "providers": {
                    "planner-model": {
                        "type": "openai_compatible",
                        "base_url": "http://planner/v1",
                        "model": "planner-a",
                    },
                    "reranker-model": {
                        "type": "openai_compatible",
                        "base_url": "http://reranker/v1",
                        "model": "reranker-b",
                    },
                },
                "profiles": {
                    "split-models": {
                        "description": "separate planner and reranker",
                        "planner": {
                            "provider": "planner-model",
                            "prompt_path": "prompts/planner.txt",
                            "prompt_version": "p1",
                        },
                        "reranker": {
                            "provider": "reranker-model",
                            "prompt_path": "prompts/reranker.txt",
                            "prompt_version": "r1",
                            "concurrency": 2,
                            "default_top_n": 2,
                            "default_frame_count": 0,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _choice(content, yes_logprob=None, no_logprob=None):
    value = {"message": {"content": content}}
    if yes_logprob is not None and no_logprob is not None:
        value["logprobs"] = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "Yes", "logprob": yes_logprob},
                        {"token": "No", "logprob": no_logprob},
                    ]
                }
            ]
        }
    return {"choices": [value]}


def _orchestrator(tmp_path):
    config = _write_config(tmp_path)
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        orchestration_enabled=True,
        orchestration_config_path=config,
        orchestration_profile="split-models",
        orchestration_trace_path=tmp_path / "runtime" / "traces.jsonl",
    )
    settings.ensure_dirs()
    engine = FakeSearchEngine()
    orchestrator = SearchOrchestrator(settings, FakeCatalog(), engine)
    orchestrator._load_registry()
    return orchestrator, engine


def test_retrieval_plan_clamps_dependent_limits_and_text_frames():
    plan = RetrievalPlan.model_validate(
        {
            "modalities": ["visual", "asr"],
            "candidate_limit": 10,
            "result_limit": 20,
            "rerank": {
                "enabled": True,
                "strategy": "text",
                "top_n": 50,
                "frame_count": 8,
            },
        }
    )

    assert plan.result_limit == 10
    assert plan.rerank.top_n == 10
    assert plan.rerank.frame_count == 0


def test_temporal_plan_normalizes_subqueries_and_uses_at_least_eight_frames():
    plan = RetrievalPlan.model_validate(
        {
            "modalities": ["visual"],
            "visual_subqueries": [
                " players in stands ",
                "players clink cartons",
                "players in stands",
            ],
            "rerank": {
                "enabled": True,
                "strategy": "multimodal",
                "top_n": 10,
                "frame_count": 4,
                "window_seconds": 10,
            },
        }
    )

    assert plan.visual_subqueries == ["players in stands", "players clink cartons"]
    assert plan.rerank.window_seconds == 10
    assert plan.rerank.frame_count == 8


def test_temporal_candidate_expands_around_original_interval(tmp_path):
    orchestrator, _engine = _orchestrator(tmp_path)

    class VideoCatalog:
        def get_video(self, video_id):
            return {"id": video_id, "duration": 64.12}

    orchestrator.catalog = VideoCatalog()
    expanded = orchestrator._expanded_candidate(
        {
            "video_id": "video-1",
            "start_time": 40.0,
            "end_time": 45.0,
            "clip_url": "/old",
        },
        10,
    )

    assert expanded["start_time"] == 37.5
    assert expanded["end_time"] == 47.5
    assert expanded["original_start_time"] == 40.0
    assert expanded["original_end_time"] == 45.0
    assert "start=37.500&end=47.500" in expanded["clip_url"]


def test_reranker_uses_expanded_verification_but_preserves_result_boundaries(tmp_path):
    orchestrator, _engine = _orchestrator(tmp_path)

    class VideoCatalog(FakeCatalog):
        def get_video(self, video_id):
            return {"id": video_id, "duration": 120}

    orchestrator.catalog = VideoCatalog()
    _name, profile = orchestrator._profile("split-models")
    plan = RetrievalPlan.model_validate({
        "modalities": ["visual"],
        "candidate_limit": 3,
        "result_limit": 3,
        "rerank": {
            "enabled": True,
            "top_n": 2,
            "frame_count": 0,
            "window_seconds": 10,
            "score_weight": 1,
        },
    })
    results = [
        {"video_id": "video-1", "start_time": 10, "end_time": 12, "score": 0.9},
        {"video_id": "video-1", "start_time": 20, "end_time": 22, "score": 0.8},
        {"video_id": "video-1", "start_time": 30, "end_time": 32, "score": 0.85},
    ]
    outcomes = iter([
        {"status": "ok", "rerank_score": 0.1},
        {"status": "ok", "rerank_score": 0.2},
    ])
    orchestrator._score_candidate = lambda *_args, **_kwargs: next(outcomes)

    final, trace = orchestrator._run_reranker(profile, "query", plan, results)

    assert [item["original_rank"] for item in final] == [3, 2, 1]
    by_rank = {item["original_rank"]: item for item in final}
    assert (by_rank[1]["start_time"], by_rank[1]["end_time"]) == (10, 12)
    assert (by_rank[2]["start_time"], by_rank[2]["end_time"]) == (20, 22)
    assert trace["candidates"][0]["verification_start_time"] == 6
    assert trace["candidates"][0]["verification_end_time"] == 16
    assert trace["status"] == "ok"


def test_reranker_all_candidate_failures_are_not_reported_as_ok(tmp_path):
    orchestrator, _engine = _orchestrator(tmp_path)
    _name, profile = orchestrator._profile("split-models")
    plan = RetrievalPlan.model_validate({
        "modalities": ["visual"],
        "rerank": {"enabled": True, "top_n": 2, "frame_count": 0},
    })
    results = FakeSearchEngine().search()
    orchestrator._score_candidate = lambda *_args, **_kwargs: {
        "status": "error",
        "error": "reranker unavailable",
    }

    with pytest.raises(OrchestrationError, match="every selected candidate"):
        orchestrator._run_reranker(profile, "query", plan, results)


def test_reranker_partial_failure_is_explicit(tmp_path):
    orchestrator, _engine = _orchestrator(tmp_path)
    _name, profile = orchestrator._profile("split-models")
    plan = RetrievalPlan.model_validate({
        "modalities": ["visual"],
        "rerank": {"enabled": True, "top_n": 2, "frame_count": 0},
    })
    outcomes = iter([
        {"status": "ok", "rerank_score": 0.9},
        {"status": "error", "error": "bad frame"},
    ])
    orchestrator._score_candidate = lambda *_args, **_kwargs: next(outcomes)

    _final, trace = orchestrator._run_reranker(
        profile, "query", plan, FakeSearchEngine().search()
    )

    assert trace["status"] == "partial"
    assert trace["successful_count"] == 1
    assert trace["error_count"] == 1


def test_extract_json_object_accepts_safe_python_literal_fallback():
    value = _extract_json_object("{'modalities': ['visual'], 'rerank': {'enabled': True}}")

    assert value["modalities"] == ["visual"]
    assert value["rerank"]["enabled"] is True


def test_binary_score_keeps_best_logprob_across_whitespace_token_variants():
    response = {
        "choices": [
            {
                "message": {"content": "Yes"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.01},
                                {"token": "No", "logprob": -6.0},
                                {"token": " Yes", "logprob": -12.0},
                                {"token": " No", "logprob": -11.0},
                            ]
                        }
                    ]
                },
            }
        ]
    }

    score = SearchOrchestrator._binary_score(response, "Yes", "No")

    assert score > 0.99


def test_profile_registry_exposes_independent_models(tmp_path):
    orchestrator, _engine = _orchestrator(tmp_path)

    response = orchestrator.profiles()

    profile = response["profiles"][0]
    assert profile["planner"]["model"] == "planner-a"
    assert profile["reranker"]["model"] == "reranker-b"
    assert profile["planner"]["prompt_version"] == "p1"
    assert profile["reranker"]["prompt_version"] == "r1"


def test_planner_selects_routes_and_reranker_reorders_with_trace(tmp_path):
    orchestrator, engine = _orchestrator(tmp_path)
    planner_output = {
        "query_intent": "spoken_visual_action",
        "modalities": ["visual", "asr", "face"],
        "alpha": 0.4,
        "visual_profile": "precision",
        "visual_subqueries": ["person speaking indoors", "visible speaker"],
        "candidate_limit": 2,
        "channel_limits": {"visual": 99, "asr": 4, "ocr": 8},
        "result_limit": 2,
        "merge_gap": 1.0,
        "max_result_seconds": 12,
        "rerank": {
            "enabled": True,
            "strategy": "text",
            "top_n": 2,
            "frame_count": 0,
            "score_weight": 0.8,
        },
        "rationale": ["combine visual and spoken evidence"],
    }
    planner = FakeProvider(
        "planner-model",
        "planner-a",
        [_choice(json.dumps(planner_output))],
    )
    reranker = FakeProvider(
        "reranker-model",
        "reranker-b",
        [
            _choice("No", yes_logprob=-3.0, no_logprob=-0.1),
            _choice("Yes", yes_logprob=-0.1, no_logprob=-3.0),
        ],
    )
    orchestrator._providers = {
        "planner-model": planner,
        "reranker-model": reranker,
    }

    outcome = orchestrator.search(
        "室内有人讲话",
        None,
        ["visual", "face", "asr", "ocr"],
        ["video-1"],
        0.5,
        10,
    )

    search_call = engine.calls[0]
    assert search_call[2] == ["visual", "asr"]
    assert search_call[4:9] == (0.4, 2, 1.0, 12.0, "precision")
    assert search_call[9] == {"visual": 30, "asr": 4}
    assert search_call[10] == ["person speaking indoors", "visible speaker"]
    assert outcome["results"][0]["start_time"] == 10
    assert outcome["results"][0]["original_rank"] == 2
    assert outcome["execution"]["planner"]["model"] == "planner-a"
    assert planner.requests[0]["response_format"]["type"] == "json_schema"
    assert outcome["execution"]["reranker"]["model"] == "reranker-b"
    assert outcome["execution"]["reranker"]["prompt_version"] == "r1"
    traces = (
        tmp_path / "runtime" / "traces.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(traces) == 1
    assert json.loads(traces[0])["request_id"] == outcome["execution"]["request_id"]


def test_planner_failure_falls_back_to_explicit_request(tmp_path):
    orchestrator, engine = _orchestrator(tmp_path)
    orchestrator._providers["planner-model"] = FakeProvider(
        "planner-model",
        "planner-a",
        [RuntimeError("planner unavailable")],
    )

    outcome = orchestrator.search(
        "精确台词",
        None,
        ["asr"],
        ["video-1"],
        0.7,
        7,
        reranker_mode="off",
    )

    assert engine.calls[0][2] == ["asr"]
    assert engine.calls[0][4] == 0.7
    assert engine.calls[0][5] == 7
    assert outcome["execution"]["planner"]["status"] == "error"


def test_openai_provider_uses_environment_model_and_endpoint(monkeypatch):
    registry = OrchestrationRegistry.model_validate(
        {
            "providers": {
                "model": {
                    "type": "openai_compatible",
                    "base_url": "http://default/v1",
                    "base_url_env": "MODEL_URL",
                    "model": "default-model",
                    "model_env": "MODEL_NAME",
                }
            },
            "profiles": {},
        }
    )
    monkeypatch.setenv("MODEL_URL", "http://override/v1")
    monkeypatch.setenv("MODEL_NAME", "override-model")

    provider = OpenAICompatibleProvider("model", registry.providers["model"])

    assert provider.base_url == "http://override/v1"
    assert provider.descriptor["model"] == "override-model"


def test_search_api_returns_execution_trace(monkeypatch):
    import app.main as main

    class FakeOrchestrator:
        def search(self, *args, **kwargs):
            assert kwargs["profile_name"] == "split-models"
            assert kwargs["planner_mode"] == "force"
            assert kwargs["reranker_mode"] == "off"
            return {
                "results": [],
                "execution": {
                    "request_id": "trace-1",
                    "plan": {"modalities": ["asr"]},
                },
            }

    monkeypatch.setattr(context, "search_orchestrator", FakeOrchestrator())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/search",
            data={
                "query_text": "精确台词",
                "modalities": "asr",
                "video_ids": "[]",
                "orchestration_profile": "split-models",
                "planner_mode": "force",
                "reranker_mode": "off",
            },
        )

    assert response.status_code == 200
    assert response.json()["execution"]["request_id"] == "trace-1"


def test_search_api_preserves_explicit_empty_video_scope(monkeypatch):
    import app.main as main

    calls: dict[str, object] = {}

    class FakeCatalog:
        def recover_color_grading_finalizations(self):
            return 0

        def resolve_video_scope(self, video_ids, folder_ids):
            calls["resolve_video_scope"] = (video_ids, folder_ids)
            return video_ids

        def list_videos(self):
            return [{"id": "video-1"}]

    class FakeOrchestrator:
        def search(self, *args, **kwargs):
            calls["search"] = {"args": args, "kwargs": kwargs}
            return {"results": [], "execution": {"request_id": "trace-empty"}}

    monkeypatch.setattr(context, "catalog", FakeCatalog())
    monkeypatch.setattr(context, "search_orchestrator", FakeOrchestrator())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/search",
            data={
                "query_text": "精确台词",
                "modalities": "asr",
                "video_ids": "[]",
            },
        )

    assert response.status_code == 200
    assert calls["resolve_video_scope"] == ([], None)
    assert calls["search"]["args"][3] == []
    assert response.json()["scope"]["video_ids"] == []
    assert response.json()["scope"]["resolved_video_count"] == 0


def test_search_api_keeps_missing_video_scope_as_none(monkeypatch):
    import app.main as main

    calls: dict[str, object] = {}

    class FakeCatalog:
        def recover_color_grading_finalizations(self):
            return 0

        def resolve_video_scope(self, video_ids, folder_ids):
            calls["resolve_video_scope"] = (video_ids, folder_ids)
            return None

        def list_videos(self):
            return [{"id": "video-1"}]

    class FakeOrchestrator:
        def search(self, *args, **kwargs):
            calls["search"] = {"args": args, "kwargs": kwargs}
            return {"results": [], "execution": {"request_id": "trace-none"}}

    monkeypatch.setattr(context, "catalog", FakeCatalog())
    monkeypatch.setattr(context, "search_orchestrator", FakeOrchestrator())

    with TestClient(main.app) as client:
        response = client.post(
            "/api/search",
            data={
                "query_text": "精确台词",
                "modalities": "asr",
            },
        )

    assert response.status_code == 200
    assert calls["resolve_video_scope"] == (None, None)
    assert calls["search"]["args"][3] is None
    assert response.json()["scope"]["video_ids"] == []
    assert response.json()["scope"]["resolved_video_count"] == 1
