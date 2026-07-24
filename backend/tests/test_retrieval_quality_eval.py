import importlib.util
from pathlib import Path


def _module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "retrieval_quality_eval.py"
    spec = importlib.util.spec_from_file_location("retrieval_quality_eval", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_answer_and_no_answer_queries():
    module = _module()
    queries = [
        {
            "query_id": "q1",
            "query": "雨夜街景",
            "split": "holdout",
            "query_type": "scene",
            "positives": [{"video_id": "v1", "start": 10, "end": 20}],
        },
        {"query_id": "q2", "query": "不存在的镜头", "split": "holdout", "positives": []},
    ]
    results = [
        {
            "query_id": "q1",
            "results": [
                {"video_id": "v2", "start_time": 10, "end_time": 20, "above_threshold": True},
                {"video_id": "v1", "start_time": 12, "end_time": 18, "above_threshold": True},
            ],
        },
        {
            "query_id": "q2",
            "results": [{"video_id": "v3", "start_time": 0, "end_time": 5, "above_threshold": True}],
        },
    ]

    report = module.evaluate(queries, results)

    assert report["overall"]["recall_at_1"] == 0.0
    assert report["overall"]["recall_at_5"] == 1.0
    assert report["overall"]["mrr"] == 0.5
    assert report["overall"]["mean_first_hit_tiou"] == 0.6
    assert report["overall"]["no_answer_false_accept_rate"] == 1.0
    assert report["details"][0]["false_positive_rate"]["5"] == 0.5


def test_below_threshold_results_do_not_count_as_hits_or_false_accepts():
    module = _module()
    queries = [
        {"id": "answer", "targets": [{"video_id": "v1"}]},
        {"id": "none", "targets": []},
    ]
    results = [
        {"id": "answer", "results": [{"video_id": "v1", "above_threshold": False}]},
        {"id": "none", "results": [{"video_id": "v2", "above_threshold": False}]},
    ]

    report = module.evaluate(queries, results)

    assert report["overall"]["recall_at_10"] == 0.0
    assert report["overall"]["no_answer_false_accept_rate"] == 0.0


def test_temporal_threshold_requires_overlap_and_tiou():
    module = _module()
    queries = [{"id": "q", "targets": [{"video_id": "v", "start_ms": 10_000, "end_ms": 20_000}]}]
    results = [{"id": "q", "results": [{"video_id": "v", "start": 19.5, "end": 30.0}]}]

    loose = module.evaluate(queries, results, min_overlap_seconds=0.5, min_tiou=0.0)
    strict = module.evaluate(queries, results, min_overlap_seconds=1.0, min_tiou=0.1)

    assert loose["overall"]["recall_at_1"] == 1.0
    assert strict["overall"]["recall_at_1"] == 0.0
