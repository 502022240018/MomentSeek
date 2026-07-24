from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "vlm_rerank_annotation_server.py"
SPEC = importlib.util.spec_from_file_location("vlm_rerank_annotation_server", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_store(tmp_path: Path):
    write_jsonl(tmp_path / "candidates" / "candidate_sets.jsonl", [{"query_id": "q", "candidates": []}])
    write_jsonl(tmp_path / "annotations.jsonl", [{"candidate_id": "c", "relevance": None, "reason": ""}])
    return MODULE.DatasetStore(tmp_path)


def test_store_updates_annotation_atomically(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.update({"candidate_id": "c", "relevance": 3, "constraint_matches": {"动作": True}, "reason": "完整"})
    saved = MODULE.read_jsonl(tmp_path / "annotations.jsonl")[0]
    assert saved["relevance"] == 3
    assert saved["constraint_matches"] == {"动作": True}
    assert not (tmp_path / "annotations.jsonl.tmp").exists()


def test_store_rejects_unknown_candidate_and_bad_score(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        store.update({"candidate_id": "missing", "relevance": 1})
    with pytest.raises(ValueError, match="0..3"):
        store.update({"candidate_id": "c", "relevance": 4})


def test_ui_auto_saves_score_and_constraint_buttons() -> None:
    assert "function setScore(n){syncDraft();current().a.relevance=n;show(qi,ci);save(false)}" in MODULE.HTML
    assert "a.constraint_matches[q.constraints[k]]=v;show(qi,ci);save(false)" in MODULE.HTML
