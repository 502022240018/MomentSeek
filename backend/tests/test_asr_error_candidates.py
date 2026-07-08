import importlib.util
from pathlib import Path

import numpy as np


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_export_flags_short_suspect_terms_and_repetition(tmp_path):
    module = _load_script("asr_error_candidates")
    index_dir = tmp_path / "runtime-server" / "indexes" / "video-1"
    index_dir.mkdir(parents=True)
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[1000, 2000], [3000, 5000], [6000, 9000]], dtype=np.int32),
        texts=np.asarray(["赵正宵", "我说过 我说过 我说过", "今天我们讨论书籍收藏"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )

    records = list(module.iter_candidate_records(tmp_path / "runtime-server"))

    by_text = {record["asr_text"]: record for record in records}
    assert "short_cjk_token" in by_text["赵正宵"]["suspect_reasons"]
    assert "repeated_phrase" in by_text["我说过 我说过 我说过"]["suspect_reasons"]
    assert "今天我们讨论书籍收藏" not in by_text


def test_candidate_export_writes_jsonl_and_html(tmp_path):
    module = _load_script("asr_error_candidates")
    index_dir = tmp_path / "runtime-server" / "indexes" / "video-1"
    index_dir.mkdir(parents=True)
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[1000, 2000]], dtype=np.int32),
        texts=np.asarray(["黄拔"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )

    jsonl_path, html_path = module.export_candidates(tmp_path / "runtime-server", tmp_path / "analysis")

    assert jsonl_path.exists()
    assert html_path.exists()
    assert "黄拔" in jsonl_path.read_text(encoding="utf-8")
    assert "short_cjk_token" in html_path.read_text(encoding="utf-8")
