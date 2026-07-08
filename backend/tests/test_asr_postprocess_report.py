import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "asr_postprocess_report.py"
    spec = importlib.util.spec_from_file_location("asr_postprocess_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asr_postprocess_report_reads_existing_indexes_and_writes_html(tmp_path):
    module = _load_report_module()
    runtime = tmp_path / "runtime-server"
    index_dir = runtime / "indexes" / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "video_id": "video-1",
                "duration_ms": 10000,
                "segment_ms": 5000,
                "channels": {"asr": {"file": "asr.npz"}},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[0, 400], [800, 1200], [3200, 3700]], dtype=np.int32),
        texts=np.asarray(["今天", "我们聊一本书", "下一段"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )

    report_path = module.build_report(runtime, tmp_path / "analysis")
    html = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "ASR Postprocess Strategy Report" in html
    assert "video-1" in html
    assert "gap_only" in html
    assert "bucket_bonus" in html
