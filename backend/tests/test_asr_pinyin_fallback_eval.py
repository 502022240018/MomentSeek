import importlib.util
import json
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


def _fake_converter(text: str) -> list[str]:
    mapping = {
        "赵": "zhao",
        "正": "zheng",
        "骁": "xiao",
        "宵": "xiao",
        "张": "zhang",
        "章": "zhang",
        "三": "san",
        "散": "san",
        "足": "zu",
        "球": "qiu",
        "场": "chang",
    }
    return [mapping[character] for character in text if character in mapping]


def test_pinyin_similarity_matches_homophone_named_entity():
    module = _load_script("asr_pinyin_fallback_eval")

    assert module.pinyin_similarity("赵正骁", "赵正宵", converter=_fake_converter) >= 0.95
    assert module.pinyin_similarity("赵正骁", "足球场", converter=_fake_converter) < 0.5


def test_pinyin_eval_reports_rescued_case(tmp_path):
    module = _load_script("asr_pinyin_fallback_eval")
    runtime = tmp_path / "runtime-server"
    index_dir = runtime / "indexes" / "video-1"
    index_dir.mkdir(parents=True)
    np.savez_compressed(
        index_dir / "asr.npz",
        chunk_times_ms=np.asarray([[1000, 2000], [3000, 4000]], dtype=np.int32),
        texts=np.asarray(["章散", "足球场上有人踢球"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
                {
                    "id": "case-1",
                    "video_id": "video-1",
                    "query": "张三",
                "target_start_ms": 1000,
                "target_end_ms": 2000,
                "should_hit": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.evaluate_file(
        runtime,
        eval_path,
        tmp_path / "analysis",
        converter=_fake_converter,
        top_k=1,
    )

    assert result["summary"]["cases"] == 1
    assert result["summary"]["baseline_recall_at_k"] == 0
    assert result["summary"]["pinyin_recall_at_k"] == 1
    assert Path(result["json_path"]).exists()
    assert Path(result["html_path"]).exists()
