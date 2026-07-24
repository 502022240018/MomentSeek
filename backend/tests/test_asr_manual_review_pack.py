import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_review_pack_writes_single_editable_html_with_embedded_jsonl(tmp_path):
    module = _load_script("asr_manual_review_pack")
    candidates = [
        {
            "video_id": "video-1",
            "video_name": "测试视频.mp4",
            "chunk_id": 3,
            "start_ms": 1000,
            "end_ms": 2600,
            "duration_ms": 1600,
            "asr_text": "黄拔",
            "normalized_text": "黄拔",
            "suspect_reasons": ["short_cjk_token"],
            "manual_label": "",
            "correct_text": "",
            "query_should_hit": "",
            "notes": "",
        }
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n", encoding="utf-8")
    upload_dir = tmp_path / "runtime-server" / "uploads"
    upload_dir.mkdir(parents=True)
    (upload_dir / "video-1.mp4").write_bytes(b"fake")
    output_path = tmp_path / "analysis" / "review.html"

    result = module.create_review_pack(candidate_path, output_path, upload_dir=upload_dir, sample_size=1)

    html = output_path.read_text(encoding="utf-8")
    assert result["records"] == 1
    assert "黄拔" in html
    assert "id=\"review-data\"" in html
    assert "manual_label" in html
    assert "correct_text" in html
    assert "query_should_hit" in html
    assert "video-1.mp4#t=0.500,3.100" in html
    assert "导出 JSONL" in html


def test_review_pack_applies_media_start_offset_to_video_seek(tmp_path):
    module = _load_script("asr_manual_review_pack")
    candidates = [
        {
            "video_id": "video-1",
            "video_name": "测试视频.mp4",
            "chunk_id": 3,
            "start_ms": 1000,
            "end_ms": 2600,
            "duration_ms": 1600,
            "asr_text": "顾小君",
            "normalized_text": "顾小君",
            "suspect_reasons": ["short_cjk_token"],
            "manual_label": "",
            "correct_text": "",
            "query_should_hit": "",
            "notes": "",
        }
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n", encoding="utf-8")
    upload_dir = tmp_path / "runtime-server" / "uploads"
    upload_dir.mkdir(parents=True)
    (upload_dir / "video-1.mp4").write_bytes(b"fake")
    output_path = tmp_path / "analysis" / "review.html"

    module.create_review_pack(
        candidate_path,
        output_path,
        upload_dir=upload_dir,
        sample_size=1,
        media_start_offsets={"video-1": 9.877},
    )

    html = output_path.read_text(encoding="utf-8")
    assert "video-1.mp4#t=10.377,12.977" in html
    assert "media offset +9.877s" in html
    assert '"media_start_offset_ms": 9877' in html
    assert '"video_seek_start_ms": 10377' in html


def test_review_pack_balances_focus_videos_and_reasons():
    module = _load_script("asr_manual_review_pack")
    records = []
    for index in range(12):
        video_id = "focus-a" if index < 6 else "other-b"
        reason = "mixed_script" if index % 2 == 0 else "short_cjk_token"
        records.append({
            "video_id": video_id,
            "video_name": video_id,
            "chunk_id": index,
            "start_ms": index * 1000,
            "end_ms": index * 1000 + 1000,
            "duration_ms": 1000,
            "asr_text": f"文本{index}",
            "normalized_text": f"文本{index}",
            "suspect_reasons": [reason],
            "manual_label": "",
            "correct_text": "",
            "query_should_hit": "",
            "notes": "",
        })

    selected = module.select_review_records(records, sample_size=6, focus_video_ids=["focus-a"])

    assert len(selected) == 6
    assert any(row["video_id"] == "focus-a" for row in selected)
    assert any(row["video_id"] == "other-b" for row in selected)
    reasons = {row["suspect_reasons"][0] for row in selected}
    assert {"mixed_script", "short_cjk_token"} <= reasons
