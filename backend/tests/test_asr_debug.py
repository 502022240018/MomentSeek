import json

from app.indexing.asr_debug import write_asr_debug_artifacts
from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk


def test_debug_writer_does_not_create_directory_when_disabled(tmp_path):
    write_asr_debug_artifacts(
        debug_dir=tmp_path / "debug",
        enabled=False,
        save_raw_transcript=True,
        raw_items=[
            RawTranscriptItem(0, 0, 1000, "你好", "fixture"),
        ],
        retrieval_chunks=[
            RetrievalChunk(0, 0, 1000, "你好", [0]),
        ],
        repair_stats={"word_boundary_repairs": 0},
    )

    assert not (tmp_path / "debug").exists()


def test_debug_writer_saves_requested_artifacts(tmp_path):
    write_asr_debug_artifacts(
        debug_dir=tmp_path / "debug",
        enabled=True,
        save_raw_transcript=True,
        raw_items=[
            RawTranscriptItem(0, 0, 1000, "孤", "fixture"),
            RawTranscriptItem(1, 1200, 2000, "独", "fixture"),
        ],
        retrieval_chunks=[
            RetrievalChunk(0, 0, 2000, "孤独", [0, 1], quality_flags=["cjk_boundary_repair"]),
        ],
        repair_stats={"word_boundary_repairs": 1, "fake_gap_repairs": 0},
    )

    raw = json.loads((tmp_path / "debug" / "asr_raw_transcript.json").read_text(encoding="utf-8"))
    chunks = json.loads((tmp_path / "debug" / "asr_retrieval_chunks.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "debug" / "asr_repair_report.json").read_text(encoding="utf-8"))

    assert raw[0]["text"] == "孤"
    assert chunks[0]["text"] == "孤独"
    assert report["word_boundary_repairs"] == 1
