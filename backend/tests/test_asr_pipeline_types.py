from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk


def test_raw_transcript_item_dict_roundtrip_uses_ms_fields():
    item = RawTranscriptItem(
        item_id=3,
        start_ms=1200,
        end_ms=2450,
        text="孤独敏感又倔强。",
        source="funasr",
        unit_id=1,
        diagnostics={"timestamp_jumps": 0},
    )

    payload = item.to_dict()

    assert payload == {
        "item_id": 3,
        "start_ms": 1200,
        "end_ms": 2450,
        "text": "孤独敏感又倔强。",
        "source": "funasr",
        "unit_id": 1,
        "diagnostics": {"timestamp_jumps": 0},
    }
    assert RawTranscriptItem.from_dict(payload) == item


def test_retrieval_chunk_exports_legacy_seconds_for_existing_semantic_code():
    chunk = RetrievalChunk(
        chunk_id=0,
        start_ms=1000,
        end_ms=4200,
        text="是不是很难受啊",
        source_item_ids=[7, 8],
        semantic_eligible=True,
        semantic_reason="ok",
        quality_flags=["cjk_boundary_repair"],
    )

    payload = chunk.to_search_dict()

    assert payload["start_ms"] == 1000
    assert payload["end_ms"] == 4200
    assert payload["start_time"] == 1.0
    assert payload["end_time"] == 4.2
    assert payload["text"] == "是不是很难受啊"
    assert payload["source_chunk_ids"] == [7, 8]
    assert payload["semantic_eligible"] is True
    assert payload["semantic_reason"] == "ok"
