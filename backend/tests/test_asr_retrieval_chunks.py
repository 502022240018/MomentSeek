from app.indexing.asr_pipeline_types import RawTranscriptItem
from app.indexing.asr_retrieval_chunks import RetrievalChunkConfig, build_retrieval_chunks


def test_default_config_uses_final_8_12_window():
    config = RetrievalChunkConfig()

    assert config.normal_gap_ms == 500
    assert config.short_gap_ms == 1000
    assert config.same_bucket_gap_ms == 1000
    assert config.target_max_duration_ms == 8000
    assert config.soft_max_duration_ms == 12000
    assert config.merge_max_duration_ms == 12000
    assert config.same_unit_only is True


def _raw(
    index: int,
    start_ms: int,
    end_ms: int,
    text: str,
    *,
    unit_id: int | None = None,
) -> RawTranscriptItem:
    return RawTranscriptItem(
        item_id=index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        source="fixture",
        unit_id=unit_id,
    )


def test_builder_carries_distinct_asr_metadata():
    chunks, stats = build_retrieval_chunks(
        [
            RawTranscriptItem(
                item_id=0,
                start_ms=0,
                end_ms=800,
                text="what are y",
                source="fixture",
                emotion="neutral",
                audio_event="speech",
            ),
            RawTranscriptItem(
                item_id=1,
                start_ms=900,
                end_ms=1800,
                text="ou doing",
                source="fixture",
                emotion="happy",
                audio_event="speech|bgm",
            ),
        ],
    )

    assert len(chunks) == 1
    assert chunks[0].text == "what are y ou doing"
    assert chunks[0].emotion == "neutral|happy"
    assert chunks[0].audio_event == "speech|bgm"
    assert chunks[0].to_search_dict()["emotion"] == "neutral|happy"
    assert chunks[0].to_search_dict()["audio_event"] == "speech|bgm"


def test_builder_does_not_merge_cjk_single_character_boundary_across_long_gap():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 3_901_100, 3_914_180, "一个人唤醒了,他是我从来没有见过的那种男生,孤"),
            _raw(1, 3_918_120, 3_924_180, "独敏感又倔强。"),
        ],
    )

    assert [chunk.text for chunk in chunks] == [
        "一个人唤醒了,他是我从来没有见过的那种男生,孤",
        "独敏感又倔强。",
    ]
    assert [chunk.source_item_ids for chunk in chunks] == [[0], [1]]
    assert stats["merged_items"] == 0


def test_builder_does_not_use_cjk_short_tail_to_bypass_merge_duration_limit():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 3_944_040, 3_956_760, "是不是很"),
            _raw(1, 3_963_550, 3_978_150, "难受啊,你永"),
            _raw(2, 3_978_270, 3_991_830, "远别再让我看见你"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["是不是很", "难受啊,你永", "远别再让我看见你"]
    assert stats["merged_items"] == 0


def test_builder_does_not_cross_sentence_end_for_normal_pause():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 1200, "我到了。"),
            _raw(1, 5000, 6500, "下一件事开始。"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["我到了。", "下一件事开始。"]
    assert stats["merged_items"] == 0


def test_builder_keeps_space_when_latin_fragments_merge():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 800, "what are y"),
            _raw(1, 1600, 2400, "ou doing"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["what are y ou doing"]
    assert stats["merged_items"] == 1


def test_builder_does_not_treat_complete_latin_words_as_word_boundary_break():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 1000, 2500, "hello world"),
            _raw(1, 5000, 7000, "green field"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["hello world", "green field"]
    assert stats["merged_items"] == 0


def test_builder_keeps_space_before_short_complete_latin_word():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 400, "today"),
            _raw(1, 800, 1200, "we discuss books"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["today we discuss books"]
    assert stats["merged_items"] == 1


def test_builder_does_not_merge_across_decode_units():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 2000, "we need to control", unit_id=0),
            _raw(1, 2050, 4000, "this somehow", unit_id=1),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["we need to control", "this somehow"]
    assert stats["cross_unit_merge_blocks"] == 1


def test_builder_does_not_create_a_chunk_longer_than_merge_limit():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 7000, "the first unfinished thought", unit_id=0),
            _raw(1, 7100, 13000, "continues beyond the limit", unit_id=0),
        ],
    )

    assert len(chunks) == 2
    assert stats["merged_items"] == 0


def test_builder_marks_boundary_length_and_embedding_rejection_in_quality_flags():
    chunks, stats = build_retrieval_chunks(
        [_raw(0, 0, 13_000, "但是", unit_id=0)],
    )

    assert chunks[0].semantic_eligible is False
    assert chunks[0].semantic_reason == "filler"
    assert chunks[0].quality_flags == [
        "non_terminal_boundary",
        "long_chunk",
        "embedding_ineligible:filler",
    ]
    assert stats["long_chunks"] == 1
    assert stats["low_boundary_chunks"] == 1
    assert stats["semantic_ineligible_chunks"] == 1
