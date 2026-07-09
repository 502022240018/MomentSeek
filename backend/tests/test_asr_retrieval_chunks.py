from app.indexing.asr_pipeline_types import RawTranscriptItem
from app.indexing.asr_retrieval_chunks import RetrievalChunkConfig, build_retrieval_chunks


def _raw(index: int, start_ms: int, end_ms: int, text: str) -> RawTranscriptItem:
    return RawTranscriptItem(
        item_id=index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        source="fixture",
    )


def test_builder_carries_distinct_asr_metadata_into_merged_chunk():
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
        config=RetrievalChunkConfig(false_gap_repair_ms=2000),
    )

    assert stats["word_boundary_repairs"] == 1
    assert len(chunks) == 1
    assert chunks[0].text == "what are you doing"
    assert chunks[0].emotion == "neutral|happy"
    assert chunks[0].audio_event == "speech|bgm"
    assert chunks[0].to_search_dict()["emotion"] == "neutral|happy"
    assert chunks[0].to_search_dict()["audio_event"] == "speech|bgm"


def test_builder_repairs_cjk_single_character_boundary_across_false_gap():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 3_901_100, 3_914_180, "一个人唤醒了,他是我从来没有见过的那种男生,孤"),
            _raw(1, 3_918_120, 3_924_180, "独敏感又倔强。"),
        ],
        config=RetrievalChunkConfig(false_gap_repair_ms=8000, hard_max_duration_ms=35000),
    )

    assert [chunk.text for chunk in chunks] == ["一个人唤醒了,他是我从来没有见过的那种男生,孤独敏感又倔强。"]
    assert chunks[0].source_item_ids == [0, 1]
    assert "cjk_boundary_repair" in chunks[0].quality_flags
    assert stats["word_boundary_repairs"] == 1
    assert stats["fake_gap_repairs"] == 1


def test_builder_repairs_cjk_short_tail_boundary():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 3_944_040, 3_956_760, "是不是很"),
            _raw(1, 3_963_550, 3_978_150, "难受啊,你永"),
            _raw(2, 3_978_270, 3_991_830, "远别再让我看见你"),
        ],
        config=RetrievalChunkConfig(false_gap_repair_ms=8000, hard_max_duration_ms=60000),
    )

    assert [chunk.text for chunk in chunks] == ["是不是很难受啊,你永远别再让我看见你"]
    assert stats["word_boundary_repairs"] == 2


def test_builder_does_not_cross_sentence_end_for_normal_pause():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 1200, "我到了。"),
            _raw(1, 5000, 6500, "下一件事开始。"),
        ],
        config=RetrievalChunkConfig(false_gap_repair_ms=8000),
    )

    assert [chunk.text for chunk in chunks] == ["我到了。", "下一件事开始。"]
    assert stats["fake_gap_repairs"] == 0


def test_builder_repairs_latin_word_boundary():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 800, "what are y"),
            _raw(1, 1600, 2400, "ou doing"),
        ],
        config=RetrievalChunkConfig(false_gap_repair_ms=2000),
    )

    assert [chunk.text for chunk in chunks] == ["what are you doing"]
    assert stats["word_boundary_repairs"] == 1


def test_builder_does_not_treat_complete_latin_words_as_word_boundary_break():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 1000, 2500, "hello world"),
            _raw(1, 5000, 7000, "green field"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["hello world", "green field"]
    assert stats["word_boundary_repairs"] == 0


def test_builder_keeps_space_before_short_complete_latin_word():
    chunks, stats = build_retrieval_chunks(
        [
            _raw(0, 0, 400, "today"),
            _raw(1, 800, 1200, "we discuss books"),
        ],
    )

    assert [chunk.text for chunk in chunks] == ["today we discuss books"]
    assert stats["word_boundary_repairs"] == 0
