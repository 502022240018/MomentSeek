from app.indexing.asr_transcript_parser import parse_funasr_raw_transcript, raw_items_from_chunks


def test_sensevoice_tags_become_chunk_metadata_not_retrieval_text():
    items, diagnostics = parse_funasr_raw_transcript(
        [{"text": "<|zh|><|HAPPY|><|BGM|><|withitn|>hello world", "start": 120, "end": 980}],
        is_sensevoice=True,
    )

    assert diagnostics["raw_items"] == 1
    assert len(items) == 1
    assert items[0].text == "hello world"
    assert items[0].emotion == "happy"
    assert items[0].audio_event == "bgm"


def _timestamps_for_timed_chars(text: str, step_ms: int = 900) -> list[list[int]]:
    timed = [char for char in text if char.strip() and (char.isalnum() or "\u3400" <= char <= "\u9fff")]
    return [[index * step_ms, index * step_ms + 700] for index, _char in enumerate(timed)]


def test_sensevoice_timestamp_text_can_split_at_safe_punctuation():
    text = "<|en|><|NEUTRAL|><|BGM|><|withitn|>hello world. next part, trailing words."

    items, diagnostics = parse_funasr_raw_transcript(
        [{"text": text, "timestamp": _timestamps_for_timed_chars(text, step_ms=900)}],
        is_sensevoice=True,
        split_timestamp_text=True,
    )

    assert [item.text for item in items] == ["hello world.", "next part, trailing words."]
    assert [item.emotion for item in items] == ["neutral", "neutral"]
    assert [item.audio_event for item in items] == ["bgm", "bgm"]
    assert items[0].end_ms <= items[1].start_ms
    assert diagnostics["timestamp_split_items"] == 2


def test_funasr_parser_keeps_long_sentence_as_raw_item():
    text = "一个人唤醒了,他是我从来没有见过的那种男生,孤独敏感又倔强。"

    items, diagnostics = parse_funasr_raw_transcript(
        [{"text": text, "timestamp": _timestamps_for_timed_chars(text)}],
        is_sensevoice=True,
    )

    assert len(items) == 1
    assert items[0].text == text
    assert items[0].start_ms == 0
    assert items[0].end_ms > 12000
    assert diagnostics["raw_items"] == 1
    assert diagnostics["timestamp_mismatch_items"] == 0


def test_funasr_parser_uses_sentence_info_when_available():
    items, diagnostics = parse_funasr_raw_transcript(
        [{"sentence_info": [{"start": 500, "end": 1600, "text": "你好"}]}],
        is_sensevoice=False,
    )

    assert [item.to_dict() for item in items] == [
        {
            "item_id": 0,
            "start_ms": 500,
            "end_ms": 1600,
            "text": "你好",
            "source": "funasr_sentence",
        }
    ]
    assert diagnostics["raw_items"] == 1


def test_raw_items_from_legacy_chunks_preserves_input_order():
    items = raw_items_from_chunks(
        [
            {"start_time": 1.2, "end_time": 2.5, "text": "第一句"},
            {"start_ms": 3300, "end_ms": 4100, "text": "第二句"},
        ],
        source="sidecar",
    )

    assert [item.start_ms for item in items] == [1200, 3300]
    assert [item.end_ms for item in items] == [2500, 4100]
    assert [item.text for item in items] == ["第一句", "第二句"]
