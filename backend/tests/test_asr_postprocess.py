from app.indexing.asr_postprocess import AsrPostprocessConfig, postprocess_asr_chunks


def test_postprocess_merges_short_adjacent_chunks_inside_gap_threshold():
    chunks = [
        {"start_time": 0.0, "end_time": 0.4, "text": "今天"},
        {"start_time": 0.8, "end_time": 1.2, "text": "我们聊一本书"},
        {"start_time": 3.0, "end_time": 3.5, "text": "下一段"},
    ]

    processed, stats = postprocess_asr_chunks(
        chunks,
        config=AsrPostprocessConfig(normal_gap_ms=700, short_gap_ms=1500),
    )

    assert [item["text"] for item in processed] == ["今天 我们聊一本书", "下一段"]
    assert stats["raw_chunks"] == 3
    assert stats["processed_chunks"] == 2


def test_postprocess_uses_segment_bonus_without_making_it_a_hard_boundary():
    chunks = [
        {"start_time": 4.6, "end_time": 4.9, "text": "这个镜头"},
        {"start_time": 5.4, "end_time": 6.1, "text": "还在说同一件事"},
    ]

    processed, stats = postprocess_asr_chunks(
        chunks,
        segment_ids=[0, 1],
        config=AsrPostprocessConfig(normal_gap_ms=200, short_gap_ms=300, cross_segment_short_gap_ms=900),
    )

    assert [item["text"] for item in processed] == ["这个镜头 还在说同一件事"]
    assert stats["cross_segment_merges"] == 1


def test_gap_only_strategy_does_not_use_same_segment_bonus():
    from app.indexing.asr_postprocess import strategy_config

    chunks = [
        {"start_time": 0.0, "end_time": 1.0, "text": "第一句话已经表达了完整意思"},
        {"start_time": 2.0, "end_time": 3.0, "text": "第二句话也表达了完整意思"},
    ]

    gap_only, _gap_stats = postprocess_asr_chunks(
        chunks,
        segment_ids=[0, 0],
        config=strategy_config("gap_only"),
    )
    bucket_bonus, _bucket_stats = postprocess_asr_chunks(
        chunks,
        segment_ids=[0, 0],
        config=strategy_config("bucket_bonus"),
    )

    assert [item["text"] for item in gap_only] == ["第一句话已经表达了完整意思", "第二句话也表达了完整意思"]
    assert [item["text"] for item in bucket_bonus] == ["第一句话已经表达了完整意思 第二句话也表达了完整意思"]


def test_postprocess_marks_low_information_chunks_as_semantic_ineligible():
    chunks = [
        {"start_time": 0.0, "end_time": 0.2, "text": "嗯"},
        {"start_time": 1.0, "end_time": 2.0, "text": "足球场上有人射门"},
    ]

    processed, stats = postprocess_asr_chunks(chunks)

    assert processed[0]["semantic_eligible"] is False
    assert processed[1]["semantic_eligible"] is True
    assert stats["semantic_ineligible_chunks"] == 1


def test_postprocess_keeps_abnormally_long_low_information_chunk_out_of_semantic_embeddings():
    chunks = [
        {"start_time": 0.0, "end_time": 14.0, "text": "嗯 嗯 嗯 嗯 嗯"},
    ]

    processed, stats = postprocess_asr_chunks(
        chunks,
        config=AsrPostprocessConfig(hard_max_duration_ms=8000),
    )

    assert len(processed) == 1
    assert processed[0]["semantic_eligible"] is False
    assert stats["long_low_info_chunks"] == 1
