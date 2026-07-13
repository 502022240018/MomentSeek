from app.indexing.asr_text import (
    asr_text_profile,
    normalize_asr_text,
    normalize_search_text,
    semantic_text_quality,
)


def test_normalize_asr_text_folds_fullwidth_and_traditional_chinese():
    assert normalize_asr_text("  這裏有ＡＩ和臺詞  ") == "这里有AI和台词"


def test_normalize_asr_text_removes_repeated_noise_tokens():
    assert normalize_asr_text("嗯，嗯，嗯，我们开始吧") == "我们开始吧"


def test_normalize_search_text_matches_chinese_variants():
    assert normalize_search_text("臺灣 乾淨") == normalize_search_text("台湾 干净")


def test_semantic_text_quality_filters_low_information_fragments():
    assert semantic_text_quality("嗯").eligible is False
    assert semantic_text_quality("对").eligible is False
    assert semantic_text_quality("今天我们要讨论这本书的结构").eligible is True


def test_semantic_text_quality_rejects_connector_only_fragments_without_rejecting_sentences():
    assert semantic_text_quality("And.").reason == "filler"
    assert semantic_text_quality("但是。").reason == "filler"
    assert semantic_text_quality("and but").reason == "filler"
    assert semantic_text_quality("嗯啊").reason == "filler"
    assert semantic_text_quality("但是我不同意。 ").eligible is True


def test_semantic_text_quality_keeps_meaningful_short_commands():
    assert semantic_text_quality("Stop!").eligible is True


def test_semantic_text_quality_rejects_impossible_text_rate():
    quality = semantic_text_quality("我们明天上午一起去图书馆", duration_ms=200)

    assert quality.eligible is False
    assert quality.reason == "impossible_text_rate"
    assert semantic_text_quality("我们明天上午一起去图书馆", duration_ms=0).reason == "impossible_text_rate"


def test_semantic_text_quality_uses_words_not_letters_for_short_english_commands():
    assert semantic_text_quality("Your head!", duration_ms=460).eligible is True
    assert semantic_text_quality(
        "we should go to the library tomorrow",
        duration_ms=200,
    ).reason == "impossible_text_rate"


def test_semantic_text_quality_does_not_rate_reject_normal_duration():
    assert semantic_text_quality("我们明天上午一起去图书馆", duration_ms=2400).eligible is True


def test_asr_text_profile_counts_language_mix():
    profile = asr_text_profile(["今天讲足球", "book and movie"])
    assert profile["chunks"] == 2
    assert profile["cjk_chars"] >= 4
    assert profile["latin_chars"] >= 12
