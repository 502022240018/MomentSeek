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


def test_asr_text_profile_counts_language_mix():
    profile = asr_text_profile(["今天讲足球", "book and movie"])
    assert profile["chunks"] == 2
    assert profile["cjk_chars"] >= 4
    assert profile["latin_chars"] >= 12
