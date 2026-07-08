from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


_FALLBACK_ZH_VARIANT_FOLD = str.maketrans({
    "臺": "台",
    "灣": "湾",
    "這": "这",
    "裏": "里",
    "裡": "里",
    "詞": "词",
    "對": "对",
    "書": "书",
    "講": "讲",
    "語": "语",
    "國": "国",
    "門": "门",
    "們": "们",
    "會": "会",
    "學": "学",
    "畫": "画",
    "時": "时",
    "間": "间",
    "後": "后",
    "發": "发",
    "與": "与",
    "為": "为",
    "說": "说",
    "個": "个",
    "體": "体",
    "電": "电",
    "腦": "脑",
    "車": "车",
    "風": "风",
    "頭": "头",
    "乾": "干",
    "淨": "净",
})

_LEADING_REPEATED_FILLERS_RE = re.compile(
    r"^(?:(?:嗯+|啊+|呃+|额+|哦+|噢+|唉+|诶+|em+|uh+|um)[，,、。.!?\s]*){2,}",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_SEARCH_DROP_RE = re.compile(r"[\W_]+", re.UNICODE)
_FILLER_RE = re.compile(r"^(嗯+|啊+|呃+|额+|哦+|噢+|唉+|诶+|em+|uh+|um+)$", re.IGNORECASE)
_LOW_INFO_WORDS = {
    "对",
    "是",
    "好",
    "好的",
    "行",
    "嗯嗯",
    "ok",
    "okay",
    "yes",
    "no",
}


@dataclass(frozen=True)
class SemanticTextQuality:
    eligible: bool
    reason: str


@lru_cache(maxsize=1)
def _opencc_converter():
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:
        return None


def _fold_chinese_variants(text: str) -> str:
    converter = _opencc_converter()
    if converter is not None:
        return converter.convert(text)
    return text.translate(_FALLBACK_ZH_VARIANT_FOLD)


def normalize_asr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    normalized = _fold_chinese_variants(normalized)
    without_prefix_fillers = _LEADING_REPEATED_FILLERS_RE.sub("", normalized).strip()
    if without_prefix_fillers:
        normalized = without_prefix_fillers
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_search_text(text: str) -> str:
    normalized = normalize_asr_text(text).casefold()
    return _SEARCH_DROP_RE.sub("", normalized)


def semantic_text_quality(text: str) -> SemanticTextQuality:
    normalized = normalize_asr_text(text)
    compact = normalize_search_text(normalized)
    if not compact:
        return SemanticTextQuality(False, "empty")
    if compact in _LOW_INFO_WORDS or _FILLER_RE.fullmatch(compact):
        return SemanticTextQuality(False, "filler")
    cjk_chars = sum(1 for character in compact if "\u4e00" <= character <= "\u9fff")
    latin_digits = sum(1 for character in compact if character.isascii() and character.isalnum())
    if cjk_chars + latin_digits < 2:
        return SemanticTextQuality(False, "too_short")
    return SemanticTextQuality(True, "ok")


def asr_text_profile(texts: Iterable[str]) -> dict[str, int]:
    chunks = 0
    empty_chunks = 0
    cjk_chars = 0
    latin_chars = 0
    digit_chars = 0
    semantic_eligible_chunks = 0
    for text in texts:
        chunks += 1
        normalized = normalize_asr_text(text)
        if not normalized:
            empty_chunks += 1
        if semantic_text_quality(normalized).eligible:
            semantic_eligible_chunks += 1
        for character in normalized:
            if "\u4e00" <= character <= "\u9fff":
                cjk_chars += 1
            elif character.isascii() and character.isalpha():
                latin_chars += 1
            elif character.isascii() and character.isdigit():
                digit_chars += 1
    return {
        "chunks": chunks,
        "empty_chunks": empty_chunks,
        "cjk_chars": cjk_chars,
        "latin_chars": latin_chars,
        "digit_chars": digit_chars,
        "semantic_eligible_chunks": semantic_eligible_chunks,
    }
