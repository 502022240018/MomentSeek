# ASR Index Quality Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the ASR retrieval channel by adding lightweight, deterministic post-processing before semantic embedding: text normalization, short-fragment merging, low-information chunk handling, explicit Whisper transcribe guardrails, and a repeatable tuning report over existing indexed material.

**Architecture:** Keep runtime search simple. ASR indexing produces post-processed chunks in the existing `asr.npz` schema, with `embedding_chunk_indices` pointing only to chunks that receive semantic embeddings. Search continues to read `chunk_times_ms`, `texts`, `embeddings`, and `embedding_chunk_indices`; the quality work happens before arrays are saved. Parameter tuning is an offline report script that replays strategies over existing ASR arrays without changing production indexes.

**Tech Stack:** Python, pytest, NumPy, OpenAI Whisper, optional OpenCC, sentence-transformers, FastAPI backend, local Docker CUDA runtime.

---

## Context

Current ASR indexing flow:

- `backend/app/indexing/asr.py` extracts audio, runs FunASR or Whisper, and writes `asr.npz`.
- `backend/app/indexing/text_semantic.py` embeds ASR text chunks with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `backend/app/search.py` combines lexical ASR matching and semantic ASR retrieval.
- `backend/app/indexing/pipeline_manifest.py` records ASR index metadata.

Current quality gaps:

- Old ASR indexes can contain translated English text even when source speech is Chinese. New Whisper calls must explicitly use `task="transcribe"` and the manifest must record task/language evidence.
- Raw Whisper chunks may be too fragmented for retrieval, especially short connective fragments.
- Semantic embeddings should be generated from cleaned and merged chunks, not raw ASR output.
- Some chunks are low-information and should remain visible in transcript text but not consume semantic embedding rows.
- Merge parameters need to be tested on existing material before becoming defaults.

Non-goals:

- Do not introduce an LLM as a runtime dependency.
- Do not require server NPU access.
- Do not change the public search API response shape in this task.
- Do not rewrite existing visual, face, or OCR indexing logic.

---

## File Structure

Create:

```text
backend/app/indexing/asr_text.py
backend/app/indexing/asr_postprocess.py
backend/tests/test_asr_text.py
backend/tests/test_asr_postprocess.py
backend/tests/test_asr_semantic_filtering.py
scripts/asr_postprocess_report.py
docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md
```

Modify:

```text
backend/app/indexing/asr.py
backend/app/indexing/text_semantic.py
backend/app/indexing/pipeline_manifest.py
backend/app/search.py
backend/tests/test_transcript.py
docs/RETRIEVAL_CHANNELS.md
docs/ISSUES_AND_ROADMAP.md
docs/OPERATIONS_AND_LESSONS.md
```

---

## Task 1: Add ASR Text Normalization

- [ ] Add tests in `backend/tests/test_asr_text.py`.

```python
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


def test_normalize_search_text_matches_existing_chinese_variant_behavior():
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
```

- [ ] Implement `backend/app/indexing/asr_text.py`.

```python
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


_FALLBACK_ZH_VARIANT_FOLD = str.maketrans(
    {
        "臺": "台",
        "台": "台",
        "灣": "湾",
        "裏": "里",
        "裡": "里",
        "後": "后",
        "發": "发",
        "髮": "发",
        "於": "于",
        "與": "与",
        "為": "为",
        "說": "说",
        "這": "这",
        "個": "个",
        "們": "们",
        "詞": "词",
        "書": "书",
        "畫": "画",
        "時": "时",
        "間": "间",
        "會": "会",
        "學": "学",
        "國": "国",
        "語": "语",
        "門": "门",
        "風": "风",
        "頭": "头",
        "車": "车",
        "電": "电",
        "腦": "脑",
        "體": "体",
    }
)

_FILLER_RE = re.compile(r"^(嗯+|啊+|呃+|额+|哦+|噢+|唉+|诶+|em+|uh+|um+)$", re.IGNORECASE)
_REPEATED_FILLER_PREFIX_RE = re.compile(r"^(嗯|啊|呃|额|哦|噢|唉|诶)[，,、\s]+(?:\1[，,、\s]+)+")
_SPACE_RE = re.compile(r"\s+")
_SEARCH_KEEP_RE = re.compile(r"[\W_]+", re.UNICODE)


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
    normalized = _REPEATED_FILLER_PREFIX_RE.sub("", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_search_text(text: str) -> str:
    normalized = normalize_asr_text(text).casefold()
    normalized = _SEARCH_KEEP_RE.sub("", normalized)
    return normalized


def semantic_text_quality(text: str) -> SemanticTextQuality:
    normalized = normalize_asr_text(text)
    compact = normalize_search_text(normalized)
    if not compact:
        return SemanticTextQuality(False, "empty")
    if _FILLER_RE.fullmatch(compact):
        return SemanticTextQuality(False, "filler")
    cjk_chars = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    latin_digits = sum(1 for ch in compact if ch.isascii() and ch.isalnum())
    if cjk_chars + latin_digits < 2:
        return SemanticTextQuality(False, "too_short")
    return SemanticTextQuality(True, "ok")


def asr_text_profile(texts: Iterable[str]) -> dict[str, int]:
    chunks = 0
    cjk_chars = 0
    latin_chars = 0
    digit_chars = 0
    empty_chunks = 0
    for text in texts:
        chunks += 1
        normalized = normalize_asr_text(text)
        if not normalized:
            empty_chunks += 1
        for ch in normalized:
            if "\u4e00" <= ch <= "\u9fff":
                cjk_chars += 1
            elif ch.isascii() and ch.isalpha():
                latin_chars += 1
            elif ch.isascii() and ch.isdigit():
                digit_chars += 1
    return {
        "chunks": chunks,
        "empty_chunks": empty_chunks,
        "cjk_chars": cjk_chars,
        "latin_chars": latin_chars,
        "digit_chars": digit_chars,
    }
```

- [ ] Update `backend/app/search.py` so `normalize_text()` delegates to `normalize_search_text()` while keeping the public function name used by tests.

```python
from app.indexing.asr_text import normalize_search_text


def normalize_text(text: str) -> str:
    return normalize_search_text(text)
```

- [ ] Run targeted tests.

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_asr_text.py backend/tests/test_transcript.py -q
```

---

## Task 2: Add Deterministic ASR Chunk Post-processing

- [ ] Add tests in `backend/tests/test_asr_postprocess.py`.

```python
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


def test_postprocess_uses_same_segment_bonus_without_hard_boundary():
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


def test_postprocess_marks_low_information_chunks_as_semantic_ineligible():
    chunks = [
        {"start_time": 0.0, "end_time": 0.2, "text": "嗯"},
        {"start_time": 1.0, "end_time": 2.0, "text": "足球场上有人射门"},
    ]
    processed, stats = postprocess_asr_chunks(chunks)
    assert processed[0]["semantic_eligible"] is False
    assert processed[1]["semantic_eligible"] is True
    assert stats["semantic_ineligible_chunks"] == 1


def test_postprocess_splits_abnormally_long_low_information_chunk():
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
```

- [ ] Implement `backend/app/indexing/asr_postprocess.py`.

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.indexing.asr_text import normalize_asr_text, semantic_text_quality


@dataclass(frozen=True)
class AsrPostprocessConfig:
    normal_gap_ms: int = 700
    short_gap_ms: int = 1500
    same_segment_normal_gap_ms: int = 1200
    same_segment_short_gap_ms: int = 2200
    cross_segment_short_gap_ms: int = 900
    hard_max_duration_ms: int = 8000
    short_text_chars: int = 8
    max_text_chars: int = 160


def strategy_config(name: str) -> AsrPostprocessConfig:
    base = AsrPostprocessConfig()
    if name == "gap_only":
        return base
    if name == "bucket_bonus":
        return replace(base, same_segment_normal_gap_ms=1300, same_segment_short_gap_ms=2400)
    if name == "shot_bonus":
        return replace(base, same_segment_normal_gap_ms=1500, same_segment_short_gap_ms=2600)
    if name == "conservative":
        return replace(base, normal_gap_ms=450, short_gap_ms=900, cross_segment_short_gap_ms=450)
    if name == "aggressive_short":
        return replace(base, normal_gap_ms=900, short_gap_ms=2200, same_segment_short_gap_ms=3200)
    raise ValueError(f"unknown ASR postprocess strategy: {name}")


def default_strategy_names() -> list[str]:
    return ["gap_only", "bucket_bonus", "shot_bonus", "conservative", "aggressive_short"]


def _to_ms(value: Any) -> int:
    return int(round(float(value) * 1000.0))


def _chunk_times_ms(chunk: dict[str, Any]) -> tuple[int, int]:
    if "start_ms" in chunk and "end_ms" in chunk:
        return int(chunk["start_ms"]), int(chunk["end_ms"])
    return _to_ms(chunk.get("start_time", 0.0)), _to_ms(chunk.get("end_time", 0.0))


def _is_short_text(text: str, config: AsrPostprocessConfig) -> bool:
    return len(text.replace(" ", "")) <= config.short_text_chars


def _merge_allowed(
    current: dict[str, Any],
    item: dict[str, Any],
    *,
    config: AsrPostprocessConfig,
) -> tuple[bool, bool]:
    gap_ms = max(0, int(item["start_ms"]) - int(current["end_ms"]))
    same_segment = item.get("segment_id") is not None and item.get("segment_id") == current.get("segment_id")
    short_side = _is_short_text(str(current["text"]), config) or _is_short_text(str(item["text"]), config)
    if same_segment and short_side:
        return gap_ms <= config.same_segment_short_gap_ms, False
    if same_segment:
        return gap_ms <= config.same_segment_normal_gap_ms, False
    if short_side:
        return gap_ms <= config.cross_segment_short_gap_ms or gap_ms <= config.short_gap_ms, True
    return gap_ms <= config.normal_gap_ms, False


def postprocess_asr_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    segment_ids: Iterable[int | None] | None = None,
    config: AsrPostprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    config = config or AsrPostprocessConfig()
    segment_values = list(segment_ids) if segment_ids is not None else []
    normalized: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        start_ms, end_ms = _chunk_times_ms(chunk)
        text = normalize_asr_text(str(chunk.get("text") or ""))
        segment_id = segment_values[idx] if idx < len(segment_values) else None
        if not text:
            continue
        normalized.append(
            {
                "source_chunk_ids": [idx],
                "start_ms": start_ms,
                "end_ms": max(start_ms, end_ms),
                "text": text,
                "segment_id": segment_id,
            }
        )

    processed: list[dict[str, Any]] = []
    cross_segment_merges = 0
    for item in normalized:
        if not processed:
            processed.append(item)
            continue
        current = processed[-1]
        allowed, cross_segment = _merge_allowed(current, item, config=config)
        candidate_duration = int(item["end_ms"]) - int(current["start_ms"])
        candidate_text = f'{current["text"]} {item["text"]}'.strip()
        if allowed and candidate_duration <= config.hard_max_duration_ms and len(candidate_text) <= config.max_text_chars:
            current["end_ms"] = item["end_ms"]
            current["text"] = candidate_text
            current["source_chunk_ids"].extend(item["source_chunk_ids"])
            if cross_segment:
                cross_segment_merges += 1
        else:
            processed.append(item)

    semantic_ineligible = 0
    long_low_info = 0
    for item in processed:
        quality = semantic_text_quality(str(item["text"]))
        duration_ms = int(item["end_ms"]) - int(item["start_ms"])
        if duration_ms > config.hard_max_duration_ms and not quality.eligible:
            long_low_info += 1
        item["semantic_eligible"] = bool(quality.eligible)
        item["semantic_reason"] = quality.reason
        if not quality.eligible:
            semantic_ineligible += 1

    stats = {
        "raw_chunks": len(list(chunks)) if not isinstance(chunks, list) else len(chunks),
        "normalized_chunks": len(normalized),
        "processed_chunks": len(processed),
        "merged_chunks": max(0, len(normalized) - len(processed)),
        "cross_segment_merges": cross_segment_merges,
        "semantic_ineligible_chunks": semantic_ineligible,
        "long_low_info_chunks": long_low_info,
    }
    return processed, stats
```

- [ ] Fix the `raw_chunks` counter by materializing the input once at the top of `postprocess_asr_chunks()` instead of consuming non-list iterables twice.

```python
raw_chunks = list(chunks)
...
for idx, chunk in enumerate(raw_chunks):
...
"raw_chunks": len(raw_chunks),
```

- [ ] Run targeted tests.

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_asr_postprocess.py -q
```

---

## Task 3: Embed Only Post-processed Semantic-eligible Chunks

- [ ] Add tests in `backend/tests/test_asr_semantic_filtering.py`.

```python
import numpy as np

from app.indexing.text_semantic import build_text_semantic_arrays


class FakeTextEmbedder:
    dim = 3

    def encode(self, texts):
        return np.asarray([[float(i + 1), 0.0, 0.0] for i, _ in enumerate(texts)], dtype=np.float32)


def test_build_text_semantic_arrays_skips_semantic_ineligible_chunks():
    chunks = [
        {"text": "嗯", "semantic_eligible": False},
        {"text": "足球场上有人射门", "semantic_eligible": True},
        {"text": "好的", "semantic_eligible": False},
    ]
    embeddings, indices = build_text_semantic_arrays(chunks=chunks, embedder=FakeTextEmbedder())
    assert embeddings.shape == (1, 3)
    assert indices.tolist() == [1]


def test_build_text_semantic_arrays_keeps_backwards_compatible_default():
    chunks = [{"text": "足球场"}, {"text": "烤包子"}]
    embeddings, indices = build_text_semantic_arrays(chunks=chunks, embedder=FakeTextEmbedder())
    assert embeddings.shape == (2, 3)
    assert indices.tolist() == [0, 1]
```

- [ ] Modify `backend/app/indexing/text_semantic.py`.

```python
def _chunk_is_semantic_eligible(chunk: dict) -> bool:
    return bool(chunk.get("semantic_eligible", True))


eligible_items = [
    (idx, str(chunk.get("text") or "").strip())
    for idx, chunk in enumerate(chunks)
    if _chunk_is_semantic_eligible(chunk) and str(chunk.get("text") or "").strip()
]
```

- [ ] Keep array behavior stable when no semantic chunks are eligible.

```python
if not eligible_items:
    return np.zeros((0, embedder.dim), dtype=np.float16), np.zeros((0,), dtype=np.int32)
```

- [ ] Run targeted tests.

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_asr_semantic_filtering.py -q
```

---

## Task 4: Wire Post-processing into ASR Indexing and Manifest Guardrails

- [ ] Update Whisper calls in `backend/app/indexing/asr.py` to force transcription and capture detected language.

```python
options = {"fp16": device != "cpu", "task": "transcribe"}
if language and language != "auto":
    options["language"] = language
if language == "zh":
    options["initial_prompt"] = "以下是普通话简体中文转写，请输出简体中文。"
result = model.transcribe(load_wav_mono(audio_path), **options)
detected_language = str(result.get("language") or "")
```

- [ ] Change `_whisper()` to return both chunks and metadata.

```python
return chunks, {
    "task": "transcribe",
    "requested_language": language or "auto",
    "detected_language": detected_language,
}
```

- [ ] Apply ASR post-processing before semantic embedding.

```python
from app.indexing.asr_postprocess import postprocess_asr_chunks
from app.indexing.asr_text import asr_text_profile


processed_chunks, postprocess_stats = postprocess_asr_chunks(chunks, segment_ids=merge_segment_ids)
embeddings, embedding_chunk_indices = build_text_semantic_arrays(
    chunks=processed_chunks,
    model_name=settings.asr_semantic_model,
    model_dir=settings.model_dir,
    device=semantic_device,
)
_save_asr_npz(output_path, processed_chunks, embeddings, embedding_chunk_indices)
```

- [ ] Add a helper in `backend/app/indexing/asr.py` to derive optional fixed-bucket segment hints from processed raw chunks.

```python
def _fixed_bucket_segment_ids(chunks: list[dict], bucket_ms: int) -> list[int]:
    ids: list[int] = []
    for chunk in chunks:
        start_ms = int(round(float(chunk.get("start_time", 0.0)) * 1000.0))
        ids.append(max(0, start_ms // bucket_ms))
    return ids
```

- [ ] Use fixed 5s bucket hints by default, because this matches the visual segment contract and is cheap.

```python
merge_segment_ids = _fixed_bucket_segment_ids(chunks, bucket_ms=5000)
```

- [ ] Return traceable ASR quality metadata from `build_asr_index()`.

```python
return {
    "chunks": len(processed_chunks),
    "raw_chunks": len(chunks),
    "semantic_chunks": int(len(embedding_chunk_indices)),
    "engine": engine_used,
    "model": model_name,
    "task": "transcribe",
    "requested_language": language,
    "detected_language": detected_language,
    "postprocess_stats": postprocess_stats,
    "text_profile": asr_text_profile(item["text"] for item in processed_chunks),
}
```

- [ ] Update `backend/app/indexing/pipeline_manifest.py` to store guardrail metadata.

```python
"task": str(result.get("task") or "transcribe"),
"requested_language": str(result.get("requested_language") or options.get("asr_language") or settings.asr_language),
"detected_language": str(result.get("detected_language") or ""),
"postprocess_stats": result.get("postprocess_stats") or {},
"text_profile": result.get("text_profile") or {},
```

- [ ] Keep `language` in the manifest for compatibility, mapping it to `detected_language` when present and requested language otherwise.

```python
"language": str(
    result.get("detected_language")
    or result.get("requested_language")
    or options.get("asr_language")
    or settings.asr_language
),
```

- [ ] Run ASR and manifest tests.

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_transcript.py backend/tests/test_index_schema_v3.py backend/tests/test_asr_postprocess.py backend/tests/test_asr_semantic_filtering.py -q
```

---

## Task 5: Add Offline Strategy Tuning Report

- [ ] Create `scripts/asr_postprocess_report.py`.

```python
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.indexing.asr_postprocess import default_strategy_names, postprocess_asr_chunks, strategy_config


def _load_asr_chunks(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    times = data["chunk_times_ms"].astype(np.int64)
    texts = data["texts"].astype(str)
    chunks: list[dict] = []
    for idx, text in enumerate(texts):
        chunks.append(
            {
                "start_ms": int(times[idx][0]),
                "end_ms": int(times[idx][1]),
                "text": str(text),
            }
        )
    return chunks


def _video_title(index_dir: Path) -> str:
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        return index_dir.name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return index_dir.name
    return str(manifest.get("title") or manifest.get("filename") or index_dir.name)


def _examples(raw_chunks: list[dict], processed_chunks: list[dict], limit: int = 8) -> list[str]:
    rows: list[str] = []
    for idx, item in enumerate(processed_chunks[:limit]):
        source_ids = item.get("source_chunk_ids") or []
        raw_text = " | ".join(str(raw_chunks[i].get("text") or "") for i in source_ids if i < len(raw_chunks))
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(raw_text)}</td>"
            f"<td>{html.escape(str(item.get('text') or ''))}</td>"
            f"<td>{int(item.get('start_ms', 0))}-{int(item.get('end_ms', 0))}</td>"
            f"<td>{html.escape(str(item.get('semantic_reason') or ''))}</td>"
            "</tr>"
        )
    return rows


def build_report(runtime_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"asr_postprocess_report_{stamp}.html"
    sections: list[str] = []
    for asr_path in sorted(runtime_dir.glob("indexes/*/asr.npz")):
        index_dir = asr_path.parent
        raw_chunks = _load_asr_chunks(asr_path)
        title = _video_title(index_dir)
        strategy_rows: list[str] = []
        example_blocks: list[str] = []
        segment_ids = [int(chunk["start_ms"]) // 5000 for chunk in raw_chunks]
        for strategy in default_strategy_names():
            processed, stats = postprocess_asr_chunks(
                raw_chunks,
                segment_ids=segment_ids,
                config=strategy_config(strategy),
            )
            short_chunks = sum(1 for item in processed if len(str(item.get("text") or "").replace(" ", "")) <= 8)
            semantic_chunks = sum(1 for item in processed if item.get("semantic_eligible", True))
            strategy_rows.append(
                "<tr>"
                f"<td>{html.escape(strategy)}</td>"
                f"<td>{stats['processed_chunks']}</td>"
                f"<td>{stats['merged_chunks']}</td>"
                f"<td>{short_chunks}</td>"
                f"<td>{semantic_chunks}</td>"
                f"<td>{stats['semantic_ineligible_chunks']}</td>"
                "</tr>"
            )
            example_blocks.append(
                f"<h4>{html.escape(strategy)}</h4>"
                "<table><tr><th>#</th><th>raw</th><th>processed</th><th>ms</th><th>semantic</th></tr>"
                + "".join(_examples(raw_chunks, processed))
                + "</table>"
            )
        sections.append(
            f"<h2>{html.escape(title)}</h2>"
            "<table><tr><th>strategy</th><th>chunks</th><th>merged</th><th>short</th><th>semantic</th><th>semantic skipped</th></tr>"
            + "".join(strategy_rows)
            + "</table>"
            + "".join(example_blocks)
        )
    report_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>ASR Postprocess Report</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.45}"
        "table{border-collapse:collapse;margin:12px 0 24px;width:100%}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{background:#f2f2f2}</style>"
        "<h1>ASR Postprocess Strategy Report</h1>"
        + "".join(sections),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="runtime-server")
    parser.add_argument("--out", default="runtime/analysis")
    args = parser.parse_args()
    path = build_report(Path(args.runtime), Path(args.out))
    print(path)


if __name__ == "__main__":
    main()
```

- [ ] Run the strategy report against existing local runtime indexes.

```powershell
$env:PYTHONPATH='backend'
python scripts/asr_postprocess_report.py --runtime runtime-server --out runtime/analysis
```

- [ ] Read the generated HTML report and select the default strategy.

Selection criteria:

- Chunks should become fewer and more semantically complete, but not collapse unrelated sentences.
- `semantic_chunks / processed_chunks` should stay high while removing obvious filler.
- Short chunk count should drop meaningfully.
- Merged examples should read naturally as retrieval units.
- For current material, prefer the most conservative strategy that fixes obvious fragmentation.

- [ ] Record the selected default and the evidence in `docs/experiments/asr/2026-07-07-asr-postprocess-tuning.md`.

```markdown
# ASR Postprocess Tuning, 2026-07-07

## Material

- Existing local indexed videos under `runtime-server/indexes`.

## Strategies

- `gap_only`
- `bucket_bonus`
- `shot_bonus`
- `conservative`
- `aggressive_short`

## Selected Default

`bucket_bonus`

## Reason

The selected strategy reduces short fragments while keeping merged chunks inside natural local context. It uses 5s fixed visual buckets as positive confidence, but still allows near-boundary short fragments to merge when the gap is small.

## Report

- Generated report: `runtime/analysis/asr_postprocess_report_<timestamp>.html`
```

---

## Task 6: Add Regression Tests for Index Output Compatibility

- [ ] Add or update an ASR indexing unit test that runs with a sidecar transcript and verifies post-processed output.

```python
def test_sidecar_asr_index_postprocesses_and_preserves_schema(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    sidecar = tmp_path / "video.srt"
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\n今天\n\n"
        "2\n00:00:00,800 --> 00:00:01,200\n我们聊一本书\n",
        encoding="utf-8",
    )
    output = tmp_path / "asr.npz"
    result = build_asr_index(video, output, engine="sidecar")
    data = np.load(output, allow_pickle=True)
    assert data["texts"].tolist() == ["今天 我们聊一本书"]
    assert data["chunk_times_ms"].shape == (1, 2)
    assert "embedding_chunk_indices" in data.files
    assert result["raw_chunks"] == 2
    assert result["chunks"] == 1
```

- [ ] Add a manifest unit assertion that `task`, `requested_language`, `detected_language`, `postprocess_stats`, and `text_profile` are present for ASR.

```python
assert asr_entry["task"] == "transcribe"
assert asr_entry["requested_language"] in {"auto", "zh", "en", "ja", "ko"}
assert "detected_language" in asr_entry
assert isinstance(asr_entry["postprocess_stats"], dict)
assert isinstance(asr_entry["text_profile"], dict)
```

- [ ] Run schema and search tests.

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests/test_index_schema_v3.py backend/tests/test_search.py backend/tests/test_transcript.py -q
```

---

## Task 7: Rebuild One ASR Index Locally and Smoke-test Retrieval

- [ ] Rebuild ASR for one short video first, using the local Docker CUDA backend if the API is running.

```powershell
curl.exe -s http://127.0.0.1:18301/api/health
```

- [ ] Submit one ASR-only job for a short indexed video after code is updated.

```powershell
curl.exe -s -X POST http://127.0.0.1:18301/api/index/jobs `
  -H "Content-Type: application/json" `
  -d "{\"video_ids\":[\"<short_video_id>\"],\"modalities\":[\"asr\"],\"asr_model\":\"small\",\"asr_language\":\"auto\"}"
```

- [ ] Poll only that job until completion when validating the implementation.

```powershell
curl.exe -s http://127.0.0.1:18301/api/index/jobs/<job_id>
```

- [ ] Inspect the resulting local ASR arrays.

```powershell
python - <<'PY'
from pathlib import Path
import numpy as np

for path in Path("runtime-server/indexes").glob("*/asr.npz"):
    data = np.load(path, allow_pickle=True)
    print(path.parent.name, data["texts"].shape, data["embeddings"].shape, data["embedding_chunk_indices"].shape)
PY
```

- [ ] Run at least three ASR search smoke tests.

```powershell
curl.exe -s -X POST http://127.0.0.1:18301/api/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"足球场上有人踢球\",\"channels\":[\"asr\"],\"top_k\":5}"
curl.exe -s -X POST http://127.0.0.1:18301/api/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"这本书讲了什么\",\"channels\":[\"asr\"],\"top_k\":5}"
curl.exe -s -X POST http://127.0.0.1:18301/api/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"旅行和朋友聊天\",\"channels\":[\"asr\"],\"top_k\":5}"
```

---

## Task 8: Update Project Documentation

- [ ] Update `docs/RETRIEVAL_CHANNELS.md` ASR section.

Content to include:

- ASR semantic embeddings are generated after text normalization and chunk post-processing.
- Whisper task is always `transcribe`.
- `asr_language=auto` is still allowed for mixed-language uploads.
- Manifest records requested and detected language.
- Runtime search does not use an LLM.

- [ ] Update `docs/ISSUES_AND_ROADMAP.md`.

Content to include:

- Move ASR translated-output guardrail from active risk to mitigated after verification.
- Add remaining improvement item: ASR postprocess defaults should be revisited after more multilingual material.

- [ ] Update `docs/OPERATIONS_AND_LESSONS.md`.

Content to include:

- When ASR text appears in the wrong language, inspect `index_manifest.json` first for `task`, `requested_language`, and `detected_language`.
- Rebuild ASR-only before rebuilding all modalities.
- Keep ASR tuning reports under `runtime/analysis` and summarize durable conclusions under `docs/experiments/asr`.

---

## Task 9: Final Verification

- [ ] Run Python tests covering the changed surface.

```powershell
$env:PYTHONPATH='backend'
python -m pytest `
  backend/tests/test_asr_text.py `
  backend/tests/test_asr_postprocess.py `
  backend/tests/test_asr_semantic_filtering.py `
  backend/tests/test_transcript.py `
  backend/tests/test_index_schema_v3.py `
  backend/tests/test_search.py `
  -q
```

- [ ] Run the ASR postprocess report.

```powershell
$env:PYTHONPATH='backend'
python scripts/asr_postprocess_report.py --runtime runtime-server --out runtime/analysis
```

- [ ] If Docker code changed, rebuild or restart the local development backend through the established local Docker workflow, then verify health.

```powershell
docker ps --filter name=momentseek-mvp-app
curl.exe -s http://127.0.0.1:18301/api/health
```

- [ ] Inspect git diff.

```powershell
git status --short
git diff -- backend/app/indexing/asr.py backend/app/indexing/asr_text.py backend/app/indexing/asr_postprocess.py backend/app/indexing/text_semantic.py backend/app/indexing/pipeline_manifest.py backend/app/search.py
```

- [ ] Commit in small logical commits only after tests pass.

Suggested commits:

```text
test: cover asr text cleanup and chunk merging
feat: postprocess asr chunks before semantic embedding
chore: add asr postprocess tuning report
docs: document asr guardrails and tuning workflow
```

---

## Risks and Controls

- **Risk:** Over-merging makes search hits too broad.
  **Control:** Default to `bucket_bonus` only if report examples show coherent chunks; otherwise use `gap_only`.

- **Risk:** OpenCC dependency is unavailable in a deployment.
  **Control:** `asr_text.py` uses optional import with deterministic fallback mapping.

- **Risk:** Old ASR indexes still contain English translated text.
  **Control:** New manifest fields expose task and language state; old indexes need ASR-only rebuild.

- **Risk:** Parameter tuning report is mistaken for production runtime.
  **Control:** Keep it under `scripts/` and document that it reads existing indexes and writes only analysis files.

