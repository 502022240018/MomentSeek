# ASR Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ASR 索引从“模型输出直接后处理入库”重构为分层 pipeline，让 parser、retrieval chunk builder、debug artifact、semantic embedding 各自职责清楚，并修复中文词内断裂导致的检索文本质量问题。

**Architecture:** 新增轻量 pipeline 类型层、FunASR raw transcript parser、retrieval chunk builder 和 debug artifact writer。`build_asr_index()` 继续作为生产入口，但只把 retrieval chunks 写进 `asr.npz`；raw transcript 与 speech units 仅在 debug 开关开启时落盘。实验和生产路径都复用同一套 parser 与 chunk builder。

**Tech Stack:** Python 3.11, NumPy, pytest, pydantic-settings, FunASR/SenseVoiceSmall, faster-whisper, sentence-transformers MiniLM, existing MomentSeek backend modules.

---

## File Structure

- Create: `backend/app/indexing/asr_pipeline_types.py`
  - ASR pipeline 的共享 dataclass 与 dict 转换函数。
  - 不加载模型，不做文件 I/O。
- Create: `backend/app/indexing/asr_transcript_parser.py`
  - 解析 FunASR/SenseVoice/sidecar/Whisper 风格结果为 `RawTranscriptItem`。
  - parser 不再按 8s/12s 规则切最终 retrieval chunk。
- Create: `backend/app/indexing/asr_retrieval_chunks.py`
  - 从 `RawTranscriptItem[]` 构造 `RetrievalChunk[]`。
  - 负责文本归一化、短句合并、CJK/Latin 边界保护、false timestamp gap 修复和 semantic eligibility。
- Create: `backend/app/indexing/asr_debug.py`
  - 根据 debug 开关写 `debug/asr_raw_transcript.json`、`debug/asr_retrieval_chunks.json`、`debug/asr_repair_report.json`。
  - debug 关闭时不创建目录。
- Modify: `backend/app/indexing/asr.py`
  - 保留模型调用入口，删除 parser 与 retrieval chunking 混合逻辑。
  - `build_asr_index()` 串起新 pipeline，并保持 `asr.npz` schema 为 `chunk_times_ms/texts/embeddings/embedding_chunk_indices`。
- Modify: `backend/app/settings.py`
  - 默认 `asr_language` 改为 `auto`。
  - 新增 `asr_debug_artifacts`、`asr_save_raw_transcript`、`asr_vad_strategy`。
- Modify: `backend/app/stage_runner.py`
  - 把 ASR debug 和 VAD 设置传给 `build_asr_index()`。
- Modify: `backend/app/indexer_daemon.py`
  - 把 ASR debug 和 VAD 设置传给 daemon 模式的 `build_asr_index()`。
- Modify: `backend/app/indexing/pipeline_manifest.py`
  - manifest 记录 `language_route`、`route_reason`、`vad_strategy` 和新 chunk builder stats。
- Modify: `backend/tests/test_transcript.py`
  - 调整旧 parser 断言，保留 build_asr_index 的集成测试。
- Create: `backend/tests/test_asr_pipeline_types.py`
- Create: `backend/tests/test_asr_transcript_parser.py`
- Create: `backend/tests/test_asr_retrieval_chunks.py`
- Create: `backend/tests/test_asr_debug.py`

Execution note: 当前工作树已有未提交改动。每个任务提交前必须运行 `git diff --cached --name-only`，确认 staged 文件只包含本任务列出的文件。

---

### Task 1: Add Pipeline Types

**Files:**
- Create: `backend/app/indexing/asr_pipeline_types.py`
- Create: `backend/tests/test_asr_pipeline_types.py`

- [ ] **Step 1: Write failing tests for shared types**

Create `backend/tests/test_asr_pipeline_types.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify missing module failure**

Run from `video_retrieval_mvp/backend`:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_pipeline_types.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.asr_pipeline_types'`.

- [ ] **Step 3: Implement pipeline dataclasses**

Create `backend/app/indexing/asr_pipeline_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpeechUnit:
    unit_id: int
    start_ms: int
    end_ms: int
    core_start_ms: int
    core_end_ms: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "core_start_ms": self.core_start_ms,
            "core_end_ms": self.core_end_ms,
            "source": self.source,
        }


@dataclass(frozen=True)
class RawTranscriptItem:
    item_id: int
    start_ms: int
    end_ms: int
    text: str
    source: str
    unit_id: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": self.item_id,
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "text": self.text,
            "source": self.source,
        }
        if self.unit_id is not None:
            payload["unit_id"] = int(self.unit_id)
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawTranscriptItem":
        return cls(
            item_id=int(payload["item_id"]),
            start_ms=int(payload["start_ms"]),
            end_ms=int(payload["end_ms"]),
            text=str(payload["text"]),
            source=str(payload.get("source") or "unknown"),
            unit_id=None if payload.get("unit_id") is None else int(payload["unit_id"]),
            diagnostics=dict(payload.get("diagnostics") or {}),
        )


@dataclass(frozen=True)
class RetrievalChunk:
    chunk_id: int
    start_ms: int
    end_ms: int
    text: str
    source_item_ids: list[int]
    semantic_eligible: bool = True
    semantic_reason: str = "ok"
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": int(self.chunk_id),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "text": self.text,
            "source_item_ids": [int(value) for value in self.source_item_ids],
            "semantic_eligible": bool(self.semantic_eligible),
            "semantic_reason": self.semantic_reason,
            "quality_flags": list(self.quality_flags),
        }

    def to_search_dict(self) -> dict[str, Any]:
        return {
            "source_chunk_ids": [int(value) for value in self.source_item_ids],
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "start_time": int(self.start_ms) / 1000.0,
            "end_time": int(self.end_ms) / 1000.0,
            "text": self.text,
            "semantic_eligible": bool(self.semantic_eligible),
            "semantic_reason": self.semantic_reason,
            "quality_flags": list(self.quality_flags),
        }
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_pipeline_types.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit task 1**

Run:

```powershell
git add backend/app/indexing/asr_pipeline_types.py backend/tests/test_asr_pipeline_types.py
git diff --cached --name-only
git commit -m "refactor(asr): add pipeline transcript types"
```

Expected staged files:

```text
backend/app/indexing/asr_pipeline_types.py
backend/tests/test_asr_pipeline_types.py
```

---

### Task 2: Split FunASR Parser From Retrieval Chunking

**Files:**
- Create: `backend/app/indexing/asr_transcript_parser.py`
- Modify: `backend/app/indexing/asr.py`
- Modify: `backend/tests/test_transcript.py`
- Create: `backend/tests/test_asr_transcript_parser.py`

- [ ] **Step 1: Write parser tests that reject duration-based word splitting**

Create `backend/tests/test_asr_transcript_parser.py`:

```python
from app.indexing.asr_transcript_parser import parse_funasr_raw_transcript, raw_items_from_chunks


def _timestamps_for_timed_chars(text: str, step_ms: int = 900) -> list[list[int]]:
    timed = [char for char in text if char.strip() and (char.isalnum() or "\u3400" <= char <= "\u9fff")]
    return [[index * step_ms, index * step_ms + 700] for index, _char in enumerate(timed)]


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
```

- [ ] **Step 2: Run parser tests and verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_transcript_parser.py
```

Expected: FAIL with missing `app.indexing.asr_transcript_parser`.

- [ ] **Step 3: Implement raw transcript parser**

Create `backend/app/indexing/asr_transcript_parser.py`:

```python
from __future__ import annotations

import re
from typing import Any, Iterable

from app.indexing.asr_pipeline_types import RawTranscriptItem


def _clean_funasr_text(text: str, is_sensevoice: bool) -> str:
    text = str(text or "").strip()
    if not is_sensevoice:
        return text
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return str(rich_transcription_postprocess(text)).strip()
    except Exception:
        return text


def _timed_char(value: str) -> bool:
    return bool(re.match(r"[\w\u3400-\u9fff]", value, flags=re.UNICODE))


def _valid_timestamp_pairs(timestamps: object) -> list[tuple[int, int]]:
    if not isinstance(timestamps, list):
        return []
    pairs: list[tuple[int, int]] = []
    for pair in timestamps:
        if isinstance(pair, list) and len(pair) >= 2 and pair[0] is not None and pair[1] is not None:
            pairs.append((int(pair[0]), int(pair[1])))
    return pairs


def _item_from_seconds(index: int, chunk: dict[str, Any], source: str) -> RawTranscriptItem | None:
    text = str(chunk.get("text") or "").strip()
    if not text:
        return None
    if "start_ms" in chunk:
        start_ms = int(chunk.get("start_ms") or 0)
        end_ms = int(chunk.get("end_ms", start_ms) or start_ms)
    else:
        start_ms = int(round(float(chunk.get("start_time", chunk.get("start", 0))) * 1000.0))
        end_ms = int(round(float(chunk.get("end_time", chunk.get("end", start_ms / 1000.0))) * 1000.0))
    return RawTranscriptItem(
        item_id=index,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        text=text,
        source=source,
    )


def raw_items_from_chunks(chunks: Iterable[dict[str, Any]], *, source: str) -> list[RawTranscriptItem]:
    items: list[RawTranscriptItem] = []
    for chunk in chunks:
        item = _item_from_seconds(len(items), chunk, source)
        if item is not None:
            items.append(item)
    return items


def parse_funasr_raw_transcript(result: object, *, is_sensevoice: bool) -> tuple[list[RawTranscriptItem], dict[str, int]]:
    items: list[RawTranscriptItem] = []
    timestamp_mismatch_items = 0
    timestamp_jump_warnings = 0

    def add_item(start_ms: int, end_ms: int, text: str, source: str, diagnostics: dict[str, Any] | None = None) -> None:
        cleaned = _clean_funasr_text(text, is_sensevoice)
        if cleaned and end_ms >= start_ms:
            items.append(
                RawTranscriptItem(
                    item_id=len(items),
                    start_ms=int(start_ms),
                    end_ms=int(end_ms),
                    text=cleaned,
                    source=source,
                    diagnostics=diagnostics or {},
                )
            )

    source_items = result if isinstance(result, list) else [result]
    for raw in source_items:
        if not isinstance(raw, dict):
            continue
        sentence_info = raw.get("sentence_info") or []
        if isinstance(sentence_info, list) and sentence_info:
            for sentence in sentence_info:
                if not isinstance(sentence, dict):
                    continue
                start_ms = int(sentence.get("start", sentence.get("start_ms", 0)) or 0)
                end_ms = int(sentence.get("end", sentence.get("end_ms", start_ms)) or start_ms)
                add_item(start_ms, end_ms, str(sentence.get("text", sentence.get("sentence", ""))), "funasr_sentence")
            continue

        text = str(raw.get("text") or "").strip()
        if not text:
            continue

        timestamps = _valid_timestamp_pairs(raw.get("timestamp"))
        words = raw.get("words")
        timestamp_text = "".join(str(word) for word in words) if isinstance(words, list) and len(words) == len(timestamps) else text
        timed_count = sum(1 for char in timestamp_text if _timed_char(char))
        diagnostics: dict[str, Any] = {}
        if timestamps and timed_count >= 8 and len(timestamps) < int(timed_count * 0.4):
            timestamp_mismatch_items += 1
            diagnostics["timestamp_mismatch"] = True
            timestamps = []
        if timestamps:
            starts = [pair[0] for pair in timestamps]
            jumps = sum(1 for left, right in zip(starts, starts[1:]) if right - left > 5000)
            timestamp_jump_warnings += jumps
            if jumps:
                diagnostics["timestamp_jumps"] = jumps
            add_item(timestamps[0][0], timestamps[-1][1], text, "funasr_timestamp", diagnostics)
            continue

        start_ms = int(raw.get("start", raw.get("start_ms", 0)) or 0)
        end_ms = int(raw.get("end", raw.get("end_ms", start_ms)) or start_ms)
        add_item(start_ms, end_ms, text, "funasr_text", diagnostics)

    return items, {
        "raw_items": len(items),
        "timestamp_mismatch_items": timestamp_mismatch_items,
        "timestamp_jump_warnings": timestamp_jump_warnings,
    }
```

- [ ] **Step 4: Replace `_parse_funasr_chunks` implementation in `asr.py` with wrapper**

In `backend/app/indexing/asr.py`, add import:

```python
from app.indexing.asr_transcript_parser import parse_funasr_raw_transcript, raw_items_from_chunks
```

Replace `_parse_funasr_chunks()` body with this compatibility wrapper:

```python
def _parse_funasr_chunks(result: object, *, is_sensevoice: bool) -> list[dict]:
    raw_items, _diagnostics = parse_funasr_raw_transcript(result, is_sensevoice=is_sensevoice)
    return [item.to_dict() for item in raw_items]
```

Leave `_clean_funasr_text()` in `asr.py` only if another function still imports it during this task; otherwise remove it after tests pass.

- [ ] **Step 5: Update old transcript parser test expectation**

In `backend/tests/test_transcript.py`, replace `test_funasr_timestamped_text_splits_long_items_by_punctuation` with:

```python
def test_funasr_timestamped_text_parser_returns_raw_item_without_duration_split():
    text = "hello world. next part."
    timed = [char for char in text if char.isalnum()]
    timestamps = [[index * 1000, index * 1000 + 800] for index in range(len(timed))]

    chunks = asr._parse_funasr_chunks(
        [{"text": text, "timestamp": timestamps}],
        is_sensevoice=True,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world. next part."
    assert chunks[0]["start_ms"] == 0
    assert chunks[0]["end_ms"] > 12000
```

- [ ] **Step 6: Run parser and transcript tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_transcript_parser.py tests/test_transcript.py
```

Expected: parser tests pass; transcript tests may still fail where `build_asr_index()` expects old postprocess stats. Those failures are resolved in Task 5. No parser duration-split test should fail.

- [ ] **Step 7: Commit task 2**

Run:

```powershell
git add backend/app/indexing/asr_transcript_parser.py backend/app/indexing/asr.py backend/tests/test_asr_transcript_parser.py backend/tests/test_transcript.py
git diff --cached --name-only
git commit -m "refactor(asr): split raw transcript parser"
```

Expected staged files:

```text
backend/app/indexing/asr.py
backend/app/indexing/asr_transcript_parser.py
backend/tests/test_asr_transcript_parser.py
backend/tests/test_transcript.py
```

---

### Task 3: Add Retrieval Chunk Builder With Boundary Repair

**Files:**
- Create: `backend/app/indexing/asr_retrieval_chunks.py`
- Create: `backend/tests/test_asr_retrieval_chunks.py`

- [ ] **Step 1: Write failing tests for CJK and Latin boundary repair**

Create `backend/tests/test_asr_retrieval_chunks.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_retrieval_chunks.py
```

Expected: FAIL with missing `app.indexing.asr_retrieval_chunks`.

- [ ] **Step 3: Implement retrieval chunk builder**

Create `backend/app/indexing/asr_retrieval_chunks.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk
from app.indexing.asr_text import normalize_asr_text, normalize_search_text, semantic_text_quality


@dataclass(frozen=True)
class RetrievalChunkConfig:
    normal_gap_ms: int = 700
    short_gap_ms: int = 1800
    same_bucket_gap_ms: int = 1800
    false_gap_repair_ms: int = 8000
    target_max_duration_ms: int = 18000
    soft_max_duration_ms: int = 25000
    hard_max_duration_ms: int = 35000
    short_text_chars: int = 8
    max_text_chars: int = 180
    bucket_ms: int = 5000


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？"))


def _ends_soft_punctuation(text: str) -> bool:
    return text.rstrip().endswith((",", ";", "，", "；", "、", ":"))


def _compact_length(text: str) -> int:
    return len(normalize_search_text(text))


def _last_run_after_punctuation(text: str) -> str:
    stripped = text.rstrip()
    boundary = max(stripped.rfind(mark) for mark in ".!?。！？,;，；、:：")
    return stripped[boundary + 1 :].strip()


def _first_run_before_punctuation(text: str) -> str:
    stripped = text.lstrip()
    positions = [index for index, char in enumerate(stripped) if char in ".!?。！？,;，；、:："]
    if not positions:
        return stripped
    return stripped[: positions[0]].strip()


def _is_short_text(text: str, config: RetrievalChunkConfig) -> bool:
    return _compact_length(text) <= config.short_text_chars


def _same_bucket(left: RawTranscriptItem, right: RawTranscriptItem, config: RetrievalChunkConfig) -> bool:
    return left.start_ms // config.bucket_ms == right.start_ms // config.bucket_ms


def _needs_cjk_boundary_repair(left_text: str, right_text: str) -> bool:
    if _ends_sentence(left_text):
        return False
    left_tail = _last_run_after_punctuation(left_text)
    right_head = _first_run_before_punctuation(right_text)
    if not left_tail or not right_head:
        return False
    if not _is_cjk(left_tail[-1]) or not _is_cjk(right_head[0]):
        return False
    if len(left_tail) <= 4:
        return True
    if _ends_soft_punctuation(left_text):
        return True
    return False


def _needs_latin_boundary_repair(left_text: str, right_text: str) -> bool:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left or not right:
        return False
    if _ends_sentence(left_text):
        return False
    return left[-1].isalpha() and right[0].isalpha()


def _join_text(left_text: str, right_text: str, *, boundary_repair: bool) -> str:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left:
        return right
    if not right:
        return left
    if boundary_repair and (_is_cjk(left[-1]) and _is_cjk(right[0])):
        return f"{left}{right}"
    if boundary_repair and left[-1].isalpha() and right[0].isalpha():
        return f"{left}{right}"
    if _ends_soft_punctuation(left):
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_decision(
    current: RetrievalChunk,
    item: RawTranscriptItem,
    *,
    config: RetrievalChunkConfig,
) -> tuple[bool, bool, list[str]]:
    gap_ms = max(0, item.start_ms - current.end_ms)
    candidate_duration_ms = max(current.end_ms, item.end_ms) - current.start_ms
    cjk_repair = _needs_cjk_boundary_repair(current.text, item.text)
    latin_repair = _needs_latin_boundary_repair(current.text, item.text)
    boundary_repair = cjk_repair or latin_repair
    flags: list[str] = []
    if cjk_repair:
        flags.append("cjk_boundary_repair")
    if latin_repair:
        flags.append("latin_boundary_repair")
    if gap_ms > config.short_gap_ms and boundary_repair:
        flags.append("fake_gap_repair")

    candidate_text = _join_text(current.text, item.text, boundary_repair=boundary_repair)
    if _compact_length(candidate_text) > config.max_text_chars:
        return False, boundary_repair, flags
    if candidate_duration_ms > config.hard_max_duration_ms:
        return False, boundary_repair, flags
    if boundary_repair and gap_ms <= config.false_gap_repair_ms:
        return True, boundary_repair, flags
    if _ends_sentence(current.text):
        return False, boundary_repair, flags
    if _is_short_text(current.text, config) or _is_short_text(item.text, config):
        return gap_ms <= config.short_gap_ms, boundary_repair, flags
    if gap_ms <= config.normal_gap_ms:
        return True, boundary_repair, flags
    if candidate_duration_ms <= config.target_max_duration_ms and gap_ms <= config.same_bucket_gap_ms:
        return True, boundary_repair, flags
    return False, boundary_repair, flags


def build_retrieval_chunks(
    raw_items: Iterable[RawTranscriptItem],
    *,
    config: RetrievalChunkConfig | None = None,
) -> tuple[list[RetrievalChunk], dict[str, int]]:
    config = config or RetrievalChunkConfig()
    normalized: list[RawTranscriptItem] = []
    dropped_empty = 0
    for item in raw_items:
        text = normalize_asr_text(item.text)
        if not text:
            dropped_empty += 1
            continue
        normalized.append(
            RawTranscriptItem(
                item_id=item.item_id,
                start_ms=item.start_ms,
                end_ms=max(item.start_ms, item.end_ms),
                text=text,
                source=item.source,
                unit_id=item.unit_id,
                diagnostics=item.diagnostics,
            )
        )

    chunks: list[RetrievalChunk] = []
    word_boundary_repairs = 0
    fake_gap_repairs = 0
    merged_items = 0

    for item in normalized:
        if not chunks:
            chunks.append(
                RetrievalChunk(
                    chunk_id=0,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                    source_item_ids=[item.item_id],
                )
            )
            continue
        current = chunks[-1]
        allowed, boundary_repair, flags = _merge_decision(current, item, config=config)
        if not allowed:
            chunks.append(
                RetrievalChunk(
                    chunk_id=len(chunks),
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    text=item.text,
                    source_item_ids=[item.item_id],
                )
            )
            continue
        merged_text = _join_text(current.text, item.text, boundary_repair=boundary_repair)
        merged_flags = list(dict.fromkeys([*current.quality_flags, *flags]))
        chunks[-1] = RetrievalChunk(
            chunk_id=current.chunk_id,
            start_ms=current.start_ms,
            end_ms=max(current.end_ms, item.end_ms),
            text=merged_text,
            source_item_ids=[*current.source_item_ids, item.item_id],
            quality_flags=merged_flags,
        )
        merged_items += 1
        if boundary_repair:
            word_boundary_repairs += 1
        if "fake_gap_repair" in flags:
            fake_gap_repairs += 1

    final_chunks: list[RetrievalChunk] = []
    semantic_ineligible = 0
    long_chunks = 0
    for chunk in chunks:
        quality = semantic_text_quality(chunk.text)
        duration_ms = chunk.end_ms - chunk.start_ms
        if duration_ms > config.soft_max_duration_ms:
            long_chunks += 1
        if not quality.eligible:
            semantic_ineligible += 1
        final_chunks.append(
            RetrievalChunk(
                chunk_id=len(final_chunks),
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                text=chunk.text,
                source_item_ids=chunk.source_item_ids,
                semantic_eligible=bool(quality.eligible),
                semantic_reason=quality.reason,
                quality_flags=chunk.quality_flags,
            )
        )

    return final_chunks, {
        "raw_items": len(list(raw_items)) if not isinstance(raw_items, list) else len(raw_items),
        "normalized_items": len(normalized),
        "retrieval_chunks": len(final_chunks),
        "dropped_empty_items": dropped_empty,
        "merged_items": merged_items,
        "word_boundary_repairs": word_boundary_repairs,
        "fake_gap_repairs": fake_gap_repairs,
        "long_chunks": long_chunks,
        "semantic_ineligible_chunks": semantic_ineligible,
    }
```

- [ ] **Step 4: Fix one-pass iterable stats if tests expose generator issue**

If `raw_items` can be a generator, adjust the first lines of `build_retrieval_chunks()` to materialize once:

```python
source_items = list(raw_items)
normalized: list[RawTranscriptItem] = []
```

Then use `source_items` in the loop and stats:

```python
for item in source_items:
    text = normalize_asr_text(item.text)
```

Stats line:

```python
"raw_items": len(source_items),
```

- [ ] **Step 5: Run retrieval chunk tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_retrieval_chunks.py
```

Expected: `4 passed`.

- [ ] **Step 6: Commit task 3**

Run:

```powershell
git add backend/app/indexing/asr_retrieval_chunks.py backend/tests/test_asr_retrieval_chunks.py
git diff --cached --name-only
git commit -m "feat(asr): build retrieval chunks from raw transcript"
```

Expected staged files:

```text
backend/app/indexing/asr_retrieval_chunks.py
backend/tests/test_asr_retrieval_chunks.py
```

---

### Task 4: Add Debug Artifact Writer

**Files:**
- Create: `backend/app/indexing/asr_debug.py`
- Create: `backend/tests/test_asr_debug.py`

- [ ] **Step 1: Write debug artifact tests**

Create `backend/tests/test_asr_debug.py`:

```python
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
```

- [ ] **Step 2: Run debug tests and verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_debug.py
```

Expected: FAIL with missing `app.indexing.asr_debug`.

- [ ] **Step 3: Implement debug writer**

Create `backend/app/indexing/asr_debug.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from app.indexing.asr_pipeline_types import RawTranscriptItem, RetrievalChunk, SpeechUnit


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_asr_debug_artifacts(
    *,
    debug_dir: str | Path,
    enabled: bool,
    save_raw_transcript: bool,
    raw_items: Sequence[RawTranscriptItem],
    retrieval_chunks: Sequence[RetrievalChunk],
    repair_stats: Mapping[str, object],
    speech_units: Sequence[SpeechUnit] | None = None,
) -> None:
    if not enabled:
        return
    target = Path(debug_dir)
    target.mkdir(parents=True, exist_ok=True)
    if speech_units is not None:
        _write_json(target / "asr_speech_units.json", [unit.to_dict() for unit in speech_units])
    if save_raw_transcript:
        _write_json(target / "asr_raw_transcript.json", [item.to_dict() for item in raw_items])
    _write_json(target / "asr_retrieval_chunks.json", [chunk.to_dict() for chunk in retrieval_chunks])
    _write_json(target / "asr_repair_report.json", dict(repair_stats))
```

- [ ] **Step 4: Run debug tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_debug.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit task 4**

Run:

```powershell
git add backend/app/indexing/asr_debug.py backend/tests/test_asr_debug.py
git diff --cached --name-only
git commit -m "feat(asr): add optional debug artifacts"
```

Expected staged files:

```text
backend/app/indexing/asr_debug.py
backend/tests/test_asr_debug.py
```

---

### Task 5: Integrate Pipeline Into `build_asr_index`

**Files:**
- Modify: `backend/app/indexing/asr.py`
- Modify: `backend/tests/test_transcript.py`
- Modify: `backend/tests/test_index_schema_v3.py`

- [ ] **Step 1: Add integration tests for sidecar-to-retrieval path**

Append to `backend/tests/test_transcript.py`:

```python
def test_sidecar_asr_pipeline_repairs_cjk_boundary_and_keeps_npz_schema(tmp_path, monkeypatch):
    sidecar = tmp_path / "broken.srt"
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n孤\n\n"
        "2\n00:00:04,940 --> 00:00:06,000\n独敏感又倔强。\n",
        encoding="utf-8",
    )

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        assert [chunk["text"] for chunk in chunks] == ["孤独敏感又倔强。"]
        return {
            "embeddings": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="sidecar",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        sidecar_path=str(sidecar),
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "asr.npz", allow_pickle=False) as data:
        assert set(data.files) == {"chunk_times_ms", "texts", "embeddings", "embedding_chunk_indices"}
        assert data["texts"].tolist() == ["孤独敏感又倔强。"]
    assert result["raw_items"] == 2
    assert result["retrieval_chunks"] == 1
    assert result["chunk_builder_stats"]["word_boundary_repairs"] == 1
```

- [ ] **Step 2: Run targeted integration test and verify failure against old pipeline**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_transcript.py::test_sidecar_asr_pipeline_repairs_cjk_boundary_and_keeps_npz_schema
```

Expected: FAIL because `build_asr_index()` still uses `postprocess_asr_chunks()` and does not expose `raw_items/retrieval_chunks/chunk_builder_stats`.

- [ ] **Step 3: Modify imports and `build_asr_index()` signature**

In `backend/app/indexing/asr.py`, replace the old postprocess import:

```python
from app.indexing.asr_debug import write_asr_debug_artifacts
from app.indexing.asr_retrieval_chunks import RetrievalChunkConfig, build_retrieval_chunks
from app.indexing.asr_transcript_parser import parse_funasr_raw_transcript, raw_items_from_chunks
```

Update `build_asr_index()` signature:

```python
def build_asr_index(
    video_path: str,
    output_path: str,
    working_dir: str,
    engine: str,
    model_name: str,
    device: str,
    model_dir: str,
    language: str = "auto",
    sidecar_path: str | None = None,
    funasr_model: str = "paraformer-zh",
    funasr_model_dir: str | None = None,
    faster_whisper_model_dir: str | None = None,
    model_local_files_only: bool = True,
    semantic_enabled: bool = True,
    semantic_output_path: str | None = None,
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    semantic_device: str = "cpu",
    semantic_model_dir: str | None = None,
    semantic_batch_size: int = 32,
    semantic_local_files_only: bool = True,
    debug_artifacts_enabled: bool = False,
    save_raw_transcript: bool = False,
    debug_output_dir: str | None = None,
    vad_strategy: str = "funasr_fsmn",
) -> dict:
```

- [ ] **Step 4: Convert every model path to `RawTranscriptItem[]`**

Inside `build_asr_index()`, change raw chunk handling:

```python
    raw_parser_stats: dict[str, int] = {}

    if sidecar_path:
        raw_items = raw_items_from_chunks(load_sidecar(sidecar_path), source="sidecar")
        used_engine = "sidecar"
    else:
        try:
            audio_path = extract_audio(video_path, Path(working_dir) / "audio.wav")
        except subprocess.CalledProcessError:
            chunks: list[dict] = []
            used_engine = "no_audio"
            _save_asr_npz(output_path, chunks, np.empty((0, 0), dtype=np.float16), np.empty((0,), dtype=np.int32))
            if semantic_target is not None:
                semantic_target.unlink(missing_ok=True)
            return {
                "chunks": 0,
                "raw_chunks": 0,
                "raw_items": 0,
                "retrieval_chunks": 0,
                "engine": used_engine,
                "model": effective_model,
                "language": requested_language,
                "task": task,
                "requested_language": requested_language,
                "detected_language": detected_language,
                "language_route": "no_audio",
                "route_reason": "no audio stream found",
                "vad_strategy": vad_strategy,
                "chunk_builder_stats": {
                    "raw_items": 0,
                    "normalized_items": 0,
                    "retrieval_chunks": 0,
                    "dropped_empty_items": 0,
                    "merged_items": 0,
                    "word_boundary_repairs": 0,
                    "fake_gap_repairs": 0,
                    "long_chunks": 0,
                    "semantic_ineligible_chunks": 0,
                },
                "postprocess_stats": _empty_postprocess_stats(),
                "text_profile": asr_text_profile([]),
                "schema_version": 3,
                "decode_status": "empty",
                "semantic_status": "empty",
                "semantic_chunks": 0,
                "warning": "no audio stream found",
            }
```

For FunASR path:

```python
                funasr_result = _funasr_raw(
                    str(audio_path),
                    funasr_model,
                    device,
                    model_root=funasr_model_dir or str(Path(model_dir).parent / "funasr"),
                    local_files_only=model_local_files_only,
                    language=requested_language,
                )
                raw_items, raw_parser_stats = parse_funasr_raw_transcript(
                    funasr_result,
                    is_sensevoice=_is_sensevoice_model(funasr_model),
                )
                used_engine = "funasr"
```

If keeping `_funasr()` as a compatibility wrapper, add a new internal `_funasr_raw()` that returns model.generate output:

```python
def _funasr_raw(
    audio_path: str,
    model_name: str,
    device: str,
    model_root: str | Path | None = None,
    local_files_only: bool = True,
    language: str = "auto",
) -> object:
    model_source = resolve_modelscope_model_source(model_root, model_name, local_files_only=local_files_only)
    vad_source = resolve_modelscope_model_source(model_root, "fsmn-vad", local_files_only=local_files_only)
    is_sensevoice = _is_sensevoice_model(model_name, model_source)
    punc_source = None if is_sensevoice else resolve_modelscope_model_source(
        model_root, "ct-punc", local_files_only=local_files_only
    )
    if str(device).startswith("npu"):
        import torch_npu

    from funasr import AutoModel

    model_kwargs = {
        "model": model_source,
        "vad_model": vad_source,
        "device": device,
        "disable_update": True,
    }
    if is_sensevoice:
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    else:
        model_kwargs["punc_model"] = punc_source
    with offline_env(local_files_only):
        model = AutoModel(**model_kwargs)
    generate_kwargs = {"input": audio_path, "cache": {}, "batch_size_s": 60 if is_sensevoice else 300}
    if language and language != "auto":
        generate_kwargs["language"] = language
    if is_sensevoice:
        generate_kwargs.update(
            {
                "use_itn": True,
                "merge_vad": True,
                "merge_length_s": 15,
                "output_timestamp": True,
                "return_time_stamps": True,
            }
        )
    else:
        generate_kwargs["sentence_timestamp"] = True
    return model.generate(**generate_kwargs)
```

Then make `_funasr()` call `_funasr_raw()` plus parser:

```python
    result = _funasr_raw(audio_path, model_name, device, model_root, local_files_only, language)
    raw_items, _diagnostics = parse_funasr_raw_transcript(result, is_sensevoice=_is_sensevoice_model(model_name))
    return [item.to_dict() for item in raw_items]
```

For Whisper and faster-whisper paths:

```python
                raw_chunks, whisper_metadata = _whisper(...)
                raw_items = raw_items_from_chunks(raw_chunks, source="whisper")
```

```python
            raw_chunks, whisper_metadata = _faster_whisper(...)
            raw_items = raw_items_from_chunks(raw_chunks, source="faster_whisper")
```

- [ ] **Step 5: Build retrieval chunks before semantic embedding**

Replace `segment_ids = ...` and `postprocess_asr_chunks(...)` with:

```python
    retrieval_chunks, chunk_builder_stats = build_retrieval_chunks(
        raw_items,
        config=RetrievalChunkConfig(),
    )
    chunks = [chunk.to_search_dict() for chunk in retrieval_chunks]
    postprocess_stats = {
        "raw_chunks": len(raw_items),
        "normalized_chunks": chunk_builder_stats["normalized_items"],
        "processed_chunks": len(chunks),
        "dropped_empty_chunks": chunk_builder_stats["dropped_empty_items"],
        "merged_chunks": chunk_builder_stats["merged_items"],
        "cross_segment_merges": 0,
        "semantic_ineligible_chunks": chunk_builder_stats["semantic_ineligible_chunks"],
        "long_low_info_chunks": chunk_builder_stats["long_chunks"],
    }
```

Keep `postprocess_stats` during this transition because existing manifest/tests still read it.

- [ ] **Step 6: Write optional debug artifacts after chunk builder**

Before `_save_asr_npz(...)`, add:

```python
    debug_dir = Path(debug_output_dir) if debug_output_dir else Path(output_path).parent / "debug"
    write_asr_debug_artifacts(
        debug_dir=debug_dir,
        enabled=debug_artifacts_enabled,
        save_raw_transcript=save_raw_transcript,
        raw_items=raw_items,
        retrieval_chunks=retrieval_chunks,
        repair_stats={**chunk_builder_stats, **raw_parser_stats},
    )
```

- [ ] **Step 7: Update result dict**

In the returned `result`, include:

```python
        "chunks": len(chunks),
        "raw_chunks": len(raw_items),
        "raw_items": len(raw_items),
        "retrieval_chunks": len(chunks),
        "language_route": f"{used_engine}:{detected_language or requested_language}",
        "route_reason": "explicit language" if requested_language != "auto" else "default auto language",
        "vad_strategy": vad_strategy,
        "raw_parser_stats": raw_parser_stats,
        "chunk_builder_stats": chunk_builder_stats,
```

- [ ] **Step 8: Run targeted integration tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_transcript.py::test_sidecar_asr_pipeline_repairs_cjk_boundary_and_keeps_npz_schema tests/test_transcript.py::test_sidecar_asr_index_postprocesses_short_fragments_and_preserves_schema
```

Expected: both pass. The old sidecar test should still see `chunk_times_ms/texts/embeddings/embedding_chunk_indices`.

- [ ] **Step 9: Run ASR unit set**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_pipeline_types.py tests/test_asr_transcript_parser.py tests/test_asr_retrieval_chunks.py tests/test_asr_debug.py tests/test_transcript.py
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit task 5**

Run:

```powershell
git add backend/app/indexing/asr.py backend/tests/test_transcript.py backend/tests/test_index_schema_v3.py
git diff --cached --name-only
git commit -m "refactor(asr): integrate layered indexing pipeline"
```

Expected staged files:

```text
backend/app/indexing/asr.py
backend/tests/test_transcript.py
backend/tests/test_index_schema_v3.py
```

---

### Task 6: Add Settings, Runner Wiring, and Language Defaults

**Files:**
- Modify: `backend/app/settings.py`
- Modify: `backend/app/stage_runner.py`
- Modify: `backend/app/indexer_daemon.py`
- Modify: `backend/app/indexing/pipeline_manifest.py`
- Modify: `backend/tests/test_transcript.py`

- [ ] **Step 1: Add tests for default language and debug option propagation**

Append to `backend/tests/test_transcript.py`:

```python
def test_settings_default_asr_language_is_auto():
    from app.settings import Settings

    settings = Settings(app_data_dir="runtime-test")

    assert settings.asr_language == "auto"
    assert settings.asr_debug_artifacts is False
    assert settings.asr_save_raw_transcript is False
    assert settings.asr_vad_strategy == "funasr_fsmn"
```

- [ ] **Step 2: Run settings test and verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_transcript.py::test_settings_default_asr_language_is_auto
```

Expected: FAIL because current default is `zh` and new settings do not exist.

- [ ] **Step 3: Update ASR settings defaults**

In `backend/app/settings.py`, replace:

```python
    asr_language: str = "zh"
```

with:

```python
    asr_language: str = "auto"
    asr_vad_strategy: str = "funasr_fsmn"
    asr_debug_artifacts: bool = False
    asr_save_raw_transcript: bool = False
```

- [ ] **Step 4: Wire settings into stage runner**

In `backend/app/stage_runner.py`, in the `build_asr_index(...)` call, add:

```python
            debug_artifacts_enabled=bool(options.get("asr_debug_artifacts", settings.asr_debug_artifacts)),
            save_raw_transcript=bool(options.get("asr_save_raw_transcript", settings.asr_save_raw_transcript)),
            vad_strategy=str(options.get("asr_vad_strategy", settings.asr_vad_strategy)),
```

- [ ] **Step 5: Wire settings into daemon**

In `backend/app/indexer_daemon.py`, in the `build_asr_index(...)` call, add:

```python
            debug_artifacts_enabled=bool(options.get("asr_debug_artifacts", settings.asr_debug_artifacts)),
            save_raw_transcript=bool(options.get("asr_save_raw_transcript", settings.asr_save_raw_transcript)),
            vad_strategy=str(options.get("asr_vad_strategy", settings.asr_vad_strategy)),
```

- [ ] **Step 6: Add manifest passthrough fields**

In `backend/app/indexing/pipeline_manifest.py`, ensure the ASR manifest details include these keys when present in the ASR result:

```python
        "language_route": result.get("language_route"),
        "route_reason": result.get("route_reason"),
        "vad_strategy": result.get("vad_strategy"),
        "raw_items": result.get("raw_items"),
        "retrieval_chunks": result.get("retrieval_chunks"),
        "chunk_builder_stats": result.get("chunk_builder_stats"),
```

Use the existing manifest-building style in that file and do not change visual, face, or OCR fields.

- [ ] **Step 7: Run settings and ASR tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_transcript.py tests/test_index_schema_v3.py
```

Expected: pass.

- [ ] **Step 8: Commit task 6**

Run:

```powershell
git add backend/app/settings.py backend/app/stage_runner.py backend/app/indexer_daemon.py backend/app/indexing/pipeline_manifest.py backend/tests/test_transcript.py
git diff --cached --name-only
git commit -m "feat(asr): wire pipeline settings and route metadata"
```

Expected staged files:

```text
backend/app/indexer_daemon.py
backend/app/indexing/pipeline_manifest.py
backend/app/settings.py
backend/app/stage_runner.py
backend/tests/test_transcript.py
```

---

### Task 7: Update Documentation and Export Guidance

**Files:**
- Modify: `docs/current_retrieval_channels.md`
- Modify: `docs/ISSUES_AND_ROADMAP.md`
- Modify: `docs/handoff/CURRENT_STATUS.md`
- Modify: `docs/handoff/PROGRESS_LOG.md`

- [ ] **Step 1: Update ASR channel description**

In `docs/current_retrieval_channels.md`, update the ASR section with this content:

```markdown
### ASR

- 默认生产路径：`audio_extract -> model_transcribe -> raw_transcript -> retrieval_chunk_builder -> MiniLM semantic embedding -> asr.npz`。
- 默认引擎：`funasr` + `iic/SenseVoiceSmall`，语言默认 `auto`；涉及明确非中文或更强多语言效果时可手动选择 `faster-whisper` + `turbo`。
- `asr.npz` 仅保存检索需要字段：`chunk_times_ms`、`texts`、`embeddings`、`embedding_chunk_indices`。
- raw transcript、retrieval chunks、repair report 只在 `ASR_DEBUG_ARTIFACTS=true` 时保存到 `runtime/indexes/{video_id}/debug/`。
- chunk builder 会记录 `word_boundary_repairs`、`fake_gap_repairs`、`semantic_ineligible_chunks`，用于检查断词修复和低信息文本过滤。
```

- [ ] **Step 2: Record issue resolution and remaining work**

In `docs/ISSUES_AND_ROADMAP.md`, under ASR/retrieval quality optimization, add:

```markdown
### ASR 文本断词与 false timestamp gap

状态：已规划并实施 pipeline 分层重构。parser 不再按固定时长生成最终检索 chunk，retrieval chunk builder 负责 CJK/Latin 边界保护、短碎片合并和 false gap 修复。

仍需观察：
- 方言、多语言素材的自动路由是否应从诊断升级为自动切换。
- SenseVoiceSmall 与 faster-whisper turbo 在同一套 chunk builder 下的真实检索召回差异。
- 是否需要引入轻量分词辅助来减少 CJK 边界修复的误合并。
```

- [ ] **Step 3: Update handoff status**

In `docs/handoff/CURRENT_STATUS.md`, add:

```markdown
## ASR Pipeline

- ASR 索引已拆为 raw transcript parser 和 retrieval chunk builder。
- 默认索引不保存 raw transcript，debug 开关开启后才写 `debug/asr_raw_transcript.json`、`debug/asr_retrieval_chunks.json`、`debug/asr_repair_report.json`。
- 新索引需要重跑 ASR 才能获得修复后的检索文本。
```

- [ ] **Step 4: Add progress log entry**

In `docs/handoff/PROGRESS_LOG.md`, append:

```markdown
## 2026-07-09

- 重构 ASR pipeline：新增 raw transcript parser、retrieval chunk builder 和可选 debug artifact。
- 修复生产 ASR chunking 中由 timestamp false gap 和词内断裂导致的 `孤/独`、`永/远`、`很/难受` 类问题。
- 默认 `asr_language` 调整为 `auto`，避免把非中文素材强制走中文参数。
```

- [ ] **Step 5: Run docs placeholder scan**

Run:

```powershell
rg -n "占位|稍后补|还没写|先空着" docs/current_retrieval_channels.md docs/ISSUES_AND_ROADMAP.md docs/handoff/CURRENT_STATUS.md docs/handoff/PROGRESS_LOG.md
```

Expected: no output.

- [ ] **Step 6: Commit task 7**

Run:

```powershell
git add docs/current_retrieval_channels.md docs/ISSUES_AND_ROADMAP.md docs/handoff/CURRENT_STATUS.md docs/handoff/PROGRESS_LOG.md
git diff --cached --name-only
git commit -m "docs(asr): document layered indexing pipeline"
```

Expected staged files:

```text
docs/ISSUES_AND_ROADMAP.md
docs/current_retrieval_channels.md
docs/handoff/CURRENT_STATUS.md
docs/handoff/PROGRESS_LOG.md
```

---

### Task 8: Final Verification and Local Container Sync

**Files:**
- No code files unless a verification failure exposes a defect.

- [ ] **Step 1: Run focused ASR tests**

Run from `video_retrieval_mvp/backend`:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asr_pipeline_types.py tests/test_asr_transcript_parser.py tests/test_asr_retrieval_chunks.py tests/test_asr_debug.py tests/test_transcript.py tests/test_index_schema_v3.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run backend test suite if time is acceptable**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q
```

Expected: pass. If a non-ASR test fails, inspect whether it is caused by this refactor before changing files.

- [ ] **Step 3: Sync code into local Docker container without reinstalling dependencies**

From repo root:

```powershell
docker cp .\backend\app momentseek-mvp-app:/app/
docker cp .\backend\tests momentseek-mvp-app:/app/
```

Expected: both commands exit with code 0. This copies code only and does not reinstall packages.

- [ ] **Step 4: Run container-side import smoke test**

Run:

```powershell
docker exec momentseek-mvp-app python -c "from app.indexing.asr import build_asr_index; from app.indexing.asr_retrieval_chunks import build_retrieval_chunks; print('asr pipeline import ok')"
```

Expected:

```text
asr pipeline import ok
```

- [ ] **Step 5: Run one no-model sidecar smoke test inside container**

Run:

```powershell
docker exec momentseek-mvp-app python - <<'PY'
from pathlib import Path
import numpy as np
from app.indexing import asr

root = Path('/tmp/asr_pipeline_smoke')
root.mkdir(parents=True, exist_ok=True)
sidecar = root / 'demo.srt'
sidecar.write_text('1\n00:00:00,000 --> 00:00:01,000\n孤\n\n2\n00:00:04,940 --> 00:00:06,000\n独敏感又倔强。\n', encoding='utf-8')

def fake_semantic_arrays(**kwargs):
    chunks = kwargs['chunks']
    return {
        'embeddings': np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float16),
        'embedding_chunk_indices': np.asarray(list(range(len(chunks))), dtype=np.int32),
        'semantic_chunks': len(chunks),
        'semantic_model': 'fake',
        'semantic_status': 'complete',
    }

asr.build_text_semantic_arrays = fake_semantic_arrays
result = asr.build_asr_index(
    video_path=str(root / 'demo.mp4'),
    output_path=str(root / 'asr.npz'),
    working_dir=str(root / 'work'),
    engine='sidecar',
    model_name='small',
    device='cpu',
    model_dir=str(root / 'models'),
    sidecar_path=str(sidecar),
    semantic_model='fake',
)
with np.load(root / 'asr.npz', allow_pickle=False) as data:
    print(data['texts'].tolist())
print(result['chunk_builder_stats'])
PY
```

Expected output contains:

```text
['孤独敏感又倔强。']
```

and:

```text
'word_boundary_repairs': 1
```

- [ ] **Step 6: Check final git status**

Run:

```powershell
git status --short
```

Expected: remaining dirty files are only pre-existing unrelated experiment artifacts or files intentionally left uncommitted. If ASR implementation files remain modified, either commit the task-specific changes or explain the blocker.

---

## Self-Review

- Spec coverage:
  - Pipeline layers are covered by Tasks 1, 2, 3, and 5.
  - Production storage remains minimal through Task 5 `_save_asr_npz()` schema preservation.
  - Debug-only raw artifacts are covered by Task 4 and Task 5.
  - Default language `auto`, route metadata, and VAD strategy are covered by Task 6.
  - Documentation and handoff continuity are covered by Task 7.
  - Verification-before-completion is covered by Task 8.
- Type consistency:
  - `RawTranscriptItem`, `RetrievalChunk`, `build_retrieval_chunks()`, `parse_funasr_raw_transcript()`, and `write_asr_debug_artifacts()` are introduced before any integration task references them.
  - `RetrievalChunk.to_search_dict()` emits the legacy dict keys expected by `build_text_semantic_arrays()` and `_save_asr_npz()`.
- Storage constraint:
  - `asr.npz` keeps only `chunk_times_ms`, `texts`, `embeddings`, `embedding_chunk_indices`.
  - raw transcript is only saved when both debug artifact writing is enabled and `save_raw_transcript=True`.
