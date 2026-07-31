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
            "unit_id": int(self.unit_id),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "core_start_ms": int(self.core_start_ms),
            "core_end_ms": int(self.core_end_ms),
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
    emotion: str = ""
    audio_event: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "item_id": int(self.item_id),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "text": self.text,
            "source": self.source,
        }
        if self.unit_id is not None:
            payload["unit_id"] = int(self.unit_id)
        if self.emotion:
            payload["emotion"] = self.emotion
        if self.audio_event:
            payload["audio_event"] = self.audio_event
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RawTranscriptItem:
        return cls(
            item_id=int(payload["item_id"]),
            start_ms=int(payload["start_ms"]),
            end_ms=int(payload["end_ms"]),
            text=str(payload["text"]),
            source=str(payload.get("source") or "unknown"),
            unit_id=None if payload.get("unit_id") is None else int(payload["unit_id"]),
            emotion=str(payload.get("emotion") or ""),
            audio_event=str(payload.get("audio_event") or ""),
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
    emotion: str = ""
    audio_event: str = ""

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
            "emotion": self.emotion,
            "audio_event": self.audio_event,
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
            "emotion": self.emotion,
            "audio_event": self.audio_event,
        }
