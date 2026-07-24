"""Milvus routing helpers.

Milvus is the primary vector store. NPZ remains a local fallback and recovery
artifact, while SQLite continues to own relational metadata.
"""
from __future__ import annotations

import hashlib


def _settings():
    from app.settings import get_settings
    return get_settings()


def milvus_write_enabled() -> bool:
    settings = _settings()
    return settings.milvus_enabled and settings.milvus_write_enabled


def milvus_read_enabled() -> bool:
    settings = _settings()
    return settings.milvus_enabled and settings.milvus_read_enabled


def milvus_fallback_enabled() -> bool:
    return _settings().milvus_fallback_enabled


def milvus_shadow_compare_enabled() -> bool:
    return _settings().milvus_shadow_compare_enabled


def should_use_milvus_for_video(video_id: str) -> bool:
    """Return a stable per-video rollout decision."""
    if not milvus_read_enabled():
        return False
    percent = _settings().milvus_rollout_percent
    if percent >= 100:
        return True
    if percent <= 0:
        return False
    bucket = int.from_bytes(
        hashlib.sha256(video_id.encode("utf-8")).digest()[:4], "big"
    ) % 100
    return bucket < percent


def milvus_write_fail_policy() -> str:
    """Returns 'raise' | 'warn'."""
    return _settings().milvus_write_fail_policy
