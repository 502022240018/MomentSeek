"""Milvus feature-flag helpers.

Milvus is the sole storage backend.  Only two gates remain:
  MILVUS_WRITE_ENABLED  — must be True in production (default True).
  MILVUS_WRITE_FAIL_POLICY — "raise" (default) or "warn".
"""
from __future__ import annotations


def _settings():
    from app.settings import get_settings
    return get_settings()


def milvus_write_enabled() -> bool:
    return _settings().milvus_write_enabled


def milvus_write_fail_policy() -> str:
    """Returns 'raise' | 'warn'."""
    return _settings().milvus_write_fail_policy
