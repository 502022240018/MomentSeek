"""Strict scalar contracts for rows read from Milvus.

Online readers must not manufacture temporal metadata for malformed or legacy
rows.  Keeping these validators in one small module makes the retrieval and
speaker-management paths enforce the same fail-closed contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def required_int_field(entity: Any, field: str) -> int:
    """Read one required integral scalar without compatibility defaults."""
    value = entity.get(field)
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"missing or invalid {field}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise ValueError(f"{field} must be an integer")


def required_nonnegative_int_field(entity: Any, field: str) -> int:
    """Read a required integral scalar whose domain is non-negative."""
    value = required_int_field(entity, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def required_time_window(entity: Any) -> tuple[int, int]:
    """Return a required non-empty window, or reject the Milvus row."""
    start_ms = required_int_field(entity, "start_ms")
    end_ms = required_int_field(entity, "end_ms")
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("time window must satisfy 0 <= start_ms < end_ms")
    return start_ms, end_ms
