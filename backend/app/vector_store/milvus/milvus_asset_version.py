"""Asset-version management for Milvus versioned publication.

Every video in Milvus carries an ``asset_version`` string that is
incremented for every indexing attempt.  This provides two
safety guarantees:

1. **Stale-write isolation** — a previous or interrupted index run writes
   to a different version and cannot silently overwrite a newly published
   generation.

2. **Version-scoped deletion** — ``delete_video_version()`` on the
   client can remove exactly one generation of data without touching
   a newer write that may have already landed.

The counter is persisted in ``<video_index_dir>/milvus_meta.json``.  The
metadata keeps the reader-visible published version separate from the last
reserved attempt version.  A failed attempt must still consume its version:
otherwise a retry could mix fresh rows with leftovers from the failed write.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)

_META_FILE = "milvus_meta.json"
_DEFAULT_VERSION = "1"


def _read_meta(index_dir: Path) -> dict[str, str]:
    meta_path = index_dir / _META_FILE
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("metadata is not an object")
        return {str(key): str(value) for key, value in data.items()}
    except (ValueError, OSError) as exc:
        logger.warning(
            "Could not read Milvus version metadata from %s: %s — defaulting to '1'",
            meta_path,
            exc,
        )
        return {}


def _write_meta(index_dir: Path, data: dict[str, str]) -> None:
    """Atomically replace the version metadata after the caller holds its video lock."""
    meta_path = index_dir / _META_FILE
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=meta_path.parent,
            prefix=f".{meta_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, meta_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _version_number(value: str | None) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def current_asset_version(index_dir: Path) -> str:
    """Return the last published asset version stored in *index_dir*.

    Returns ``"1"`` when no version file exists yet (first-ever index run).
    """
    return _read_meta(index_dir).get("asset_version", _DEFAULT_VERSION)


def current_attempt_version(index_dir: Path) -> str:
    """Return the last reserved attempt version, falling back to published data.

    Old metadata contains only ``asset_version``.  Treating that value as the
    last attempt preserves a monotonic migration path without rewriting every
    existing video directory.
    """
    data = _read_meta(index_dir)
    return data.get("last_attempt_version", data.get("asset_version", _DEFAULT_VERSION))


def next_asset_version(index_dir: Path) -> str:
    """Return the next attempt version without reserving it.

    New write paths must use :func:`reserve_next_attempt_version` instead.
    This compatibility helper deliberately remains side-effect free.
    """
    numbers = [
        number
        for number in (
            _version_number(current_asset_version(index_dir)),
            _version_number(current_attempt_version(index_dir)),
        )
        if number is not None
    ]
    return str(max(numbers, default=1) + 1)


def reserve_next_attempt_version(index_dir: Path) -> str:
    """Persist and return a fresh version before any Milvus row is written.

    The caller must hold the shared per-video publication lock.  The metadata
    replacement is atomic, and failed attempts intentionally leave their
    reservation behind so a retry cannot reuse rows from an interrupted write.
    """
    data = _read_meta(index_dir)
    published = data.get("asset_version", _DEFAULT_VERSION)
    previous_attempt = data.get("last_attempt_version", published)
    numbers = [
        number
        for number in (_version_number(published), _version_number(previous_attempt))
        if number is not None
    ]
    version = str(max(numbers, default=1) + 1)
    data["asset_version"] = published
    data["last_attempt_version"] = version
    _write_meta(index_dir, data)
    return version


def publish_asset_version(index_dir: Path, version: str) -> None:
    """Advance the published version without forgetting newer reservations."""
    data = _read_meta(index_dir)
    version = str(version)
    previous_attempt = data.get(
        "last_attempt_version", data.get("asset_version", _DEFAULT_VERSION)
    )
    if (_version_number(previous_attempt) or 0) < (_version_number(version) or 0):
        previous_attempt = version
    data["asset_version"] = version
    data["last_attempt_version"] = previous_attempt
    _write_meta(index_dir, data)


def bump_asset_version(index_dir: Path) -> str:
    """Compatibility helper for offline recovery tools.

    The caller is responsible for holding the per-video stage lock before
    calling this function to avoid races in multi-process environments.

    Version numbers are decimal integers (``"1"``, ``"2"``, …).  If the
    stored value is not a parseable integer (legacy migration edge-case)
    the counter restarts at ``"2"``.
    """
    old = current_asset_version(index_dir)
    new = reserve_next_attempt_version(index_dir)
    publish_asset_version(index_dir, new)
    logger.debug("asset_version bumped %s → %s in %s", old, new, index_dir)
    return new
