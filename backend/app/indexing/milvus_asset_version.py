"""Asset-version management for Milvus dual-write.

Every video in Milvus carries an ``asset_version`` string that is
incremented each time the video is re-indexed.  This provides two
safety guarantees:

1. **Stale retry isolation** — write-queue jobs enqueued during a
   previous index run carry the version that was current *then*.
   After a re-index the version is bumped, so a stale retry writes
   to different PKs and cannot silently overwrite fresh data.
   ``cancel_pending_for_video()`` on the write-queue should be called
   *before* bumping so that stale retries are discarded entirely.

2. **Version-scoped deletion** — ``delete_video_version()`` on the
   client can remove exactly one generation of data without touching
   a newer write that may have already landed.

The counter is persisted in ``<video_index_dir>/milvus_meta.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_META_FILE = "milvus_meta.json"


def current_asset_version(index_dir: Path) -> str:
    """Return the current asset_version stored in *index_dir*.

    Returns ``"1"`` when no version file exists yet (first-ever index run).
    """
    meta_path = index_dir / _META_FILE
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text("utf-8"))
            return str(data["asset_version"])
        except (KeyError, ValueError, OSError) as exc:
            logger.warning(
                "Could not read asset_version from %s: %s — defaulting to '1'",
                meta_path, exc,
            )
    return "1"


def bump_asset_version(index_dir: Path) -> str:
    """Increment the stored asset_version and return the **new** value.

    The caller is responsible for holding the per-video stage lock before
    calling this function to avoid races in multi-process environments.

    Version numbers are decimal integers (``"1"``, ``"2"``, …).  If the
    stored value is not a parseable integer (legacy migration edge-case)
    the counter restarts at ``"2"``.
    """
    meta_path = index_dir / _META_FILE
    old = current_asset_version(index_dir)
    try:
        new = str(int(old) + 1)
    except ValueError:
        new = "2"  # non-integer legacy value — restart from 2
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"asset_version": new}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.debug("asset_version bumped %s → %s in %s", old, new, index_dir)
    return new
