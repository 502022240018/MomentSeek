"""Per-video-stage advisory lock for Milvus re-index operations.

Concurrent re-index of the same (video_id, stage) pair would interleave
delete and write operations, potentially leaving Milvus in an inconsistent
state.  This module provides a lightweight cross-platform file lock that
serialises access at the granularity of one stage per video.

Usage::

    from app.indexing.milvus_stage_lock import video_stage_lock

    with video_stage_lock(video_index_dir, video_id="vid123", stage="visual"):
        # safe to delete + re-index here
        ...

The lock is advisory (non-mandatory) and is held for the duration of the
``with`` block.  It is automatically released — even on exception — so a
crashed indexer process releases the lock when its file descriptor is closed
by the OS.

Platform notes
--------------
* **Linux / macOS**: uses ``fcntl.flock()`` (POSIX advisory locks).
* **Windows**: uses ``msvcrt.locking()`` with a 1-byte lock region.

Both modes are non-blocking: if the lock is already held,
``StageLockError`` is raised immediately rather than waiting.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class StageLockError(RuntimeError):
    """Raised when the per-video stage lock cannot be acquired."""


if sys.platform == "win32":
    import msvcrt

    def _lock(fh) -> None:  # type: ignore[type-arg]
        # Lock 1 byte at offset 0; raises OSError on contention.
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

    def _unlock(fh) -> None:  # type: ignore[type-arg]
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        except OSError:
            pass

else:
    import fcntl  # noqa: PLC0415 (import-outside-toplevel — platform-specific)

    def _lock(fh) -> None:  # type: ignore[type-arg]
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh) -> None:  # type: ignore[type-arg]
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            pass


@contextlib.contextmanager
def video_stage_lock(index_dir: Path, video_id: str, stage: str):
    """Acquire an exclusive advisory lock for *video_id* + *stage*.

    Args:
        index_dir: The video's index directory (used to place the lock file).
        video_id:  Video identifier — included in the lock filename for
                   debuggability.
        stage:     Index stage name (``"visual"``, ``"asr"``, etc.).

    Raises:
        StageLockError: if the lock file is already held by another process.
    """
    lock_path = index_dir / f".{stage}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w", encoding="utf-8")  # noqa: WPS515
    try:
        _lock(fh)
    except (IOError, OSError) as exc:
        fh.close()
        raise StageLockError(
            f"Cannot acquire stage lock for video={video_id} stage={stage}. "
            f"Another process is already indexing this stage. "
            f"Lock file: {lock_path}"
        ) from exc

    logger.debug("Stage lock acquired: video=%s stage=%s pid=%d", video_id, stage, os.getpid())
    try:
        yield
    finally:
        _unlock(fh)
        fh.close()
        with contextlib.suppress(OSError):
            lock_path.unlink()
        logger.debug("Stage lock released: video=%s stage=%s", video_id, stage)
