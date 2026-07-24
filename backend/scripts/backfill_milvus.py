#!/usr/bin/env python3
"""Mandatory Milvus backfill — push all existing NPZ indexes into Milvus.

Usage (from backend/):
    python -m scripts.backfill_milvus [--index-dir runtime/indexes] [--asset-version 1]
                                       [--modalities visual,asr,ocr,face,speaker]
                                       [--dry-run] [--resume]

Rationale
---------
Once MILVUS_READ_ENABLED is turned on, any video whose NPZ has not been
backfilled will trigger NPZ fallback on every query, masking Milvus coverage
gaps and making performance comparisons meaningless.  This script must be run
to completion before read traffic is migrated.

Resume support
--------------
Completed video+modality pairs are tracked in a sidecar file
(backfill_milvus_progress.jsonl) so the script can be safely interrupted and
restarted.

Idempotency and verification
----------------------------
Each video+modality is replaced as one recovery unit: old rows are deleted,
the retained NPZ is inserted, the collection is flushed, and the persisted
row count is verified.  Re-running the command therefore does not accumulate
duplicate asset versions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as  python -m scripts.backfill_milvus  from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_milvus")

MODALITY_TO_NPZ = {
    "visual":  "visual.npz",
    "asr":     "asr.npz",
    "ocr":     "ocr.npz",
    "face":    "face.npz",
    "speaker": "speaker.npz",
}


def _progress_key(video_id: str, modality: str, asset_version: str) -> str:
    return f"{video_id}:{modality}:{asset_version}"


def _expected_count(npz_path: Path, modality: str) -> int:
    """Derive the exact number of rows the modality indexer will write."""
    count_fields = {
        "visual": "frame_embeddings",
        "asr": "chunk_times_ms",
        "ocr": "frame_times_ms",
        "face": "embeddings",
        "speaker": "utterance_embeddings",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        field = count_fields[modality]
        if field not in data.files:
            raise ValueError(f"{npz_path.name} missing required field: {field}")
        return len(data[field])


def _load_progress(path: Path) -> set[str]:
    """Return version-scoped keys already completed successfully."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
                if rec.get("status") == "done" and rec.get("asset_version") is not None:
                    done.add(_progress_key(
                        str(rec["video_id"]),
                        str(rec["modality"]),
                        str(rec["asset_version"]),
                    ))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _record_progress(
    path: Path,
    video_id: str,
    modality: str,
    asset_version: str,
    status: str,
    detail: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "video_id": video_id,
            "modality": modality,
            "asset_version": asset_version,
            "status":   status,
            "detail":   detail,
            "ts":       time.time(),
        }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill existing NPZ indexes into Milvus")
    parser.add_argument("--index-dir",
                        help="Root directory containing per-video indexes (default: app setting)")
    parser.add_argument("--asset-version", default="1",
                        help="Asset version to stamp on all backfilled records")
    parser.add_argument("--modalities",    default="visual,asr,ocr,face,speaker",
                        help="Comma-separated list of modalities to backfill")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Discover what would be done but do not write to Milvus")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip video+modality pairs already marked done in the progress file")
    parser.add_argument("--progress-file",
                        help="Resume sidecar (default: <app-data>/backfill_milvus_progress.jsonl)")
    args = parser.parse_args()

    from app.settings import get_settings

    settings = get_settings()
    index_dir     = Path(args.index_dir) if args.index_dir else settings.index_dir
    asset_version = args.asset_version
    modalities    = [m.strip() for m in args.modalities.split(",") if m.strip()]
    unknown_modalities = sorted(set(modalities) - set(MODALITY_TO_NPZ))
    if unknown_modalities:
        parser.error(f"unknown modalities: {', '.join(unknown_modalities)}")
    progress_path = (
        Path(args.progress_file)
        if args.progress_file
        else settings.app_data_dir / "backfill_milvus_progress.jsonl"
    )
    done_keys     = _load_progress(progress_path) if args.resume else set()

    if not index_dir.is_dir():
        logger.error("index-dir does not exist: %s", index_dir)
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN — no data will be written to Milvus")

    # Initialise Milvus client (skip in dry-run)
    client = None
    if not args.dry_run:
        from app.indexing.milvus_client import get_milvus_client
        client = get_milvus_client()
        logger.info("Milvus connected")

    from app.indexing.milvus_indexer import MilvusWriteContext, _INDEXERS

    video_dirs = sorted(p for p in index_dir.iterdir() if p.is_dir())
    logger.info("Found %d video directories, modalities: %s", len(video_dirs), modalities)

    total_ok = total_skip = total_fail = total_missing = 0

    for video_dir in video_dirs:
        video_id = video_dir.name
        for modality in modalities:
            key = _progress_key(video_id, modality, asset_version)
            npz_name = MODALITY_TO_NPZ[modality]
            npz_path = video_dir / npz_name

            if not npz_path.exists():
                total_missing += 1
                logger.debug("MISSING %s/%s", video_id, npz_name)
                continue

            if args.resume and key in done_keys:
                if args.dry_run:
                    total_skip += 1
                    continue
                expected = _expected_count(npz_path, modality)
                persisted = client.count_video_modality(video_id, modality)
                if persisted == expected:
                    total_skip += 1
                    continue
                logger.warning(
                    "Resume marker is stale for %s/%s: expected=%d persisted=%d; rewriting",
                    video_id, modality, expected, persisted,
                )

            if args.dry_run:
                logger.info("DRY-RUN would upsert %s/%s", video_id, npz_name)
                total_ok += 1
                continue

            ctx = MilvusWriteContext(
                video_id=video_id,
                asset_version=asset_version,
                client=client,
            )
            try:
                # Validate the primary row axis before deleting a possibly
                # healthy existing generation.
                expected = _expected_count(npz_path, modality)
                deleted = client.delete_video_modality(video_id, modality)
                if deleted < 0:
                    raise RuntimeError("failed to delete existing Milvus rows")
                count = _INDEXERS[modality].upsert_from_npz(ctx, npz_path)
                if count != expected:
                    raise RuntimeError(
                        f"indexer count mismatch: expected={expected}, wrote={count}"
                    )
                client.collection_for(modality).flush()
                persisted = client.count_video_modality(video_id, modality)
                if persisted != count:
                    raise RuntimeError(
                        f"row count mismatch: wrote={count}, persisted={persisted}"
                    )
                logger.info(
                    "OK %s/%s deleted=%d count=%d",
                    video_id, modality, deleted, count,
                )
                _record_progress(
                    progress_path, video_id, modality, asset_version,
                    "done", f"deleted={deleted},count={count}",
                )
                total_ok += 1
            except Exception as exc:
                logger.error("FAIL %s/%s: %s", video_id, modality, exc)
                _record_progress(
                    progress_path, video_id, modality, asset_version, "fail", str(exc)
                )
                total_fail += 1

    logger.info(
        "Backfill complete — ok=%d  skipped=%d  missing=%d  fail=%d",
        total_ok, total_skip, total_missing, total_fail,
    )
    if total_fail:
        logger.warning("Some writes failed; rerun with --resume to retry.")
        sys.exit(2)


if __name__ == "__main__":
    main()
