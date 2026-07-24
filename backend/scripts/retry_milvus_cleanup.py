#!/usr/bin/env python3
"""Retry durable Milvus cleanup tasks left by deleted videos."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retry_milvus_cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from app.db import Catalog
    from app.settings import get_settings

    settings = get_settings()
    catalog = Catalog(settings.db_path)
    pending = catalog.list_milvus_cleanup_queue()
    logger.info("Pending Milvus cleanup tasks: %d", len(pending))
    if args.dry_run:
        for item in pending:
            logger.info("DRY-RUN video=%s attempts=%s", item["video_id"], item["attempts"])
        return

    from app.indexing.milvus_client import get_milvus_client

    client = get_milvus_client()
    failures = 0
    for item in pending:
        video_id = item["video_id"]
        try:
            counts = client.delete_video(video_id)
            failed = [name for name, count in counts.items() if count < 0]
            if failed:
                raise RuntimeError("failed collections: " + ", ".join(sorted(failed)))
            catalog.complete_milvus_cleanup(video_id)
            logger.info("CLEANED video=%s counts=%s", video_id, counts)
        except Exception as exc:
            catalog.enqueue_milvus_cleanup(video_id, str(exc))
            logger.error("FAIL video=%s: %s", video_id, exc)
            failures += 1
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
