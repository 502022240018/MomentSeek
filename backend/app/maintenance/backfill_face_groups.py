from __future__ import annotations

import logging

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.identity.face_gallery_service import ensure_video_face_groups, published_face_version


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    created = skipped = failed = 0
    for video in catalog.list_videos():
        if "face" not in (video.get("indexed_modalities") or []):
            continue
        try:
            version = published_face_version(settings.index_dir, video["id"])
            if ensure_video_face_groups(
                video["id"], version, settings.face_gallery_cosine_threshold
            ):
                created += 1
            else:
                skipped += 1
        except Exception:
            failed += 1
            logging.exception("Face-group backfill failed for %s", video["id"])
    logging.info("Face-group backfill complete: created=%d skipped=%d failed=%d", created, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
