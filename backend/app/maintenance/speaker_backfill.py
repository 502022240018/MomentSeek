from __future__ import annotations

import argparse
import json
import sys
import time

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.indexing.stage_executor import execute_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speaker indexes for videos that already have ASR indexes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    args = parser.parse_args()
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    summary = []
    failed = 0
    for video in catalog.list_videos():
        if args.video_ids and video["id"] not in args.video_ids:
            continue
        if "speaker" in video["indexed_modalities"] and not args.force:
            continue
        started = time.perf_counter()
        print(json.dumps({"status": "started", "video_id": video["id"], "name": video["name"]}, ensure_ascii=False), flush=True)
        try:
            result = execute_stage(
                "speaker",
                video,
                {"asr_speaker_enabled": True},
                settings,
            )
            modalities = sorted({*video["indexed_modalities"], "speaker"})
            catalog.update_video(video["id"], indexed_modalities=modalities)
            item = {"status": "completed", "video_id": video["id"], "name": video["name"], **result}
        except Exception as exc:
            item = {"status": "failed", "video_id": video["id"], "name": video["name"], "error": str(exc)}
            failed += 1
        item["wall_seconds"] = round(time.perf_counter() - started, 3)
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "done", "results": summary}, ensure_ascii=False), flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
