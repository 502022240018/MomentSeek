from __future__ import annotations

import argparse
import json
import time

from app.db import Catalog
from app.indexing.pipeline_manifest import write_stage_manifest
from app.indexing.speaker import build_speaker_index
from app.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speaker indexes for videos that already have ASR indexes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    args = parser.parse_args()
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    summary = []
    for video in catalog.list_videos():
        if args.video_ids and video["id"] not in args.video_ids:
            continue
        if "speaker" in video["indexed_modalities"]:
            catalog.update_video(
                video["id"],
                indexed_modalities=[item for item in video["indexed_modalities"] if item != "speaker"],
            )
        index_dir = settings.index_dir / video["id"]
        asr_path = index_dir / "asr.npz"
        output_path = index_dir / "speaker.npz"
        if not asr_path.exists() or (output_path.exists() and not args.force):
            continue
        started = time.perf_counter()
        print(json.dumps({"status": "started", "video_id": video["id"], "name": video["name"]}, ensure_ascii=False), flush=True)
        try:
            result = build_speaker_index(
                video_path=str(settings.resolve_path(video["file_path"])),
                asr_path=str(asr_path), output_path=str(output_path),
                working_dir=str(index_dir / "work"),
                model_repo=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)),
                model_cache_dir=str(settings.resolve_path(settings.app_model_dir / settings.speaker_model_cache_dir)),
                device=settings.speaker_device,
            )
            write_stage_manifest(
                "speaker", index_dir=index_dir, video=video,
                options={"asr_speaker_enabled": True}, settings=settings, result=result,
            )
            modalities = [item for item in video["indexed_modalities"] if item != "speaker"]
            catalog.update_video(video["id"], indexed_modalities=modalities)
            item = {"status": "completed", "video_id": video["id"], "name": video["name"], **result}
        except Exception as exc:
            item = {"status": "failed", "video_id": video["id"], "name": video["name"], "error": str(exc)}
        item["wall_seconds"] = round(time.perf_counter() - started, 3)
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "done", "results": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
