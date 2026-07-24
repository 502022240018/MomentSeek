from __future__ import annotations

import argparse
import json

from app.db import Catalog
from app.settings import get_settings
from app.stage_executor import (
    _pre_delete_modality,  # noqa: F401 - compatibility re-export
    _setup_milvus_context,  # noqa: F401 - compatibility re-export
    execute_stage,
)


def run(stage: str, job_id: str) -> dict:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    job = catalog.get_job(job_id)
    if not job:
        raise KeyError(f"任务不存在: {job_id}")
    video = catalog.get_video(job["video_id"])
    if not video:
        raise KeyError(f"视频不存在: {job['video_id']}")
    return execute_stage(stage, video, job.get("options") or {}, settings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["visual", "face", "asr", "speaker", "ocr"])
    parser.add_argument("job_id")
    arguments = parser.parse_args()
    result = run(arguments.stage, arguments.job_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
