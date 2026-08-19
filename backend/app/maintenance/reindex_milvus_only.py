"""Rebuild existing catalog videos into the Milvus-only online index.

This is an explicit maintenance command, not a runtime fallback.  It rebuilds
each selected source video through the normal indexing stages, which write and
verify a new Milvus asset version before publishing its Catalog pointer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

from app.catalog.db import Catalog
from app.core.settings import Settings, get_settings
from app.indexing.stage_executor import execute_stage


DEFAULT_MODALITIES = ("visual", "face", "asr", "ocr")
ALLOWED_MODALITIES = frozenset((*DEFAULT_MODALITIES, "speaker"))
STAGE_ORDER = ("visual", "face", "asr", "speaker", "ocr")


@dataclass(frozen=True)
class MigrationTarget:
    video: dict
    source_exists: bool


def normalize_modalities(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize user input into an execution order without duplicate speaker work."""
    raw = (
        DEFAULT_MODALITIES
        if value is None
        else (value.split(",") if isinstance(value, str) else value)
    )
    selected = {str(item).strip().lower() for item in raw if str(item).strip()}
    unknown = selected - ALLOWED_MODALITIES
    if unknown:
        raise ValueError(f"不支持的通道: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("至少选择一个索引通道")
    # ASR can build speaker atomically from the newly published ASR version.
    # Running a second standalone speaker stage in that case only wastes time.
    if "asr" in selected:
        selected.discard("speaker")
    return tuple(stage for stage in STAGE_ORDER if stage in selected)


def migration_targets(
    catalog: Catalog, settings: Settings, video_ids: list[str] | None = None
) -> list[MigrationTarget]:
    selected = set(video_ids or [])
    targets = []
    for video in catalog.list_videos():
        if selected and video["id"] not in selected:
            continue
        source_path = settings.resolve_path(video["file_path"])
        targets.append(
            MigrationTarget(video=video, source_exists=source_path.is_file())
        )
    missing = selected - {target.video["id"] for target in targets}
    if missing:
        raise ValueError(f"目录中不存在 video_id: {', '.join(sorted(missing))}")
    return targets


def published_modalities(
    stages: tuple[str, ...], result_by_stage: dict[str, dict]
) -> set[str]:
    completed = set(stages)
    if "asr" in stages and isinstance(
        result_by_stage.get("asr", {}).get("speaker"), dict
    ):
        completed.add("speaker")
    return completed


def run_migration(
    catalog: Catalog,
    settings: Settings,
    targets: list[MigrationTarget],
    stages: tuple[str, ...],
) -> tuple[list[dict], int]:
    """Run selected normal indexing stages sequentially and update catalog state."""
    results: list[dict] = []
    failed = 0
    for target in targets:
        video = target.video
        item: dict = {"video_id": video["id"], "name": video["name"], "stages": {}}
        if not target.source_exists:
            item.update(
                status="skipped", error="源视频不存在，未修改已有 Milvus 发布版本"
            )
            failed += 1
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
            continue
        started = time.perf_counter()
        stage_results: dict[str, dict] = {}
        try:
            for stage in stages:
                options = {"asr_speaker_enabled": True} if stage == "asr" else {}
                stage_started = time.perf_counter()
                stage_result = execute_stage(stage, video, options, settings)
                stage_results[stage] = stage_result
                item["stages"][stage] = {
                    "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
                    "milvus_asset_version": stage_result.get("milvus_asset_version"),
                    "milvus_row_count": stage_result.get("milvus_row_count"),
                }
            catalog.update_video(video["id"], status="ready")
            item["status"] = "completed"
        except Exception as exc:
            # A failed replacement must not make a previously published index
            # look unavailable in the catalog.  Stage publication is atomic per
            # modality, so the last Catalog publication stays readable.
            item.update(status="failed", error=str(exc))
            failed += 1
        item["wall_seconds"] = round(time.perf_counter() - started, 3)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return results, failed


def verify_published_versions(
    catalog: Catalog,
    settings: Settings,
    targets: list[MigrationTarget],
    stages: tuple[str, ...],
) -> tuple[list[dict], int]:
    """Check Catalog pointers against persisted rows in the declared Milvus version."""
    if not settings.milvus_enabled:
        raise RuntimeError("Milvus-only 验收需要 MILVUS_ENABLED=true")
    from app.vector_store.milvus.milvus_client import get_milvus_client

    expected = set(stages)
    if "asr" in expected:
        expected.add("speaker")
    client = get_milvus_client()
    results: list[dict] = []
    failures = 0
    for target in targets:
        video_id = target.video["id"]
        channels = {
            publication["modality"]: publication
            for publication in catalog.list_modality_publications([video_id])
        }
        item = {"video_id": video_id, "status": "completed", "channels": {}}
        for modality in sorted(expected):
            channel = channels.get(modality) or {}
            version = channel.get("asset_version")
            declared = channel.get("row_count")
            if not version or declared is None:
                item["channels"][modality] = {"status": "missing_publish_pointer"}
                item["status"] = "failed"
                continue
            actual = client.count_video_modality_version(
                video_id, modality, str(version)
            )
            state = "ok" if actual == int(declared) else "row_count_mismatch"
            item["channels"][modality] = {
                "status": state,
                "asset_version": str(version),
                "catalog_rows": int(declared),
                "milvus_rows": actual,
            }
            if state != "ok":
                item["status"] = "failed"
        if item["status"] == "failed":
            failures += 1
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="重建既有视频的 Milvus-only 正式索引")
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="只处理指定视频，可重复传入",
    )
    parser.add_argument(
        "--modalities",
        default=",".join(DEFAULT_MODALITIES),
        help="逗号分隔：visual,face,asr,speaker,ocr；默认重建全部正式通道",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只检查 catalog 与源视频，不写任何数据"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只核验已发布 Catalog 指针和 Milvus 行数",
    )
    parser.add_argument(
        "--execute", action="store_true", help="确认执行重建；未传入时只做 dry-run"
    )
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("--dry-run 与 --verify-only 不能同时使用")
    if args.execute and (args.dry_run or args.verify_only):
        parser.error("--execute 不能与 --dry-run 或 --verify-only 同时使用")

    stages = normalize_modalities(args.modalities)
    settings = get_settings()
    if not settings.milvus_enabled or not settings.milvus_write_enabled:
        print(
            "[ERROR] Milvus-only 迁移要求 MILVUS_ENABLED=true 且 MILVUS_WRITE_ENABLED=true",
            file=sys.stderr,
        )
        return 2
    catalog = Catalog(settings.db_path)
    try:
        targets = migration_targets(catalog, settings, args.video_ids)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if not targets:
        print("[ERROR] 没有可处理的视频", file=sys.stderr)
        return 2

    if args.verify_only:
        _, failed = verify_published_versions(catalog, settings, targets, stages)
    elif args.execute:
        _, failed = run_migration(catalog, settings, targets, stages)
    else:
        for target in targets:
            print(
                json.dumps(
                    {
                        "video_id": target.video["id"],
                        "name": target.video["name"],
                        "source_exists": target.source_exists,
                        "stages": stages,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        failed = sum(not target.source_exists for target in targets)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
