"""Promote verified publication pointers from a staging Catalog.

This is a control-plane migration: Milvus rows are never copied or deleted.
Every source pointer is checked against persisted rows before an atomic,
per-video publication switch in the target Catalog.  The command is read-only
unless ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.catalog.db import Catalog
from app.core.settings import Settings, get_settings
from app.maintenance.migrate_index_publications import (
    _scan_version_counts,
    _scan_visual_health,
)


def _signature(publication: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if publication is None:
        return None
    return (
        str(publication.get("modality") or ""),
        str(publication.get("asset_version") or ""),
        int(publication.get("row_count") or 0),
        str(publication.get("status") or ""),
        json.dumps(publication.get("metadata") or {}, sort_keys=True),
    )


def _validate_source(
    *,
    source_publications: list[dict[str, Any]],
    client: Any,
    timeout: float,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_by_modality: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for publication in source_publications:
        status = str(publication.get("status") or "").casefold()
        if status != "ready":
            continue
        video_id = str(publication.get("video_id") or "").strip()
        modality = str(publication.get("modality") or "").strip().casefold()
        asset_version = str(publication.get("asset_version") or "").strip()
        row_count = int(publication.get("row_count") or 0)
        if not video_id or not modality or not asset_version or row_count < 0:
            raise ValueError(f"invalid source publication: {publication!r}")
        grouped[video_id].append(publication)
        expected_by_modality[modality][(video_id, asset_version)] = row_count

    for modality, expected in expected_by_modality.items():
        video_ids = {video_id for video_id, _ in expected}
        scanned = _scan_version_counts(client, modality, video_ids, timeout=timeout)
        for (video_id, version), row_count in expected.items():
            actual = int(scanned.get(video_id, {}).get(version, 0))
            if actual != row_count:
                raise RuntimeError(
                    f"{video_id}/{modality}@{version}: "
                    f"Catalog rows={row_count}, Milvus rows={actual}"
                )

    visual_expected = expected_by_modality.get("visual", {})
    if visual_expected:
        health = _scan_visual_health(
            client,
            {video_id for video_id, _ in visual_expected},
            timeout=timeout,
        )
        for (video_id, version), row_count in visual_expected.items():
            state = health.get((video_id, version)) or health.get((video_id, "*")) or {}
            if (
                state.get("schema_error")
                or int(state.get("row_count") or 0) != row_count
                or int(state.get("invalid_rows") or 0)
                or list(state.get("inconsistent_segments") or [])
            ):
                raise RuntimeError(
                    f"{video_id}/visual@{version}: invalid visual health {state!r}"
                )

    for video_id, publications in grouped.items():
        ready = {str(item["modality"]): item for item in publications}
        speaker = ready.get("speaker")
        if speaker:
            asr = ready.get("asr")
            source_asr = str(
                (speaker.get("metadata") or {}).get("source_asr_asset_version") or ""
            )
            if not asr or source_asr != str(asr.get("asset_version") or ""):
                raise RuntimeError(
                    f"{video_id}/speaker: source ASR version does not match ready ASR"
                )
        publications.sort(key=lambda item: str(item["modality"]))
    return dict(grouped)


def promote(
    *,
    source_catalog: Catalog,
    target_catalog: Catalog,
    client: Any,
    target_index_root: Path,
    execute: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Validate all ready staging pointers and atomically promote per video."""
    if source_catalog.path.resolve() == target_catalog.path.resolve():
        raise ValueError("source and target Catalog must be different files")
    grouped = _validate_source(
        source_publications=source_catalog.list_modality_publications(),
        client=client,
        timeout=timeout,
    )
    target_videos = {str(video["id"]) for video in target_catalog.list_videos()}
    report: dict[str, Any] = {
        "execute": bool(execute),
        "promoted": [],
        "already_current": [],
        "dry_run_ready": [],
        "errors": [],
    }
    from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

    for video_id, publications in sorted(grouped.items()):
        try:
            if video_id not in target_videos:
                raise KeyError(f"video does not exist in target Catalog: {video_id}")
            expected_source = {
                str(item["modality"]): _signature(item) for item in publications
            }
            with video_stage_lock(
                target_index_root / video_id,
                video_id=video_id,
                stage="publish",
            ):
                current = {
                    str(item["modality"]): item
                    for item in target_catalog.list_modality_publications([video_id])
                }
                conflicts = {
                    modality: _signature(current.get(modality))
                    for modality, signature in expected_source.items()
                    if current.get(modality) is not None
                    and _signature(current[modality]) != signature
                }
                if conflicts:
                    raise RuntimeError(
                        f"target publication conflict for {video_id}: {conflicts!r}"
                    )
                if all(
                    _signature(current.get(modality)) == signature
                    for modality, signature in expected_source.items()
                ):
                    report["already_current"].append({"video_id": video_id})
                    continue
                latest_source = {
                    str(item["modality"]): _signature(item)
                    for item in source_catalog.list_modality_publications([video_id])
                    if str(item.get("status") or "").casefold() == "ready"
                }
                if latest_source != expected_source:
                    raise RuntimeError(
                        f"source publications changed during promotion for {video_id}"
                    )
                payloads = [
                    {
                        "modality": item["modality"],
                        "asset_version": item["asset_version"],
                        "row_count": item["row_count"],
                        "status": "ready",
                        "metadata": item.get("metadata") or {},
                    }
                    for item in publications
                ]
                if not execute:
                    report["dry_run_ready"].append(
                        {"video_id": video_id, "modalities": sorted(expected_source)}
                    )
                    continue
                target_catalog.publish_modalities(video_id, payloads)
                persisted = {
                    str(item["modality"]): _signature(item)
                    for item in target_catalog.list_modality_publications([video_id])
                }
                if any(
                    persisted.get(modality) != signature
                    for modality, signature in expected_source.items()
                ):
                    raise RuntimeError(f"post-publish verification failed for {video_id}")
                report["promoted"].append(
                    {"video_id": video_id, "modalities": sorted(expected_source)}
                )
        except Exception as exc:
            report["errors"].append({"video_id": video_id, "error": str(exc)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote verified publication pointers from a staging Catalog"
    )
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    settings: Settings = get_settings()
    if not args.source_db.is_file():
        parser.error(f"--source-db does not exist: {args.source_db}")
    if not settings.milvus_enabled:
        print("[ERROR] MILVUS_ENABLED=true is required", file=sys.stderr)
        return 2
    from app.vector_store.milvus.milvus_client import get_milvus_client

    report = promote(
        source_catalog=Catalog(args.source_db),
        target_catalog=Catalog(settings.db_path),
        client=get_milvus_client(),
        target_index_root=settings.index_dir,
        execute=bool(args.execute),
        timeout=float(settings.milvus_query_timeout_seconds),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
