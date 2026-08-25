"""One-time import of legacy index declarations into the Catalog control plane.

Old manifests identify which modalities were indexed, but many deployments did
not record the Milvus ``asset_version`` or persisted row count.  This command
therefore treats Milvus as the data-plane source of truth: it scans each needed
collection, groups rows by ``(video_id, asset_version)``, verifies unambiguous
versions, and only then publishes Catalog pointers.

Legacy JSON is read only by this explicit migration command.  Normal indexing
and retrieval code never imports this module, and NPZ files are never read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.vector_store.milvus.milvus_client import get_milvus_client


CHANNELS = ("visual", "face", "asr", "speaker", "ocr")
LEGACY_EMPTY_VERSION = "legacy-empty"
_SCAN_BATCH_SIZE = 2_000
_VISUAL_TIME_FIELDS = (
    "video_id",
    "asset_version",
    "frame_idx",
    "timestamp_ms",
    "segment_id",
    "segment_start_ms",
    "segment_end_ms",
)


def _legacy_channels(index_dir: Path) -> dict:
    path = index_dir / "index_manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels") or {}
    if not isinstance(channels, dict):
        raise ValueError(f"invalid channels object: {path}")
    return channels


def _video_filter(video_ids: Iterable[str]) -> str:
    values = [str(video_id) for video_id in dict.fromkeys(video_ids)]
    if not values:
        raise ValueError("Milvus scan requires at least one video_id")
    return f"video_id in {json.dumps(values, ensure_ascii=False)}"


def _iter_query_rows(
    collection,
    *,
    expr: str,
    output_fields: list[str],
    timeout: float,
) -> Iterator[dict[str, Any]]:
    """Stream all matching rows without Milvus' one-query row cap."""
    if hasattr(collection, "query_iterator"):
        iterator = collection.query_iterator(
            batch_size=_SCAN_BATCH_SIZE,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
        )
        try:
            while True:
                try:
                    page = iterator.next()
                except StopIteration:
                    break
                if not page:
                    break
                yield from page
        finally:
            iterator.close()
        return

    offset = 0
    while True:
        page = collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=_SCAN_BATCH_SIZE,
            offset=offset,
            timeout=timeout,
        )
        if not page:
            break
        yield from page
        if len(page) < _SCAN_BATCH_SIZE:
            break
        offset += len(page)


def _scan_version_counts(
    client,
    modality: str,
    video_ids: set[str],
    *,
    timeout: float,
) -> dict[str, dict[str, int]]:
    """Return actual row counts grouped by video and asset version."""
    if not video_ids:
        return {}
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    collection = client.collection_for(modality)
    for row in _iter_query_rows(
        collection,
        expr=_video_filter(sorted(video_ids)),
        output_fields=["video_id", "asset_version"],
        timeout=timeout,
    ):
        video_id = str(row.get("video_id") or "")
        version = str(row.get("asset_version") or "").strip()
        if video_id not in video_ids:
            continue
        if not version:
            raise RuntimeError(
                f"{video_id}/{modality}: Milvus row has an empty asset_version"
            )
        grouped[video_id][version] += 1
    return {
        video_id: dict(versions)
        for video_id, versions in grouped.items()
    }


def _schema_field_names(collection) -> set[str] | None:
    schema = getattr(collection, "schema", None)
    fields = getattr(schema, "fields", None)
    if fields is None:
        return None
    return {
        str(getattr(field, "name", "") or "")
        for field in fields
        if getattr(field, "name", None)
    }


def _scan_visual_health(
    client,
    video_ids: set[str],
    *,
    timeout: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate persisted visual time bounds before a version becomes readable."""
    if not video_ids:
        return {}
    collection = client.collection_for("visual")
    available = _schema_field_names(collection)
    required = set(_VISUAL_TIME_FIELDS)
    if available is not None and not required.issubset(available):
        missing = sorted(required - available)
        return {
            (video_id, "*"): {
                "schema_error": f"missing visual time fields: {', '.join(missing)}"
            }
            for video_id in video_ids
        }

    health: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _iter_query_rows(
        collection,
        expr=_video_filter(sorted(video_ids)),
        output_fields=list(_VISUAL_TIME_FIELDS),
        timeout=timeout,
    ):
        video_id = str(row.get("video_id") or "")
        version = str(row.get("asset_version") or "").strip()
        if video_id not in video_ids or not version:
            continue
        key = (video_id, version)
        state = health.setdefault(
            key,
            {
                "row_count": 0,
                "invalid_rows": 0,
                "bounds_by_segment": defaultdict(set),
            },
        )
        state["row_count"] += 1
        try:
            frame_idx = int(row["frame_idx"])
            timestamp_ms = int(row["timestamp_ms"])
            segment_id = int(row["segment_id"])
            start_ms = int(row["segment_start_ms"])
            end_ms = int(row["segment_end_ms"])
        except (KeyError, TypeError, ValueError, OverflowError):
            state["invalid_rows"] += 1
            continue
        if (
            frame_idx < 0
            or timestamp_ms < 0
            or segment_id < 0
            or start_ms < 0
            or end_ms <= start_ms
            or timestamp_ms < start_ms
            or timestamp_ms > end_ms
        ):
            state["invalid_rows"] += 1
            continue
        state["bounds_by_segment"][segment_id].add((start_ms, end_ms))

    for state in health.values():
        state["inconsistent_segments"] = sorted(
            segment_id
            for segment_id, bounds in state.pop("bounds_by_segment").items()
            if len(bounds) != 1
        )
    return health


def _manifest_version(channel: dict) -> str:
    return str(
        channel.get("milvus_asset_version")
        or channel.get("asset_version")
        or ""
    ).strip()


def _manifest_row_count(channel: dict) -> int | None:
    value = channel.get("milvus_row_count", channel.get("row_count"))
    if value is None or value == "":
        return None
    return int(value)


def _select_version(
    *,
    video_id: str,
    modality: str,
    channel: dict,
    version_counts: dict[str, int],
) -> tuple[str, int, str]:
    explicit_version = _manifest_version(channel)
    declared_rows = _manifest_row_count(channel)

    if explicit_version:
        if version_counts and explicit_version not in version_counts:
            discovered = ", ".join(sorted(version_counts))
            raise RuntimeError(
                f"{video_id}/{modality}@{explicit_version}: manifest version has no "
                f"Milvus rows; discovered versions: {discovered}"
            )
        persisted = int(version_counts.get(explicit_version, 0))
        if declared_rows is not None and persisted != declared_rows:
            raise RuntimeError(
                f"{video_id}/{modality}@{explicit_version}: "
                f"manifest rows={declared_rows}, Milvus rows={persisted}"
            )
        return explicit_version, persisted, "manifest"

    if not version_counts:
        if declared_rows not in (None, 0):
            raise RuntimeError(
                f"{video_id}/{modality}: manifest rows={declared_rows}, Milvus rows=0"
            )
        return LEGACY_EMPTY_VERSION, 0, "legacy_empty"
    if len(version_counts) != 1:
        details = ", ".join(
            f"{version}:{rows}" for version, rows in sorted(version_counts.items())
        )
        raise RuntimeError(
            f"{video_id}/{modality}: multiple Milvus asset versions ({details}); "
            "refusing to guess the published version"
        )
    version, rows = next(iter(version_counts.items()))
    if declared_rows is not None and rows != declared_rows:
        raise RuntimeError(
            f"{video_id}/{modality}@{version}: "
            f"manifest rows={declared_rows}, Milvus rows={rows}"
        )
    return version, int(rows), "milvus_scan"


def migrate(*, apply: bool, video_ids: set[str] | None = None) -> dict:
    settings = get_settings()
    catalog = Catalog(settings.db_path)
    client = get_milvus_client()
    from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

    report = {
        "apply": apply,
        "migrated": [],
        "skipped": [],
        "requires_rebuild": [],
        "errors": [],
    }
    selected = set(video_ids or ())
    found: set[str] = set()
    timeout = float(settings.milvus_query_timeout_seconds)

    for video in catalog.list_videos():
        video_id = str(video["id"])
        if selected and video_id not in selected:
            continue
        found.add(video_id)
        try:
            index_dir = settings.index_dir / video_id
            # Use the exact same per-video publication lock as normal indexing.
            # All Milvus observations used to select a pointer happen while the
            # lock is held, so a slower legacy scan cannot overwrite a newer
            # indexing attempt that honours the platform publication contract.
            with video_stage_lock(index_dir, video_id=video_id, stage="publish"):
                channels = _legacy_channels(index_dir)
                if not channels:
                    report["skipped"].append(
                        {"video_id": video_id, "reason": "no_manifest"}
                    )
                    continue
                declarations = {
                    modality: channel
                    for modality in CHANNELS
                    if isinstance((channel := channels.get(modality)), dict)
                }
                for modality, channel in declarations.items():
                    try:
                        # Deliberately rescan this one video inside the lock.  A
                        # process-wide pre-scan would become stale before publish.
                        scanned = _scan_version_counts(
                            client, modality, {video_id}, timeout=timeout
                        )
                        per_version = scanned.get(video_id, {})
                        visual_health: dict[tuple[str, str], dict[str, Any]] = {}
                        if modality == "visual" and per_version:
                            visual_health = _scan_visual_health(
                                client, {video_id}, timeout=timeout
                            )
                        source_asr_asset_version = None
                        if modality == "speaker":
                            if isinstance(declarations.get("asr"), dict):
                                asr_counts = _scan_version_counts(
                                    client, "asr", {video_id}, timeout=timeout
                                ).get(video_id, {})
                                source_asr_asset_version, _, _ = _select_version(
                                    video_id=video_id,
                                    modality="asr",
                                    channel=declarations["asr"],
                                    version_counts=asr_counts,
                                )
                                if apply:
                                    asr_publication = catalog.get_modality_publication(
                                        video_id, "asr"
                                    )
                                    current_asr_version = (
                                        str(asr_publication["asset_version"])
                                        if asr_publication
                                        and asr_publication.get("status") == "ready"
                                        else None
                                    )
                                    if current_asr_version != source_asr_asset_version:
                                        raise RuntimeError(
                                            f"{video_id}/speaker: selected ASR "
                                            f"{source_asr_asset_version} is not the current "
                                            f"ready ASR publication ({current_asr_version})"
                                        )
                        _migrate_one_publication(
                            report=report,
                            catalog=catalog,
                            client=client,
                            video_id=video_id,
                            modality=modality,
                            channel=channel,
                            version_counts=per_version,
                            visual_health=visual_health,
                            source_asr_asset_version=source_asr_asset_version,
                            apply=apply,
                        )
                    except Exception as exc:
                        logging.exception(
                            "Publication migration failed for %s/%s",
                            video_id,
                            modality,
                        )
                        report["errors"].append(
                            {
                                "video_id": video_id,
                                "modality": modality,
                                "error": str(exc),
                            }
                        )
        except Exception as exc:
            logging.exception("Publication migration failed for %s", video_id)
            report["errors"].append({"video_id": video_id, "error": str(exc)})

    for missing in sorted(selected - found):
        report["errors"].append(
            {"video_id": missing, "error": "video does not exist in Catalog"}
        )
    return report


def _migrate_one_publication(
    *,
    report: dict,
    catalog: Catalog,
    client,
    video_id: str,
    modality: str,
    channel: dict,
    version_counts: dict[str, int],
    visual_health: dict[tuple[str, str], dict[str, Any]],
    source_asr_asset_version: str | None = None,
    apply: bool,
) -> None:
    version, persisted, source = _select_version(
        video_id=video_id,
        modality=modality,
        channel=channel,
        version_counts=version_counts,
    )
    # Count aggregation and the direct Milvus count must agree before the
    # Catalog pointer can be switched.
    actual = client.count_video_modality_version(video_id, modality, version)
    if actual != persisted:
        raise RuntimeError(
            f"{video_id}/{modality}@{version}: "
            f"scan rows={persisted}, Milvus rows={actual}"
        )

    metadata = {
        key: value
        for key, value in channel.items()
        if key
        not in {
            "file",
            "asset_version",
            "row_count",
            "milvus_asset_version",
            "milvus_row_count",
        }
    }
    if modality == "speaker":
        if not source_asr_asset_version:
            raise RuntimeError(
                f"{video_id}/speaker: no ready or declared ASR publication "
                "is available for source-version validation"
            )
        metadata["source_asr_asset_version"] = source_asr_asset_version
    if modality == "visual" and persisted:
        health = visual_health.get((video_id, version)) or visual_health.get(
            (video_id, "*")
        ) or {}
        invalid_rows = int(health.get("invalid_rows") or 0)
        inconsistent = list(health.get("inconsistent_segments") or [])
        schema_error = health.get("schema_error")
        scanned_rows = int(health.get("row_count") or 0)
        if schema_error or scanned_rows != persisted or invalid_rows or inconsistent:
            reason = schema_error or (
                f"visual time rows invalid={invalid_rows}, "
                f"scanned={scanned_rows}, expected={persisted}, "
                f"inconsistent_segments={inconsistent}"
            )
            disabled_metadata = {
                **metadata,
                "migration_state": "requires_rebuild",
                "reason": reason,
            }
            if apply:
                # Preserve the verified source pointer for diagnosis, while
                # keeping it invisible to normal retrieval.
                catalog.publish_modality(
                    video_id,
                    modality,
                    asset_version=version,
                    row_count=persisted,
                    metadata=disabled_metadata,
                    status="disabled",
                )
            report["requires_rebuild"].append(
                {
                    "video_id": video_id,
                    "modality": modality,
                    "asset_version": version,
                    "row_count": persisted,
                    "reason": reason,
                    "status": "disabled" if apply else "pending",
                }
            )
            return

    if apply:
        catalog.publish_modality(
            video_id,
            modality,
            asset_version=version,
            row_count=persisted,
            metadata=metadata,
        )
    report["migrated"].append(
        {
            "video_id": video_id,
            "modality": modality,
            "asset_version": version,
            "row_count": persisted,
            "version_source": source,
            "status": "published" if apply else "would_publish",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write verified publication rows. Without this flag the command is read-only.",
    )
    parser.add_argument("--video-id", action="append", dest="video_ids")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = migrate(
        apply=bool(args.apply),
        video_ids=set(args.video_ids) if args.video_ids else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] or report["requires_rebuild"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
