"""Repair legacy visual time bounds by copying Milvus rows to a new version.

This is an explicit, one-time maintenance command.  It never reads NPZ files,
never mutates the source rows, and never deletes old asset versions.  The source
embedding is reused byte-for-byte at the Python/Milvus boundary; only the PK,
asset version, and explicit fixed-window bounds are changed.

The command is dry-run by default.  ``--execute`` is required before any row is
written or any Catalog pointer is published.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from app.catalog.db import Catalog
from app.core.settings import Settings, get_settings
from app.vector_store.milvus.milvus_indexer import _upsert_batched
from app.vector_store.milvus.milvus_schema import EMBEDDING_DIMS, visual_pk


logger = logging.getLogger(__name__)

_MANIFEST_NAME = "index_manifest.json"
_MIGRATION_PROTOCOL = "milvus-fixed-window-v1"
_MIGRATION_NAMESPACE = uuid.UUID("17012acb-5906-4a09-b1e5-98a14dd6a20f")
_META_FIELDS = (
    "pk",
    "video_id",
    "asset_version",
    "model_version",
    "frame_idx",
    "timestamp_ms",
    "segment_id",
    "segment_start_ms",
    "segment_end_ms",
)
_COPY_FIELDS = (*_META_FIELDS, "embedding")


@dataclass(frozen=True)
class FixedWindowConfig:
    segment_ms: int
    manifest_path: Path


@dataclass(frozen=True)
class FramePlan:
    frame_idx: int
    timestamp_ms: int
    segment_id: int
    segment_start_ms: int
    segment_end_ms: int
    model_version: str


@dataclass(frozen=True)
class ValidatedRows:
    plans: dict[int, FramePlan]
    bounds_state: str
    model_version: str

    @property
    def row_count(self) -> int:
        return len(self.plans)


def _strict_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a positive integer")
    number = int(value)
    if not math.isfinite(float(value)) or float(value) != float(number) or number <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return number


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"visual row has invalid integer field {field}: {value!r}")
    return int(value)


def _strategy_values(payload: dict[str, Any], publication: dict[str, Any]) -> list[str]:
    channels = payload.get("channels")
    visual = channels.get("visual") if isinstance(channels, dict) else None
    values = [
        payload.get("segment_strategy"),
        payload.get("visual_segment_strategy"),
        publication.get("segment_strategy"),
    ]
    if isinstance(visual, dict):
        values.extend((visual.get("segment_strategy"), visual.get("visual_segment_strategy")))
    return [str(value).strip().casefold() for value in values if str(value or "").strip()]


def load_legacy_fixed_config(
    index_dir: Path,
    *,
    video_id: str,
    duration_ms: int,
    publication: dict[str, Any],
) -> FixedWindowConfig:
    """Load only the legacy fixed-window facts required for this migration."""
    path = index_dir / _MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"{video_id}: missing legacy {_MANIFEST_NAME}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{video_id}: invalid legacy manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{video_id}: legacy manifest must be a JSON object")
    if payload.get("schema_version") != 3:
        raise ValueError(
            f"{video_id}: unsupported legacy manifest schema="
            f"{payload.get('schema_version')!r}"
        )
    manifest_video_id = str(payload.get("video_id") or "").strip()
    if manifest_video_id != video_id:
        raise ValueError(
            f"{video_id}: legacy manifest belongs to video_id={manifest_video_id}"
        )
    manifest_duration_ms = _strict_positive_int(
        payload.get("duration_ms"), field="manifest.duration_ms"
    )
    if manifest_duration_ms != duration_ms:
        raise ValueError(
            f"{video_id}: Catalog duration_ms={duration_ms} conflicts with "
            f"manifest duration_ms={manifest_duration_ms}"
        )
    segment_ms = _strict_positive_int(payload.get("segment_ms"), field="segment_ms")
    channels = payload.get("channels")
    visual = channels.get("visual") if isinstance(channels, dict) else None
    if not isinstance(visual, dict):
        raise ValueError(f"{video_id}: legacy manifest is missing visual channel")
    manifest_version = str(visual.get("milvus_asset_version") or "").strip()
    if manifest_version and manifest_version != publication["asset_version"]:
        raise ValueError(
            f"{video_id}: Catalog visual version={publication['asset_version']} "
            f"conflicts with manifest version={manifest_version}"
        )
    if visual.get("milvus_row_count") is not None:
        manifest_rows = _strict_positive_int(
            visual.get("milvus_row_count"), field="manifest.visual.milvus_row_count"
        )
        if manifest_rows != int(publication["row_count"]):
            raise ValueError(
                f"{video_id}: Catalog visual rows={publication['row_count']} "
                f"conflicts with manifest rows={manifest_rows}"
            )
    strategies = _strategy_values(payload, publication)
    if any(strategy != "fixed" for strategy in strategies):
        raise ValueError(
            f"{video_id}: only fixed-window visual indexes can be repaired; "
            f"found strategies={strategies}"
        )
    publication_segment_ms = publication.get("segment_ms")
    if publication_segment_ms is not None:
        declared = _strict_positive_int(
            publication_segment_ms, field="publication.segment_ms"
        )
        if declared != segment_ms:
            raise ValueError(
                f"{video_id}: conflicting segment_ms manifest={segment_ms} "
                f"publication={declared}"
            )
    return FixedWindowConfig(segment_ms=segment_ms, manifest_path=path)


def _duration_ms(video: dict[str, Any]) -> int:
    duration = video.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise ValueError(f"{video.get('id')}: Catalog duration is missing or invalid")
    duration_float = float(duration)
    if not math.isfinite(duration_float) or duration_float <= 0:
        raise ValueError(f"{video.get('id')}: Catalog duration must be positive")
    duration_ms = int(round(duration_float * 1000.0))
    if duration_ms <= 0:
        raise ValueError(f"{video.get('id')}: Catalog duration rounds to zero")
    return duration_ms


def _expr(video_id: str, asset_version: str) -> str:
    return (
        f"video_id == {json.dumps(str(video_id), ensure_ascii=False)} and "
        f"asset_version == {json.dumps(str(asset_version), ensure_ascii=False)}"
    )


def _iter_rows(
    collection: Any,
    *,
    video_id: str,
    asset_version: str,
    output_fields: Iterable[str],
    batch_size: int,
    timeout: float,
) -> Iterator[dict[str, Any]]:
    """Stream a version with Milvus QueryIterator; never use a bounded query."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    query_iterator = getattr(collection, "query_iterator", None)
    if not callable(query_iterator):
        raise RuntimeError(
            "Milvus Collection.query_iterator is required for complete visual migration"
        )
    iterator = query_iterator(
        expr=_expr(video_id, asset_version),
        output_fields=list(output_fields),
        batch_size=int(batch_size),
        timeout=float(timeout),
    )
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Milvus returned a non-object visual row: {row!r}")
                yield row
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _plan_for_row(
    row: dict[str, Any],
    *,
    video_id: str,
    asset_version: str,
    duration_ms: int,
    segment_ms: int,
) -> tuple[FramePlan, str]:
    if str(row.get("video_id") or "") != video_id:
        raise ValueError(f"visual row video_id mismatch: {row.get('video_id')!r}")
    if str(row.get("asset_version") or "") != asset_version:
        raise ValueError(
            f"visual row asset_version mismatch: {row.get('asset_version')!r}"
        )
    frame_idx = _required_int(row, "frame_idx")
    timestamp_ms = _required_int(row, "timestamp_ms")
    segment_id = _required_int(row, "segment_id")
    if frame_idx < 0:
        raise ValueError(f"visual row has negative frame_idx={frame_idx}")
    if timestamp_ms < 0 or timestamp_ms > duration_ms:
        raise ValueError(
            f"frame_idx={frame_idx}: timestamp_ms={timestamp_ms} outside "
            f"[0,{duration_ms}]"
        )
    # Legacy samplers may emit one terminal frame exactly at video duration.
    # It belongs to the final non-empty fixed window, never to a new zero-width
    # bucket.  No timestamp beyond duration is accepted.
    effective_timestamp_ms = min(timestamp_ms, duration_ms - 1)
    expected_segment = effective_timestamp_ms // segment_ms
    legacy_terminal_segment = timestamp_ms // segment_ms
    allowed_source_segments = (
        {expected_segment, legacy_terminal_segment}
        if timestamp_ms == duration_ms
        else {expected_segment}
    )
    if segment_id not in allowed_source_segments:
        raise ValueError(
            f"frame_idx={frame_idx}: segment_id={segment_id} does not match "
            f"timestamp_ms//segment_ms={expected_segment}"
        )
    segment_id = expected_segment
    start_ms = segment_id * segment_ms
    end_ms = min(start_ms + segment_ms, duration_ms)
    if not (
        0 <= start_ms <= timestamp_ms <= end_ms <= duration_ms
        and (timestamp_ms < end_ms or timestamp_ms == duration_ms)
    ):
        raise ValueError(
            f"frame_idx={frame_idx}: computed bounds [{start_ms},{end_ms}] are invalid"
        )
    model_version = str(row.get("model_version") or "").strip()
    if not model_version:
        raise ValueError(f"frame_idx={frame_idx}: model_version is missing")

    raw_start = row.get("segment_start_ms")
    raw_end = row.get("segment_end_ms")
    explicit_valid = (
        isinstance(raw_start, Integral)
        and not isinstance(raw_start, bool)
        and isinstance(raw_end, Integral)
        and not isinstance(raw_end, bool)
        and 0 <= int(raw_start) < int(raw_end) <= duration_ms
    )
    if explicit_valid:
        actual = (int(raw_start), int(raw_end))
        expected = (start_ms, end_ms)
        if actual != expected:
            raise ValueError(
                f"frame_idx={frame_idx}: explicit bounds={actual} conflict with "
                f"fixed-window bounds={expected}"
            )
        bounds_state = "valid"
    else:
        bounds_state = "invalid"
    return (
        FramePlan(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            segment_id=segment_id,
            segment_start_ms=start_ms,
            segment_end_ms=end_ms,
            model_version=model_version,
        ),
        bounds_state,
    )


def validate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    video_id: str,
    asset_version: str,
    duration_ms: int,
    segment_ms: int,
    expected_count: int,
) -> ValidatedRows:
    """Validate row identity, fixed-window math, and uniform bounds state."""
    plans: dict[int, FramePlan] = {}
    states: set[str] = set()
    models: set[str] = set()
    for row in rows:
        plan, state = _plan_for_row(
            row,
            video_id=video_id,
            asset_version=asset_version,
            duration_ms=duration_ms,
            segment_ms=segment_ms,
        )
        if plan.frame_idx in plans:
            raise ValueError(f"duplicate visual frame_idx={plan.frame_idx}")
        plans[plan.frame_idx] = plan
        states.add(state)
        models.add(plan.model_version)
    if len(plans) != int(expected_count):
        raise ValueError(
            f"{video_id}@{asset_version}: iterated rows={len(plans)} "
            f"expected={expected_count}"
        )
    if not plans:
        raise ValueError(f"{video_id}@{asset_version}: visual version is empty")
    expected_frame_ids = set(range(len(plans)))
    if set(plans) != expected_frame_ids:
        missing = sorted(expected_frame_ids - set(plans))[:10]
        extra = sorted(set(plans) - expected_frame_ids)[:10]
        raise ValueError(
            f"{video_id}@{asset_version}: frame_idx must be contiguous; "
            f"missing={missing} extra={extra}"
        )
    if len(models) != 1:
        raise ValueError(
            f"{video_id}@{asset_version}: multiple model versions={sorted(models)}"
        )
    if len(states) != 1:
        raise ValueError(
            f"{video_id}@{asset_version}: mixed valid and invalid time bounds"
        )
    return ValidatedRows(
        plans=plans,
        bounds_state=next(iter(states)),
        model_version=next(iter(models)),
    )


def _publication_source(catalog: Any, video_id: str) -> dict[str, Any]:
    publication = catalog.get_modality_publication(video_id, "visual")
    if not isinstance(publication, dict):
        raise ValueError(f"{video_id}: no Catalog visual publication")
    status = str(publication.get("status") or "").strip().casefold()
    metadata = publication.get("metadata")
    migration_state = str(
        publication.get("migration_state")
        or (metadata.get("migration_state") if isinstance(metadata, dict) else "")
        or ""
    ).strip().casefold()
    if status == "ready":
        pass
    elif status == "disabled" and migration_state == "requires_rebuild":
        pass
    else:
        raise ValueError(
            f"{video_id}: unsupported Catalog visual source status={status or '<missing>'} "
            f"migration_state={migration_state or '<missing>'}"
        )
    version = str(publication.get("asset_version") or "").strip()
    if not version:
        raise ValueError(f"{video_id}: Catalog visual asset_version is missing")
    row_count = _strict_positive_int(
        publication.get("row_count"), field="publication.row_count"
    )
    return {
        **publication,
        "asset_version": version,
        "row_count": row_count,
        "status": status,
        "migration_state": migration_state,
    }


def _copy_metadata(
    publication: dict[str, Any],
    *,
    source_version: str,
    segment_ms: int,
) -> dict[str, Any]:
    metadata = dict(publication.get("metadata") or {})
    for key in ("model_key", "embedding_space", "sample_fps", "decode_status"):
        value = publication.get(key)
        if value is not None:
            metadata.setdefault(key, value)
    metadata.update(
        {
            "segment_strategy": "fixed",
            "segment_times": "explicit",
            "segment_ms": int(segment_ms),
            "time_bounds_migration": _MIGRATION_PROTOCOL,
            "migrated_from_asset_version": source_version,
            "migration_state": "completed",
        }
    )
    return metadata


def target_asset_version(
    *, video_id: str, source_version: str, segment_ms: int
) -> str:
    """Return the stable target version for one migration input contract.

    A retry after a process or network interruption therefore upserts the same
    PKs and completes the same unpublished version instead of leaking another
    random partial version.
    """
    identity = "\0".join(
        (_MIGRATION_PROTOCOL, str(video_id), str(source_version), str(int(segment_ms)))
    )
    return uuid.uuid5(_MIGRATION_NAMESPACE, identity).hex


def migrate_visual_video(
    *,
    catalog: Any,
    client: Any,
    video: dict[str, Any],
    index_dir: Path,
    execute: bool = False,
    batch_size: int = 512,
    timeout: float = 30.0,
    version_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Validate and optionally publish one video's repaired visual version."""
    video_id = str(video.get("id") or "").strip()
    if not video_id:
        raise ValueError("Catalog video id is missing")
    duration_ms = _duration_ms(video)
    source = _publication_source(catalog, video_id)
    source_version = source["asset_version"]
    config = load_legacy_fixed_config(
        index_dir,
        video_id=video_id,
        duration_ms=duration_ms,
        publication=source,
    )
    collection = client.collection_for("visual")
    declared_count = int(source["row_count"])
    persisted_source_count = client.count_video_modality_version(
        video_id, "visual", source_version
    )
    if persisted_source_count != declared_count:
        raise RuntimeError(
            f"{video_id}@{source_version}: Catalog rows={declared_count}, "
            f"Milvus rows={persisted_source_count}"
        )
    source_rows = validate_rows(
        _iter_rows(
            collection,
            video_id=video_id,
            asset_version=source_version,
            output_fields=_META_FIELDS,
            batch_size=batch_size,
            timeout=timeout,
        ),
        video_id=video_id,
        asset_version=source_version,
        duration_ms=duration_ms,
        segment_ms=config.segment_ms,
        expected_count=declared_count,
    )
    base_result = {
        "video_id": video_id,
        "source_asset_version": source_version,
        "source_status": source["status"],
        "row_count": declared_count,
        "duration_ms": duration_ms,
        "segment_ms": config.segment_ms,
    }
    if source_rows.bounds_state == "valid":
        return {**base_result, "status": "already_valid"}
    if not execute:
        return {**base_result, "status": "dry_run_ready"}

    new_version = str(
        version_factory()
        if version_factory is not None
        else target_asset_version(
            video_id=video_id,
            source_version=source_version,
            segment_ms=config.segment_ms,
        )
    ).strip()
    if not new_version:
        raise ValueError(f"{video_id}: generated asset_version is empty")
    try:
        normalized_version = uuid.UUID(new_version).hex
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{video_id}: generated asset_version is not a UUID") from exc
    if normalized_version != new_version:
        raise ValueError(
            f"{video_id}: generated asset_version must be a 32-character UUID hex"
        )
    if new_version == source_version:
        raise ValueError(f"{video_id}: generated asset_version reuses the source version")

    copied = 0
    seen: set[int] = set()
    for batch in _row_batches(
        _iter_rows(
            collection,
            video_id=video_id,
            asset_version=source_version,
            output_fields=_COPY_FIELDS,
            batch_size=batch_size,
            timeout=timeout,
        ),
        batch_size=batch_size,
    ):
        new_rows: list[dict[str, Any]] = []
        for row in batch:
            plan, _ = _plan_for_row(
                row,
                video_id=video_id,
                asset_version=source_version,
                duration_ms=duration_ms,
                segment_ms=config.segment_ms,
            )
            expected = source_rows.plans.get(plan.frame_idx)
            if expected != plan or plan.frame_idx in seen:
                raise RuntimeError(
                    f"{video_id}: source rows changed between validation and copy "
                    f"at frame_idx={plan.frame_idx}"
                )
            embedding = row.get("embedding")
            try:
                embedding_dim = len(embedding)
            except TypeError as exc:
                raise ValueError(
                    f"{video_id}: frame_idx={plan.frame_idx} embedding is missing"
                ) from exc
            if embedding_dim != EMBEDDING_DIMS["visual"]:
                raise ValueError(
                    f"{video_id}: frame_idx={plan.frame_idx} embedding dim="
                    f"{embedding_dim}, expected={EMBEDDING_DIMS['visual']}"
                )
            seen.add(plan.frame_idx)
            new_rows.append(
                {
                    "pk": visual_pk(
                        video_id, new_version, plan.frame_idx, plan.model_version
                    ),
                    "video_id": video_id,
                    "asset_version": new_version,
                    "model_version": plan.model_version,
                    "frame_idx": plan.frame_idx,
                    "timestamp_ms": plan.timestamp_ms,
                    "segment_id": plan.segment_id,
                    "segment_start_ms": plan.segment_start_ms,
                    "segment_end_ms": plan.segment_end_ms,
                    "embedding": embedding,
                }
            )
        copied += _upsert_batched(collection, new_rows, "visual")
    if copied != declared_count or seen != set(source_rows.plans):
        raise RuntimeError(
            f"{video_id}: copied rows={copied}, expected={declared_count}"
        )
    collection.flush()

    persisted_new_count = client.count_video_modality_version(
        video_id, "visual", new_version
    )
    if persisted_new_count != declared_count:
        raise RuntimeError(
            f"{video_id}@{new_version}: copied rows={declared_count}, "
            f"persisted rows={persisted_new_count}"
        )
    new_rows = validate_rows(
        _iter_rows(
            collection,
            video_id=video_id,
            asset_version=new_version,
            output_fields=_META_FIELDS,
            batch_size=batch_size,
            timeout=timeout,
        ),
        video_id=video_id,
        asset_version=new_version,
        duration_ms=duration_ms,
        segment_ms=config.segment_ms,
        expected_count=declared_count,
    )
    if new_rows.bounds_state != "valid" or new_rows.plans != source_rows.plans:
        raise RuntimeError(f"{video_id}@{new_version}: post-write validation failed")

    latest = _publication_source(catalog, video_id)
    if (
        latest["status"] != source["status"]
        or
        latest["asset_version"] != source_version
        or int(latest["row_count"]) != declared_count
    ):
        raise RuntimeError(
            f"{video_id}: Catalog visual pointer changed during migration; refusing publish"
        )
    catalog.publish_modality(
        video_id,
        "visual",
        asset_version=new_version,
        row_count=declared_count,
        status="ready",
        metadata=_copy_metadata(
            source,
            source_version=source_version,
            segment_ms=config.segment_ms,
        ),
    )
    return {
        **base_result,
        "status": "migrated",
        "asset_version": new_version,
    }


def _row_batches(
    rows: Iterable[dict[str, Any]], *, batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def migrate(
    *,
    catalog: Any,
    client: Any,
    index_root: Path,
    execute: bool = False,
    video_ids: set[str] | None = None,
    batch_size: int = 512,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run the migration per video; errors never change that video's pointer."""
    from app.vector_store.milvus.milvus_stage_lock import video_stage_lock

    report: dict[str, Any] = {
        "execute": bool(execute),
        "migrated": [],
        "already_valid": [],
        "dry_run_ready": [],
        "errors": [],
    }
    selected = set(video_ids or [])
    found: set[str] = set()
    for video in catalog.list_videos():
        video_id = str(video.get("id") or "")
        if selected and video_id not in selected:
            continue
        found.add(video_id)
        index_dir = index_root / video_id
        try:
            with video_stage_lock(index_dir, video_id=video_id, stage="publish"):
                result = migrate_visual_video(
                    catalog=catalog,
                    client=client,
                    video=video,
                    index_dir=index_dir,
                    execute=execute,
                    batch_size=batch_size,
                    timeout=timeout,
                )
            report[result["status"]].append(result)
        except Exception as exc:
            logger.exception("Visual time migration failed for %s", video_id)
            report["errors"].append({"video_id": video_id, "error": str(exc)})
    for missing in sorted(selected - found):
        report["errors"].append(
            {"video_id": missing, "error": "video does not exist in Catalog"}
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Milvus-to-Milvus repair of legacy fixed visual time bounds"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write and publish verified UUID versions; default is a read-only dry run.",
    )
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    settings: Settings = get_settings()
    if not settings.milvus_enabled:
        print("[ERROR] MILVUS_ENABLED=true is required", file=sys.stderr)
        return 2
    if args.execute and not settings.milvus_write_enabled:
        print("[ERROR] --execute requires MILVUS_WRITE_ENABLED=true", file=sys.stderr)
        return 2
    from app.vector_store.milvus.milvus_client import get_milvus_client

    report = migrate(
        catalog=Catalog(settings.db_path),
        client=get_milvus_client(),
        index_root=settings.index_dir,
        execute=bool(args.execute),
        video_ids=set(args.video_ids) if args.video_ids else None,
        batch_size=int(args.batch_size),
        timeout=float(settings.milvus_query_timeout_seconds),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
