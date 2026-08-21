"""Publish versioned Face groups for legacy Milvus-only Face track indexes.

This is a one-time operational migration, not an online fallback. It is a
dry-run by default and never changes Face track rows or their asset version.
``--apply`` writes a deterministic derived group generation, validates it, and
then adds the group pointer to the existing ready Face Catalog publication.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.catalog.db import Catalog
from app.core.settings import Settings, get_settings
from app.encoders.face import FaceEncoder
from app.identity.face_gallery import (
    FACE_GROUP_ALGORITHM_VERSION,
    FaceGroup,
    cluster_face_tracks,
    face_group_arrays,
    face_group_model_version,
    select_major_face_groups,
)
from app.indexing.common import normalize
from app.indexing.modalities.face.faces import face_detection_quality
from app.media.media import extract_frame
from app.vector_store.milvus.milvus_client import ExistingMilvusCollectionsClient
from app.vector_store.milvus.milvus_indexer import (
    MilvusWriteContext,
    upsert_face_group_rows,
)
from app.vector_store.milvus.milvus_stage_lock import video_stage_lock


logger = logging.getLogger(__name__)

_TRACK_FIELDS = ("track_idx", "start_ms", "end_ms", "best_ms", "embedding")
_LEGACY_SOURCE_FIELDS = ("asset_version", "model_version", *_TRACK_FIELDS)
_GROUP_VERIFY_FIELDS = (
    "group_idx", "representative_track_idx", "start_ms", "end_ms", "best_ms",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "representative_quality",
    "duration_ms", "occurrence_count", "importance_score", "embedding",
)


def _expr(video_id: str, asset_version: str, group_version: str | None = None) -> str:
    parts = [
        f"video_id == {json.dumps(video_id, ensure_ascii=False)}",
        f"asset_version == {json.dumps(asset_version, ensure_ascii=False)}",
    ]
    if group_version is not None:
        parts.append(
            f"model_version == {json.dumps(group_version, ensure_ascii=False)}"
        )
    return " and ".join(parts)


def _iter_rows(
    collection: Any,
    *,
    expr: str,
    output_fields: Iterable[str],
    batch_size: int,
    timeout: float,
) -> Iterator[dict[str, Any]]:
    query_iterator = getattr(collection, "query_iterator", None)
    if not callable(query_iterator):
        raise RuntimeError("Milvus query_iterator is required for complete Face migration")
    iterator = query_iterator(
        expr=expr,
        output_fields=list(output_fields),
        batch_size=batch_size,
        timeout=timeout,
    )
    try:
        while True:
            page = iterator.next()
            if not page:
                break
            for row in page:
                if not isinstance(row, dict):
                    raise ValueError(f"Milvus returned a non-object Face row: {row!r}")
                yield row
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Face row has invalid {field}: {value!r}")
    return int(value)


def _load_tracks(
    client: Any,
    *,
    video_id: str,
    asset_version: str,
    expected_count: int,
    batch_size: int,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = list(_iter_rows(
        client.collection_for("face"),
        expr=_expr(video_id, asset_version),
        output_fields=_TRACK_FIELDS,
        batch_size=batch_size,
        timeout=timeout,
    ))
    if len(rows) != expected_count:
        raise RuntimeError(
            f"{video_id}@{asset_version}: Catalog Face rows={expected_count}, "
            f"iterated rows={len(rows)}"
        )
    rows.sort(key=lambda row: _required_int(row, "track_idx"))
    observed = [_required_int(row, "track_idx") for row in rows]
    if observed != list(range(expected_count)):
        raise ValueError(f"{video_id}@{asset_version}: track_idx is not contiguous")
    vectors: list[np.ndarray] = []
    times: list[list[int]] = []
    for row in rows:
        start_ms = _required_int(row, "start_ms")
        end_ms = _required_int(row, "end_ms")
        best_ms = _required_int(row, "best_ms")
        if start_ms < 0 or end_ms <= start_ms or not start_ms <= best_ms <= end_ms:
            raise ValueError(
                f"{video_id}: track_idx={row['track_idx']} has invalid time bounds"
            )
        vector = np.asarray(row.get("embedding"), dtype=np.float32)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError(
                f"{video_id}: track_idx={row['track_idx']} has invalid embedding"
            )
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError(
                f"{video_id}: track_idx={row['track_idx']} has zero embedding"
            )
        vectors.append(vector / norm)
        times.append([start_ms, end_ms, best_ms])
    return (
        np.asarray(vectors, dtype=np.float32).reshape((-1, 512)),
        np.asarray(times, dtype=np.int64).reshape((-1, 3)),
    )


def _discover_legacy_face_source(
    client: Any,
    *,
    video_id: str,
    model_key: str,
    batch_size: int,
    timeout: float,
) -> dict[str, Any]:
    rows = list(_iter_rows(
        client.collection_for("face"),
        expr=f"video_id == {json.dumps(video_id, ensure_ascii=False)}",
        output_fields=_LEGACY_SOURCE_FIELDS,
        batch_size=batch_size,
        timeout=timeout,
    ))
    if not rows:
        raise ValueError(f"{video_id}: no legacy Face track rows")
    asset_versions = {
        str(row.get("asset_version") or "").strip()
        for row in rows
    }
    model_versions = {
        str(row.get("model_version") or "").strip()
        for row in rows
    }
    if "" in asset_versions or len(asset_versions) != 1:
        raise ValueError(
            f"{video_id}: legacy Face asset version is ambiguous: "
            f"{sorted(asset_versions)}"
        )
    if "" in model_versions or len(model_versions) != 1:
        raise ValueError(
            f"{video_id}: legacy Face model version is ambiguous: "
            f"{sorted(model_versions)}"
        )
    asset_version = next(iter(asset_versions))
    persisted = client.count_video_modality_version(
        video_id,
        "face",
        asset_version,
    )
    if persisted != len(rows):
        raise RuntimeError(
            f"{video_id}@{asset_version}: iterated Face rows={len(rows)}, "
            f"Milvus rows={persisted}"
        )
    return {
        "status": "legacy-unpublished",
        "asset_version": asset_version,
        "row_count": len(rows),
        "metadata": {
            "model_key": model_key,
            "embedding_space": "arcface-identity",
            "legacy_track_model_version": next(iter(model_versions)),
        },
    }


def refine_group_representatives(
    groups: list[FaceGroup],
    *,
    frame_at_ms: Callable[[int], np.ndarray],
    encoder: Any,
    identity_threshold: float,
    max_groups: int,
    min_duration_ms: int,
    min_occurrence_count: int,
) -> tuple[list[FaceGroup], int]:
    """Target only major representative frames; never rescan the whole video."""
    selected = select_major_face_groups(
        groups,
        limit=max_groups,
        min_duration_ms=min_duration_ms,
        min_occurrence_count=min_occurrence_count,
    )
    replacements: dict[int, FaceGroup] = {}
    for group in selected:
        frame = frame_at_ms(group.best_ms)
        if frame is None or not getattr(frame, "size", 0):
            continue
        height, width = frame.shape[:2]
        matches = []
        for face in encoder.detect(frame):
            embedding = normalize(np.asarray(face.normed_embedding, dtype=np.float32))
            matches.append((float(embedding @ group.embedding), face))
        score, face = max(matches, key=lambda item: item[0], default=(-1.0, None))
        if face is None or score < identity_threshold:
            continue
        bbox = np.asarray(face.bbox, dtype=np.float32)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        normalized_bbox = tuple(float(value) for value in (
            bbox / np.asarray([width, height, width, height], dtype=np.float32)
        ))
        quality = face_detection_quality(face, frame, bbox)
        replacements[group.group_idx] = replace(
            group,
            bbox=normalized_bbox,
            quality=quality,
            importance_score=(
                group.importance_score
                - 0.20 * group.quality
                + 0.20 * float(np.clip(quality, 0.0, 1.0))
            ),
        )
    return [replacements.get(group.group_idx, group) for group in groups], len(replacements)


def _refine_from_video(
    groups: list[FaceGroup],
    *,
    video_path: Path,
    settings: Settings,
    encoder: Any,
) -> tuple[list[FaceGroup], int]:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="face-group-migration-",
        dir=settings.index_dir,
    ) as directory:
        scratch = Path(directory)

        def frame_at_ms(best_ms: int) -> np.ndarray:
            path = scratch / f"{best_ms}.jpg"
            extract_frame(video_path, path, best_ms, max_width=960)
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"无法读取迁移代表帧: {best_ms}ms")
            return frame

        return refine_group_representatives(
            groups,
            frame_at_ms=frame_at_ms,
            encoder=encoder,
            identity_threshold=settings.face_identity_threshold,
            max_groups=settings.face_gallery_max_groups,
            min_duration_ms=int(round(
                settings.face_gallery_min_duration_seconds * 1000
            )),
            min_occurrence_count=settings.face_gallery_min_occurrences,
        )


def _verify_group_generation(
    client: Any,
    *,
    video_id: str,
    asset_version: str,
    group_version: str,
    expected_count: int,
    batch_size: int,
    timeout: float,
) -> None:
    persisted = client.count_face_groups_version(
        video_id,
        asset_version,
        group_version,
    )
    if persisted != expected_count:
        raise RuntimeError(
            f"{video_id}: Face group rows={persisted}, expected={expected_count}"
        )
    rows = list(_iter_rows(
        client.collection("face_groups"),
        expr=_expr(video_id, asset_version, group_version),
        output_fields=_GROUP_VERIFY_FIELDS,
        batch_size=batch_size,
        timeout=timeout,
    ))
    if len(rows) != expected_count:
        raise RuntimeError(f"{video_id}: incomplete Face group verification scan")
    indices = sorted(_required_int(row, "group_idx") for row in rows)
    if indices != list(range(expected_count)):
        raise RuntimeError(f"{video_id}: Face group_idx generation is incomplete")
    for row in rows:
        track_idx = _required_int(row, "representative_track_idx")
        start_ms = _required_int(row, "start_ms")
        end_ms = _required_int(row, "end_ms")
        best_ms = _required_int(row, "best_ms")
        duration_ms = _required_int(row, "duration_ms")
        occurrence_count = _required_int(row, "occurrence_count")
        bbox = np.asarray([
            row.get("bbox_x1"),
            row.get("bbox_y1"),
            row.get("bbox_x2"),
            row.get("bbox_y2"),
        ], dtype=np.float32)
        quality = float(row.get("representative_quality"))
        importance = float(row.get("importance_score"))
        vector = np.asarray(row.get("embedding"), dtype=np.float32)
        missing_bbox = bool(np.all(bbox == -1.0))
        valid_bbox = bool(
            np.isfinite(bbox).all()
            and np.all((bbox >= 0.0) & (bbox <= 1.0))
            and bbox[2] > bbox[0]
            and bbox[3] > bbox[1]
        )
        if (
            track_idx < 0
            or start_ms < 0
            or end_ms <= start_ms
            or not start_ms <= best_ms <= end_ms
            or duration_ms <= 0
            or occurrence_count <= 0
            or not missing_bbox and not valid_bbox
            or not math.isfinite(quality)
            or not 0.0 <= quality <= 1.0
            or not math.isfinite(importance)
            or vector.shape != (512,)
            or not np.isfinite(vector).all()
            or float(np.linalg.norm(vector)) <= 1e-12
        ):
            raise RuntimeError(f"{video_id}: invalid persisted Face group row")


def migrate_face_groups_video(
    *,
    catalog: Any,
    client: Any,
    video: dict[str, Any],
    settings: Settings,
    apply: bool = False,
    replace_existing: bool = False,
    bootstrap_legacy_publication: bool = False,
    refine_representatives: bool = True,
    batch_size: int = 512,
    timeout: float = 30.0,
    encoder_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    video_id = str(video.get("id") or "").strip()
    if not video_id:
        raise ValueError("Catalog video id is missing")
    source = catalog.get_modality_publication(video_id, "face")
    source_was_published = bool(source and source.get("status") == "ready")
    if not source_was_published:
        if not bootstrap_legacy_publication:
            raise ValueError(f"{video_id}: no ready Face publication")
        source = _discover_legacy_face_source(
            client,
            video_id=video_id,
            model_key=settings.face_model,
            batch_size=batch_size,
            timeout=timeout,
        )
    assert source is not None
    asset_version = str(source["asset_version"])
    track_count = int(source["row_count"])
    group_version = face_group_model_version(
        settings.face_gallery_cosine_threshold
    )
    if (
        source_was_published
        and source.get("group_version") == group_version
        and replace_existing
    ):
        raise ValueError(
            f"{video_id}: refusing to replace the published immutable "
            f"Face group generation {group_version}"
        )
    if (
        not replace_existing
        and source.get("group_version") == group_version
        and source.get("group_row_count") is not None
    ):
        _verify_group_generation(
            client,
            video_id=video_id,
            asset_version=asset_version,
            group_version=group_version,
            expected_count=int(source["group_row_count"]),
            batch_size=batch_size,
            timeout=timeout,
        )
        return {
            "video_id": video_id,
            "asset_version": asset_version,
            "group_version": group_version,
            "status": "already_current",
            "group_row_count": int(source["group_row_count"]),
        }

    persisted_tracks = client.count_video_modality_version(
        video_id,
        "face",
        asset_version,
    )
    if persisted_tracks != track_count:
        raise RuntimeError(
            f"{video_id}: Catalog Face rows={track_count}, "
            f"Milvus rows={persisted_tracks}"
        )
    embeddings, times = _load_tracks(
        client,
        video_id=video_id,
        asset_version=asset_version,
        expected_count=track_count,
        batch_size=batch_size,
        timeout=timeout,
    )
    groups = cluster_face_tracks(
        embeddings,
        times,
        cosine_threshold=settings.face_gallery_cosine_threshold,
    )
    result = {
        "video_id": video_id,
        "asset_version": asset_version,
        "group_version": group_version,
        "track_row_count": track_count,
        "group_row_count": len(groups),
        "publication_bootstrapped": not source_was_published,
    }
    if not apply:
        return {**result, "status": "dry_run_ready"}

    refined_count = 0
    if refine_representatives and groups:
        factory = encoder_factory or (lambda: FaceEncoder(
            settings.face_model,
            settings.face_provider,
            settings.npu_device_id,
            str(settings.app_model_dir / "insightface"),
            settings.face_ort_intra_op_threads,
            settings.face_ort_inter_op_threads,
        ))
        groups, refined_count = _refine_from_video(
            groups,
            video_path=settings.resolve_path(video["file_path"]),
            settings=settings,
            encoder=factory(),
        )
    arrays = face_group_arrays(groups)
    ctx = MilvusWriteContext(
        video_id=video_id,
        asset_version=asset_version,
        client=client,
    )
    replaced_group_rows = 0
    if replace_existing:
        collection = client.collection("face_groups")
        deleted = collection.delete(_expr(video_id, asset_version, group_version))
        collection.flush()
        replaced_group_rows = int(getattr(deleted, "delete_count", 0))
        remaining = client.count_face_groups_version(
            video_id,
            asset_version,
            group_version,
        )
        if remaining:
            raise RuntimeError(
                f"{video_id}: failed to clear unpublished Face group generation; "
                f"remaining={remaining}"
            )
    written = upsert_face_group_rows(
        ctx,
        group_model_version=group_version,
        **arrays,
    )
    if written != len(groups):
        raise RuntimeError(
            f"{video_id}: wrote Face groups={written}, expected={len(groups)}"
        )
    _verify_group_generation(
        client,
        video_id=video_id,
        asset_version=asset_version,
        group_version=group_version,
        expected_count=len(groups),
        batch_size=batch_size,
        timeout=timeout,
    )
    latest = catalog.get_modality_publication(video_id, "face")
    publication_changed = (
        source_was_published
        and (
            not latest
            or latest.get("status") != "ready"
            or str(latest.get("asset_version")) != asset_version
            or int(latest.get("row_count", -1)) != track_count
        )
    ) or (not source_was_published and latest is not None)
    if publication_changed:
        raise RuntimeError(
            f"{video_id}: Face publication changed during migration; refusing publish"
        )
    metadata = dict(source.get("metadata") or {})
    metadata.update({
        "group_version": group_version,
        "group_row_count": len(groups),
        "group_algorithm": FACE_GROUP_ALGORITHM_VERSION,
        "group_cosine_threshold": settings.face_gallery_cosine_threshold,
        "group_source": "legacy-face-tracks",
        "group_refined_representatives": refined_count,
        "publication_bootstrapped": not source_was_published,
    })
    catalog.publish_modality(
        video_id,
        "face",
        asset_version=asset_version,
        row_count=track_count,
        status="ready",
        metadata=metadata,
    )
    return {
        **result,
        "status": "migrated",
        "refined_representatives": refined_count,
        "replaced_group_rows": replaced_group_rows,
    }


def run_migration(
    *,
    settings: Settings,
    catalog: Catalog,
    client: Any,
    video_id: str | None,
    apply: bool,
    replace_existing: bool,
    bootstrap_legacy_publication: bool,
    refine_representatives: bool,
) -> dict[str, Any]:
    if bootstrap_legacy_publication and not video_id:
        raise ValueError("--bootstrap-legacy-publication requires --video-id")
    if video_id:
        video = catalog.get_video(video_id)
        if not video:
            raise ValueError(f"video not found: {video_id}")
        videos = [video]
    else:
        videos = catalog.list_videos()
    report: dict[str, Any] = {"apply": apply, "results": [], "errors": []}
    encoder_holder: list[Any] = []

    def shared_encoder() -> Any:
        if not encoder_holder:
            encoder_holder.append(FaceEncoder(
                settings.face_model,
                settings.face_provider,
                settings.npu_device_id,
                str(settings.app_model_dir / "insightface"),
                settings.face_ort_intra_op_threads,
                settings.face_ort_inter_op_threads,
            ))
        return encoder_holder[0]

    for video in videos:
        publication = catalog.get_modality_publication(video["id"], "face")
        if (
            (not publication or publication.get("status") != "ready")
            and not bootstrap_legacy_publication
        ):
            continue
        try:
            with video_stage_lock(
                settings.index_dir / video["id"],
                video_id=video["id"],
                stage="publish",
            ):
                report["results"].append(migrate_face_groups_video(
                    catalog=catalog,
                    client=client,
                    video=video,
                    settings=settings,
                    apply=apply,
                    replace_existing=replace_existing,
                    bootstrap_legacy_publication=bootstrap_legacy_publication,
                    refine_representatives=refine_representatives,
                    encoder_factory=shared_encoder,
                ))
        except Exception as exc:
            logger.exception("Face group migration failed for %s", video["id"])
            report["errors"].append({"video_id": video["id"], "error": str(exc)})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write and publish groups")
    parser.add_argument("--video-id")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="clear only an unpublished target generation before rebuilding",
    )
    parser.add_argument(
        "--bootstrap-legacy-publication",
        action="store_true",
        help="explicitly bootstrap one unpublished legacy Face asset",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="skip targeted representative-frame detection",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    client = ExistingMilvusCollectionsClient(("face_embeddings", "face_groups"))
    try:
        report = run_migration(
            settings=settings,
            catalog=Catalog(settings.db_path),
            client=client,
            video_id=args.video_id,
            apply=args.apply,
            replace_existing=args.replace,
            bootstrap_legacy_publication=args.bootstrap_legacy_publication,
            refine_representatives=not args.skip_refine,
        )
    finally:
        client.close()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
