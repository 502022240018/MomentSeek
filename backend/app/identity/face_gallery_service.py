from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.settings import get_settings
from app.media.media import extract_frame
from app.vector_store.milvus.milvus_client import get_milvus_client
from app.vector_store.milvus.milvus_schema import entity_face_sample_pk


GROUP_FIELDS = [
    "group_idx", "representative_track_idx", "start_ms", "end_ms", "best_ms",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "representative_quality",
    "duration_ms", "occurrence_count", "importance_score",
]


class FaceGroupMigrationRequired(RuntimeError):
    """Face tracks are published but no read-only group generation is active."""


@dataclass(frozen=True)
class PublishedFaceGeneration:
    asset_version: str
    group_version: str
    group_row_count: int


def _query_all(collection, *, expr: str, output_fields: list[str]) -> list[dict]:
    """Read every matching Milvus row without the 16,384-row query cap."""
    timeout = get_settings().milvus_query_timeout_seconds
    if not hasattr(collection, "query_iterator"):
        return collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=16_384,
            timeout=timeout,
        )
    rows: list[dict] = []
    iterator = collection.query_iterator(
        batch_size=2000,
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
            rows.extend(page)
    finally:
        iterator.close()
    return rows


def published_face_generation(catalog, video_id: str) -> PublishedFaceGeneration:
    publication = catalog.get_modality_publication(video_id, "face")
    if not publication or publication.get("status") != "ready":
        raise FileNotFoundError("该视频尚未发布 Face Milvus 索引")
    group_version = str(publication.get("group_version") or "").strip()
    group_row_count = publication.get("group_row_count")
    if not group_version or group_row_count is None:
        raise FaceGroupMigrationRequired("该视频的人物分组尚待一次性迁移")
    try:
        count = int(group_row_count)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Face publication 的 group_row_count 无效") from exc
    if count < 0:
        raise RuntimeError("Face publication 的 group_row_count 不能为负数")
    return PublishedFaceGeneration(
        asset_version=str(publication["asset_version"]),
        group_version=group_version,
        group_row_count=count,
    )


def _expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _group_expr(video_id: str, asset_version: str, group_version: str) -> str:
    return (
        f'video_id == "{_expr_value(video_id)}" and '
        f'asset_version == "{_expr_value(asset_version)}" and '
        f'model_version == "{_expr_value(group_version)}"'
    )


def video_face_groups(
    catalog,
    video_id: str,
    *,
    limit: int,
    min_duration_ms: int,
    min_occurrence_count: int,
) -> dict:
    if limit <= 0 or min_duration_ms < 0 or min_occurrence_count <= 0:
        raise ValueError("Face gallery display parameters are invalid")
    generation = published_face_generation(catalog, video_id)
    rows = _query_all(
        get_milvus_client().collection("face_groups"),
        expr=_group_expr(
            video_id,
            generation.asset_version,
            generation.group_version,
        ),
        output_fields=GROUP_FIELDS,
    )
    if len(rows) != generation.group_row_count:
        raise RuntimeError(
            "Face group publication mismatch: "
            f"expected={generation.group_row_count} persisted={len(rows)}"
        )
    indices: list[int] = []
    for row in rows:
        try:
            group_idx = int(row["group_idx"])
            start_ms = int(row["start_ms"])
            end_ms = int(row["end_ms"])
            best_ms = int(row["best_ms"])
            duration_ms = int(row["duration_ms"])
            occurrence_count = int(row["occurrence_count"])
            importance_score = float(row["importance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Face group row has invalid required fields") from exc
        if (
            group_idx < 0
            or start_ms < 0
            or end_ms <= start_ms
            or not start_ms <= best_ms <= end_ms
            or duration_ms < 0
            or occurrence_count <= 0
            or not np.isfinite(importance_score)
        ):
            raise RuntimeError(f"Face group row is invalid: group_idx={group_idx}")
        indices.append(group_idx)
    if sorted(indices) != list(range(generation.group_row_count)):
        raise RuntimeError("Face group publication has duplicate or missing group_idx")
    rows.sort(key=lambda row: (
        -float(row.get("importance_score", 0)),
        int(row.get("start_ms", 0)),
        int(row["group_idx"]),
    ))
    eligible = [
        row for row in rows
        if int(row.get("duration_ms", 0)) >= min_duration_ms
        or int(row.get("occurrence_count", 0)) >= min_occurrence_count
    ]
    displayed = eligible[:limit]
    bindings = catalog.face_identity_bindings(
        video_id,
        generation.asset_version,
        generation.group_version,
    )
    for row in displayed:
        group_idx = int(row["group_idx"])
        row["thumbnail_url"] = (
            f"/api/videos/{video_id}/face-gallery/{group_idx}/thumbnail"
            f"?asset_version={generation.asset_version}"
            f"&group_version={generation.group_version}"
        )
        row["media_url"] = (
            f"/api/videos/{video_id}/media#t="
            f"{max(0, int(row['best_ms'])) / 1000:.3f}"
        )
        binding = bindings.get(group_idx)
        row["entity_id"] = binding["entity_id"] if binding else None
        row["entity_name"] = binding["entity_name"] if binding else None
    return {
        "video_id": video_id,
        "asset_version": generation.asset_version,
        "group_version": generation.group_version,
        "total_group_count": len(rows),
        "eligible_group_count": len(eligible),
        "displayed_group_count": len(displayed),
        "groups": displayed,
    }


def get_face_group(
    video_id: str,
    asset_version: str,
    group_version: str,
    group_idx: int,
    *,
    embedding: bool = False,
) -> dict | None:
    fields = [*GROUP_FIELDS, *(["embedding"] if embedding else [])]
    rows = get_milvus_client().collection("face_groups").query(
        expr=(
            f"{_group_expr(video_id, asset_version, group_version)} and "
            f"group_idx == {int(group_idx)}"
        ),
        output_fields=fields,
        limit=1,
        timeout=get_settings().milvus_query_timeout_seconds,
    )
    return rows[0] if rows else None


def ensure_group_thumbnail(
    settings,
    video: dict,
    asset_version: str,
    group_version: str,
    group: dict,
) -> Path:
    group_idx = int(group["group_idx"])
    version_key = hashlib.sha256(group_version.encode("utf-8")).hexdigest()[:12]
    destination = (
        settings.frame_cache_dir / video["id"] / "face-gallery"
        / asset_version / version_key / f"{group_idx:06d}.jpg"
    )
    if destination.is_file() and destination.stat().st_size:
        return destination
    raw_path = destination.with_name(f"{group_idx:06d}.source.jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    extract_frame(
        settings.resolve_path(video["file_path"]),
        raw_path,
        int(group["best_ms"]),
    )
    frame = cv2.imread(str(raw_path))
    if frame is None:
        raise RuntimeError("无法读取人物代表帧")
    bbox = np.asarray([
        group.get(f"bbox_{axis}", -1.0)
        for axis in ("x1", "y1", "x2", "y2")
    ])
    if (
        np.all((bbox >= 0) & (bbox <= 1))
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    ):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox * np.asarray([width, height, width, height])
        pad = 0.22 * max(x2 - x1, y2 - y1)
        crop = frame[
            max(0, int(y1 - pad)):min(height, int(y2 + pad)),
            max(0, int(x1 - pad)):min(width, int(x2 + pad)),
        ]
        if crop.size:
            frame = crop
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError("无法写入人物代表图")
    raw_path.unlink(missing_ok=True)
    return destination


def attach_group_to_entity(
    catalog,
    video_id: str,
    asset_version: str,
    group_version: str,
    group_idx: int,
    entity_id: str,
) -> dict:
    entity = catalog.get_entity(entity_id)
    if not entity:
        raise KeyError("人物不存在")
    group = get_face_group(
        video_id,
        asset_version,
        group_version,
        group_idx,
        embedding=True,
    )
    if not group:
        raise KeyError("人脸分组不存在或已过期")
    sample_id = hashlib.sha256(
        f"{video_id}\0{asset_version}\0{group_version}\0{group_idx}".encode("utf-8")
    ).hexdigest()[:32]
    row = {
        "pk": entity_face_sample_pk(entity_id, sample_id),
        "entity_id": entity_id,
        "sample_id": sample_id,
        "source_video_id": video_id,
        "source_asset_version": asset_version,
        "source_group_idx": group_idx,
        "quality": float(group.get("representative_quality", 0)),
        "embedding": list(group["embedding"]),
    }
    collection = get_milvus_client().collection("entity_face_samples")
    # sample_id includes the immutable group generation. Rebinding the current
    # group replaces only that generation's sample and never deletes a sample
    # learned from an older grouping of the same Face asset/group_idx.
    collection.delete(f'sample_id == "{_expr_value(sample_id)}"')
    collection.upsert([row])
    collection.flush()
    catalog.bind_face_identity(
        video_id,
        asset_version,
        group_version,
        group_idx,
        entity_id,
    )
    return {
        "sample_id": sample_id,
        "entity_id": entity_id,
        "entity_name": entity["name"],
    }


def delete_entity_face_samples(entity_id: str) -> int:
    collection = get_milvus_client().collection("entity_face_samples")
    result = collection.delete(f'entity_id == "{_expr_value(entity_id)}"')
    collection.flush()
    return int(getattr(result, "delete_count", 0))


def copy_thumbnail_as_reference(thumbnail: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(thumbnail, destination)
