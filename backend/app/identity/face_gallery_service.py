from __future__ import annotations

import shutil
import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.identity.face_gallery import cluster_face_tracks
from app.indexing.manifest import load_index_manifest
from app.media.media import extract_frame
from app.vector_store.milvus.milvus_client import get_milvus_client
from app.vector_store.milvus.milvus_schema import (
    MODEL_VERSIONS,
    entity_face_sample_pk,
    face_group_pk,
)


GROUP_FIELDS = [
    "group_idx", "representative_track_idx", "start_ms", "end_ms", "best_ms",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "representative_quality",
    "duration_ms", "occurrence_count", "importance_score",
]


def published_face_version(index_dir: Path, video_id: str) -> str:
    manifest = load_index_manifest(index_dir / video_id) or {}
    channel = (manifest.get("channels") or {}).get("face") or {}
    version = channel.get("milvus_asset_version")
    if version is None:
        raise FileNotFoundError("该视频尚未发布 Face Milvus 索引")
    return str(version)


def _expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _group_expr(video_id: str, asset_version: str) -> str:
    return (
        f'video_id == "{_expr_value(video_id)}" and '
        f'asset_version == "{_expr_value(asset_version)}"'
    )


def _upsert_group_rows(video_id: str, asset_version: str, groups) -> None:
    if not groups:
        return
    client = get_milvus_client()
    model_version = MODEL_VERSIONS["face"]
    rows = [{
        "pk": face_group_pk(video_id, asset_version, group.group_idx, model_version),
        "video_id": video_id,
        "asset_version": asset_version,
        "model_version": model_version,
        "group_idx": group.group_idx,
        "representative_track_idx": group.representative_track_idx,
        "start_ms": group.start_ms,
        "end_ms": group.end_ms,
        "best_ms": group.best_ms,
        "bbox_x1": group.bbox[0], "bbox_y1": group.bbox[1],
        "bbox_x2": group.bbox[2], "bbox_y2": group.bbox[3],
        "representative_quality": group.quality,
        "duration_ms": group.duration_ms,
        "occurrence_count": group.occurrence_count,
        "importance_score": group.importance_score,
        "embedding": group.embedding.tolist(),
    } for group in groups]
    collection = client.collection("face_groups")
    collection.upsert(rows)
    collection.flush()


def ensure_video_face_groups(video_id: str, asset_version: str, threshold: float) -> bool:
    """Backfill legacy published face tracks directly from Milvus, without NPZ."""
    client = get_milvus_client()
    expr = _group_expr(video_id, asset_version)
    existing = client.collection("face_groups").query(expr=expr, output_fields=["count(*)"])
    if existing and int(existing[0].get("count(*)", 0)):
        return False
    tracks = client.collection_for("face").query(
        expr=expr,
        output_fields=["track_idx", "start_ms", "end_ms", "best_ms", "embedding"],
        limit=16384,
    )
    tracks.sort(key=lambda row: int(row["track_idx"]))
    if not tracks:
        return False
    groups = cluster_face_tracks(
        np.asarray([row["embedding"] for row in tracks], dtype=np.float32),
        np.asarray([[row["start_ms"], row["end_ms"], row["best_ms"]] for row in tracks], dtype=np.int64),
        cosine_threshold=threshold,
    )
    _upsert_group_rows(video_id, asset_version, groups)
    return True


def video_face_groups(index_dir: Path, catalog, video_id: str, threshold: float) -> dict:
    asset_version = published_face_version(index_dir, video_id)
    backfilled = ensure_video_face_groups(video_id, asset_version, threshold)
    rows = get_milvus_client().collection("face_groups").query(
        expr=_group_expr(video_id, asset_version), output_fields=GROUP_FIELDS, limit=16384
    )
    rows.sort(key=lambda row: (-float(row.get("importance_score", 0)), int(row["group_idx"])))
    bindings = catalog.face_identity_bindings(video_id, asset_version)
    for row in rows:
        group_idx = int(row["group_idx"])
        row["thumbnail_url"] = f"/api/videos/{video_id}/face-gallery/{group_idx}/thumbnail?asset_version={asset_version}"
        row["media_url"] = f"/api/videos/{video_id}/media#t={max(0, int(row['best_ms'])) / 1000:.3f}"
        binding = bindings.get(group_idx)
        row["entity_id"] = binding["entity_id"] if binding else None
        row["entity_name"] = binding["entity_name"] if binding else None
    return {
        "video_id": video_id,
        "asset_version": asset_version,
        "groups": rows,
        "legacy_backfilled": backfilled,
    }


def get_face_group(video_id: str, asset_version: str, group_idx: int, *, embedding: bool = False) -> dict | None:
    fields = [*GROUP_FIELDS, *( ["embedding"] if embedding else [])]
    rows = get_milvus_client().collection("face_groups").query(
        expr=f'{_group_expr(video_id, asset_version)} and group_idx == {int(group_idx)}',
        output_fields=fields,
        limit=1,
    )
    return rows[0] if rows else None


def ensure_group_thumbnail(settings, video: dict, asset_version: str, group: dict) -> Path:
    group_idx = int(group["group_idx"])
    destination = settings.frame_cache_dir / video["id"] / "face-gallery" / asset_version / f"{group_idx:06d}.jpg"
    if destination.is_file() and destination.stat().st_size:
        return destination
    raw_path = destination.with_name(f"{group_idx:06d}.source.jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    extract_frame(settings.resolve_path(video["file_path"]), raw_path, int(group["best_ms"]))
    frame = cv2.imread(str(raw_path))
    if frame is None:
        raise RuntimeError("无法读取人物代表帧")
    bbox = np.asarray([group.get(f"bbox_{axis}", -1.0) for axis in ("x1", "y1", "x2", "y2")])
    if np.all((bbox >= 0) & (bbox <= 1)) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox * np.asarray([width, height, width, height])
        pad = 0.22 * max(x2 - x1, y2 - y1)
        crop = frame[max(0, int(y1 - pad)):min(height, int(y2 + pad)), max(0, int(x1 - pad)):min(width, int(x2 + pad))]
        if crop.size:
            frame = crop
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError("无法写入人物代表图")
    raw_path.unlink(missing_ok=True)
    return destination


def attach_group_to_entity(catalog, video_id: str, asset_version: str, group_idx: int, entity_id: str) -> dict:
    entity = catalog.get_entity(entity_id)
    if not entity:
        raise KeyError("人物不存在")
    group = get_face_group(video_id, asset_version, group_idx, embedding=True)
    if not group:
        raise KeyError("人脸分组不存在或已过期")
    sample_id = hashlib.sha256(
        f"{video_id}\0{asset_version}\0{group_idx}".encode("utf-8")
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
    collection.delete(
        f'source_video_id == "{_expr_value(video_id)}" and '
        f'source_asset_version == "{_expr_value(asset_version)}" and '
        f'source_group_idx == {int(group_idx)}'
    )
    collection.upsert([row])
    collection.flush()
    catalog.bind_face_identity(video_id, asset_version, group_idx, entity_id)
    return {"sample_id": sample_id, "entity_id": entity_id, "entity_name": entity["name"]}


def delete_entity_face_samples(entity_id: str) -> int:
    collection = get_milvus_client().collection("entity_face_samples")
    result = collection.delete(f'entity_id == "{_expr_value(entity_id)}"')
    collection.flush()
    return int(getattr(result, "delete_count", 0))


def copy_thumbnail_as_reference(thumbnail: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(thumbnail, destination)
