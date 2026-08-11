from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import cv2

from app.encoders.face import FaceEncoder
from app.indexing.common import atomic_save_npz, normalize
from app.media.media import read_frames

if TYPE_CHECKING:
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(1e-6, first_area + second_area - intersection))


@dataclass
class Track:
    number: int
    start: float
    end: float
    bbox: np.ndarray
    embeddings: list[np.ndarray] = field(default_factory=list)
    best_quality: float = 0
    best_time: float = 0
    best_crop: np.ndarray | None = None
    best_bbox: np.ndarray = field(default_factory=lambda: np.full(4, -1.0, dtype=np.float32))
    detection_count: int = 0


def _expire_face_tracks(active: list[Track], timestamp: float, max_gap: float) -> tuple[list[Track], list[Track]]:
    retained, expired = [], []
    for track in active:
        (retained if timestamp - track.end <= max_gap else expired).append(track)
    return retained, expired


def _best_face_track_match(
    active: list[Track],
    used_tracks: set[int],
    embedding: np.ndarray,
    bbox: np.ndarray,
    cosine_threshold: float,
) -> tuple[Track | None, int | None]:
    candidates = []
    for index, track in enumerate(active):
        if index in used_tracks:
            continue
        track_embedding = normalize(np.mean(track.embeddings, axis=0))
        cosine = float(np.dot(embedding, track_embedding))
        candidates.append((0.85 * cosine + 0.15 * _iou(bbox, track.bbox), cosine, index))
    match = max(candidates, default=None)
    if match and match[1] >= cosine_threshold:
        return active[match[2]], int(match[2])
    return None, None


def _update_best_face_crop(track: Track, face, frame: np.ndarray, bbox: np.ndarray, timestamp: float) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    height, width = frame.shape[:2]
    relative_size = min(1.0, np.sqrt(area / max(1.0, height * width)) / 0.35)
    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    sharpness = 0.0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0)
    border_margin = min(x1, y1, width - x2, height - y2)
    border_score = min(1.0, max(0.0, border_margin) / max(1.0, 0.08 * min(width, height)))
    quality = 0.50 * float(face.det_score) + 0.25 * relative_size + 0.15 * sharpness + 0.10 * border_score
    if quality <= track.best_quality:
        return
    pad = max(4, int(0.15 * max(x2 - x1, y2 - y1)))
    track.best_crop = frame[
        max(0, y1 - pad):min(height, y2 + pad),
        max(0, x1 - pad):min(width, x2 + pad),
    ].copy()
    track.best_quality = quality
    track.best_time = timestamp
    track.best_bbox = np.asarray([x1 / width, y1 / height, x2 / width, y2 / height], dtype=np.float32)


def _face_track_arrays(tracks: list[Track]) -> tuple[list[np.ndarray], list[list[int]]]:
    embeddings, track_times_ms = [], []
    for track in tracks:
        if not track.embeddings:
            continue
        embeddings.append(normalize(np.mean(track.embeddings, axis=0)))
        track_times_ms.append([
            int(round(track.start * 1000)),
            int(round(track.end * 1000)),
            int(round(track.best_time * 1000)),
        ])
    return embeddings, track_times_ms


def build_face_index(
    video_path: str,
    output_path: str,
    model_name: str,
    sample_fps: float,
    provider: str,
    device_id: int,
    model_root: str | None = None,
    max_gap: float = 1.5,
    cosine_threshold: float = 0.35,
    gallery_cosine_threshold: float = 0.52,
    encoder: "FaceEncoder | None" = None,
    decode_height: int = 0,
    prefer_ffmpeg: bool = True,
    ort_intra_op_threads: int = 8,
    ort_inter_op_threads: int = 1,
    milvus_ctx: "MilvusWriteContext | None" = None,
) -> dict:
    # encoder may be supplied by the warm pool (model already resident); otherwise
    # load it for this call (the process_exit path).
    if encoder is None:
        encoder = FaceEncoder(
            model_name,
            provider,
            device_id,
            model_root,
            ort_intra_op_threads,
            ort_inter_op_threads,
        )
    active: list[Track] = []
    finished: list[Track] = []
    next_number = 0
    detections = 0

    for timestamp, frame in read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg):
        active, expired = _expire_face_tracks(active, timestamp, max_gap)
        finished.extend(expired)
        used_tracks: set[int] = set()
        faces = sorted(encoder.detect(frame), key=lambda item: float(item.det_score), reverse=True)
        detections += len(faces)
        for face in faces:
            embedding = normalize(face.normed_embedding)
            bbox = np.asarray(face.bbox, dtype=np.float32)
            track, matched_index = _best_face_track_match(
                active, used_tracks, embedding, bbox, cosine_threshold
            )
            if track is not None and matched_index is not None:
                used_tracks.add(matched_index)
                track.end = timestamp + 1 / sample_fps
                track.bbox = bbox
                track.embeddings.append(embedding)
            else:
                track = Track(next_number, timestamp, timestamp + 1 / sample_fps, bbox, [embedding])
                next_number += 1
                active.append(track)
                used_tracks.add(len(active) - 1)
            track.detection_count += 1
            _update_best_face_crop(track, face, frame, bbox, timestamp)
    finished.extend(active)
    embeddings, track_times_ms = _face_track_arrays(finished)

    dimension = len(embeddings[0]) if embeddings else 512
    embedding_array = (
        np.stack(embeddings).astype(np.float32)
        if embeddings else np.empty((0, dimension), np.float32)
    )
    track_times_array = np.asarray(track_times_ms, dtype=np.int32).reshape((-1, 3))
    valid_tracks = [track for track in finished if track.embeddings]
    qualities = np.asarray([track.best_quality for track in valid_tracks], dtype=np.float32)
    bboxes = np.asarray([track.best_bbox for track in valid_tracks], dtype=np.float32).reshape((-1, 4))
    detection_counts = np.asarray([track.detection_count for track in valid_tracks], dtype=np.int32)
    from app.identity.face_gallery import cluster_face_tracks, face_group_arrays
    groups = cluster_face_tracks(
        embedding_array,
        track_times_array,
        qualities=qualities,
        bboxes=bboxes,
        detection_counts=detection_counts,
        cosine_threshold=gallery_cosine_threshold,
    )
    group_arrays = face_group_arrays(groups)
    milvus_rows = None
    if milvus_ctx is not None:
        from app.vector_store.milvus.milvus_indexer import write_modality_from_memory

        milvus_rows = write_modality_from_memory(
            milvus_ctx,
            "face",
            {
                "embeddings": embedding_array,
                "track_times_ms": track_times_array,
                **group_arrays,
            },
        )
    # Retained only as an offline recovery artifact; no runtime path reads it.
    atomic_save_npz(
        output_path,
        embeddings=embedding_array,
        track_times_ms=track_times_array,
    )
    return {
        "tracks": len(embeddings),
        "face_groups": len(groups),
        "detections": detections,
        "provider": encoder.provider,
        "schema_version": 3,
        "decode_status": "complete" if embeddings else "empty",
        "milvus_rows": milvus_rows,
    }


def encode_face_reference(
    path: str, model_name: str, provider: str = "cpu", device_id: int = 0, model_root: str | None = None
) -> np.ndarray:
    return FaceEncoder(model_name, provider, device_id, model_root).encode_reference(path)
