from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-12)


@dataclass(frozen=True)
class FaceGroup:
    group_idx: int
    track_indices: tuple[int, ...]
    representative_track_idx: int
    start_ms: int
    end_ms: int
    best_ms: int
    bbox: tuple[float, float, float, float]
    quality: float
    duration_ms: int
    occurrence_count: int
    importance_score: float
    embedding: np.ndarray


def cluster_face_tracks(
    embeddings: np.ndarray,
    track_times_ms: np.ndarray,
    *,
    qualities: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    detection_counts: np.ndarray | None = None,
    cosine_threshold: float = 0.52,
) -> list[FaceGroup]:
    """Conservatively merge face tracks belonging to the same person.

    Tracks are processed in temporal order and are joined only when their
    vector agrees with both the current centroid and a member of the group.
    This favours identity precision: two people being merged is harder to
    correct than one person appearing as two cards.
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    times = np.asarray(track_times_ms, dtype=np.int64)
    if vectors.ndim != 2 or len(vectors) == 0:
        return []
    vectors = np.stack([_normalize(row) for row in vectors])
    count = len(vectors)
    quality_values = (
        np.asarray(qualities, dtype=np.float32)
        if qualities is not None and len(qualities) == count
        else np.full(count, 0.5, dtype=np.float32)
    )
    bbox_values = (
        np.asarray(bboxes, dtype=np.float32)
        if bboxes is not None and len(bboxes) == count
        else np.full((count, 4), -1.0, dtype=np.float32)
    )
    detections = (
        np.asarray(detection_counts, dtype=np.int64)
        if detection_counts is not None and len(detection_counts) == count
        else np.ones(count, dtype=np.int64)
    )

    members: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for index in np.argsort(times[:, 0], kind="stable"):
        best_group = -1
        best_score = -1.0
        for group_idx, group_members in enumerate(members):
            centroid_score = float(vectors[index] @ centroids[group_idx])
            member_score = max(float(vectors[index] @ vectors[item]) for item in group_members)
            score = min(centroid_score, member_score)
            if score >= cosine_threshold and score > best_score:
                best_group, best_score = group_idx, score
        if best_group < 0:
            members.append([int(index)])
            centroids.append(vectors[index].copy())
        else:
            members[best_group].append(int(index))
            weights = np.maximum(quality_values[members[best_group]], 0.1)
            centroids[best_group] = _normalize(
                np.average(vectors[members[best_group]], axis=0, weights=weights)
            )

    groups: list[FaceGroup] = []
    for group_members, centroid in zip(members, centroids):
        centrality = vectors[group_members] @ centroid
        representative_scores = 0.70 * quality_values[group_members] + 0.30 * centrality
        representative = group_members[int(np.argmax(representative_scores))]
        duration_ms = int(sum(max(0, int(times[i, 1]) - int(times[i, 0])) for i in group_members))
        occurrence_count = int(sum(max(1, int(detections[i])) for i in group_members))
        importance = float(
            0.55 * min(1.0, np.log1p(duration_ms / 1000.0) / np.log(31.0))
            + 0.25 * min(1.0, np.log1p(occurrence_count) / np.log(21.0))
            + 0.20 * float(quality_values[representative])
        )
        groups.append(FaceGroup(
            group_idx=0,
            track_indices=tuple(group_members),
            representative_track_idx=int(representative),
            start_ms=int(min(times[i, 0] for i in group_members)),
            end_ms=int(max(times[i, 1] for i in group_members)),
            best_ms=int(times[representative, 2]),
            bbox=tuple(float(value) for value in bbox_values[representative]),
            quality=float(quality_values[representative]),
            duration_ms=duration_ms,
            occurrence_count=occurrence_count,
            importance_score=importance,
            embedding=centroid.astype(np.float32),
        ))

    groups.sort(key=lambda item: (-item.importance_score, item.start_ms))
    return [FaceGroup(**{**item.__dict__, "group_idx": idx}) for idx, item in enumerate(groups)]


def face_group_arrays(groups: list[FaceGroup]) -> dict[str, np.ndarray]:
    return {
        "group_embeddings": np.stack([group.embedding for group in groups]).astype(np.float32)
        if groups else np.empty((0, 512), dtype=np.float32),
        "group_track_indices": np.asarray([group.representative_track_idx for group in groups], dtype=np.int32),
        "group_times_ms": np.asarray(
            [[group.start_ms, group.end_ms, group.best_ms] for group in groups], dtype=np.int64
        ).reshape((-1, 3)),
        "group_bboxes": np.asarray([group.bbox for group in groups], dtype=np.float32).reshape((-1, 4)),
        "group_qualities": np.asarray([group.quality for group in groups], dtype=np.float32),
        "group_durations_ms": np.asarray([group.duration_ms for group in groups], dtype=np.int64),
        "group_occurrence_counts": np.asarray([group.occurrence_count for group in groups], dtype=np.int64),
        "group_importance_scores": np.asarray([group.importance_score for group in groups], dtype=np.float32),
    }
