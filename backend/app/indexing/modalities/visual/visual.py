from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from app.encoders.visual import (
    ClipEncoder,
    resolve_device,
)
from app.indexing.common import atomic_save_npz, normalize
from app.media.media import probe_video, read_frames

if TYPE_CHECKING:
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext


SHOT_DETECTORS = {"simple", "pyscenedetect_content", "pyscenedetect_adaptive"}
SHOT_DETECTOR_ALIASES = {
    "content": "pyscenedetect_content",
    "pyscene_content": "pyscenedetect_content",
    "pyscenedetect": "pyscenedetect_content",
    "adaptive": "pyscenedetect_adaptive",
    "pyscene_adaptive": "pyscenedetect_adaptive",
}


def normalize_shot_detector(value: str | None) -> str:
    raw = (value or "simple").strip().lower()
    normalized = SHOT_DETECTOR_ALIASES.get(raw, raw)
    if normalized not in SHOT_DETECTORS:
        allowed = ", ".join(sorted(SHOT_DETECTORS))
        raise ValueError(f"Unknown shot_detector={value!r}. Allowed: {allowed}")
    return normalized


def _split_long_segments(segments: list[tuple[int, int]], min_segment_ms: int, max_segment_ms: int) -> list[tuple[int, int]]:
    if max_segment_ms <= 0:
        return segments
    result: list[tuple[int, int]] = []
    for start_ms, end_ms in segments:
        cursor = start_ms
        while end_ms - cursor > max_segment_ms:
            if end_ms - (cursor + max_segment_ms) < min_segment_ms:
                break
            result.append((cursor, cursor + max_segment_ms))
            cursor += max_segment_ms
        if end_ms > cursor:
            result.append((cursor, end_ms))
    return result


def _normalize_segments(
    segments: list[tuple[int, int]],
    duration_ms: int,
    min_segment_ms: int,
    max_segment_ms: int,
) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in sorted(segments):
        start_ms = max(0, min(duration_ms, int(raw_start)))
        end_ms = max(0, min(duration_ms, int(raw_end)))
        if end_ms <= start_ms:
            continue
        if start_ms > cursor:
            cleaned.append((cursor, start_ms))
        start_ms = max(start_ms, cursor)
        if end_ms > start_ms:
            cleaned.append((start_ms, end_ms))
            cursor = end_ms
    if duration_ms > cursor:
        cleaned.append((cursor, duration_ms))
    if not cleaned:
        return []

    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in cleaned:
        if end_ms - start_ms < min_segment_ms and merged:
            previous_start, _previous_end = merged[-1]
            merged[-1] = (previous_start, end_ms)
        else:
            merged.append((start_ms, end_ms))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_segment_ms:
        first_start, _first_end = merged[0]
        _second_start, second_end = merged[1]
        merged[1] = (first_start, second_end)
        merged.pop(0)
    return _split_long_segments(merged, min_segment_ms, max_segment_ms)


def detect_shot_segments(
    video_path: str,
    duration_seconds: float,
    sample_fps: float = 2.0,
    threshold: float = 0.45,
    min_segment_seconds: float = 0.8,
    max_segment_seconds: float = 8.0,
    decode_height: int = 0,
    prefer_ffmpeg: bool = True,
) -> list[tuple[int, int]]:
    """Detect coarse shot boundaries using sampled-frame grayscale differences.

    This intentionally keeps the first implementation dependency-light. It is a
    fallback-friendly detector: callers can drop back to fixed windows whenever it
    returns no usable segments or raises.
    """
    duration_ms = max(0, int(round(float(duration_seconds or 0) * 1000)))
    if duration_ms <= 0:
        return []
    min_segment_ms = max(1, int(round(float(min_segment_seconds) * 1000)))
    max_segment_ms = max(min_segment_ms, int(round(float(max_segment_seconds) * 1000)))
    boundaries = [0]
    previous_gray = None
    last_boundary_ms = 0
    for timestamp, frame in read_frames(
        video_path,
        max(0.2, float(sample_fps)),
        out_height=decode_height,
        prefer_ffmpeg=prefer_ffmpeg,
    ):
        timestamp_ms = int(round(float(timestamp) * 1000))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            if gray.shape != previous_gray.shape:
                gray = cv2.resize(gray, (previous_gray.shape[1], previous_gray.shape[0]))
            difference = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)
            if difference >= threshold and timestamp_ms - last_boundary_ms >= min_segment_ms:
                boundaries.append(min(duration_ms, max(0, timestamp_ms)))
                last_boundary_ms = timestamp_ms
        previous_gray = gray
    boundaries.append(duration_ms)
    raw_segments = [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]
    return _normalize_segments(raw_segments, duration_ms, min_segment_ms, max_segment_ms)


def _timecode_to_ms(value: Any) -> int:
    seconds = getattr(value, "seconds", None)
    if seconds is None:
        seconds = value.get_seconds()
    return int(round(float(seconds) * 1000))


def detect_pyscenedetect_segments(
    video_path: str,
    duration_seconds: float,
    detector: str = "pyscenedetect_content",
    threshold: float = 0.20,
    min_segment_seconds: float = 0.8,
    max_segment_seconds: float = 8.0,
) -> list[tuple[int, int]]:
    """Detect shot boundaries with PySceneDetect, then normalize to app segments."""
    detector = normalize_shot_detector(detector)
    if detector == "simple":
        return detect_shot_segments(
            video_path,
            duration_seconds=duration_seconds,
            threshold=threshold,
            min_segment_seconds=min_segment_seconds,
            max_segment_seconds=max_segment_seconds,
        )

    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    duration_ms = max(0, int(round(float(duration_seconds or 0) * 1000)))
    if duration_ms <= 0:
        return []
    min_segment_ms = max(1, int(round(float(min_segment_seconds) * 1000)))
    max_segment_ms = max(min_segment_ms, int(round(float(max_segment_seconds) * 1000)))
    fps = max(1.0, float(probe_video(video_path).fps or 0))
    min_scene_len = max(1, int(round(float(min_segment_seconds) * fps)))
    normalized_threshold = max(0.01, min(1.0, float(threshold)))

    if detector == "pyscenedetect_adaptive":
        scene_detector = AdaptiveDetector(
            adaptive_threshold=max(0.1, normalized_threshold * 15.0),
            min_scene_len=min_scene_len,
            min_content_val=15.0,
        )
    else:
        scene_detector = ContentDetector(
            threshold=max(1.0, normalized_threshold * 135.0),
            min_scene_len=min_scene_len,
        )

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(scene_detector)
    manager.detect_scenes(video=video, show_progress=False)
    raw_segments = [
        (_timecode_to_ms(start), _timecode_to_ms(end))
        for start, end in manager.get_scene_list()
    ]
    return _normalize_segments(raw_segments, duration_ms, min_segment_ms, max_segment_ms)


def _fixed_segments(duration_ms: int, segment_ms: int, max_bucket: int) -> list[tuple[int, int]]:
    segments_total = max(1, int(math.ceil(duration_ms / segment_ms)) if duration_ms > 0 else 1, max_bucket + 1)
    end_limit = duration_ms if duration_ms > 0 else segments_total * segment_ms
    return [
        (segment_id * segment_ms, min((segment_id + 1) * segment_ms, end_limit))
        for segment_id in range(segments_total)
    ]


def _segment_id_for_timestamp(timestamp_ms: int, segment_times_ms: np.ndarray) -> int:
    starts = segment_times_ms[:, 0]
    index = int(np.searchsorted(starts, timestamp_ms, side="right") - 1)
    index = max(0, min(index, len(segment_times_ms) - 1))
    if timestamp_ms > int(segment_times_ms[index, 1]) and index < len(segment_times_ms) - 1:
        index += 1
    return index


def _resolve_visual_segmentation(
    video_path: str,
    *,
    segment_seconds: float,
    segment_strategy: str,
    duration_seconds: float | int | None,
    min_segment_seconds: float,
    max_segment_seconds: float,
    shot_detector: str,
    shot_detector_threshold: float,
    decode_height: int,
    prefer_ffmpeg: bool,
) -> tuple[int, int, np.ndarray | None, str, str, str]:
    segment_ms = max(1, int(round(float(segment_seconds) * 1000)))
    requested_strategy = (segment_strategy or "fixed").strip().lower()
    if requested_strategy not in {"fixed", "shot"}:
        requested_strategy = "fixed"
    try:
        requested_detector = normalize_shot_detector(shot_detector)
    except ValueError:
        requested_detector = "simple"
    duration_ms = int(round(float(duration_seconds or 0) * 1000))
    if requested_strategy != "shot" or duration_ms <= 0:
        return segment_ms, duration_ms, None, "fixed", requested_detector, "inferred_from_segment_ms"
    min_segment_ms = max(1, int(round(float(min_segment_seconds) * 1000)))
    max_segment_ms = max(min_segment_ms, int(round(float(max_segment_seconds) * 1000)))
    try:
        if requested_detector == "simple":
            shot_segments = detect_shot_segments(
                video_path,
                duration_seconds=float(duration_seconds or 0),
                threshold=shot_detector_threshold,
                min_segment_seconds=min_segment_seconds,
                max_segment_seconds=max_segment_seconds,
                decode_height=decode_height,
                prefer_ffmpeg=prefer_ffmpeg,
            )
        else:
            shot_segments = detect_pyscenedetect_segments(
                video_path,
                duration_seconds=float(duration_seconds or 0),
                detector=requested_detector,
                threshold=shot_detector_threshold,
                min_segment_seconds=min_segment_seconds,
                max_segment_seconds=max_segment_seconds,
            )
    except Exception:
        shot_segments = []
    normalized = _normalize_segments(shot_segments, duration_ms, min_segment_ms, max_segment_ms)
    if not normalized:
        return segment_ms, duration_ms, None, "fixed", requested_detector, "inferred_from_segment_ms"
    return segment_ms, duration_ms, np.asarray(normalized, dtype=np.int32), "shot", requested_detector, "explicit"


def _flush_visual_batch(
    encoder: "ClipEncoder",
    pending_frames: list[np.ndarray],
    pending_meta: list[tuple[int, int]],
    frame_embeddings: list[np.ndarray],
    frame_times_ms: list[int],
) -> None:
    if not pending_frames:
        return
    vectors = encoder.encode_frames(pending_frames)
    for (_bucket, timestamp_ms), vector in zip(pending_meta, vectors):
        frame_embeddings.append(normalize(vector))
        frame_times_ms.append(timestamp_ms)
    pending_frames.clear()
    pending_meta.clear()


def _encode_visual_frames(
    video_path: str,
    encoder: "ClipEncoder",
    *,
    sample_fps: float,
    batch_size: int,
    decode_height: int,
    prefer_ffmpeg: bool,
    segment_ms: int,
    explicit_segment_times: np.ndarray | None,
) -> tuple[list[np.ndarray], list[int], list[int], int]:
    frame_embeddings: list[np.ndarray] = []
    frame_times_ms: list[int] = []
    frame_segment_ids: list[int] = []
    pending_frames: list[np.ndarray] = []
    pending_meta: list[tuple[int, int]] = []
    total_frames = 0
    frames = read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg)
    for timestamp, frame in frames:
        timestamp_ms = int(round(float(timestamp) * 1000))
        bucket = (
            _segment_id_for_timestamp(timestamp_ms, explicit_segment_times)
            if explicit_segment_times is not None
            else timestamp_ms // segment_ms
        )
        pending_frames.append(frame)
        pending_meta.append((bucket, timestamp_ms))
        frame_segment_ids.append(int(bucket))
        total_frames += 1
        if len(pending_frames) >= batch_size:
            _flush_visual_batch(encoder, pending_frames, pending_meta, frame_embeddings, frame_times_ms)
    _flush_visual_batch(encoder, pending_frames, pending_meta, frame_embeddings, frame_times_ms)
    return frame_embeddings, frame_times_ms, frame_segment_ids, total_frames


def _visual_index_payload(
    frame_embeddings: list[np.ndarray],
    frame_times_ms: list[int],
    frame_segment_ids: list[int],
    *,
    duration_ms: int,
    segment_ms: int,
    sample_fps: float,
    explicit_segment_times: np.ndarray | None,
) -> tuple[dict, int, int, int]:
    frame_times = np.asarray(frame_times_ms, dtype=np.int32)
    segment_ids = np.asarray(frame_segment_ids, dtype=np.int32)
    embeddings = np.stack(frame_embeddings).astype(np.float32)
    order = np.argsort(frame_times)
    frame_times, segment_ids, embeddings = frame_times[order], segment_ids[order], embeddings[order]
    if duration_ms <= 0:
        duration_ms = int(frame_times.max()) + max(1, int(round(1000 / sample_fps)))
    max_bucket = int(segment_ids.max()) if len(segment_ids) else 0
    segments_total = (
        int(len(explicit_segment_times))
        if explicit_segment_times is not None
        else len(_fixed_segments(duration_ms, segment_ms, max_bucket))
    )
    offsets = np.searchsorted(
        segment_ids, np.arange(segments_total + 1, dtype=np.int32), side="left"
    ).astype(np.int32)
    segments_with_frames = int(len(np.unique(segment_ids)))
    empty_segments = max(0, segments_total - segments_with_frames)
    payload = {
        "frame_embeddings": embeddings.astype(np.float16),
        "frame_times_ms": frame_times.astype(np.int32),
        "segment_frame_offsets": offsets,
    }
    if explicit_segment_times is not None:
        payload["segment_times_ms"] = explicit_segment_times.astype(np.int32)
    return payload, segments_total, segments_with_frames, empty_segments


def build_visual_index(
    video_path: str,
    output_path: str,
    model_name: str,
    pretrained: str,
    sample_fps: float,
    segment_seconds: float,
    batch_size: int,
    npu_enabled: bool,
    npu_device_id: int,
    cuda_enabled: bool = False,
    encoder: "ClipEncoder | None" = None,
    visual_model: str | None = None,
    model_cache_dir: str | Path | None = None,
    decode_height: int = 0,
    prefer_ffmpeg: bool = True,
    duration_seconds: float | int | None = None,
    segment_strategy: str = "fixed",
    min_segment_seconds: float = 0.8,
    max_segment_seconds: float = 8.0,
    shot_detector: str = "simple",
    shot_detector_threshold: float = 0.20,
    milvus_ctx: "MilvusWriteContext | None" = None,
) -> dict:
    # encoder may be supplied by the warm pool (model already resident); otherwise
    # load it for this call (the process_exit path).
    if encoder is None:
        encoder = ClipEncoder(
            model_name,
            pretrained,
            resolve_device(npu_enabled, npu_device_id, cuda_enabled),
            visual_model=visual_model,
            model_cache_dir=model_cache_dir,
        )
    device = encoder.device
    segment_ms, duration_ms, explicit_segment_times, active_strategy, active_detector, segment_time_source = (
        _resolve_visual_segmentation(
            video_path,
            segment_seconds=segment_seconds,
            segment_strategy=segment_strategy,
            duration_seconds=duration_seconds,
            min_segment_seconds=min_segment_seconds,
            max_segment_seconds=max_segment_seconds,
            shot_detector=shot_detector,
            shot_detector_threshold=shot_detector_threshold,
            decode_height=decode_height,
            prefer_ffmpeg=prefer_ffmpeg,
        )
    )
    frame_embeddings, frame_times_ms, frame_segment_ids, total_frames = _encode_visual_frames(
        video_path,
        encoder,
        sample_fps=sample_fps,
        batch_size=batch_size,
        decode_height=decode_height,
        prefer_ffmpeg=prefer_ffmpeg,
        segment_ms=segment_ms,
        explicit_segment_times=explicit_segment_times,
    )
    if not frame_embeddings:
        raise RuntimeError("未从视频抽取到画面")
    payload, segments_total, segments_with_frames, empty_segments = _visual_index_payload(
        frame_embeddings,
        frame_times_ms,
        frame_segment_ids,
        duration_ms=duration_ms,
        segment_ms=segment_ms,
        sample_fps=sample_fps,
        explicit_segment_times=explicit_segment_times,
    )
    atomic_save_npz(output_path, **payload)
    if milvus_ctx is not None:
        from app.vector_store.milvus.milvus_indexer import write_modality_to_milvus

        write_modality_to_milvus(milvus_ctx, "visual", output_path)
    return {
        "segments_total": segments_total,
        "segments_with_frames": segments_with_frames,
        "empty_segments": empty_segments,
        "frames": total_frames,
        "schema_version": 3,
        "device": device,
        "visual_model": encoder.model_key,
        "model": encoder.model_label,
        "decode_status": "complete" if empty_segments == 0 else "partial",
        "segment_strategy": active_strategy,
        "segment_times": segment_time_source,
        "shot_detector": active_detector,
    }
