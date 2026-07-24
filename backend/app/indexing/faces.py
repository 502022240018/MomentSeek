from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import cv2
import numpy as np

from app.indexing.common import atomic_save_npz, normalize
from app.media import read_frames

if TYPE_CHECKING:
    from app.indexing.milvus_indexer import MilvusWriteContext


def _has_non_empty_onnx(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() and item.stat().st_size > 0 and item.suffix.lower() == ".onnx" for item in path.rglob("*"))


def _resolve_insightface_root(root: str | None, model_name: str) -> Path:
    root_path = Path(root or "~/.insightface").expanduser()
    canonical_model_dir = root_path / "models" / model_name
    if _has_non_empty_onnx(canonical_model_dir):
        return root_path

    models_parent_dir = root_path / model_name
    if _has_non_empty_onnx(models_parent_dir) and root_path.name == "models":
        return root_path.parent

    if _has_non_empty_onnx(root_path) and root_path.name == model_name and root_path.parent.name == "models":
        return root_path.parent.parent

    raise FileNotFoundError(f"本地 InsightFace 模型缺失: expected {canonical_model_dir}")


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(1e-6, first_area + second_area - intersection))


class FaceEncoder:
    def __init__(
        self,
        model_name: str,
        provider: str = "cpu",
        device_id: int = 0,
        root: str | None = None,
        ort_intra_op_threads: int = 8,
        ort_inter_op_threads: int = 1,
    ):
        face_root = _resolve_insightface_root(root, model_name)
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider == "cann" and "CANNExecutionProvider" not in available:
            raise RuntimeError(
                "Face 已配置为 CANN NPU，但 onnxruntime 未提供 CANNExecutionProvider；"
                f"available_providers={available}。为避免产品环境静默回落 CPU，任务已终止。"
            )
        from insightface.app import FaceAnalysis

        if provider == "cann":
            providers = [("CANNExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
            ctx_id = device_id
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        analysis_options = {
            "name": model_name,
            "providers": providers,
            "root": str(face_root),
            "allowed_modules": ["detection", "recognition"],
        }
        if provider != "cann":
            # The Ascend ORT build currently fails CANN graph initialization
            # when an explicit SessionOptions object is supplied. Bound the
            # CPU query encoder, but leave CANN session creation to the EP.
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = max(1, int(ort_intra_op_threads))
            session_options.inter_op_num_threads = max(1, int(ort_inter_op_threads))
            analysis_options["sess_options"] = session_options
        self.app = FaceAnalysis(**analysis_options)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.provider = provider

    def detect(self, frame_bgr: np.ndarray):
        return self.app.get(frame_bgr)

    def encode_reference(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise OSError(f"无法读取参考图: {path}")
        faces = self.detect(image)
        if not faces:
            raise ValueError("参考图中未检测到人脸")
        face = max(faces, key=lambda item: float(np.prod(item.bbox[2:] - item.bbox[:2])))
        return normalize(face.normed_embedding)


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
    quality = float(face.det_score) * float(np.sqrt(area))
    if quality <= track.best_quality:
        return
    pad = max(4, int(0.15 * max(x2 - x1, y2 - y1)))
    height, width = frame.shape[:2]
    track.best_crop = frame[
        max(0, y1 - pad):min(height, y2 + pad),
        max(0, x1 - pad):min(width, x2 + pad),
    ].copy()
    track.best_quality = quality
    track.best_time = timestamp


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
            _update_best_face_crop(track, face, frame, bbox, timestamp)
    finished.extend(active)
    embeddings, track_times_ms = _face_track_arrays(finished)

    dimension = len(embeddings[0]) if embeddings else 512
    atomic_save_npz(
        output_path,
        embeddings=np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, dimension), np.float32),
        track_times_ms=np.asarray(track_times_ms, dtype=np.int32).reshape((-1, 3)),
    )
    if milvus_ctx is not None:
        from app.indexing.milvus_indexer import write_modality_to_milvus

        write_modality_to_milvus(milvus_ctx, "face", output_path)
    return {
        "tracks": len(embeddings),
        "detections": detections,
        "provider": encoder.provider,
        "schema_version": 3,
        "decode_status": "complete" if embeddings else "empty",
    }


def encode_face_reference(
    path: str, model_name: str, provider: str = "cpu", device_id: int = 0, model_root: str | None = None
) -> np.ndarray:
    return FaceEncoder(model_name, provider, device_id, model_root).encode_reference(path)
