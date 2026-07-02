from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.indexing.common import atomic_save_npz, normalize
from app.media import read_frames, save_thumbnail


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(1e-6, first_area + second_area - intersection))


class FaceEncoder:
    def __init__(self, model_name: str, provider: str = "cpu", device_id: int = 0, root: str | None = None):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        if provider == "cann" and "CANNExecutionProvider" in available:
            providers = [("CANNExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
            ctx_id = device_id
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        self.app = FaceAnalysis(name=model_name, providers=providers, root=root or "~/.insightface")
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.provider = provider if provider == "cann" and "CANNExecutionProvider" in available else "cpu"

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
    best_crop: np.ndarray | None = None


def build_face_index(
    video_path: str,
    output_path: str,
    thumbnail_dir: str,
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
) -> dict:
    # encoder may be supplied by the warm pool (model already resident); otherwise
    # load it for this call (the process_exit path).
    if encoder is None:
        encoder = FaceEncoder(model_name, provider, device_id, model_root)
    active: list[Track] = []
    finished: list[Track] = []
    next_number = 0
    detections = 0

    for timestamp, frame in read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg):
        retained = []
        for track in active:
            if timestamp - track.end <= max_gap:
                retained.append(track)
            else:
                finished.append(track)
        active = retained
        used_tracks: set[int] = set()
        faces = sorted(encoder.detect(frame), key=lambda item: float(item.det_score), reverse=True)
        detections += len(faces)
        for face in faces:
            embedding = normalize(face.normed_embedding)
            bbox = np.asarray(face.bbox, dtype=np.float32)
            candidates = []
            for index, track in enumerate(active):
                if index in used_tracks:
                    continue
                track_embedding = normalize(np.mean(track.embeddings, axis=0))
                cosine = float(np.dot(embedding, track_embedding))
                candidates.append((0.85 * cosine + 0.15 * _iou(bbox, track.bbox), cosine, index))
            match = max(candidates, default=None)
            if match and match[1] >= cosine_threshold:
                track = active[match[2]]
                used_tracks.add(match[2])
                track.end = timestamp + 1 / sample_fps
                track.bbox = bbox
                track.embeddings.append(embedding)
            else:
                track = Track(next_number, timestamp, timestamp + 1 / sample_fps, bbox, [embedding])
                next_number += 1
                active.append(track)
                used_tracks.add(len(active) - 1)

            x1, y1, x2, y2 = bbox.astype(int)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            quality = float(face.det_score) * float(np.sqrt(area))
            if quality > track.best_quality:
                pad = max(4, int(0.15 * max(x2 - x1, y2 - y1)))
                height, width = frame.shape[:2]
                track.best_crop = frame[
                    max(0, y1 - pad):min(height, y2 + pad),
                    max(0, x1 - pad):min(width, x2 + pad),
                ].copy()
                track.best_quality = quality
    finished.extend(active)

    embeddings, starts, ends, thumbnails, qualities = [], [], [], [], []
    thumbnail_dir = Path(thumbnail_dir)
    for track in finished:
        if not track.embeddings:
            continue
        thumbnail = thumbnail_dir / f"face_{track.number:06d}.jpg"
        if track.best_crop is not None and track.best_crop.size:
            save_thumbnail(track.best_crop, thumbnail, max_width=240)
        embeddings.append(normalize(np.mean(track.embeddings, axis=0)))
        starts.append(track.start)
        ends.append(track.end)
        thumbnails.append(thumbnail.name if thumbnail.exists() else "")
        qualities.append(track.best_quality)

    dimension = len(embeddings[0]) if embeddings else 512
    atomic_save_npz(
        output_path,
        embeddings=np.stack(embeddings).astype(np.float32) if embeddings else np.empty((0, dimension), np.float32),
        start_times=np.asarray(starts, np.float32),
        end_times=np.asarray(ends, np.float32),
        thumbnails=np.asarray(thumbnails, dtype="U128"),
        qualities=np.asarray(qualities, np.float32),
        model=np.asarray([model_name]),
    )
    return {"tracks": len(embeddings), "detections": detections, "provider": encoder.provider}


def encode_face_reference(
    path: str, model_name: str, provider: str = "cpu", device_id: int = 0, model_root: str | None = None
) -> np.ndarray:
    return FaceEncoder(model_name, provider, device_id, model_root).encode_reference(path)
