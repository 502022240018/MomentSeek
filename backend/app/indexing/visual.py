from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.indexing.common import atomic_save_npz, normalize
from app.media import iter_sampled_frames, save_thumbnail


def resolve_device(npu_enabled: bool, npu_device_id: int, cuda_enabled: bool = False) -> str:
    if cuda_enabled:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    if not npu_enabled:
        return "cpu"
    try:
        import torch_npu  # noqa: F401

        return f"npu:{npu_device_id}"
    except ImportError as exc:
        raise RuntimeError("NPU_ENABLED=true，但当前环境没有 torch_npu") from exc


class ClipEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        source = pretrained
        if pretrained not in {"openai", "laion2b_s34b_b79k"} and not Path(pretrained).exists():
            raise FileNotFoundError(f"CLIP 权重不存在: {pretrained}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=source, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        tensors = [
            self.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            for frame in frames_bgr
        ]
        with torch.inference_mode():
            batch = torch.stack(tensors).to(self.device)
            output = self.model.encode_image(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = torch.nn.functional.normalize(output, dim=-1)
        return output.float().cpu().numpy()

    def encode_query(self, text: str | None, image_path: str | None, alpha: float = 0.5) -> np.ndarray:
        torch = self.torch
        parts = []
        with torch.inference_mode():
            if text:
                tokens = self.tokenizer([text]).to(self.device)
                encoded = torch.nn.functional.normalize(self.model.encode_text(tokens), dim=-1)
                parts.append((alpha if image_path else 1.0, encoded))
            if image_path:
                tensor = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
                encoded = self.model.encode_image(tensor)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                parts.append(((1 - alpha) if text else 1.0, encoded))
            if not parts:
                raise ValueError("视觉查询需要文本或参考图")
            vector = sum(weight * value for weight, value in parts)
            vector = torch.nn.functional.normalize(vector, dim=-1)
        return vector.float().cpu().numpy()[0]


def build_visual_index(
    video_path: str,
    output_path: str,
    thumbnail_dir: str,
    model_name: str,
    pretrained: str,
    sample_fps: float,
    segment_seconds: float,
    batch_size: int,
    npu_enabled: bool,
    npu_device_id: int,
    cuda_enabled: bool = False,
) -> dict:
    device = resolve_device(npu_enabled, npu_device_id, cuda_enabled)
    encoder = ClipEncoder(model_name, pretrained, device)
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    times: dict[int, list[float]] = defaultdict(list)
    thumbnails: dict[int, str] = {}
    pending_frames: list[np.ndarray] = []
    pending_meta: list[tuple[int, float]] = []
    thumbnail_dir = Path(thumbnail_dir)
    total_frames = 0

    def flush() -> None:
        nonlocal pending_frames, pending_meta
        if not pending_frames:
            return
        vectors = encoder.encode_frames(pending_frames)
        for (bucket, timestamp), vector in zip(pending_meta, vectors):
            buckets[bucket].append(vector)
            times[bucket].append(timestamp)
        pending_frames, pending_meta = [], []

    for timestamp, frame in iter_sampled_frames(video_path, sample_fps):
        bucket = int(timestamp // segment_seconds)
        if bucket not in thumbnails:
            thumbnail = thumbnail_dir / f"visual_{bucket:06d}.jpg"
            save_thumbnail(frame, thumbnail)
            thumbnails[bucket] = thumbnail.name
        pending_frames.append(frame)
        pending_meta.append((bucket, timestamp))
        total_frames += 1
        if len(pending_frames) >= batch_size:
            flush()
    flush()
    if not buckets:
        raise RuntimeError("未从视频抽取到画面")

    bucket_ids = sorted(buckets)
    embeddings = np.stack([normalize(np.mean(buckets[bucket], axis=0)) for bucket in bucket_ids])
    starts = np.asarray([bucket * segment_seconds for bucket in bucket_ids], dtype=np.float32)
    ends = np.asarray([max(times[bucket]) + 1 / sample_fps for bucket in bucket_ids], dtype=np.float32)
    thumbnail_names = np.asarray([thumbnails[bucket] for bucket in bucket_ids], dtype="U128")
    atomic_save_npz(
        output_path,
        embeddings=embeddings.astype(np.float32),
        start_times=starts,
        end_times=ends,
        thumbnails=thumbnail_names,
        model=np.asarray([model_name]),
    )
    return {"segments": len(bucket_ids), "frames": total_frames, "device": device}
