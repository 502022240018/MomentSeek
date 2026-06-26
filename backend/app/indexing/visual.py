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
        self._init_cv2_preprocess()

    def _init_cv2_preprocess(self) -> None:
        """Mirror the model's torchvision preprocess with cv2 for the frame path.

        The CLIP preprocess (PIL resize→crop→normalize) is pure CPU and dominates
        visual indexing — at 720p it is ~14ms/frame vs ~0.9ms NPU encode. cv2.resize
        is 30-40% faster and stays within 0.996 cosine of PIL (measured on 910B), so
        ranking is unaffected. We read resize/crop/mean/std off the actual transform
        pipeline so this matches whatever CLIP weights are loaded; if the pipeline is
        unexpected we leave ``_cv2_ok=False`` and fall back to the PIL path.
        """
        self._cv2_ok = False
        try:
            size = crop = None
            mean = std = None
            for transform in self.preprocess.transforms:
                name = type(transform).__name__
                if name == "Resize":
                    value = transform.size
                    size = value if isinstance(value, int) else min(value)
                elif name == "CenterCrop":
                    value = transform.size
                    crop = value if isinstance(value, int) else min(value)
                elif name == "Normalize":
                    mean = np.asarray(transform.mean, dtype=np.float32)
                    std = np.asarray(transform.std, dtype=np.float32)
            if size and crop and mean is not None and std is not None:
                self._resize_size, self._crop_size = int(size), int(crop)
                self._norm_mean, self._norm_std = mean, std
                self._cv2_ok = True
        except Exception:
            self._cv2_ok = False

    def _preprocess_cv2(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = self._resize_size / min(height, width)
        resized = cv2.resize(
            rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
        rows, cols = resized.shape[:2]
        top, left = (rows - self._crop_size) // 2, (cols - self._crop_size) // 2
        crop = resized[top:top + self._crop_size, left:left + self._crop_size].astype(np.float32) / 255.0
        crop = (crop - self._norm_mean) / self._norm_std
        return self.torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))

    def encode_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        if self._cv2_ok:
            tensors = [self._preprocess_cv2(frame) for frame in frames_bgr]
        else:
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
    encoder: "ClipEncoder | None" = None,
) -> dict:
    # encoder may be supplied by the warm pool (model already resident); otherwise
    # load it for this call (the process_exit path).
    if encoder is None:
        encoder = ClipEncoder(model_name, pretrained, resolve_device(npu_enabled, npu_device_id, cuda_enabled))
    device = encoder.device
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
