from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.indexing.common import atomic_save_npz, normalize
from app.media import read_frames, save_thumbnail


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


@dataclass(frozen=True)
class VisualModelConfig:
    key: str
    label: str
    backend: str
    model_name: str | None = None
    pretrained: str | None = None
    model_id: str | None = None


DEFAULT_VISUAL_MODEL = "siglip2-so400m-384"

VISUAL_MODEL_REGISTRY: dict[str, VisualModelConfig] = {
    "siglip2-so400m-384": VisualModelConfig(
        key="siglip2-so400m-384",
        label="SigLIP2 So400m patch14 384",
        backend="hf",
        model_id="google/siglip2-so400m-patch14-384",
    ),
    "chinese-clip-vit-b16": VisualModelConfig(
        key="chinese-clip-vit-b16",
        label="ChineseCLIP ViT-B/16",
        backend="hf",
        model_id="OFA-Sys/chinese-clip-vit-base-patch16",
    ),
    "openclip-vit-b32": VisualModelConfig(
        key="openclip-vit-b32",
        label="OpenCLIP ViT-B/32 openai",
        backend="openclip",
        model_name="ViT-B-32",
        pretrained="openai",
    ),
    "openclip-vit-b16": VisualModelConfig(
        key="openclip-vit-b16",
        label="OpenCLIP ViT-B/16 openai",
        backend="openclip",
        model_name="ViT-B-16",
        pretrained="openai",
    ),
    "openclip-vit-l14": VisualModelConfig(
        key="openclip-vit-l14",
        label="OpenCLIP ViT-L/14 openai",
        backend="openclip",
        model_name="ViT-L-14",
        pretrained="openai",
    ),
}

VISUAL_MODEL_ALIASES = {
    "siglip2": "siglip2-so400m-384",
    "siglip2-so400m": "siglip2-so400m-384",
    "google/siglip2-so400m-patch14-384": "siglip2-so400m-384",
    "chinese": "chinese-clip-vit-b16",
    "chineseclip": "chinese-clip-vit-b16",
    "chinese-clip-vit-b-16": "chinese-clip-vit-b16",
    "chinese-vit-b-16": "chinese-clip-vit-b16",
    "chinese vit-b-16": "chinese-clip-vit-b16",
    "ofa-sys/chinese-clip-vit-base-patch16": "chinese-clip-vit-b16",
    "openclip-b32": "openclip-vit-b32",
    "vit-b-32": "openclip-vit-b32",
    "vit-b/32": "openclip-vit-b32",
    "openclip-b16": "openclip-vit-b16",
    "vit-b-16": "openclip-vit-b16",
    "vit-b/16": "openclip-vit-b16",
    "openclip-l14": "openclip-vit-l14",
    "vit-l-14": "openclip-vit-l14",
    "vit-l/14": "openclip-vit-l14",
}


def normalize_visual_model(value: str | None) -> str:
    raw = (value or DEFAULT_VISUAL_MODEL).strip()
    key = raw.lower()
    normalized = VISUAL_MODEL_ALIASES.get(key, key)
    if normalized not in VISUAL_MODEL_REGISTRY:
        allowed = ", ".join(sorted(VISUAL_MODEL_REGISTRY))
        raise ValueError(f"Unknown visual_model={raw!r}. Allowed: {allowed}")
    return normalized


def visual_model_from_legacy(model_name: str | None, pretrained: str | None) -> str:
    model = (model_name or "").strip()
    source = (pretrained or "").strip()
    if model == "ViT-B-16" and source == "openai":
        return "openclip-vit-b16"
    if model == "ViT-L-14" and source == "openai":
        return "openclip-vit-l14"
    return "openclip-vit-b32"


def visual_model_config(value: str | None) -> VisualModelConfig:
    return VISUAL_MODEL_REGISTRY[normalize_visual_model(value)]


def _hf_cached_snapshot_path(model_cache_dir: str | Path | None, model_id: str) -> Path | None:
    if not model_cache_dir:
        return None
    cache_dir = Path(model_cache_dir)
    repo_name = f"models--{model_id.replace('/', '--')}"
    for repo_dir in (cache_dir / "hub" / repo_name, cache_dir / repo_name):
        snapshots = repo_dir / "snapshots"
        if not repo_dir.exists() or not snapshots.exists():
            continue
        ref = repo_dir / "refs" / "main"
        if ref.exists():
            snapshot = snapshots / ref.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return snapshot
        complete_snapshots = []
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            has_config = (snapshot / "config.json").exists()
            has_weights = (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists()
            if has_config and has_weights:
                complete_snapshots.append(snapshot)
        if complete_snapshots:
            return sorted(complete_snapshots, key=lambda path: path.name)[0]
        for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.name):
            if snapshot.is_dir():
                return snapshot
    return None


def _hf_model_cached(model_cache_dir: str | Path | None, model_id: str) -> bool:
    return _hf_cached_snapshot_path(model_cache_dir, model_id) is not None


def _hf_load_source(model_cache_dir: str | Path | None, model_id: str) -> tuple[str, bool]:
    snapshot = _hf_cached_snapshot_path(model_cache_dir, model_id)
    if snapshot is not None:
        return str(snapshot), True
    return model_id, False


def _resolve_openclip_pretrained(model_name: str, pretrained: str) -> str:
    if pretrained in {"openai", "laion2b_s34b_b79k"}:
        filename = f"{model_name}.{pretrained}.bin"
        for candidate in (
            Path("/app/models") / filename,
            Path("models") / filename,
            Path.cwd() / "models" / filename,
        ):
            if candidate.exists():
                return str(candidate)
    return pretrained


class OpenClipEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model_key = visual_model_from_legacy(model_name, pretrained)
        self.model_label = VISUAL_MODEL_REGISTRY[self.model_key].label
        self.backend = "openclip"
        source = _resolve_openclip_pretrained(model_name, pretrained)
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


class HfVisualEncoder:
    def __init__(
        self,
        config: VisualModelConfig,
        device: str,
        model_cache_dir: str | Path | None = None,
    ):
        import torch

        if device.startswith("npu"):
            import torch_npu  # noqa: F401

            torch.npu.set_device(device)
        elif device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(device)

        if model_cache_dir:
            # Keep using the deployment cache layout: HF_HOME/hub/models--...
            os.environ["HF_HOME"] = str(model_cache_dir)

        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.config = config
        self.model_key = config.key
        self.model_label = config.label
        self.backend = config.backend
        self.model_id = config.model_id or config.key
        self.dtype = torch.bfloat16 if device.startswith(("npu", "cuda")) else torch.float32
        load_source, local_files_only = _hf_load_source(model_cache_dir, self.model_id)
        offline_env = {}
        if local_files_only:
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
                offline_env[name] = os.environ.get(name)
                os.environ[name] = "1"
        try:
            self.processor = AutoProcessor.from_pretrained(
                load_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                load_source,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                local_files_only=local_files_only,
            ).to(device)
        finally:
            for name, old_value in offline_env.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value
        self.model.eval()

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if key == "pixel_values" and hasattr(value, "is_floating_point") and value.is_floating_point():
                    value = value.to(dtype=self.dtype)
            moved[key] = value
        return moved

    def _processor_text_kwargs(self) -> dict[str, Any]:
        # SigLIP/SigLIP2 expects max-length text padding. Dynamic padding made
        # short prompts behave almost random in our retrieval sweep.
        if "siglip" in self.model_key:
            kwargs: dict[str, Any] = {"padding": "max_length", "truncation": True}
            text_config = getattr(getattr(self.model, "config", None), "text_config", None)
            max_length = getattr(text_config, "max_position_embeddings", None)
            if max_length is None:
                tokenizer = getattr(self.processor, "tokenizer", None)
                max_length = getattr(tokenizer, "model_max_length", None)
                if max_length and max_length > 100_000:
                    max_length = None
            if max_length:
                kwargs["max_length"] = max_length
            return kwargs
        return {"padding": True, "truncation": True}

    def _image_features(self, images: list[Image.Image]):
        torch = self.torch
        batch = self.processor(images=images, return_tensors="pt")
        batch = self._move_batch(dict(batch))
        with torch.inference_mode():
            if hasattr(self.model, "get_image_features"):
                output = self.model.get_image_features(**batch)
            else:
                output = self.model(**batch).image_embeds
            if isinstance(output, (tuple, list)):
                output = output[0]
            return torch.nn.functional.normalize(output, dim=-1)

    def _text_features(self, texts: list[str]):
        torch = self.torch
        batch = self.processor(text=texts, return_tensors="pt", **self._processor_text_kwargs())
        batch = self._move_batch(dict(batch))
        with torch.inference_mode():
            if hasattr(self.model, "get_text_features"):
                output = self.model.get_text_features(**batch)
            elif hasattr(self.model, "text_model") and hasattr(self.model, "text_projection"):
                text_outputs = self.model.text_model(**batch, return_dict=True)
                pooled = getattr(text_outputs, "pooler_output", None)
                if pooled is None:
                    pooled = text_outputs.last_hidden_state[:, 0]
                output = self.model.text_projection(pooled)
            else:
                output = self.model(**batch).text_embeds
            if isinstance(output, (tuple, list)):
                output = output[0]
            return torch.nn.functional.normalize(output, dim=-1)

    def encode_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
        output = self._image_features(images)
        return output.float().cpu().numpy()

    def encode_query(self, text: str | None, image_path: str | None, alpha: float = 0.5) -> np.ndarray:
        torch = self.torch
        parts = []
        if text:
            encoded = self._text_features([text])
            parts.append((alpha if image_path else 1.0, encoded))
        if image_path:
            image = Image.open(image_path).convert("RGB")
            encoded = self._image_features([image])
            parts.append(((1 - alpha) if text else 1.0, encoded))
        if not parts:
            raise ValueError("瑙嗚鏌ヨ闇€瑕佹枃鏈垨鍙傝€冨浘")
        with torch.inference_mode():
            vector = sum(weight * value for weight, value in parts)
            vector = torch.nn.functional.normalize(vector, dim=-1)
        return vector.float().cpu().numpy()[0]


class ClipEncoder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        visual_model: str | None = None,
        model_cache_dir: str | Path | None = None,
    ):
        config = visual_model_config(visual_model or visual_model_from_legacy(model_name, pretrained))
        self.config = config
        if config.backend == "openclip":
            encoder = OpenClipEncoder(config.model_name or model_name, config.pretrained or pretrained, device)
        elif config.backend == "hf":
            encoder = HfVisualEncoder(config, device, model_cache_dir=model_cache_dir)
        else:
            raise ValueError(f"Unsupported visual model backend: {config.backend}")
        self._encoder = encoder
        self.device = encoder.device
        self.model_key = encoder.model_key
        self.model_label = encoder.model_label
        self.backend = encoder.backend
        self.model_id = getattr(encoder, "model_id", None)

    def encode_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        return self._encoder.encode_frames(frames_bgr)

    def encode_query(self, text: str | None, image_path: str | None, alpha: float = 0.5) -> np.ndarray:
        return self._encoder.encode_query(text, image_path, alpha=alpha)


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
    visual_model: str | None = None,
    model_cache_dir: str | Path | None = None,
    decode_height: int = 0,
    prefer_ffmpeg: bool = True,
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
    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    times: dict[int, list[float]] = defaultdict(list)
    thumbnails: dict[int, str] = {}
    frame_embeddings: list[np.ndarray] = []
    frame_times: list[float] = []
    frame_segment_ids: list[int] = []
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
            frame_embeddings.append(vector)
            frame_times.append(timestamp)
            frame_segment_ids.append(bucket)
        pending_frames, pending_meta = [], []

    for timestamp, frame in read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg):
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
        schema_version=np.asarray([2], dtype=np.int16),
        segment_ids=np.asarray(bucket_ids, dtype=np.int32),
        embeddings=embeddings.astype(np.float32),
        start_times=starts,
        end_times=ends,
        thumbnails=thumbnail_names,
        frame_embeddings=np.stack(frame_embeddings).astype(np.float32),
        frame_times=np.asarray(frame_times, dtype=np.float32),
        frame_segment_ids=np.asarray(frame_segment_ids, dtype=np.int32),
        model=np.asarray([encoder.model_label]),
        visual_model=np.asarray([encoder.model_key]),
        model_backend=np.asarray([encoder.backend]),
        model_id=np.asarray([encoder.model_id or ""]),
    )
    return {
        "segments": len(bucket_ids),
        "frames": total_frames,
        "schema_version": 2,
        "device": device,
        "visual_model": encoder.model_key,
        "model": encoder.model_label,
    }
