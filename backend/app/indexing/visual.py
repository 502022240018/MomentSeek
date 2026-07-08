from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.indexing.common import atomic_save_npz, normalize
from app.media import probe_video, read_frames, save_thumbnail


def resolve_device(npu_enabled: bool, npu_device_id: int, cuda_enabled: bool = False) -> str:
    if cuda_enabled:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
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
    duration_seconds: float | int | None = None,
    segment_strategy: str = "fixed",
    min_segment_seconds: float = 0.8,
    max_segment_seconds: float = 8.0,
    shot_detector: str = "simple",
    shot_detector_threshold: float = 0.20,
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
    thumbnails: dict[int, str] = {}
    frame_embeddings: list[np.ndarray] = []
    frame_times_ms: list[int] = []
    pending_frames: list[np.ndarray] = []
    pending_meta: list[tuple[int, int]] = []
    thumbnail_dir = Path(thumbnail_dir)
    total_frames = 0
    segment_ms = max(1, int(round(float(segment_seconds) * 1000)))
    requested_strategy = (segment_strategy or "fixed").strip().lower()
    if requested_strategy not in {"fixed", "shot"}:
        requested_strategy = "fixed"
    try:
        requested_detector = normalize_shot_detector(shot_detector)
    except ValueError:
        requested_detector = "simple"
    duration_ms = int(round(float(duration_seconds or 0) * 1000))
    explicit_segment_times: np.ndarray | None = None
    active_strategy = "fixed"
    active_detector = requested_detector
    segment_time_source = "inferred_from_segment_ms"
    if requested_strategy == "shot" and duration_ms > 0:
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
        if normalized:
            explicit_segment_times = np.asarray(normalized, dtype=np.int32)
            active_strategy = "shot"
            active_detector = requested_detector
            segment_time_source = "explicit"

    def flush() -> None:
        nonlocal pending_frames, pending_meta
        if not pending_frames:
            return
        vectors = encoder.encode_frames(pending_frames)
        for (_bucket, timestamp_ms), vector in zip(pending_meta, vectors):
            frame_embeddings.append(normalize(vector))
            frame_times_ms.append(timestamp_ms)
        pending_frames, pending_meta = [], []

    frame_segment_ids: list[int] = []
    for timestamp, frame in read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg):
        timestamp_ms = int(round(float(timestamp) * 1000))
        if explicit_segment_times is not None:
            bucket = _segment_id_for_timestamp(timestamp_ms, explicit_segment_times)
        else:
            bucket = timestamp_ms // segment_ms
        if bucket not in thumbnails:
            thumbnail = thumbnail_dir / f"visual_{bucket:06d}.jpg"
            save_thumbnail(frame, thumbnail)
            thumbnails[bucket] = thumbnail.name
        pending_frames.append(frame)
        pending_meta.append((bucket, timestamp_ms))
        frame_segment_ids.append(int(bucket))
        total_frames += 1
        if len(pending_frames) >= batch_size:
            flush()
    flush()
    if not frame_embeddings:
        raise RuntimeError("未从视频抽取到画面")

    frame_times_ms_array = np.asarray(frame_times_ms, dtype=np.int32)
    frame_segment_ids_array = np.asarray(frame_segment_ids, dtype=np.int32)
    embeddings = np.stack(frame_embeddings).astype(np.float32)
    order = np.argsort(frame_times_ms_array)
    frame_times_ms_array = frame_times_ms_array[order]
    frame_segment_ids_array = frame_segment_ids_array[order]
    embeddings = embeddings[order]
    inferred_duration_ms = int(frame_times_ms_array.max()) + max(1, int(round(1000 / sample_fps)))
    if duration_ms <= 0:
        duration_ms = inferred_duration_ms
    max_bucket = int(frame_segment_ids_array.max()) if len(frame_segment_ids_array) else 0
    if explicit_segment_times is not None:
        segments_total = int(len(explicit_segment_times))
    else:
        segments_total = len(_fixed_segments(duration_ms, segment_ms, max_bucket))
    segment_frame_offsets = np.searchsorted(
        frame_segment_ids_array,
        np.arange(segments_total + 1, dtype=np.int32),
        side="left",
    ).astype(np.int32)
    segments_with_frames = int(len(np.unique(frame_segment_ids_array)))
    empty_segments = max(0, segments_total - segments_with_frames)
    payload = {
        "frame_embeddings": embeddings.astype(np.float16),
        "frame_times_ms": frame_times_ms_array.astype(np.int32),
        "segment_frame_offsets": segment_frame_offsets,
    }
    if explicit_segment_times is not None:
        payload["segment_times_ms"] = explicit_segment_times.astype(np.int32)
    atomic_save_npz(output_path, **payload)
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
