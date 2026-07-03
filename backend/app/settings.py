from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_data_dir: Path = Path("runtime")
    app_model_dir: Path = Path("models")
    app_public_url: str = "http://127.0.0.1:8000"

    env_profile: str | None = None
    release_id: str | None = None
    git_commit: str | None = None
    image_tag: str | None = None
    model_manifest: str | None = None
    release_manifest_path: Path | None = None

    npu_enabled: bool = False
    npu_device_id: int = 0
    cuda_enabled: bool = False
    ascend_visible_devices: str | None = None
    ascend_rt_visible_devices: str | None = None
    torch_device_backend_autoload: str | None = None
    model_idle_policy: str = "process_exit"

    # Indexing execution mode:
    #   "subprocess" (default) — API spawns a per-job worker; models load+exit per
    #     stage (process_exit). Safe, no resident NPU memory.
    #   "daemon" — API only enqueues jobs and starts the warm-pool indexer daemon,
    #     which keeps CLIP/InsightFace resident and releases them after idle. Skips
    #     ~14.5s model load + kernel compile per job. Holds ~2.3GB while active.
    indexer_mode: str = "subprocess"
    indexer_idle_timeout_seconds: float = 300.0
    indexer_poll_seconds: float = 2.0

    # Frame source for indexing decode:
    #   "ffmpeg" (default) — ffmpeg multithreaded decode + fps/scale in one C pass,
    #     decoding directly to a small size (decode + preprocess is ~89% of visual
    #     and ~58% of face, all CPU). Falls back to cv2 if ffmpeg can't start.
    #   "cv2" — original single-threaded cv2 full-resolution decode.
    frame_reader: str = "ffmpeg"
    # Decode height fed to each stage (0 = source resolution). Visual only needs
    # 224 for CLIP, so 256 is plenty; face detector resizes to 640 internally, so
    # 720 keeps detection while cutting decode + pipe bytes. Source is never upscaled.
    visual_decode_height: int = 256
    face_decode_height: int = 720

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    visual_model: str = "siglip2-so400m-384"
    visual_hf_cache_dir: Path = Path("runtime/hf_cache")
    visual_sample_fps: float = 5.0
    visual_segment_seconds: float = 5.0
    visual_batch_size: int = 32

    face_model: str = "buffalo_l"
    face_sample_fps: float = 2.0
    face_provider: str = "cpu"

    asr_engine: str = "auto"
    asr_model: str = "small"
    asr_zh_model: str = "paraformer-zh"
    asr_device: str = "auto"
    asr_language: str = "zh"
    asr_semantic_enabled: bool = True
    asr_semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Keep semantic text embeddings on CPU by default; sentence-transformers on
    # Ascend NPU is not guaranteed, and chunk embedding is cheap compared with ASR.
    asr_semantic_device: str = "cpu"
    asr_semantic_batch_size: int = 32
    # Shared servers should not hang an indexing job while downloading from
    # Hugging Face. Pre-cache/mount the model, or set this false for local dev.
    asr_semantic_local_files_only: bool = True

    ocr_engine: str = "rapidocr"
    ocr_device: str = "auto"
    # RapidOCR 3.9's default PP-OCRv6 ONNX models attach to CANN on the Ascend
    # container but return empty OCR results in practice. PP-OCRv4 English mobile
    # is the current verified NPU baseline for video text overlays.
    ocr_version: str = "PP-OCRv4"
    ocr_det_lang: str = "en"
    ocr_rec_lang: str = "en"
    ocr_model_type: str = "mobile"
    ocr_sample_fps: float = 0.05
    ocr_decode_height: int = 720
    ocr_min_confidence: float = 0.5
    ocr_semantic_enabled: bool = True
    ocr_npu_self_test: bool = True

    @property
    def db_path(self) -> Path:
        return self.app_data_dir / "catalog.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.app_data_dir / "indexes"

    @property
    def thumbnail_dir(self) -> Path:
        return self.app_data_dir / "thumbnails"

    @property
    def clip_cache_dir(self) -> Path:
        return self.app_data_dir / "clips"

    @property
    def query_dir(self) -> Path:
        return self.app_data_dir / "queries"

    def ensure_dirs(self) -> None:
        for directory in (
            self.app_data_dir,
            self.app_model_dir,
            self.upload_dir,
            self.index_dir,
            self.thumbnail_dir,
            self.clip_cache_dir,
            self.query_dir,
            self.resolve_path(self.visual_hf_cache_dir),
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path

        candidates: list[Path] = []
        if path.parts and path.parts[0] == self.app_data_dir.name:
            candidates.append(self.app_data_dir.parent / path)
        candidates.extend([Path.cwd() / path, self.app_data_dir / path])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0].resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
