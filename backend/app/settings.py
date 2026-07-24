from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
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
    # Indexing execution mode:
    #   "subprocess" (default) — API spawns a per-job worker; models load+exit per
    #     stage (process_exit). Safe, no resident NPU memory.
    #   "daemon" — API only enqueues jobs and starts the single warm-pool indexer
    #     daemon. Jobs and their channels run serially; pooled CLIP/InsightFace
    #     models can stay resident and skip reload/kernel compilation.
    indexer_mode: Literal["subprocess", "daemon"] = "subprocess"
    # Worker isolation inside daemon mode:
    #   "legacy" keeps every runtime/model in the daemon process itself.
    #   "isolated" gives each modality a persistent child process and its own
    #   NPU context, while the daemon remains the single serial scheduler.
    npu_worker_mode: Literal["legacy", "isolated"] = "legacy"
    # <= 0 keeps pooled models resident until daemon/container shutdown.
    indexer_idle_timeout_seconds: float = 300.0
    indexer_poll_seconds: float = 2.0
    indexer_worker_start_timeout_seconds: float = 30.0
    indexer_stage_max_attempts: int = 2

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
    visual_segment_strategy: str = "fixed"
    visual_min_segment_seconds: float = 0.8
    visual_max_segment_seconds: float = 8.0
    visual_shot_detector: str = "simple"
    visual_shot_threshold: float = 0.20
    visual_batch_size: int = 32

    face_model: str = "buffalo_l"
    face_sample_fps: float = 2.0
    face_provider: str = "cpu"
    # ONNX Runtime otherwise creates one intra-op thread per physical CPU core
    # for every InsightFace session. On the shared Ascend host that means
    # hundreds of threads for the detector + recognizer alone.
    face_ort_intra_op_threads: int = 8
    face_ort_inter_op_threads: int = 1

    asr_engine: str = "auto"
    # Used by ASR_ENGINE=whisper or ASR_ENGINE=faster-whisper. In auto mode this
    # is also the lightweight language probe model and the non-Chinese ASR path.
    asr_model: str = "turbo"
    asr_zh_model: str = "iic/SenseVoiceSmall"
    asr_device: str = "auto"
    asr_language: str = "auto"
    asr_vad_strategy: str = "silero_12s"
    asr_debug_artifacts: bool = False
    asr_save_raw_transcript: bool = False
    asr_model_local_files_only: bool = True
    asr_semantic_enabled: bool = True
    asr_semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Keep semantic text embeddings on CPU by default; sentence-transformers on
    # Ascend NPU is not guaranteed, and chunk embedding is cheap compared with ASR.
    asr_semantic_device: str = "cpu"
    asr_semantic_batch_size: int = 32
    # Shared servers should not hang an indexing job while downloading from
    # Hugging Face. Pre-cache/mount the model, or set this false for local dev.
    asr_semantic_local_files_only: bool = True

    speaker_device: str = "auto"
    speaker_model_repo: str = "3D-Speaker"
    speaker_model_cache_dir: str = "3dspeaker-cache"

    ocr_engine: str = "rapidocr"
    ocr_device: str = "auto"
    ocr_version: str = "PP-OCRv6"
    ocr_det_lang: str = "ch"
    ocr_rec_lang: str = "ch"
    ocr_model_type: str = "small"
    ocr_sample_fps: float = 0.5
    ocr_decode_height: int = 720
    ocr_min_confidence: float = 0.5
    ocr_semantic_enabled: bool = True
    ocr_npu_self_test: bool = True
    ocr_acl_model_dir: str = "rapidocr/ascend/910b4-cann9-profile"

    # Optional query orchestration layer. The JSON profile registry keeps
    # planner and reranker providers independent so they can share one VLM or
    # use separate specialist models.
    orchestration_enabled: bool = False
    orchestration_config_path: Path = Path("deploy/orchestration/qwen35-vllm.json")
    orchestration_profile: str = "qwen35-unified"
    orchestration_fail_open: bool = True
    orchestration_trace_enabled: bool = True
    orchestration_trace_path: Path = Path("runtime/orchestration-traces.jsonl")

    # Milvus is the primary vector store. SQLite remains the metadata/catalog
    # database and NPZ files are retained as a local recovery/search fallback.
    milvus_enabled: bool = True
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    # Bound fail-open retrieval latency. A request stops retrying Milvus after
    # its first failed operation and serves remaining videos from NPZ.
    milvus_query_timeout_seconds: float = 3.0
    milvus_read_enabled: bool = True
    milvus_write_enabled: bool = True
    milvus_fallback_enabled: bool = True
    milvus_shadow_compare_enabled: bool = False
    milvus_rollout_percent: int = 100
    # Local NPZ is written before Milvus, so "warn" preserves service
    # availability and leaves a recoverable artifact for later backfill.
    milvus_write_fail_policy: Literal["raise", "warn"] = "warn"

    @field_validator("indexer_mode", mode="before")
    @classmethod
    def normalize_indexer_mode(cls, value: object) -> object:
        # Backward-compatible alias used by older environment profiles.
        if isinstance(value, str) and value.strip().casefold() == "process_exit":
            return "subprocess"
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("npu_worker_mode", mode="before")
    @classmethod
    def normalize_npu_worker_mode(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("milvus_rollout_percent")
    @classmethod
    def validate_milvus_rollout_percent(cls, value: int) -> int:
        if not 0 <= value <= 100:
            raise ValueError("milvus_rollout_percent 必须在 0 到 100 之间")
        return value

    @field_validator("milvus_query_timeout_seconds")
    @classmethod
    def validate_milvus_query_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("milvus_query_timeout_seconds 必须大于 0")
        return value

    @property
    def model_idle_policy(self) -> Literal["process_exit", "idle_release", "resident"]:
        """Describe the effective model lifetime without a second configuration knob."""
        if self.indexer_mode == "subprocess":
            return "process_exit"
        if self.indexer_idle_timeout_seconds <= 0:
            return "resident"
        return "idle_release"

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
    def clip_cache_dir(self) -> Path:
        return self.app_data_dir / "clips"

    @property
    def frame_cache_dir(self) -> Path:
        return self.app_data_dir / "frame_cache"

    @property
    def legacy_thumbnail_dir(self) -> Path:
        """Legacy cache location retained only for cleanup during migration."""
        return self.app_data_dir / "thumbnails"

    @property
    def query_dir(self) -> Path:
        return self.app_data_dir / "queries"

    def ensure_dirs(self) -> None:
        for directory in (
            self.app_data_dir,
            self.app_model_dir,
            self.upload_dir,
            self.index_dir,
            self.clip_cache_dir,
            self.frame_cache_dir,
            self.query_dir,
            self.resolve_path(self.app_model_dir / self.speaker_model_cache_dir),
            self.resolve_path(self.visual_hf_cache_dir),
            self.resolve_path(self.orchestration_trace_path).parent,
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
