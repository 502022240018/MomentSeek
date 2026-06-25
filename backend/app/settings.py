from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_data_dir: Path = Path("runtime")
    app_model_dir: Path = Path("models")
    app_public_url: str = "http://127.0.0.1:8300"

    npu_enabled: bool = False
    npu_device_id: int = 7
    cuda_enabled: bool = False
    model_idle_policy: str = "process_exit"

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
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
    def query_dir(self) -> Path:
        return self.app_data_dir / "queries"

    def ensure_dirs(self) -> None:
        for directory in (
            self.app_data_dir,
            self.app_model_dir,
            self.upload_dir,
            self.index_dir,
            self.thumbnail_dir,
            self.query_dir,
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
