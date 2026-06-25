from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class IndexRequest(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["visual", "face", "asr"])
    visual_sample_fps: float | None = Field(default=None, gt=0, le=10)
    visual_segment_seconds: float | None = Field(default=None, gt=0, le=60)
    face_sample_fps: float | None = Field(default=None, gt=0, le=15)
    asr_model: str | None = None
    asr_language: str | None = None

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: list[str]) -> list[str]:
        allowed = {"visual", "face", "asr"}
        normalized = list(dict.fromkeys(item.lower() for item in value))
        if not normalized or any(item not in allowed for item in normalized):
            raise ValueError("modalities 只能包含 visual、face、asr")
        return normalized

    @field_validator("asr_model")
    @classmethod
    def validate_asr_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        allowed = {"tiny", "base", "small", "medium", "large", "large-v3"}
        if normalized not in allowed:
            raise ValueError("asr_model 只能是 tiny、base、small、medium、large、large-v3")
        return normalized

    @field_validator("asr_language")
    @classmethod
    def validate_asr_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        aliases = {"chinese": "zh", "中文": "zh", "english": "en", "英文": "en"}
        normalized = aliases.get(normalized, normalized)
        allowed = {"auto", "zh", "en"}
        if normalized not in allowed:
            raise ValueError("asr_language 只能是 auto、zh、en")
        return normalized


class VideoRenameRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("视频名称不能为空")
        if len(normalized) > 200:
            raise ValueError("视频名称过长")
        return normalized


class HealthResponse(BaseModel):
    status: str
    version: str
    npu_enabled: bool
    npu_device_id: int | None
    cuda_enabled: bool = False
    model_idle_policy: str
