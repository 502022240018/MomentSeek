from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class IndexRequest(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["visual", "face", "asr"])
    visual_sample_fps: float | None = Field(default=None, gt=0, le=10)
    visual_segment_seconds: float | None = Field(default=None, gt=0, le=60)
    face_sample_fps: float | None = Field(default=None, gt=0, le=15)

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: list[str]) -> list[str]:
        allowed = {"visual", "face", "asr"}
        normalized = list(dict.fromkeys(item.lower() for item in value))
        if not normalized or any(item not in allowed for item in normalized):
            raise ValueError("modalities 只能包含 visual、face、asr")
        return normalized


class HealthResponse(BaseModel):
    status: str
    version: str
    npu_enabled: bool
    npu_device_id: int | None
    model_idle_policy: str

