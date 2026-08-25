from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class IndexRequest(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["visual", "face", "asr"])
    visual_model: str | None = None
    visual_sample_fps: float | None = Field(default=None, gt=0, le=10)
    visual_segment_seconds: float | None = Field(default=None, gt=0, le=60)
    visual_segment_strategy: str | None = None
    visual_min_segment_seconds: float | None = Field(default=None, gt=0, le=60)
    visual_max_segment_seconds: float | None = Field(default=None, gt=0, le=60)
    visual_shot_detector: str | None = None
    visual_shot_threshold: float | None = Field(default=None, gt=0, le=1)
    face_sample_fps: float | None = Field(default=None, gt=0, le=15)
    ocr_sample_fps: float | None = Field(default=None, gt=0, le=5)
    asr_engine: str | None = None
    asr_model: str | None = None
    asr_language: str | None = None
    asr_speaker_enabled: bool | None = None

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: list[str]) -> list[str]:
        allowed = {"visual", "face", "asr", "ocr"}
        normalized = list(dict.fromkeys(item.lower() for item in value))
        if not normalized or any(item not in allowed for item in normalized):
            raise ValueError("modalities 只能包含 visual、face、asr、ocr")
        return normalized

    @field_validator("visual_model")
    @classmethod
    def validate_visual_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        aliases = {
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
            "openclip-b16": "openclip-vit-b16",
            "vit-b-16": "openclip-vit-b16",
            "openclip-l14": "openclip-vit-l14",
            "vit-l-14": "openclip-vit-l14",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "siglip2-so400m-384",
            "chinese-clip-vit-b16",
            "openclip-vit-b32",
            "openclip-vit-b16",
            "openclip-vit-l14",
        }
        if normalized not in allowed:
            raise ValueError("visual_model must be one of: " + ", ".join(sorted(allowed)))
        return normalized

    @field_validator("visual_segment_strategy")
    @classmethod
    def validate_visual_segment_strategy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        allowed = {"fixed", "shot"}
        if normalized not in allowed:
            raise ValueError("visual_segment_strategy 只能是 fixed 或 shot")
        return normalized

    @field_validator("visual_shot_detector")
    @classmethod
    def validate_visual_shot_detector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        aliases = {
            "content": "pyscenedetect_content",
            "pyscene_content": "pyscenedetect_content",
            "pyscenedetect": "pyscenedetect_content",
            "adaptive": "pyscenedetect_adaptive",
            "pyscene_adaptive": "pyscenedetect_adaptive",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {"simple", "pyscenedetect_content", "pyscenedetect_adaptive"}
        if normalized not in allowed:
            raise ValueError("visual_shot_detector must be simple, pyscenedetect_content, or pyscenedetect_adaptive")
        return normalized

    @field_validator("asr_model")
    @classmethod
    def validate_asr_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        allowed = {"tiny", "base", "small", "medium", "large", "large-v3", "turbo", "large-v3-turbo"}
        if normalized not in allowed:
            raise ValueError("asr_model 只能是 tiny、base、small、medium、large、large-v3、turbo、large-v3-turbo")
        return normalized

    @field_validator("asr_engine")
    @classmethod
    def validate_asr_engine(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower().replace("_", "-")
        aliases = {"fasterwhisper": "faster-whisper"}
        normalized = aliases.get(normalized, normalized)
        allowed = {"auto", "whisper", "funasr", "faster-whisper"}
        if normalized not in allowed:
            raise ValueError("asr_engine must be one of: auto, whisper, funasr, faster-whisper")
        return normalized

    @field_validator("asr_language")
    @classmethod
    def validate_asr_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        aliases = {
            "chinese": "zh",
            "中文": "zh",
            "mandarin": "zh",
            "cantonese": "yue",
            "粤语": "yue",
            "english": "en",
            "英文": "en",
            "spanish": "es",
            "西语": "es",
            "portuguese": "pt",
            "葡语": "pt",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {"auto", "zh", "yue", "en", "es", "pt", "ja", "ko", "fr", "de", "it", "ru"}
        if normalized not in allowed:
            raise ValueError("asr_language 只能是 auto、zh、yue、en、es、pt、ja、ko、fr、de、it、ru")
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


class FolderCreateRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("文件夹名称不能为空")
        if len(value) > 100:
            raise ValueError("文件夹名称过长")
        return value


class FolderRenameRequest(FolderCreateRequest):
    pass


class VideoFolderMembershipRequest(BaseModel):
    video_ids: list[str] = Field(min_length=1)
    folder_ids: list[str] = Field(default_factory=list)
    operation: Literal["add", "remove", "replace"] = "replace"

    @field_validator("video_ids", "folder_ids")
    @classmethod
    def validate_ids(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))
        if len(normalized) != len(value):
            raise ValueError("文件夹和视频 ID 必须为非空字符串且不能重复")
        return normalized


class SpeakerUpdateRequest(BaseModel):
    display_name: str | None = None
    representative_utterance_index: int | None = Field(default=None, ge=0)
    hidden: bool = False


class UtteranceUpdateRequest(BaseModel):
    corrected_track_id: int | None = Field(default=None, ge=-1)
    searchable: bool = True


class VoiceSearchRequest(BaseModel):
    query_video_id: str
    query_utterance_index: int = Field(ge=0)
    video_ids: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=200)


class VoiceOnlyEntityRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_voice_entity_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("人物名称不能为空")
        return value


class EntityUpdateRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_entity_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("人物名称不能为空")
        return value


class VoiceSampleRequest(BaseModel):
    video_id: str
    utterance_index: int = Field(ge=0)
    bind_track_id: int | None = Field(default=None, ge=0)


class FaceGroupLibraryRequest(BaseModel):
    asset_version: str
    group_version: str
    entity_id: str | None = None
    new_entity_name: str | None = None

    @field_validator("asset_version", "group_version")
    @classmethod
    def validate_asset_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("版本不能为空")
        return value

    @field_validator("entity_id", "new_entity_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class HealthResponse(BaseModel):
    status: str
    version: str
    app_version: str | None = None
    env_profile: str | None = None
    release_id: str | None = None
    git_commit: str | None = None
    image_tag: str | None = None
    model_manifest: str | None = None
    npu_enabled: bool
    npu_device_id: int | None
    cuda_enabled: bool = False
    model_idle_policy: str
    indexer_mode: str
    npu_worker_mode: str
    orchestration_enabled: bool = False
    orchestration_profile: str | None = None
    milvus_enabled: bool = False
    milvus_primary: bool = False
    query_models: dict = Field(default_factory=dict)
