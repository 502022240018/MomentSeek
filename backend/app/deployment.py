from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.settings import Settings


def load_release_manifest(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _first_optional_str(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text is not None:
            return text
    return None


def _image_tag_from_manifest(manifest: dict[str, Any], env_profile: str) -> str | None:
    image = manifest.get("image")
    if not isinstance(image, dict):
        return _optional_str(image)
    if env_profile.endswith(".ascend"):
        return _optional_str(image.get("ascend"))
    if env_profile.endswith(".cuda"):
        return _optional_str(image.get("cuda"))
    return _first_optional_str(image.get("cpu"), image.get("cuda"), image.get("ascend"))


def _model_manifest_from_release(manifest: dict[str, Any]) -> str | None:
    models = manifest.get("models")
    if isinstance(models, dict):
        return _optional_str(models.get("manifest"))
    return None


def build_deployment_info(settings: Settings) -> dict[str, str | None]:
    manifest = load_release_manifest(settings.release_manifest_path)
    env_profile = _optional_str(settings.env_profile) or _optional_str(manifest.get("env_profile")) or "dev.cpu"
    return {
        "env_profile": env_profile,
        "release_id": _optional_str(settings.release_id) or _optional_str(manifest.get("release_id")),
        "git_commit": _optional_str(settings.git_commit) or _optional_str(manifest.get("git_commit")),
        "image_tag": _optional_str(settings.image_tag) or _image_tag_from_manifest(manifest, env_profile),
        "model_manifest": _optional_str(settings.model_manifest) or _model_manifest_from_release(manifest),
    }
