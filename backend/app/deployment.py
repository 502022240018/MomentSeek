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
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _image_tag_from_manifest(manifest: dict[str, Any], env_profile: str) -> str | None:
    image = manifest.get("image")
    if isinstance(image, str):
        return image
    if not isinstance(image, dict):
        return None
    if env_profile.endswith(".ascend"):
        return image.get("ascend")
    if env_profile.endswith(".cuda"):
        return image.get("cuda")
    return image.get("cpu") or image.get("cuda") or image.get("ascend")


def _model_manifest_from_release(manifest: dict[str, Any]) -> str | None:
    models = manifest.get("models")
    if isinstance(models, dict):
        value = models.get("manifest")
        return str(value) if value else None
    return None


def build_deployment_info(settings: Settings) -> dict[str, str | None]:
    manifest = load_release_manifest(settings.release_manifest_path)
    env_profile = settings.env_profile or str(manifest.get("env_profile") or "dev.cpu")
    return {
        "env_profile": env_profile,
        "release_id": settings.release_id or manifest.get("release_id"),
        "git_commit": settings.git_commit or manifest.get("git_commit"),
        "image_tag": settings.image_tag or _image_tag_from_manifest(manifest, env_profile),
        "model_manifest": settings.model_manifest or _model_manifest_from_release(manifest),
    }
