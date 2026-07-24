from __future__ import annotations

from pathlib import Path
from typing import Any

from app.indexing.common import atomic_save_json


INDEX_SCHEMA_VERSION = 3
MANIFEST_NAME = "index_manifest.json"


def manifest_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / MANIFEST_NAME


def load_index_manifest(index_dir: str | Path) -> dict[str, Any] | None:
    path = manifest_path(index_dir)
    if not path.exists():
        return None
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        return None
    channels = payload.get("channels")
    if not isinstance(channels, dict):
        payload["channels"] = {}
    return payload


def update_channel_manifest(
    index_dir: str | Path,
    *,
    video_id: str,
    duration_seconds: float | int | None,
    segment_seconds: float,
    channel: str,
    channel_manifest: dict[str, Any],
) -> dict[str, Any]:
    path = manifest_path(index_dir)
    payload = load_index_manifest(index_dir) or {
        "schema_version": INDEX_SCHEMA_VERSION,
        "video_id": video_id,
        "duration_ms": max(0, int(round(float(duration_seconds or 0) * 1000))),
        "segment_ms": max(1, int(round(float(segment_seconds) * 1000))),
        "channels": {},
    }
    payload["schema_version"] = INDEX_SCHEMA_VERSION
    payload["video_id"] = video_id
    payload["duration_ms"] = max(0, int(round(float(duration_seconds or 0) * 1000)))
    payload["segment_ms"] = max(1, int(round(float(segment_seconds) * 1000)))
    payload.setdefault("channels", {})[channel] = channel_manifest
    atomic_save_json(path, payload)
    return payload


def require_channel_manifest(index_dir: str | Path, video_name: str, channel: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_index_manifest(index_dir)
    if manifest is None:
        raise ValueError(f"视频 {video_name} 的索引版本过旧，请重跑索引")
    channels = manifest.get("channels") or {}
    channel_manifest = channels.get(channel)
    if not isinstance(channel_manifest, dict):
        raise ValueError(f"视频 {video_name} 缺少 {channel} v3 索引，请重跑该通道")
    return manifest, channel_manifest
