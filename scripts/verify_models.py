from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ENTRY_FIELDS = ("name", "kind", "id", "target")
HF_CONFIG_FILES = {
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "modules.json",
}
HF_WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "open_clip_pytorch_model.bin",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return data


def validate_manifest(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if "allow_download" in manifest and not isinstance(manifest["allow_download"], bool):
        raise ValueError(f"manifest allow_download must be a boolean: {path}")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"manifest must contain a non-empty models list: {path}")

    entries = []
    for index, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(f"manifest models[{index}] must be a JSON object: {path}")
        missing = [
            field
            for field in REQUIRED_ENTRY_FIELDS
            if field not in item or item[field] is None or not str(item[field]).strip()
        ]
        if missing:
            raise ValueError(
                f"manifest models[{index}] missing required field(s): {', '.join(missing)}"
            )
        if "required" in item and not isinstance(item["required"], bool):
            raise ValueError(f"manifest models[{index}] required must be a boolean")
        entries.append(item)
    return entries


def is_non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_valid_json_file(path: Path) -> bool:
    if not is_non_empty_file(path):
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def has_non_empty_file_with_suffix(target: Path, suffixes: set[str]) -> bool:
    return target.is_dir() and any(
        is_non_empty_file(path) and path.suffix.lower() in suffixes
        for path in target.rglob("*")
    )


def hf_cache_dirs(target: Path, model_id: str) -> list[Path]:
    repo_name = f"models--{model_id.replace('/', '--')}"
    return [target / "hub" / repo_name, target / repo_name]


def hf_snapshot_has_assets(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False

    has_config = any(is_valid_json_file(snapshot / name) for name in HF_CONFIG_FILES)
    if not has_config:
        return False

    return any(
        is_non_empty_file(path)
        and (path.name in HF_WEIGHT_FILES or path.suffix.lower() in {".bin", ".safetensors"})
        for path in snapshot.rglob("*")
    )


def hf_snapshot_exists(target: Path, model_id: str) -> bool:
    for repo_dir in hf_cache_dirs(target, model_id):
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        if any(hf_snapshot_has_assets(snapshot) for snapshot in snapshots.iterdir()):
            return True
    return False


def verify_non_hf_target(kind: str, target: Path) -> bool:
    if kind == "directory":
        return has_non_empty_file_with_suffix(
            target, {".onnx", ".pt", ".bin", ".safetensors", ".json"}
        )
    if kind == "insightface":
        return has_non_empty_file_with_suffix(target, {".onnx"})
    if kind == "whisper":
        return has_non_empty_file_with_suffix(target, {".pt", ".bin", ".safetensors"})
    if kind == "rapidocr":
        return has_non_empty_file_with_suffix(target, {".onnx", ".bin"})
    raise ValueError(f"unsupported model kind: {kind}")


def download_hf_model(target: Path, model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_id, cache_dir=str(target)))


def verify_entry(entry: dict[str, Any], allow_download: bool) -> dict[str, Any]:
    name = str(entry["name"])
    kind = str(entry["kind"])
    model_id = str(entry["id"])
    target = Path(str(entry["target"]))
    required = bool(entry.get("required", True))
    verified = False
    local_path = target

    if kind == "huggingface":
        verified = hf_snapshot_exists(target, model_id)
        if not verified and allow_download:
            local_path = download_hf_model(target, model_id)
            verified = hf_snapshot_has_assets(local_path)
    elif kind in {"directory", "insightface", "whisper", "rapidocr"}:
        verified = verify_non_hf_target(kind, target)
    else:
        raise ValueError(f"unsupported model kind for {name}: {kind}")

    if required and not verified:
        raise FileNotFoundError(f"required model is missing: {name} -> {target}")

    return {
        "name": name,
        "kind": kind,
        "id": model_id,
        "local_path": str(local_path),
        "verified": verified,
    }


def write_lock(path: Path, manifest: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "profile": manifest.get("name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": entries,
    }
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", default="models/models.lock.json")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    try:
        manifest_path = Path(args.manifest)
        manifest = load_json(manifest_path)
        model_entries = validate_manifest(manifest, manifest_path)
        allow_download = manifest.get("allow_download", False) is True and args.download
        entries = [verify_entry(item, allow_download) for item in model_entries]
        write_lock(Path(args.lock), manifest, entries)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps({"verified": len(entries), "lock": args.lock}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
