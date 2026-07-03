from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return data


def hf_snapshot_exists(target: Path, model_id: str) -> bool:
    repo_dir = target / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.exists():
        repo_dir = target / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def download_hf_model(target: Path, model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
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
            verified = True
    elif kind in {"directory", "insightface", "whisper", "rapidocr"}:
        target.mkdir(parents=True, exist_ok=True) if allow_download and kind in {"insightface", "whisper", "rapidocr"} else None
        verified = target.exists()
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
        manifest = load_json(Path(args.manifest))
        allow_download = bool(manifest.get("allow_download", False)) and args.download
        entries = [verify_entry(item, allow_download) for item in manifest.get("models", [])]
        write_lock(Path(args.lock), manifest, entries)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps({"verified": len(entries), "lock": args.lock}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
