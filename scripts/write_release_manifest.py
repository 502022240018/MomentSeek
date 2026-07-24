from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class ManifestError(Exception):
    pass


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def validate_model_manifest(path: Path) -> None:
    if not path.is_file():
        raise ManifestError(f"model manifest does not exist: {path}")
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ManifestError(f"model manifest is not valid UTF-8: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"model manifest is not valid JSON: {path}: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"could not read model manifest: {path}: {exc}") from exc


def collect_dist_files(dist_root: Path) -> list[tuple[str, Path]]:
    if not dist_root.is_dir():
        raise ManifestError(f"frontend dist directory does not exist: {dist_root}")
    files = [
        (path.relative_to(dist_root).as_posix(), path)
        for path in dist_root.rglob("*")
        if path.is_file()
    ]
    if not files:
        raise ManifestError(f"frontend dist directory is empty: {dist_root}")
    return sorted(files, key=lambda item: item[0])


def hash_files(files: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for relative_path, path in files:
        path_bytes = relative_path.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-profile", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--frontend-dist", default="frontend/dist")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty_status = run_git(["status", "--porcelain"])
    dirty = bool(dirty_status)
    if dirty and not args.allow_dirty:
        print(
            "release manifest requires a clean git worktree; commit or stash changes, "
            "or rerun with --allow-dirty",
            file=sys.stderr,
        )
        return 1

    model_manifest = Path(args.model_manifest)
    dist_root = Path(args.frontend_dist)
    try:
        validate_model_manifest(model_manifest)
        dist_files = collect_dist_files(dist_root)
    except ManifestError as exc:
        print(f"release manifest error: {exc}", file=sys.stderr)
        return 1

    commit = run_git(["rev-parse", "HEAD"])
    short = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    release_id = f"{date}-{short}"
    dist_hash = hash_files(dist_files)

    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "git_commit": commit,
        "branch": branch,
        "dirty": dirty,
        "image": {
            "ascend": f"momentseek-mvp:ascend-{short}",
            "cuda": f"momentseek-mvp:cuda-{short}",
        },
        "frontend": {
            "build_command": "npm run build",
            "dist_hash": dist_hash,
            "mounted_to": "backend/app/static",
        },
        "models": {
            "manifest": args.model_manifest,
            "mount": "/app/models",
            "lock": "models/models.lock.json",
        },
        "runtime": {
            "mount": "/app/runtime",
            "migration": "none",
        },
        "env_profile": args.env_profile,
        "verification": {
            "backend_tests": "required",
            "frontend_build": "required",
            "health": "required",
            "api_smoke": "required",
            "search_smoke": "required",
            "resource_check": "required",
        },
    }
    out = Path(args.out or f"deploy/releases/{release_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
