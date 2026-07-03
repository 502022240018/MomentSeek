from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-profile", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--frontend-dist", default="frontend/dist")
    args = parser.parse_args()

    commit = run_git(["rev-parse", "HEAD"])
    short = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    release_id = f"{date}-{short}"
    dist_root = Path(args.frontend_dist)
    dist_hash = hash_files(dist_root.rglob("*")) if dist_root.exists() else None

    manifest = {
        "schema_version": 1,
        "release_id": release_id,
        "git_commit": commit,
        "branch": branch,
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
            "smoke_search": "required",
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
