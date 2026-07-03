# Development And Deployment Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-phase GitHub-ready development, model, and deployment workflow for MomentSeek.

**Architecture:** Add explicit dev/prod profiles, model manifests, deployment metadata, and verification scripts without changing retrieval behavior. Keep runtime data and model weights outside git, and make `/api/health` expose enough release metadata to identify what a server is running.

**Tech Stack:** Python 3.11, FastAPI, Pydantic settings, PowerShell, Bash, Node/Vite, Hugging Face cache layout, Markdown docs.

---

## Scope And Order

Implement first-phase deliverables from:

```text
docs/superpowers/specs/2026-07-03-development-deployment-workflow-design.md
```

Do not touch the shared server in this plan. Server deployment happens after local scripts and docs exist, and only with the safety checks in `docs/OPERATIONS.md`.

## File Map

Create:

```text
backend/app/deployment.py
backend/tests/test_deployment.py
deploy/env/dev.cpu.example
deploy/env/dev.cuda.example
deploy/env/staging.ascend.example
deploy/env/prod.ascend.example
deploy/models/dev-full.models.json
deploy/models/ascend-prod.models.json
deploy/releases/release.example.json
scripts/bootstrap_dev.ps1
scripts/bootstrap_dev.sh
scripts/start_backend.ps1
scripts/start_backend.sh
scripts/start_frontend.ps1
scripts/start_frontend.sh
scripts/verify_models.py
scripts/smoke_check.py
scripts/write_release_manifest.py
docs/DEVELOPMENT.md
docs/DEPLOYMENT.md
docs/MODELS.md
```

Modify:

```text
backend/app/settings.py
backend/app/schemas.py
backend/app/main.py
docs/README.md
docs/ARCHITECTURE.md
docs/CURRENT.md
docs/VALIDATION.md
docs/ISSUES_AND_ROADMAP.md
```

Do not commit generated files:

```text
.env
runtime/
models/
models/models.lock.json
frontend/dist/
backend/app/static/
```

---

### Task 1: Backend Deployment Metadata In Health

**Files:**
- Create: `backend/app/deployment.py`
- Create: `backend/tests/test_deployment.py`
- Modify: `backend/app/settings.py`
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing tests for deployment metadata**

Create `backend/tests/test_deployment.py`:

```python
import json

from app.deployment import build_deployment_info, load_release_manifest
from app.settings import Settings


def test_build_deployment_info_prefers_explicit_settings(tmp_path):
    settings = Settings(
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        env_profile="dev.cuda",
        release_id="dev-local",
        git_commit="abc123",
        image_tag="momentseek:dev",
        model_manifest="deploy/models/dev-full.models.json",
    )

    info = build_deployment_info(settings)

    assert info["env_profile"] == "dev.cuda"
    assert info["release_id"] == "dev-local"
    assert info["git_commit"] == "abc123"
    assert info["image_tag"] == "momentseek:dev"
    assert info["model_manifest"] == "deploy/models/dev-full.models.json"


def test_load_release_manifest_reads_json(tmp_path):
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({
        "release_id": "2026-07-03-abc123",
        "git_commit": "abc123",
        "env_profile": "staging.ascend",
        "models": {"manifest": "deploy/models/ascend-prod.models.json"},
        "image": {"ascend": "momentseek:ascend-abc123"},
    }), encoding="utf-8")

    data = load_release_manifest(manifest)

    assert data["release_id"] == "2026-07-03-abc123"
    assert data["git_commit"] == "abc123"
    assert data["env_profile"] == "staging.ascend"
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run from `video_retrieval_mvp/backend`:

```powershell
python -m pytest tests/test_deployment.py -v
```

Expected if `pytest` is installed:

```text
ModuleNotFoundError: No module named 'app.deployment'
```

If the local Python environment lacks pytest, record the exact error and continue implementing; final verification still needs a pytest-capable environment.

- [ ] **Step 3: Add deployment settings**

Modify `backend/app/settings.py` inside `Settings`:

```python
    env_profile: str = "dev.cpu"
    release_id: str | None = None
    git_commit: str | None = None
    image_tag: str | None = None
    model_manifest: str | None = None
    release_manifest_path: Path | None = None
```

These env vars map to:

```text
ENV_PROFILE
RELEASE_ID
GIT_COMMIT
IMAGE_TAG
MODEL_MANIFEST
RELEASE_MANIFEST_PATH
```

- [ ] **Step 4: Implement deployment metadata helpers**

Create `backend/app/deployment.py`:

```python
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
```

- [ ] **Step 5: Extend health response schema**

Modify `backend/app/schemas.py`:

```python
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
```

- [ ] **Step 6: Add deployment metadata to `/api/health`**

Modify imports in `backend/app/main.py`:

```python
from app.deployment import build_deployment_info
```

Modify `health()`:

```python
@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "app_version": __version__,
        **build_deployment_info(settings),
        "npu_enabled": settings.npu_enabled,
        "npu_device_id": settings.npu_device_id if settings.npu_enabled else None,
        "cuda_enabled": settings.cuda_enabled,
        "model_idle_policy": settings.model_idle_policy,
    }
```

- [ ] **Step 7: Run focused backend tests**

Run from `video_retrieval_mvp/backend`:

```powershell
python -m pytest tests/test_deployment.py tests/test_worker.py -v
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit Task 1**

```powershell
git add backend/app/deployment.py backend/app/settings.py backend/app/schemas.py backend/app/main.py backend/tests/test_deployment.py
git commit -m "feat: expose deployment metadata in health"
```

---

### Task 2: Env Profiles And Release/Model Manifests

**Files:**
- Create: `deploy/env/dev.cpu.example`
- Create: `deploy/env/dev.cuda.example`
- Create: `deploy/env/staging.ascend.example`
- Create: `deploy/env/prod.ascend.example`
- Create: `deploy/models/dev-full.models.json`
- Create: `deploy/models/ascend-prod.models.json`
- Create: `deploy/releases/release.example.json`

- [ ] **Step 1: Create deploy directories**

Run:

```powershell
New-Item -ItemType Directory -Force deploy/env, deploy/models, deploy/releases | Out-Null
```

- [ ] **Step 2: Add `dev.cpu` env profile**

Create `deploy/env/dev.cpu.example`:

```dotenv
ENV_PROFILE=dev.cpu
APP_PORT=8000
APP_DATA_DIR=runtime
APP_MODEL_DIR=models
APP_PUBLIC_URL=http://127.0.0.1:8000
NPU_ENABLED=false
CUDA_ENABLED=false
MODEL_IDLE_POLICY=process_exit
INDEXER_MODE=subprocess
CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=openai
VISUAL_MODEL=openclip-vit-b32
VISUAL_HF_CACHE_DIR=models/hf-cache
VISUAL_SAMPLE_FPS=1.0
VISUAL_SEGMENT_SECONDS=5.0
VISUAL_BATCH_SIZE=8
FACE_MODEL=buffalo_l
FACE_SAMPLE_FPS=1.0
FACE_PROVIDER=cpu
ASR_ENGINE=whisper
ASR_MODEL=base
ASR_DEVICE=cpu
ASR_LANGUAGE=zh
ASR_SEMANTIC_ENABLED=true
ASR_SEMANTIC_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ASR_SEMANTIC_DEVICE=cpu
ASR_SEMANTIC_LOCAL_FILES_ONLY=false
OCR_ENGINE=rapidocr
OCR_DEVICE=cpu
OCR_VERSION=PP-OCRv4
OCR_SAMPLE_FPS=0.05
OCR_SEMANTIC_ENABLED=true
MODEL_MANIFEST=deploy/models/dev-full.models.json
```

- [ ] **Step 3: Add `dev.cuda` env profile**

Create `deploy/env/dev.cuda.example`:

```dotenv
ENV_PROFILE=dev.cuda
APP_PORT=8000
APP_DATA_DIR=runtime
APP_MODEL_DIR=models
APP_PUBLIC_URL=http://127.0.0.1:8000
NPU_ENABLED=false
CUDA_ENABLED=true
MODEL_IDLE_POLICY=process_exit
INDEXER_MODE=subprocess
CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=openai
VISUAL_MODEL=openclip-vit-b32
VISUAL_HF_CACHE_DIR=models/hf-cache
VISUAL_SAMPLE_FPS=2.0
VISUAL_SEGMENT_SECONDS=5.0
VISUAL_BATCH_SIZE=16
FACE_MODEL=buffalo_l
FACE_SAMPLE_FPS=1.0
FACE_PROVIDER=cpu
ASR_ENGINE=whisper
ASR_MODEL=base
ASR_DEVICE=auto
ASR_LANGUAGE=zh
ASR_SEMANTIC_ENABLED=true
ASR_SEMANTIC_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ASR_SEMANTIC_DEVICE=cpu
ASR_SEMANTIC_LOCAL_FILES_ONLY=false
OCR_ENGINE=rapidocr
OCR_DEVICE=cpu
OCR_VERSION=PP-OCRv4
OCR_SAMPLE_FPS=0.05
OCR_SEMANTIC_ENABLED=true
MODEL_MANIFEST=deploy/models/dev-full.models.json
```

- [ ] **Step 4: Add Ascend env profiles**

Create `deploy/env/staging.ascend.example`:

```dotenv
ENV_PROFILE=staging.ascend
APP_PORT=18300
APP_DATA_DIR=/app/runtime
APP_MODEL_DIR=/app/models
APP_PUBLIC_URL=http://127.0.0.1:18300
NPU_ENABLED=true
NPU_DEVICE_ID=0
CUDA_ENABLED=false
MODEL_IDLE_POLICY=process_exit
INDEXER_MODE=subprocess
ASCEND_VISIBLE_DEVICES=2
ASCEND_RT_VISIBLE_DEVICES=2
TORCH_DEVICE_BACKEND_AUTOLOAD=0
CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=/app/models/ViT-B-32.openai.bin
VISUAL_MODEL=siglip2-so400m-384
VISUAL_HF_CACHE_DIR=/app/models/hf-cache
VISUAL_SAMPLE_FPS=5.0
VISUAL_SEGMENT_SECONDS=5.0
VISUAL_BATCH_SIZE=32
FACE_MODEL=buffalo_l
FACE_SAMPLE_FPS=1.0
FACE_PROVIDER=cann
ASR_ENGINE=whisper
ASR_MODEL=small
ASR_DEVICE=auto
ASR_LANGUAGE=zh
ASR_SEMANTIC_ENABLED=true
ASR_SEMANTIC_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ASR_SEMANTIC_DEVICE=cpu
ASR_SEMANTIC_LOCAL_FILES_ONLY=true
OCR_ENGINE=rapidocr
OCR_DEVICE=auto
OCR_VERSION=PP-OCRv4
OCR_SAMPLE_FPS=0.05
OCR_SEMANTIC_ENABLED=true
MODEL_MANIFEST=deploy/models/ascend-prod.models.json
```

Create `deploy/env/prod.ascend.example` with the same values except:

```dotenv
ENV_PROFILE=prod.ascend
APP_PUBLIC_URL=http://127.0.0.1:18300
```

- [ ] **Step 5: Add dev model manifest**

Create `deploy/models/dev-full.models.json`:

```json
{
  "schema_version": 1,
  "name": "dev-full",
  "allow_download": true,
  "models": [
    {
      "name": "visual_openclip_b32",
      "kind": "huggingface",
      "id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
      "target": "models/hf-cache",
      "required": true
    },
    {
      "name": "face_buffalo_l",
      "kind": "insightface",
      "id": "buffalo_l",
      "target": "models/insightface",
      "required": true
    },
    {
      "name": "whisper_base",
      "kind": "whisper",
      "id": "base",
      "target": "models/whisper",
      "required": true
    },
    {
      "name": "text_semantic_minilm",
      "kind": "huggingface",
      "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "target": "models/text-embeddings",
      "required": true
    },
    {
      "name": "rapidocr_ppocrv4",
      "kind": "rapidocr",
      "id": "PP-OCRv4",
      "target": "models/rapidocr",
      "required": true
    }
  ]
}
```

- [ ] **Step 6: Add Ascend model manifest**

Create `deploy/models/ascend-prod.models.json`:

```json
{
  "schema_version": 1,
  "name": "ascend-prod",
  "allow_download": false,
  "models": [
    {
      "name": "visual_siglip2_so400m_384",
      "kind": "huggingface",
      "id": "google/siglip2-so400m-patch14-384",
      "target": "/app/models/hf-cache",
      "required": true
    },
    {
      "name": "face_buffalo_l",
      "kind": "directory",
      "id": "buffalo_l",
      "target": "/app/models/insightface/models/buffalo_l",
      "required": true
    },
    {
      "name": "whisper_small",
      "kind": "directory",
      "id": "small",
      "target": "/app/models/whisper",
      "required": true
    },
    {
      "name": "text_semantic_minilm",
      "kind": "huggingface",
      "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "target": "/app/models/text-embeddings",
      "required": true
    },
    {
      "name": "rapidocr_ppocrv4",
      "kind": "directory",
      "id": "PP-OCRv4",
      "target": "/app/models/rapidocr",
      "required": true
    }
  ]
}
```

- [ ] **Step 7: Add release manifest example**

Create `deploy/releases/release.example.json`:

```json
{
  "schema_version": 1,
  "release_id": "2026-07-03-local-example",
  "git_commit": "0000000000000000000000000000000000000000",
  "branch": "main",
  "image": {
    "ascend": "momentseek-mvp:ascend-20260703-example",
    "cuda": "momentseek-mvp:cuda-20260703-example"
  },
  "frontend": {
    "build_command": "npm run build",
    "dist_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "mounted_to": "backend/app/static"
  },
  "models": {
    "manifest": "deploy/models/ascend-prod.models.json",
    "mount": "/app/models",
    "lock": "models/models.lock.json"
  },
  "runtime": {
    "mount": "/app/runtime",
    "migration": "none"
  },
  "env_profile": "staging.ascend",
  "verification": {
    "backend_tests": "required",
    "frontend_build": "required",
    "health": "required",
    "smoke_search": "required",
    "resource_check": "required"
  }
}
```

- [ ] **Step 8: Validate JSON manifests**

Run:

```powershell
python -m json.tool deploy/models/dev-full.models.json | Out-Null
python -m json.tool deploy/models/ascend-prod.models.json | Out-Null
python -m json.tool deploy/releases/release.example.json | Out-Null
```

Expected: exit code `0`.

- [ ] **Step 9: Commit Task 2**

```powershell
git add deploy/env deploy/models deploy/releases
git commit -m "docs: add deployment profiles and manifests"
```

---

### Task 3: Model Verification Script

**Files:**
- Create: `scripts/verify_models.py`

- [ ] **Step 1: Create the script with manifest parsing and lock writing**

Create `scripts/verify_models.py`:

```python
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

    manifest = load_json(Path(args.manifest))
    allow_download = bool(manifest.get("allow_download", False)) and args.download
    entries = [verify_entry(item, allow_download) for item in manifest.get("models", [])]
    write_lock(Path(args.lock), manifest, entries)
    print(json.dumps({"verified": len(entries), "lock": args.lock}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run model verifier in no-download mode on dev manifest**

Run:

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json
```

Expected on a fresh machine: fails with `required model is missing` because `--download` was not passed.

- [ ] **Step 3: Run model verifier in download mode on dev manifest**

Run only when network access is acceptable:

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json --download
```

Expected:

```json
{"verified": 5, "lock": "runtime/dev-models.lock.json"}
```

If network or package availability blocks the run, record the exact error. Do not change staging/prod to permit downloads.

- [ ] **Step 4: Validate prod manifest fails fast when models are absent**

Run:

```powershell
python scripts/verify_models.py --manifest deploy/models/ascend-prod.models.json --lock runtime/prod-models.lock.json
```

Expected on a non-Ascend local machine:

```text
required model is missing
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/verify_models.py
git commit -m "feat: add model manifest verifier"
```

---

### Task 4: Smoke Check And Release Manifest Scripts

**Files:**
- Create: `scripts/smoke_check.py`
- Create: `scripts/write_release_manifest.py`

- [ ] **Step 1: Add smoke check script**

Create `scripts/smoke_check.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()

    try:
        health = get_json(f"{args.base_url.rstrip('/')}/api/health")
        jobs = get_json(f"{args.base_url.rstrip('/')}/api/jobs")
    except urllib.error.URLError as exc:
        print(f"smoke_check failed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(health, dict) or health.get("status") != "ok":
        print(f"unexpected health response: {health}", file=sys.stderr)
        return 1
    if args.require_release and not health.get("release_id"):
        print("health response does not include release_id", file=sys.stderr)
        return 1
    if not isinstance(jobs, list):
        print(f"unexpected jobs response: {jobs}", file=sys.stderr)
        return 1

    print(json.dumps({"health": health, "jobs_count": len(jobs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add release manifest writer**

Create `scripts/write_release_manifest.py`:

```python
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
```

- [ ] **Step 3: Verify script help output**

Run:

```powershell
python scripts/smoke_check.py --help
python scripts/write_release_manifest.py --help
```

Expected: both commands exit `0` and print usage text.

- [ ] **Step 4: Generate a local release manifest**

Run:

```powershell
python scripts/write_release_manifest.py --env-profile dev.cuda --model-manifest deploy/models/dev-full.models.json --out runtime/release.local.json
python -m json.tool runtime/release.local.json | Out-Null
```

Expected: `runtime/release.local.json` exists and is valid JSON. It stays untracked because `runtime/` is ignored.

- [ ] **Step 5: Commit Task 4**

```powershell
git add scripts/smoke_check.py scripts/write_release_manifest.py
git commit -m "feat: add deployment smoke and release scripts"
```

---

### Task 5: Developer Bootstrap And Start Scripts

**Files:**
- Create: `scripts/bootstrap_dev.ps1`
- Create: `scripts/bootstrap_dev.sh`
- Create: `scripts/start_backend.ps1`
- Create: `scripts/start_backend.sh`
- Create: `scripts/start_frontend.ps1`
- Create: `scripts/start_frontend.sh`

- [ ] **Step 1: Add Windows bootstrap script**

Create `scripts/bootstrap_dev.ps1`:

```powershell
param(
  [string]$Profile = "dev.cuda",
  [switch]$DownloadModels
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not (Test-Path ".env")) {
  Copy-Item "deploy/env/$Profile.example" ".env"
}

New-Item -ItemType Directory -Force runtime, models | Out-Null

python -m pip install -r backend/requirements-cpu.txt
Push-Location frontend
npm install
npm run build
Pop-Location

$manifest = if ($Profile -like "*.ascend") { "deploy/models/ascend-prod.models.json" } else { "deploy/models/dev-full.models.json" }
$args = @("scripts/verify_models.py", "--manifest", $manifest, "--lock", "models/models.lock.json")
if ($DownloadModels) { $args += "--download" }
python @args

Write-Host "Bootstrap complete. Start backend with scripts/start_backend.ps1 and frontend with scripts/start_frontend.ps1."
```

- [ ] **Step 2: Add Linux bootstrap script**

Create `scripts/bootstrap_dev.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

profile="${1:-dev.cuda}"
download_flag="${2:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ ! -f .env ]]; then
  cp "deploy/env/${profile}.example" .env
fi

mkdir -p runtime models

python -m pip install -r backend/requirements-cpu.txt
(cd frontend && npm install && npm run build)

if [[ "$profile" == *.ascend ]]; then
  manifest="deploy/models/ascend-prod.models.json"
else
  manifest="deploy/models/dev-full.models.json"
fi

args=(scripts/verify_models.py --manifest "$manifest" --lock models/models.lock.json)
if [[ "$download_flag" == "--download" ]]; then
  args+=(--download)
fi
python "${args[@]}"

echo "Bootstrap complete. Start backend with scripts/start_backend.sh and frontend with scripts/start_frontend.sh."
```

- [ ] **Step 3: Add start scripts**

Create `scripts/start_backend.ps1`:

```powershell
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $root "backend")
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Create `scripts/start_frontend.ps1`:

```powershell
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $root "frontend")
npm run dev
```

Create `scripts/start_backend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Create `scripts/start_frontend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/frontend"
npm run dev
```

- [ ] **Step 4: Set executable bits for shell scripts**

Run:

```powershell
git update-index --chmod=+x scripts/bootstrap_dev.sh scripts/start_backend.sh scripts/start_frontend.sh
```

- [ ] **Step 5: Verify scripts parse**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap_dev.ps1 -Profile dev.cpu
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_backend.ps1 -?
```

Expected: bootstrap runs until dependency/model setup or reports a clear package/network error. The start script help command may start parsing and then complain about unknown `-?`; that confirms PowerShell syntax loaded.

On Linux or Git Bash with Bash available, run:

```bash
bash -n scripts/bootstrap_dev.sh
bash -n scripts/start_backend.sh
bash -n scripts/start_frontend.sh
```

Expected: exit code `0`.

- [ ] **Step 6: Commit Task 5**

```powershell
git add scripts/bootstrap_dev.ps1 scripts/bootstrap_dev.sh scripts/start_backend.ps1 scripts/start_backend.sh scripts/start_frontend.ps1 scripts/start_frontend.sh
git commit -m "feat: add developer bootstrap scripts"
```

---

### Task 6: Development, Deployment, And Model Docs

**Files:**
- Create: `docs/DEVELOPMENT.md`
- Create: `docs/DEPLOYMENT.md`
- Create: `docs/MODELS.md`
- Modify: `docs/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CURRENT.md`
- Modify: `docs/VALIDATION.md`
- Modify: `docs/ISSUES_AND_ROADMAP.md`

- [ ] **Step 1: Add `docs/DEVELOPMENT.md`**

Create a Chinese-first document with these exact sections:

```markdown
# 开发环境

## 目标

## Profile 选择

## Windows 快速启动

## Linux 快速启动

## 模型下载策略

## 启动后端

## 启动前端

## 验证命令

## 常见问题
```

Include these commands:

```powershell
Copy-Item deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.ps1 -Profile dev.cuda -DownloadModels
scripts/start_backend.ps1
scripts/start_frontend.ps1
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

```bash
cp deploy/env/dev.cuda.example .env
scripts/bootstrap_dev.sh dev.cuda --download
scripts/start_backend.sh
scripts/start_frontend.sh
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

- [ ] **Step 2: Add `docs/DEPLOYMENT.md`**

Create a Chinese-first document with these exact sections:

```markdown
# 部署流程

## 环境分层

## 标准目录

## Release Manifest

## Deployment Record

## Staging Ascend

## Prod Ascend

## 新服务器复刻

## 回滚原则

## 共享服务器安全要求
```

Include the standard path:

```text
/opt/momentseek/
  releases/
  current -> releases/<release-id>
  runtime/
  models/
  env/
  logs/
  deployment-record.json
```

Include the no-state-change safety reminder:

```text
任何服务器状态变更前，先执行 docs/OPERATIONS.md 的只读检查，并确认没有 active indexing jobs。
```

- [ ] **Step 3: Add `docs/MODELS.md`**

Create a Chinese-first document with these exact sections:

```markdown
# 模型管理

## 总原则

## 开发模型

## Ascend Staging/Prod 模型

## 模型目录

## Model Manifest

## Models Lock

## 下载和校验

## 线上禁止运行时下载
```

Document these paths:

```text
开发默认：models/
容器内：/app/models
当前服务器宿主机：/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp/models
```

- [ ] **Step 4: Update docs index and architecture**

Modify `docs/README.md` reading order to include:

```text
docs/DEVELOPMENT.md
docs/DEPLOYMENT.md
docs/MODELS.md
```

Modify update rules:

```text
开发启动流程变化 -> docs/DEVELOPMENT.md
部署流程或 release manifest 变化 -> docs/DEPLOYMENT.md
模型清单、缓存和下载策略变化 -> docs/MODELS.md
```

Modify `docs/ARCHITECTURE.md` to add a short `部署元信息` section referencing `/api/health` fields:

```text
env_profile, release_id, git_commit, image_tag, model_manifest
```

- [ ] **Step 5: Update current status, validation, and roadmap**

Modify `docs/CURRENT.md` with a short note:

```text
多人开发与可复刻部署方案已设计，第一阶段将新增 dev.cpu/dev.cuda/staging.ascend/prod.ascend profile 和 manifest。
```

Modify `docs/VALIDATION.md` with:

```powershell
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json
```

Modify `docs/ISSUES_AND_ROADMAP.md`:

```text
ENG-007 多人开发与可复刻部署第一阶段
优先级：P1
状态：in_progress
范围：development workflow / deployment
问题或目标：
  GitHub clone 后可以开发验证，staging/prod/new-server 可以按 manifest 复刻。
下一步：
  完成 docs、env profile、model manifest、bootstrap、smoke、health metadata。
```

Add a second-stage item:

```text
ENG-008 CI/CD 与镜像化部署
优先级：P2
状态：open
范围：deployment automation
问题或目标：
  第一阶段先手动 manifest 和脚本，后续再标准化 Dockerfile、compose、GitHub Actions、自动发布和回滚。
```

- [ ] **Step 6: Verify docs anchors**

Run:

```powershell
rg -n "Windows 快速启动|Linux 快速启动|Release Manifest|Deployment Record|线上禁止运行时下载|dev\\.cpu|dev\\.cuda|staging\\.ascend|prod\\.ascend" docs/DEVELOPMENT.md docs/DEPLOYMENT.md docs/MODELS.md docs/README.md
```

Expected: each term appears in the intended docs.

- [ ] **Step 7: Commit Task 6**

```powershell
git add docs/DEVELOPMENT.md docs/DEPLOYMENT.md docs/MODELS.md docs/README.md docs/ARCHITECTURE.md docs/CURRENT.md docs/VALIDATION.md docs/ISSUES_AND_ROADMAP.md
git commit -m "docs: add development and deployment workflow"
```

---

### Task 7: Final Validation

**Files:**
- All files changed by Tasks 1-6

- [ ] **Step 1: Check git status**

Run:

```powershell
git status --short
```

Expected: clean after all task commits.

- [ ] **Step 2: Run backend tests**

Run from `video_retrieval_mvp/backend`:

```powershell
python -m pytest tests/test_deployment.py tests/test_worker.py tests/test_search.py -v
```

Expected:

```text
passed
```

- [ ] **Step 3: Run frontend build**

Run from `video_retrieval_mvp/frontend`:

```powershell
npm run build
```

Expected:

```text
tsc -b && vite build
```

and exit code `0`.

- [ ] **Step 4: Validate JSON files**

Run from repo root:

```powershell
python -m json.tool deploy/models/dev-full.models.json | Out-Null
python -m json.tool deploy/models/ascend-prod.models.json | Out-Null
python -m json.tool deploy/releases/release.example.json | Out-Null
```

Expected: exit code `0`.

- [ ] **Step 5: Run script syntax checks**

Run:

```powershell
python scripts/smoke_check.py --help
python scripts/write_release_manifest.py --help
python scripts/verify_models.py --help
bash -n scripts/bootstrap_dev.sh
bash -n scripts/start_backend.sh
bash -n scripts/start_frontend.sh
```

Expected: help commands print usage; Bash syntax checks exit `0`.

- [ ] **Step 6: Run docs boundary checks**

Run:

```powershell
rg -n "开发启动流程变化|部署流程或 release manifest 变化|模型清单、缓存和下载策略变化" docs/README.md
rg -n "ENG-007|ENG-008" docs/ISSUES_AND_ROADMAP.md
rg -n "env_profile|release_id|git_commit|image_tag|model_manifest" docs/ARCHITECTURE.md docs/DEPLOYMENT.md
```

Expected: all terms are present.

- [ ] **Step 7: Final commit if validation edits were required**

If validation required small fixes:

```powershell
git add <changed-files>
git commit -m "chore: validate development deployment workflow"
```

If no fixes were required, do not create an empty commit.
