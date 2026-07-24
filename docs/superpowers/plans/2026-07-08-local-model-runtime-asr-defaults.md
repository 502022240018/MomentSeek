# Local Model Runtime And ASR Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make runtime model loading local-only by default, and switch ASR defaults to SenseVoiceSmall with faster-whisper turbo available as an explicit option.

**Architecture:** Add a small shared local model resolver for Hugging Face snapshot layouts and faster-whisper cache layouts. Use it from ASR/runtime experiment paths so libraries receive local filesystem paths and offline environment variables instead of model IDs that can trigger network metadata calls. Keep visual/text existing behavior intact while adding regression tests.

**Tech Stack:** Python, pytest, OpenAI Whisper, faster-whisper/CTranslate2, FunASR/ModelScope, Hugging Face cache layout, Pydantic settings.

---

### Task 1: Shared Local Model Resolver

**Files:**
- Create: `backend/app/model_sources.py`
- Test: `backend/tests/test_model_sources.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

- Hugging Face cache layout `models--repo--name/refs/main -> snapshots/<sha>`.
- Hugging Face cache layout without refs, choosing a complete snapshot.
- faster-whisper aliases `small` and `turbo` resolving to cached snapshot directories.
- Missing local model raising a clear `FileNotFoundError` when local-only is required.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest backend/tests/test_model_sources.py -q
```

Expected: fails because `app.model_sources` does not exist.

- [ ] **Step 3: Implement resolver**

Implement:

- `hf_cached_snapshot_path(model_cache_dir, model_id)`
- `resolve_hf_model_source(model_cache_dir, model_id, local_files_only=True)`
- `resolve_faster_whisper_model_source(model_root, model_name, local_files_only=True)`
- `offline_env(enabled=True)` context manager setting `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` while loading local models.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest backend/tests/test_model_sources.py -q
```

Expected: all tests pass.

### Task 2: Use Resolver In Runtime Loaders

**Files:**
- Modify: `backend/app/indexing/text_semantic.py`
- Modify: `backend/app/indexing/visual.py`
- Modify: `backend/app/indexing/asr.py`
- Test: `backend/tests/test_transcript.py`

- [ ] **Step 1: Add focused ASR tests**

Add tests proving `_whisper` requires local `<model>.pt` when local-only mode is true, and that `build_asr_index` passes local-only behavior to the semantic encoder.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest backend/tests/test_transcript.py backend/tests/test_model_sources.py -q
```

Expected: new local-only Whisper test fails before implementation.

- [ ] **Step 3: Wire resolver**

Use shared resolver in text semantic and visual to remove duplicated snapshot logic. In ASR, add `model_local_files_only=True` path checking for Whisper. Keep semantic failure downgrade behavior.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest backend/tests/test_transcript.py backend/tests/test_model_sources.py backend/tests/test_visual_cache.py -q
```

Expected: all selected tests pass.

### Task 3: ASR Defaults And Optional Turbo

**Files:**
- Modify: `backend/app/settings.py`
- Modify: `.env.example`
- Modify: `deploy/models/dev-full.models.json`
- Modify: `deploy/models/ascend-prod.models.json`
- Modify: `runtime-server/analysis/asr_model_vad_ab_20260708_run.py`

- [ ] **Step 1: Update defaults**

Set:

- `ASR_ENGINE=funasr`
- `ASR_ZH_MODEL=iic/SenseVoiceSmall`
- `ASR_MODEL=turbo`
- `ASR_SEMANTIC_LOCAL_FILES_ONLY=true`

Keep faster-whisper turbo as an explicit optional engine/experiment path rather than default backend in production ASR.

- [ ] **Step 2: Make faster-whisper experiments local-only**

In the experiment runner, resolve `small`, `medium`, and `turbo` to cached snapshot directories before creating `WhisperModel`, and set offline env during model loading.

- [ ] **Step 3: Verify syntax**

Run:

```bash
python -m py_compile backend/app/settings.py backend/app/indexing/asr.py backend/app/indexing/text_semantic.py backend/app/indexing/visual.py
docker exec momentseek-mvp-app python -m py_compile /app/runtime/analysis/asr_model_vad_ab_20260708_run.py
```

Expected: no output and exit code 0.

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/MODELS.md`
- Modify: `docs/DEVELOPMENT.md`
- Modify: `docs/experiments/asr/2026-07-08-asr-best3-metric-retest.md`

- [ ] **Step 1: Document policy**

Document that runtime code is local-only. Only bootstrap/verify commands with explicit download flags may download.

- [ ] **Step 2: Document ASR default**

Document SenseVoiceSmall as default Chinese/general ASR path, and faster-whisper turbo as optional fallback for languages/effect checks.

- [ ] **Step 3: Verify docs references**

Run:

```bash
rg -n "auto.*download|自动下载|allow_download|SenseVoiceSmall|faster-whisper turbo|HF_HUB_OFFLINE" README.md docs/MODELS.md docs/DEVELOPMENT.md docs/experiments/asr/2026-07-08-asr-best3-metric-retest.md
```

Expected: references match the new runtime-local policy.

### Task 5: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run selected tests**

Run:

```bash
pytest backend/tests/test_model_sources.py backend/tests/test_transcript.py backend/tests/test_visual_cache.py backend/tests/test_worker.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Confirm git diff**

Run:

```bash
git status --short
git diff -- backend/app/model_sources.py backend/app/indexing/asr.py backend/app/indexing/text_semantic.py backend/app/indexing/visual.py backend/app/settings.py .env.example deploy/models/dev-full.models.json deploy/models/ascend-prod.models.json
```

Expected: only scoped local-model and ASR-default changes plus docs/tests.
