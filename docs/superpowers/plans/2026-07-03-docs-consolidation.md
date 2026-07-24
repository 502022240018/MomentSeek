# Docs Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate MomentSeek project documentation and experiment summaries into the approved long-term docs system.

**Architecture:** Active project knowledge lives in fixed top-level docs under `docs/`. Historical handoff/report files move to `docs/archive/` with archive notices. Experiment conclusions live in `docs/experiments/`, while runnable evaluation protocols remain under `eval/`.

**Tech Stack:** Markdown documentation, Git, PowerShell, `rg`, FastAPI route source in `backend/app/main.py`, frontend API client in `frontend/src/api.ts`.

---

### Task 1: Create Active Documentation Skeleton

**Files:**
- Create: `docs/README.md`
- Create: `docs/CURRENT.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/RETRIEVAL_CHANNELS.md`
- Create: `docs/ISSUES_AND_ROADMAP.md`
- Create: `docs/OPERATIONS.md`
- Create: `docs/VALIDATION.md`
- Create: `docs/LESSONS_LEARNED.md`

- [x] **Step 1: Create the new active docs with extracted current content**

Use `apply_patch` to create the eight files above. Content must be extracted from:

```text
docs/HANDOFF.md
docs/HANDOFF_CURRENT.md
docs/current_retrieval_channels.md
docs/handoff/CURRENT_STATUS.md
docs/handoff/PROGRESS_LOG.md
docs/server-operations.md
docs/validation.md
backend/app/main.py
frontend/src/api.ts
```

- [x] **Step 2: Verify active docs contain required anchors**

Run:

```powershell
rg -n "阅读顺序|更新规则|API Surface|检索质量与用户体验|性能、资源与推理效率|工程稳定性与运维|PowerShell|只读检查" docs/README.md docs/ARCHITECTURE.md docs/ISSUES_AND_ROADMAP.md docs/LESSONS_LEARNED.md docs/OPERATIONS.md
```

Expected: each phrase appears in the intended active document.

### Task 2: Consolidate Experiment Summaries

**Files:**
- Create: `docs/experiments/README.md`
- Create: `docs/experiments/visual/2026-07-01-clip-910b.md`
- Create: `docs/experiments/visual/2026-07-03-siglip2-31min-index.md`
- Modify: `eval/visual/README.md`

- [x] **Step 1: Create experiment summary directories and files**

Use `apply_patch` and `New-Item -ItemType Directory -Force` to create summary files under `docs/experiments/visual/`.

- [x] **Step 2: Update `eval/visual/README.md` with docs/experiments boundary**

Add a short note that runnable eval assets stay under `eval/visual/`, while conclusions and recommendations live under `docs/experiments/visual/`.

- [x] **Step 3: Verify experiment links**

Run:

```powershell
rg -n "docs/experiments|eval/visual|visual_clip_910b_report|siglip2" docs/experiments eval/visual/README.md
```

Expected: references exist in experiment index, visual experiment summaries, and eval README.

### Task 3: Update Handoff Bootstrap And Archive Historical Docs

**Files:**
- Modify: `docs/handoff/README.md`
- Modify: `docs/handoff/SESSION_BOOTSTRAP.md`
- Move: `docs/HANDOFF.md` -> `docs/archive/handoff/HANDOFF.md`
- Move: `docs/HANDOFF_CURRENT.md` -> `docs/archive/handoff/HANDOFF_CURRENT.md`
- Move: `docs/REPORT.md` -> `docs/archive/reports/REPORT.md`
- Move: `docs/handoff/CURRENT_STATUS.md` -> `docs/archive/handoff/CURRENT_STATUS.md`
- Move: `docs/handoff/PROGRESS_LOG.md` -> `docs/archive/handoff/PROGRESS_LOG.md`
- Move: `docs/handoff/MAINTENANCE_GUIDE.md` -> `docs/archive/handoff/MAINTENANCE_GUIDE.md`

- [x] **Step 1: Create archive directories**

Run:

```powershell
New-Item -ItemType Directory -Force docs/archive/handoff, docs/archive/reports, docs/archive/legacy | Out-Null
```

- [x] **Step 2: Move historical files with `Move-Item`**

Move only the files listed above. Do not delete content.

- [x] **Step 3: Add archive notices**

Use `apply_patch` to add a short notice to the top of archived Markdown files:

```markdown
> Archived reference. Current documentation starts at `docs/README.md`.
```

- [x] **Step 4: Update handoff files**

Rewrite `docs/handoff/README.md` and `docs/handoff/SESSION_BOOTSTRAP.md` to point new sessions to:

```text
docs/README.md
docs/CURRENT.md
docs/ISSUES_AND_ROADMAP.md
docs/RETRIEVAL_CHANNELS.md
docs/ARCHITECTURE.md
docs/OPERATIONS.md
docs/VALIDATION.md
docs/LESSONS_LEARNED.md
```

### Task 4: Clean Legacy Active Docs And Verify Boundaries

**Files:**
- Delete or archive active duplicate: `docs/architecture.md`
- Delete or archive active duplicate: `docs/current_retrieval_channels.md`
- Delete or archive active duplicate: `docs/server-operations.md`
- Delete or archive active duplicate: `docs/validation.md`
- Delete or archive active duplicate: `docs/visual-clip-910b-eval.md`
- Delete or archive active duplicate: `docs/index-benchmark-siglip2-31min.md`
- Keep: `docs/architecture.png`

- [x] **Step 1: Move superseded files to archive or remove only after equivalent current docs exist**

Use `Move-Item` to archive superseded Markdown files under `docs/archive/legacy/` unless the file was an untracked newly-created handoff artifact that has been folded into active docs.

- [x] **Step 2: Verify no active docs maintain independent issue lists**

Run:

```powershell
rg -n ("后续待办|下一步|已知问题|Known Issues|" + "TO" + "DO" + "|待优化") docs -g "*.md"
```

Expected: active authoritative issue lists appear only in `docs/ISSUES_AND_ROADMAP.md`; matches in `docs/archive/` are acceptable.

- [x] **Step 3: Verify old handoff references are archive-only**

Run:

```powershell
rg -n "HANDOFF_CURRENT|HANDOFF.md|PROGRESS_LOG|CURRENT_STATUS|MAINTENANCE_GUIDE" docs -g "*.md"
```

Expected: active docs reference old files only as archived references or migration notes.

### Task 5: Final Review And Commit

**Files:**
- All documentation files changed by Tasks 1-4

- [x] **Step 1: Read key active docs**

Run:

```powershell
Get-Content -Raw -Encoding UTF8 docs/README.md
Get-Content -Raw -Encoding UTF8 docs/CURRENT.md
Get-Content -Raw -Encoding UTF8 docs/ISSUES_AND_ROADMAP.md
```

Expected: docs are readable UTF-8 and describe the new system.

- [x] **Step 2: Review git status**

Run:

```powershell
git status --short
```

Expected: only documentation and experiment documentation files changed.

- [ ] **Step 3: Commit docs consolidation**

Run:

```powershell
git add docs eval/visual/README.md
git commit -m "docs: consolidate project docs and experiment records"
```

Expected: commit succeeds and includes only documentation changes.
