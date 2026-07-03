# MomentSeek Docs And Experiments Consolidation Design

## Goal

Rebuild the MomentSeek documentation and experiment-result system into a small set of long-lived, clearly owned files. Existing historical documents remain available as reference, but day-to-day project knowledge should move into stable current documents with explicit update rules.

## Current Problems

- Project knowledge is split across multiple generations of handoff documents, reports, channel notes, operation notes, and experiment summaries.
- Some documents mix current state, history, known issues, future work, commands, and experiment conclusions.
- Experiment definitions live under `eval/`, while experiment conclusions also appear in root `docs/` files.
- New Codex sessions currently need to read too many files before knowing which information is authoritative.
- Known issues and future work are scattered across `CURRENT_STATUS.md`, `PROGRESS_LOG.md`, `HANDOFF_CURRENT.md`, and topical docs.

## Target Documentation Structure

```text
docs/
  README.md
  CURRENT.md
  ARCHITECTURE.md
  RETRIEVAL_CHANNELS.md
  ISSUES_AND_ROADMAP.md
  OPERATIONS.md
  VALIDATION.md

  experiments/
    README.md
    visual/
      2026-07-01-clip-910b.md
      2026-07-03-siglip2-31min-index.md
    asr/
    ocr/
    face/

  handoff/
    README.md
    SESSION_BOOTSTRAP.md

  archive/
    handoff/
    reports/
    legacy/
```

`eval/` remains the home for runnable evaluation assets:

```text
eval/
  README.md
  visual/
    README.md
    image_retrieval/
    sequence_retrieval/
    manifests/
    outputs/
```

The repository should distinguish:

- `docs/experiments/`: human-readable experiment summaries, conclusions, recommendations, and links to outputs.
- `eval/`: datasets, manifests, query files, schemas, and instructions needed to reproduce evaluations.

## Fixed Document Responsibilities

### `docs/README.md`

The single entry point for humans and new Codex sessions.

Must include:

- Recommended reading order.
- Which file owns which kind of information.
- Update rules for new project changes.
- Pointers to current state, issues, retrieval channels, operations, validation, and experiments.

### `docs/CURRENT.md`

Current factual system snapshot only.

Must include:

- Current branch and repo context.
- Current server/container/port status.
- Current model choices and channel availability.
- Current public access mode.
- Short warnings that link to `ISSUES_AND_ROADMAP.md`.

Should not include:

- Long history.
- Detailed future work.
- Full experiment writeups.

### `docs/ARCHITECTURE.md`

Current architecture and code/system boundaries.

Must include:

- Backend/frontend/runtime overview.
- Data flow from upload to indexing to search.
- Model lifecycle and worker/indexing strategy.
- Main module responsibilities.
- Storage layout.

### `docs/RETRIEVAL_CHANNELS.md`

Authoritative contract for visual, face, ASR, and OCR channels.

Must include:

- Index frequency and model used by each channel.
- File formats and schema fields.
- Embedding dimensions and vector spaces.
- Recall granularity.
- Search and fusion behavior.
- Evidence fields returned to frontend.

### `docs/ISSUES_AND_ROADMAP.md`

The only place for known problems and future optimization items.

It has three fixed sections:

1. Retrieval Quality And UX
2. Performance, Resources, And Inference Efficiency
3. Engineering Stability And Operations

Each item should use this shape:

```text
ID:
Priority: P0 / P1 / P2 / P3
Status: open / investigating / planned / in_progress / done / deferred
Area:
Problem Or Goal:
Impact:
Evidence Or Context:
Next Step:
Related Files Or Experiments:
```

Section rules:

- Retrieval Quality And UX: search quality, false positives, ranking, chunk quality, evidence presentation, query profiles, result merging.
- Performance, Resources, And Inference Efficiency: memory management, model loading/release, warm pool, indexing speed, preprocessing speed, ASR speed, NPU utilization, parallelism, queue scheduling.
- Engineering Stability And Operations: documentation, frontend/backend maintainability, public access, auth, server SOP, shared environment safety, validation workflow, tests, stale jobs, job cancel.

### `docs/OPERATIONS.md`

Shared-server and public-access operating manual.

Must include:

- Strict read-only checks before any server operation.
- Allowed and forbidden server actions.
- MomentSeek container/process identification.
- Public tunnel modes and current mode.
- Health checks, Docker checks, NPU checks.
- How to avoid touching ComfyUI, VLLM, or other users' jobs.

### `docs/VALIDATION.md`

Validation and acceptance commands.

Must include:

- Local test commands.
- Backend/frontend smoke checks.
- Server health checks.
- Search smoke tests by modality.
- Documentation verification commands.
- Rule: no completion claim without fresh verification output.

### `docs/experiments/README.md`

Index of experiment summaries.

Must include:

- Experiment naming convention.
- Where raw outputs live.
- What belongs in an experiment summary.
- Cross-links to `eval/` definitions.

### `docs/handoff/`

New-session bootstrap only.

Must include:

- `README.md`: points to `docs/README.md` and explains that long-term knowledge lives in the fixed docs.
- `SESSION_BOOTSTRAP.md`: concise prompt for new Codex windows to read `docs/README.md`, `docs/CURRENT.md`, `docs/ISSUES_AND_ROADMAP.md`, `docs/RETRIEVAL_CHANNELS.md`, `docs/OPERATIONS.md`, and `docs/VALIDATION.md`.

Should not include:

- Separate long current-status documents.
- Separate progress logs.
- Independent issue lists.

### `docs/archive/`

Historical reference only.

Must include:

- Old handoff files.
- Old reports.
- Legacy snapshots.

Archived files are not authoritative and should start with a short archive notice pointing to the new document entry points.

## Migration Map

| Existing file | Target action |
|---|---|
| `docs/HANDOFF.md` | Archive under `docs/archive/handoff/`; extract still-current architecture, API, pipeline, and caveats into new fixed docs |
| `docs/HANDOFF_CURRENT.md` | Archive under `docs/archive/handoff/`; extract current server, topology, operations, model, data, and known issue details |
| `docs/REPORT.md` | Archive under `docs/archive/reports/`; extract useful experiment/result summary into `docs/experiments/README.md` or topic files |
| `docs/architecture.md` | Replace with `docs/ARCHITECTURE.md`; archive or keep old image only if referenced |
| `docs/current_retrieval_channels.md` | Rename or rewrite into `docs/RETRIEVAL_CHANNELS.md` |
| `docs/server-operations.md` | Rewrite into `docs/OPERATIONS.md` |
| `docs/validation.md` | Rewrite into `docs/VALIDATION.md` |
| `docs/visual-clip-910b-eval.md` | Move/rewrite into `docs/experiments/visual/2026-07-01-clip-910b.md` |
| `docs/index-benchmark-siglip2-31min.md` | Move/rewrite into `docs/experiments/visual/2026-07-03-siglip2-31min-index.md` |
| `docs/handoff/CURRENT_STATUS.md` | Fold into `docs/CURRENT.md`; archive or replace with pointer |
| `docs/handoff/PROGRESS_LOG.md` | Extract issues into `docs/ISSUES_AND_ROADMAP.md`; extract history into archive if needed |
| `docs/handoff/MAINTENANCE_GUIDE.md` | Fold update rules into `docs/README.md` |
| `docs/handoff/SESSION_BOOTSTRAP.md` | Keep and update |
| `eval/visual/*.md` | Keep under `eval/visual/` as evaluation protocol/data documentation |

## Initial Issue Classification

### Retrieval Quality And UX

- Visual multi-video false positives: SigLIP2 MaxSim improves short-object recall, but per-video percentile can rank unrelated videos' local best buckets too high during broad multi-video search.
- Visual single-frame spike risk: MaxSim can promote a 5s bucket based on one coincidentally similar frame; `visual_top3` and `visual_mean` should be evaluated as consistency signals.
- ASR chunk post-processing: overly short or overly long chunks can reduce semantic search quality.
- OCR chunk quality: current 0.05fps sampling creates coarse 20s chunks.

### Performance, Resources, And Inference Efficiency

- Model load/release overhead during indexing.
- Warm pool/indexer daemon deployment decision.
- Visual preprocessing bottleneck.
- ASR speed and model choice.
- NPU memory release and shared resource safety.
- Future parallelism and queue scheduling.

### Engineering Stability And Operations

- Documentation system consolidation.
- Public tunnel stability and clear failed-fetch diagnosis.
- Lack of auth for public demos.
- Frontend `main.tsx` size and component boundaries.
- Index completeness/status export scripts.
- Job cancel and stale-job cleanup.
- Shared-server SOP before any restart or state change.

## Update Rules

```text
System state changed -> docs/CURRENT.md
Architecture or module boundaries changed -> docs/ARCHITECTURE.md
Retrieval channel protocol/index format changed -> docs/RETRIEVAL_CHANNELS.md
Problem or future optimization discovered -> docs/ISSUES_AND_ROADMAP.md
Server/public access/deployment operation changed -> docs/OPERATIONS.md
Validation command or acceptance rule changed -> docs/VALIDATION.md
Experiment conclusion created -> docs/experiments/<area>/<date>-<topic>.md
Evaluation dataset/schema/run method changed -> eval/<area>/README.md or adjacent eval files
New Codex bootstrap changed -> docs/handoff/SESSION_BOOTSTRAP.md
```

No other document should maintain an independent issue list or future-work list. Other documents may link to `docs/ISSUES_AND_ROADMAP.md`.

## Migration Constraints

- Do not delete historical information; archive it.
- Do not overwrite user changes.
- Use `apply_patch` for manual file edits.
- Keep runtime data untouched.
- Do not perform server state changes during documentation consolidation.
- Keep new docs concise and current; detailed history belongs in archive or experiment summaries.
- Prefer links over duplicated content.

## Verification Plan

After migration:

- Run `rg -n ("后续待办|下一步|已知问题|Known Issues|" + "TO" + "DO" + "|待优化") docs` and confirm authoritative lists only live in `docs/ISSUES_AND_ROADMAP.md` or archived files.
- Run `rg -n "HANDOFF_CURRENT|HANDOFF.md|PROGRESS_LOG|CURRENT_STATUS" docs` and confirm active docs point to new entry points, not old authoritative files.
- Run `git status --short` to review the exact documentation changes.
- Read `docs/README.md`, `docs/CURRENT.md`, and `docs/ISSUES_AND_ROADMAP.md` top to bottom once before declaring the documentation system ready for review.
