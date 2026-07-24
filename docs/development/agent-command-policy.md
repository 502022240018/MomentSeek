# Agent Command Policy

This project is usually operated from Windows. The default shell may be Windows PowerShell 5.1, which has older parsing, encoding, and JSON behavior than PowerShell 7.

## Default Choices

Use PowerShell only for simple operating-system commands:

- `rg`
- `Get-ChildItem`
- `Test-Path`
- `docker ps`
- `docker exec` with a short command
- `python -m pytest`

Use Python scripts for logic-heavy work:

- parsing JSON or JSONL
- comparing evaluation runs
- building Markdown or CSV reports
- walking directories with non-trivial filters
- computing metrics, percentages, or rankings
- any command that would require nested quoting or long pipelines

Use `pwsh` instead of `powershell.exe` when PowerShell-specific logic is necessary. PowerShell 7 has more predictable behavior for modern command-line workflows.

## Durable Scripts

Repeated analysis should live in `scripts/` and have tests under `backend/tests/`. Avoid reimplementing the same report with one-off shell snippets.

Current ASR evaluation helper:

```powershell
python scripts/asr_eval_report.py --eval-dir runtime-server/analysis/asr_internal_eval_20260709_rerun_shuffle_5 --baseline-dir runtime-server/analysis/asr_internal_eval_20260709 --output runtime-server/analysis/asr_internal_eval_20260709_rerun_shuffle_5/report.md
```

## Agent Rule

When a command grows beyond a single straightforward shell operation, write or reuse a Python helper first. This reduces PowerShell 5.1 compatibility issues and leaves a reproducible artifact for future experiments.
