# Agent Instructions

## Shell Policy

The project is commonly used on Windows. Prefer `pwsh` for PowerShell-specific commands because Windows PowerShell 5.1 has older parsing, encoding, and JSON behavior.

Keep PowerShell snippets simple. Use it for commands such as `rg`, `Get-ChildItem`, `Test-Path`, `docker ps`, and `python -m pytest`.

Use Python for structured or repeated logic, including JSON parsing, metrics comparison, Markdown/CSV report generation, directory walking with filters, and any command that would require nested quoting or long pipelines.

Do not rely on WSL bash unless it has been freshly verified in the current environment.

For ASR evaluation summaries, prefer:

```powershell
python scripts/asr_eval_report.py --eval-dir <eval-dir> --baseline-dir <baseline-dir>
```
