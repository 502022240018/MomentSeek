$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location (Join-Path $RepoRoot "backend")

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
