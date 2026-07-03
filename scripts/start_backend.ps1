$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvPath = Join-Path $RepoRoot ".env"
$Port = "8000"

if (Test-Path -LiteralPath $EnvPath) {
    foreach ($Line in Get-Content -LiteralPath $EnvPath) {
        if ($Line -match "^\s*#" -or $Line -notmatch "^\s*APP_PORT\s*=\s*(.+?)\s*$") {
            continue
        }

        $Port = $Matches[1].Trim().Trim('"').Trim("'")
        break
    }
}

Set-Location $RepoRoot

python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $Port --reload
