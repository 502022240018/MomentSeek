param(
    [string]$Profile = "dev.cuda",
    [switch]$DownloadModels
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$ProfileExample = Join-Path $RepoRoot "deploy/env/$Profile.example"
if (-not (Test-Path -LiteralPath $ProfileExample)) {
    throw "Environment profile not found: deploy/env/$Profile.example"
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath $ProfileExample -Destination ".env"
    Write-Host "Created .env from deploy/env/$Profile.example"
} else {
    Write-Host ".env already exists; leaving it unchanged"
}

New-Item -ItemType Directory -Force -Path "runtime", "models" | Out-Null

Write-Host "Installing backend CPU requirements..."
python -m pip install -r "backend/requirements-cpu.txt"

if (-not (Test-Path -LiteralPath "frontend/package.json")) {
    throw "Frontend package.json not found: frontend/package.json"
}

Write-Host "Installing frontend dependencies..."
Push-Location "frontend"
try {
    npm install
    npm run build
} finally {
    Pop-Location
}

if ($Profile -like "*.ascend") {
    $ManifestName = "ascend-prod"
} else {
    $ManifestName = "dev-full"
}

$ManifestPath = "deploy/models/$ManifestName.models.json"
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Model manifest not found: $ManifestPath"
}

$VerifyArgs = @(
    "scripts/verify_models.py",
    "--manifest",
    $ManifestPath,
    "--lock",
    "models/models.lock.json"
)

if ($DownloadModels) {
    $VerifyArgs += "--download"
}

Write-Host "Verifying model manifest: $ManifestName"
python @VerifyArgs

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next steps:"
Write-Host "  1. Review .env and adjust local settings if needed."
Write-Host "  2. Start the backend: .\scripts\start_backend.ps1"
Write-Host "  3. Start the frontend: .\scripts\start_frontend.ps1"
