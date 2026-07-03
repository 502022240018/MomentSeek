#!/usr/bin/env bash
set -euo pipefail

profile="${1:-dev.cuda}"
download_arg="${2:-}"

if [[ -n "$download_arg" && "$download_arg" != "--download" ]]; then
  echo "Usage: $0 [profile] [--download]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

profile_example="deploy/env/${profile}.example"
if [[ ! -f "$profile_example" ]]; then
  echo "Environment profile not found: $profile_example" >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  cp "$profile_example" ".env"
  echo "Created .env from $profile_example"
else
  echo ".env already exists; leaving it unchanged"
fi

mkdir -p runtime models

echo "Installing backend CPU requirements..."
python -m pip install -r backend/requirements-cpu.txt

if [[ ! -f "frontend/package.json" ]]; then
  echo "Frontend package.json not found: frontend/package.json" >&2
  exit 1
fi

echo "Installing frontend dependencies..."
(
  cd frontend
  npm install
  npm run build
)

if [[ "$profile" == *.ascend ]]; then
  manifest_name="ascend-prod"
else
  manifest_name="dev-full"
fi

manifest_path="deploy/models/${manifest_name}.models.json"
if [[ ! -f "$manifest_path" ]]; then
  echo "Model manifest not found: $manifest_path" >&2
  exit 1
fi

verify_args=(
  scripts/verify_models.py
  --manifest "$manifest_path"
  --lock models/models.lock.json
)

if [[ "$download_arg" == "--download" ]]; then
  verify_args+=(--download)
fi

echo "Verifying model manifest: $manifest_name"
python "${verify_args[@]}"

cat <<'NEXT_STEPS'

Bootstrap complete.
Next steps:
  1. Review .env and adjust local settings if needed.
  2. Start the backend: ./scripts/start_backend.sh
  3. Start the frontend: ./scripts/start_frontend.sh
NEXT_STEPS
