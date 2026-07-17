#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
SOURCE_DIR="${SOURCE_DIR:-${WORK_ROOT}/platform}"
REPO_URL="${REPO_URL:-https://github.com/502022240018/MomentSeek.git}"
BRANCH="${BRANCH:-main}"
MIN_DEPLOY_COMMIT="${MIN_DEPLOY_COMMIT:-86d78ed}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nUPDATE_DEPLOY_FAILED: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nUPDATE_DEPLOY_FAILED_AT_LINE=%s\n" "$LINENO" >&2' ERR

for command_name in git curl docker; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command: $command_name"
done
mkdir -p "$WORK_ROOT" "$LOG_DIR"

log "1/5 Check GitHub connectivity"
if ! curl -ILsS --connect-timeout 5 --max-time 12 -o /dev/null \
  -w 'github http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\n' \
  https://github.com/; then
  printf 'WARNING: GitHub probe failed; continuing to the retrying Git operation.\n' >&2
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
  log "2/5 Clone source with retries"
  clone_ok=0
  for attempt in 1 2 3; do
    printf 'clone_attempt=%s/3\n' "$attempt"
    if git -c http.version=HTTP/1.1 clone \
      --depth 20 --filter=blob:none --no-tags \
      --branch "$BRANCH" --single-branch \
      "$REPO_URL" "$SOURCE_DIR"; then
      clone_ok=1
      break
    fi
    [[ ! -e "$SOURCE_DIR" ]] || fail "Clone left an unexpected target: $SOURCE_DIR"
    sleep $((attempt * 3))
  done
  [[ "$clone_ok" == 1 ]] || fail "Clone failed after 3 attempts"
elif [[ ! -d "$SOURCE_DIR/.git" ]]; then
  fail "$SOURCE_DIR exists but is not a Git repository"
else
  log "2/5 Validate existing checkout"
  current_origin="$(git -C "$SOURCE_DIR" remote get-url origin 2>/dev/null || true)"
  [[ "$current_origin" == "$REPO_URL" ]] || fail "Unexpected origin: $current_origin"
  printf '.server-build/\n' >>"$SOURCE_DIR/.git/info/exclude"
  sort -u "$SOURCE_DIR/.git/info/exclude" -o "$SOURCE_DIR/.git/info/exclude"
  git -C "$SOURCE_DIR" diff --quiet || fail "Tracked source files have local modifications"
  git -C "$SOURCE_DIR" diff --cached --quiet || fail "Source repository has staged modifications"

  log "3/5 Fetch shallow history and fast-forward, with retries"
  fetch_ok=0
  for attempt in 1 2 3; do
    printf 'fetch_attempt=%s/3\n' "$attempt"
    if git -C "$SOURCE_DIR" \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 \
      -c http.lowSpeedTime=60 \
      fetch \
      --deepen=20 --prune origin "$BRANCH"; then
      fetch_ok=1
      break
    fi
    sleep $((attempt * 3))
  done
  [[ "$fetch_ok" == 1 ]] || fail "GitHub fetch failed after 3 attempts"
  git -C "$SOURCE_DIR" checkout "$BRANCH"
  git -C "$SOURCE_DIR" merge --ff-only "origin/$BRANCH"
fi

log "4/5 Verify deployment revision and files"
git -C "$SOURCE_DIR" status -sb
git -C "$SOURCE_DIR" log -4 --oneline
git -C "$SOURCE_DIR" merge-base --is-ancestor "$MIN_DEPLOY_COMMIT" HEAD \
  || fail "Checkout does not contain required deployment commit $MIN_DEPLOY_COMMIT"
[[ -x "$SOURCE_DIR/scripts/deploy_ascend_shared_server.sh" ]] \
  || chmod +x "$SOURCE_DIR/scripts/deploy_ascend_shared_server.sh"
grep -q '"vite": "6.4.3"' "$SOURCE_DIR/frontend/package.json" \
  || fail "Validated frontend toolchain is not present"

log "5/5 Run versioned Ascend deployment"
exec "$SOURCE_DIR/scripts/deploy_ascend_shared_server.sh"
