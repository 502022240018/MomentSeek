#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/momentseek-29154}"
REPO_URL="${REPO_URL:-https://github.com/502022240018/MomentSeek.git}"
BRANCH="${BRANCH:-main}"
TARGET_DIR="${TARGET_DIR:-${WORK_ROOT}/platform}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

log "Check required commands"
for command_name in git curl; do
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command: ${command_name}"
done

log "Check host access to GitHub"
curl -ILsS --connect-timeout 5 --max-time 20 -o /dev/null \
  -w 'github http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\n' \
  https://github.com/

log "Prepare work directories under ${WORK_ROOT}"
mkdir -p \
  "$WORK_ROOT" \
  "$WORK_ROOT/logs" \
  "$WORK_ROOT/models/platform" \
  "$WORK_ROOT/runtime" \
  "$WORK_ROOT/releases" \
  "$WORK_ROOT/cache"

if [[ ! -e "$TARGET_DIR" ]]; then
  log "Clone ${REPO_URL} branch ${BRANCH} (shallow, HTTP/1.1, retry enabled)"
  clone_ok=0
  for attempt in 1 2 3; do
    printf 'clone_attempt=%s/3\n' "$attempt"
    if git \
      -c http.version=HTTP/1.1 \
      -c http.lowSpeedLimit=1 \
      -c http.lowSpeedTime=60 \
      clone \
      --depth 1 \
      --filter=blob:none \
      --no-tags \
      --branch "$BRANCH" \
      --single-branch \
      "$REPO_URL" "$TARGET_DIR"; then
      clone_ok=1
      break
    fi
    if [[ -e "$TARGET_DIR" ]]; then
      if [[ -d "$TARGET_DIR" && ! -e "$TARGET_DIR/.git" && -z "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        rmdir "$TARGET_DIR"
      else
        fail "Clone failed and left a non-empty target at ${TARGET_DIR}; refusing to remove it"
      fi
    fi
    sleep $((attempt * 2))
  done
  [[ "$clone_ok" == 1 ]] || fail "GitHub clone failed after 3 attempts"
elif [[ -d "$TARGET_DIR/.git" ]]; then
  log "Existing Git repository found; verify it before updating"
  current_url="$(git -C "$TARGET_DIR" remote get-url origin 2>/dev/null || true)"
  [[ "$current_url" == "$REPO_URL" ]] || fail "Unexpected origin: ${current_url}"

  if [[ -n "$(git -C "$TARGET_DIR" status --porcelain)" ]]; then
    git -C "$TARGET_DIR" status -sb
    fail "Repository has local changes; refusing to overwrite them"
  fi

  git -C "$TARGET_DIR" fetch --prune origin "$BRANCH"
  git -C "$TARGET_DIR" checkout "$BRANCH"
  git -C "$TARGET_DIR" merge --ff-only "origin/$BRANCH"
else
  fail "${TARGET_DIR} exists but is not a Git repository"
fi

log "Source checkout summary"
git -C "$TARGET_DIR" status -sb
git -C "$TARGET_DIR" remote -v
git -C "$TARGET_DIR" log -1 --date=iso --format='commit=%H%nshort=%h%nsubject=%s%nauthor=%an%ndate=%ad'

log "Deployment files"
for relative_path in Dockerfile.ascend compose.ascend.yml backend/requirements-ascend.txt deploy/env/prod.ascend.example; do
  if [[ -f "$TARGET_DIR/$relative_path" ]]; then
    printf 'FOUND   %s\n' "$relative_path"
  else
    printf 'MISSING %s\n' "$relative_path"
  fi
done

log "Disk usage"
du -sh "$TARGET_DIR" "$WORK_ROOT/models" "$WORK_ROOT/runtime" 2>/dev/null || true
df -h "$WORK_ROOT"

printf '\nSOURCE_READY=1\nSOURCE_DIR=%s\n' "$TARGET_DIR"
printf 'No image was built and no package was installed.\n'
