#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/momentseek-29154/models/platform}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/momentseek-29154/platform}"
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-platform}"
HOST_MANIFEST="${HOST_MANIFEST:-$PROJECT_ROOT/deploy/models/ascend-prod.models.json}"
MANIFEST="/tmp/ascend-prod.models.json"
LOG_DIR="${LOG_DIR:-/home/momentseek-29154/platform/logs}"
SKIP_SIGLIP="${SKIP_SIGLIP:-0}"
SPEAKER_SOURCE_REVISION="${SPEAKER_SOURCE_REVISION:-065629c313ea}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    "$@" && return 0
    printf 'attempt=%s/5 failed; retrying in 8 seconds\n' "$attempt" >&2
    sleep 8
  done
  return 1
}

[[ $(id -u) -eq 0 ]] || die "run this script as root"
docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || die "container not found: $CONTAINER_NAME"
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]] || die "container is not running"
mkdir -p "$MODEL_ROOT" "$LOG_DIR"
[[ -f "$HOST_MANIFEST" ]] || die "model manifest not found: $HOST_MANIFEST"
docker cp "$HOST_MANIFEST" "$CONTAINER_NAME:$MANIFEST" >/dev/null

log "1/8 Check model endpoints"
for url in \
  https://huggingface.co/ \
  https://modelscope.cn/ \
  https://github.com/ \
  https://www.modelscope.cn/models/RapidAI/RapidOCR; do
  curl -ILsS --connect-timeout 10 --max-time 30 -o /dev/null \
    -w 'url=%{url_effective} http=%{http_code} ip=%{remote_ip} time=%{time_total}s error=%{errormsg}\n' \
    "$url" || true
done

log "2/8 Download SenseVoiceSmall and MiniLM from ModelScope"
retry docker exec "$CONTAINER_NAME" python3 -c '
import shutil
from pathlib import Path
from modelscope import snapshot_download

repo = "iic/SenseVoiceSmall"
print(f"MODELSCOPE_DOWNLOAD_START={repo}", flush=True)
path = snapshot_download(repo, cache_dir="/app/models/funasr")
print(f"MODELSCOPE_DOWNLOAD_DONE={repo} path={path}", flush=True)

# ModelScope mirror of the upstream sentence-transformers repository.  Copy it
# into the cache layout expected by the platform local-only resolver.
repo = "Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2"
print(f"MODELSCOPE_DOWNLOAD_START={repo}", flush=True)
source = Path(snapshot_download(repo, cache_dir="/app/models/modelscope-cache"))
target = Path("/app/models/text-embeddings/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/modelscope")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    shutil.rmtree(target)
shutil.copytree(source, target, symlinks=False)
print(f"MODELSCOPE_DOWNLOAD_DONE={repo} path={target}", flush=True)
'

log "3/8 Download InsightFace buffalo_l from ModelScope"
FACE_DIR="$MODEL_ROOT/insightface/models/buffalo_l"
mkdir -p "$FACE_DIR"
if ! find "$FACE_DIR" -type f -name '*.onnx' -size +0c | grep -q .; then
  retry docker exec "$CONTAINER_NAME" python3 -c '
import shutil
from pathlib import Path
from modelscope import snapshot_download

source = Path(snapshot_download("LumilioPhotos/buffalo_l", cache_dir="/app/models/modelscope-cache"))
target = Path("/app/models/insightface/models/buffalo_l")
target.mkdir(parents=True, exist_ok=True)
files = list(source.rglob("*.onnx"))
if not files:
    raise RuntimeError(f"No ONNX files found in {source}")
for item in files:
    shutil.copy2(item, target / item.name)
print(f"BUFFALO_L_DONE files={len(files)} path={target}")
'
else
  printf 'SKIP: buffalo_l already exists\n'
fi

log "4/8 Download RapidOCR PP-OCRv6 assets"
OCR_DIR="$MODEL_ROOT/rapidocr"
mkdir -p "$OCR_DIR"
download_ocr() {
  local relative="$1" output="$2"
  [[ -s "$OCR_DIR/$output" ]] && { printf 'SKIP: %s\n' "$output"; return 0; }
  retry curl -fL --retry 3 --retry-delay 5 --connect-timeout 15 \
    -C - -o "$OCR_DIR/$output" \
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/$relative"
}
download_ocr "PP-OCRv6/det/PP-OCRv6_det_small.onnx" "PP-OCRv6_det_small.onnx"
download_ocr "PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx" "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
download_ocr "PP-OCRv6/rec/PP-OCRv6_rec_small.onnx" "PP-OCRv6_rec_small.onnx"

log "5/8 Download SigLIP2 (official endpoint, then mirror)"
download_siglip() {
  local endpoint="$1"
  printf 'SIGLIP_ENDPOINT=%s\n' "$endpoint"
  docker exec -e "HF_ENDPOINT=$endpoint" -e HF_HUB_DISABLE_XET=1 \
    "$CONTAINER_NAME" python3 -c '
from huggingface_hub import snapshot_download
repo = "google/siglip2-so400m-patch14-384"
print(f"HF_DOWNLOAD_START={repo}", flush=True)
path = snapshot_download(repo_id=repo, cache_dir="/app/models/hf-cache")
print(f"HF_DOWNLOAD_DONE={repo} path={path}", flush=True)
'
}
if [[ "$SKIP_SIGLIP" == 1 ]]; then
  printf 'SKIP: SigLIP2 download disabled by SKIP_SIGLIP=1\n'
elif ! retry download_siglip "https://huggingface.co"; then
  printf 'Official Hugging Face endpoint failed; trying hf-mirror.com\n'
  retry download_siglip "https://hf-mirror.com"
fi

log "6/8 Prepare pinned 3D-Speaker source and speaker models"
SPEAKER_REPO="$MODEL_ROOT/3D-Speaker"
if [[ ! -s "$SPEAKER_REPO/.momentseek-revision" ]] \
  || [[ "$(cat "$SPEAKER_REPO/.momentseek-revision")" != "$SPEAKER_SOURCE_REVISION" ]]; then
  SPEAKER_STAGE="$(mktemp -d "$MODEL_ROOT/.3dspeaker-source.XXXXXX")"
  retry curl -fL --retry 3 --retry-delay 5 --connect-timeout 15 \
    -o "$SPEAKER_STAGE/source.tar.gz" \
    "https://github.com/modelscope/3D-Speaker/archive/${SPEAKER_SOURCE_REVISION}.tar.gz"
  tar -xzf "$SPEAKER_STAGE/source.tar.gz" -C "$SPEAKER_STAGE"
  SPEAKER_EXTRACTED="$(find "$SPEAKER_STAGE" -mindepth 1 -maxdepth 1 -type d -name '3D-Speaker-*' | head -1)"
  [[ -n "$SPEAKER_EXTRACTED" ]] || die "3D-Speaker archive layout is invalid"
  printf '%s\n' "$SPEAKER_SOURCE_REVISION" >"$SPEAKER_EXTRACTED/.momentseek-revision"
  rm -rf "$SPEAKER_REPO"
  mv "$SPEAKER_EXTRACTED" "$SPEAKER_REPO"
  rm -rf "$SPEAKER_STAGE"
fi

retry docker exec "$CONTAINER_NAME" python3 -c '
from modelscope import snapshot_download

for repo, revision in (
    ("iic/speech_campplus_sv_zh_en_16k-common_advanced", "v1.0.0"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "v2.0.4"),
):
    print(f"SPEAKER_MODEL_DOWNLOAD_START={repo}@{revision}", flush=True)
    path = snapshot_download(repo, revision=revision, cache_dir="/app/models/3dspeaker-cache")
    print(f"SPEAKER_MODEL_DOWNLOAD_DONE={repo}@{revision} path={path}", flush=True)
'

log "7/8 Verify required model files"
if ! docker exec "$CONTAINER_NAME" python3 /app/scripts/verify_models.py \
  --manifest "$MANIFEST" --lock /app/models/models.lock.json; then
  if [[ "$SKIP_SIGLIP" == 1 ]]; then
    printf 'EXPECTED: full verification is incomplete until the offline SigLIP2 package is imported.\n'
  else
    die "required model verification failed"
  fi
fi
du -sh "$MODEL_ROOT"

log "8/8 Summary"
if [[ "$SKIP_SIGLIP" == 1 ]]; then
  printf 'MODEL_PREP_RESULT=PARTIAL_SIGLIP_PENDING\n'
else
  printf 'MODEL_PREP_RESULT=PASS\n'
fi
printf 'lock=%s/models.lock.json\n' "$MODEL_ROOT"
printf 'Next: restart the container only if its environment or image changed; model files are visible immediately.\n'
