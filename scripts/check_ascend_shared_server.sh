#!/usr/bin/env bash

# Read-only environment audit for deploying MomentSeek on a shared Ascend host.
# It does not install packages, download models, modify host configuration, or
# leave containers running.

set -u
set -o pipefail

IMAGE="${MOMENTSEEK_ASCEND_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:3.0.0b2-800I-A2-py311-openeuler24.03-lts}"
NPU_ID="${MOMENTSEEK_NPU_ID:-5}"
WORK_ROOT="${MOMENTSEEK_HOME:-/home/momentseek-29154}"
REPORT="${1:-momentseek_ascend_audit_$(date +%Y%m%d_%H%M%S).log}"
CONTAINER_NAME="momentseek-audit-$$"

exec > >(tee "$REPORT") 2>&1

section() {
  printf '\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

run() {
  printf '\n$ %s\n' "$*"
  "$@" || printf '[WARN] command exited with status %s\n' "$?"
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

section "MomentSeek Ascend shared-server audit"
printf 'time=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'image=%s\n' "$IMAGE"
printf 'selected_npu=%s\n' "$NPU_ID"
printf 'work_root=%s\n' "$WORK_ROOT"
printf 'report=%s\n' "$REPORT"

section "1. Host identity and resources"
run uname -a
run uname -m
run id
run bash -lc 'cat /etc/os-release 2>/dev/null || true'
run bash -lc 'python3 --version 2>&1 || true'
run bash -lc 'docker --version 2>&1 || true'
run bash -lc 'docker compose version 2>&1 || true'
run bash -lc 'free -h 2>/dev/null || true'
run bash -lc 'df -h / /home /data 2>/dev/null || true'
run bash -lc 'lscpu 2>/dev/null | head -40 || true'

section "2. Ascend host state"
run bash -lc 'npu-smi info 2>&1 || true'
run bash -lc 'ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>/dev/null || true'
for card in 5 6 7; do
  printf '\n--- NPU %s process memory ---\n' "$card"
  run bash -lc "npu-smi info -t proc-mem -i $card -c 0 2>&1 || true"
done
run bash -lc 'cat /usr/local/Ascend/driver/version.info 2>/dev/null || true'
run bash -lc 'cat /etc/ascend_install.info 2>/dev/null || true'
run bash -lc 'find /usr/local/Ascend -maxdepth 4 \( -name version.cfg -o -name version.info \) -type f -print 2>/dev/null | sort || true'

section "3. Existing Docker state (read-only)"
run bash -lc 'docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>&1 || true'
run bash -lc 'docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" 2>&1 | head -80 || true'
run bash -lc 'docker info --format "DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}" 2>&1 || true'
run bash -lc 'df -h /var/lib/docker 2>/dev/null || true'
run bash -lc 'ss -lntp 2>/dev/null | grep -E ":(18500|18510|18511)\\b" || true'

section "4. Work directory"
if [ -e "$WORK_ROOT" ]; then
  run ls -ld "$WORK_ROOT"
  run bash -lc "find '$WORK_ROOT' -maxdepth 2 -type d -print 2>/dev/null | sort"
  run bash -lc "du -sh '$WORK_ROOT' 2>/dev/null || true"
  run df -h "$WORK_ROOT"
else
  printf '[INFO] work directory does not exist: %s\n' "$WORK_ROOT"
fi

section "5. Host network and DNS"
run bash -lc 'cat /etc/resolv.conf 2>/dev/null || true'
for host in pypi.org pypi.tuna.tsinghua.edu.cn modelscope.cn github.com quay.io; do
  printf '\n--- DNS %s ---\n' "$host"
  run bash -lc "getent ahosts '$host' 2>&1 | head -6 || true"
done
for url in \
  https://pypi.org/simple/ \
  https://pypi.tuna.tsinghua.edu.cn/simple/ \
  https://modelscope.cn/ \
  https://github.com/ \
  https://quay.io/; do
  printf '\n--- HTTPS %s ---\n' "$url"
  run bash -lc "curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null -w 'http_code=%{http_code} remote_ip=%{remote_ip} total=%{time_total}s error=%{errormsg}\\n' '$url' 2>&1 || true"
done

section "6. Local offline assets"
run bash -lc 'python3 -m pip cache dir 2>/dev/null || true'
run bash -lc 'python3 -m pip cache info 2>/dev/null || true'
run bash -lc 'find /root/.cache /home /data -xdev -type f -name "*.whl" 2>/dev/null | head -200 || true'
run bash -lc 'find /root /home /data -xdev -type f \( -name "*.tar.gz" -o -name "*.tar.zst" \) -size +100M 2>/dev/null | head -100 || true'

section "7. Base image package/import audit"
run docker image inspect "$IMAGE" --format 'image_id={{.Id}} created={{.Created}} arch={{.Architecture}} os={{.Os}} size={{.Size}}'

docker run --rm \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  bash -lc 'set +e
echo "--- OS and commands ---"
cat /etc/os-release
python3 --version
command -v ffmpeg || true
ffmpeg -version 2>/dev/null | head -2 || true
command -v dnf || true
command -v yum || true
command -v curl || true

echo "--- package imports ---"
python3 - <<'"'"'PY'"'"'
import importlib
import importlib.util

packages = [
    "torch", "torch_npu", "torchvision", "transformers", "PIL", "cv2",
    "open_clip", "timm", "funasr", "silero_vad", "faster_whisper",
    "sentence_transformers", "onnxruntime", "insightface", "rapidocr",
    "scenedetect", "opencc", "pypinyin", "multipart", "pydantic_settings",
    "fastapi", "uvicorn", "numpy", "onnx",
]

for name in packages:
    exists = bool(importlib.util.find_spec(name))
    version = ""
    error = ""
    if exists:
        try:
            module = importlib.import_module(name)
            version = str(getattr(module, "__version__", ""))
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    print(f"{name:25} exists={str(exists):5} version={version:18} error={error}")

print("--- targeted imports ---")
try:
    from torchvision import transforms
    print("torchvision.transforms: PASS")
except BaseException as exc:
    print("torchvision.transforms: FAIL", type(exc).__name__, str(exc))

try:
    from transformers import Siglip2Model, AutoProcessor
    print("transformers.Siglip2Model: PASS")
except BaseException as exc:
    print("transformers.Siglip2Model: FAIL", type(exc).__name__, str(exc))

try:
    import onnxruntime as ort
    print("onnxruntime providers:", ort.get_available_providers())
except BaseException as exc:
    print("onnxruntime providers: UNAVAILABLE", type(exc).__name__, str(exc))
PY

echo "--- selected pip versions ---"
python3 -m pip list 2>/dev/null | grep -Ei "torch|npu|vision|transformers|onnx|funasr|whisper|sentence|rapidocr|insightface|open.clip|timm|scenedetect|pydantic" || true
'
BASE_AUDIT_STATUS=$?
printf 'base_image_audit_exit=%s\n' "$BASE_AUDIT_STATUS"

section "8. Selected NPU container audit"
docker run --rm \
  --name "$CONTAINER_NAME" \
  --shm-size=4g \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "$IMAGE" \
  bash -lc 'set +e
echo "--- visible devices ---"
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>/dev/null || true
npu-smi info || true

echo "--- torch_npu compute ---"
python3 - <<'"'"'PY'"'"'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("available:", torch.npu.is_available())
print("device_count:", torch.npu.device_count())
if torch.npu.is_available() and torch.npu.device_count() > 0:
    torch.npu.set_device(0)
    x = torch.arange(8, dtype=torch.float32).npu()
    y = x * 2
    torch.npu.synchronize()
    print("device:", x.device)
    print("result:", y.cpu().tolist())
    print("allocated_mb:", torch.npu.memory_allocated() / 1024 / 1024)
PY
'
NPU_AUDIT_STATUS=$?
printf 'npu_image_audit_exit=%s\n' "$NPU_AUDIT_STATUS"

section "9. Container network"
docker run --rm \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  bash -lc 'set +e
cat /etc/resolv.conf 2>/dev/null || true
for host in pypi.org pypi.tuna.tsinghua.edu.cn modelscope.cn github.com quay.io; do
  echo "--- DNS $host ---"
  getent ahosts "$host" 2>&1 | head -6 || true
done
for url in https://pypi.org/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/ https://modelscope.cn/ https://github.com/ https://quay.io/; do
  echo "--- HTTPS $url ---"
  curl -ILsS --connect-timeout 5 --max-time 15 -o /dev/null -w "http_code=%{http_code} remote_ip=%{remote_ip} total=%{time_total}s error=%{errormsg}\\n" "$url" 2>&1 || true
done
'
NETWORK_AUDIT_STATUS=$?
printf 'container_network_audit_exit=%s\n' "$NETWORK_AUDIT_STATUS"

section "10. Final snapshot"
run bash -lc 'npu-smi info 2>&1 || true'
run bash -lc 'docker ps --filter name=momentseek-audit --format "{{.Names}} {{.Status}}" 2>&1 || true'
printf '\nAudit complete. Report: %s\n' "$REPORT"
printf 'No packages were installed and no persistent container was created.\n'
