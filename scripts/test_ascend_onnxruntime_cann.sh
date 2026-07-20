#!/usr/bin/env bash
set -Eeuo pipefail

# Disposable validation only: the running platform container is never modified.
CONTAINER_NAME="${CONTAINER_NAME:-momentseek-29154-platform}"
MODEL_DIR="${MODEL_DIR:-/home/momentseek-29154/models/platform}"
NPU_ID="${NPU_ID:-5}"
ORT_CANN_VERSION="${ORT_CANN_VERSION:-1.24.4}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || fail "container not found: $CONTAINER_NAME"
[[ -d "$MODEL_DIR" ]] || fail "model directory not found: $MODEL_DIR"
IMAGE_NAME="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER_NAME")"

printf 'image=%s\nphysical_npu=%s\nonnxruntime_cann=%s\n' "$IMAGE_NAME" "$NPU_ID" "$ORT_CANN_VERSION"
npu-smi info -t proc-mem -i "$NPU_ID" -c 0 || true

docker run --rm \
  --network host \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$MODEL_DIR:/app/models:ro" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/cann-8.5.1/lib64 \
  "$IMAGE_NAME" sh -lc "
    set -e
    python3 -m pip uninstall -y onnxruntime onnxruntime-cann >/dev/null 2>&1 || true
    python3 -m pip install --no-cache-dir --no-deps onnxruntime-cann==${ORT_CANN_VERSION}
    python3 - <<'PY'
import time
import cv2
import numpy as np
import onnxruntime as ort

providers = ort.get_available_providers()
print('providers=', providers, flush=True)
if 'CANNExecutionProvider' not in providers:
    raise RuntimeError('onnxruntime-cann installed but CANNExecutionProvider is unavailable')

from app.indexing.ocr import _load_ocr
image = np.full((240, 720, 3), 255, dtype=np.uint8)
cv2.putText(image, 'QATAR WORLD CUP', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)

ocr_results = {}
for device in ('cpu', 'npu'):
    try:
        start = time.perf_counter()
        ocr, active_providers = _load_ocr(device, 0, '/app/models/rapidocr', npu_self_test=False)
        load_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        output = ocr(image, text_score=0.1, box_thresh=0.1)
        infer_elapsed = time.perf_counter() - start
        texts = list(getattr(output, 'txts', None) or [])
        ocr_results[device] = texts
        print(f'ocr_{device}_providers=', active_providers, flush=True)
        print(f'ocr_{device}_load_seconds=', round(load_elapsed, 3), flush=True)
        print(f'ocr_{device}_inference_seconds=', round(infer_elapsed, 3), flush=True)
        print(f'ocr_{device}_texts=', texts, flush=True)
    except Exception as exc:
        ocr_results[device] = None
        print(f'ocr_{device}_error={type(exc).__name__}: {exc}', flush=True)

from app.indexing.faces import FaceEncoder
face_ok = False
try:
    start = time.perf_counter()
    face = FaceEncoder('buffalo_l', 'cann', 0, '/app/models/insightface')
    load_elapsed = time.perf_counter() - start
    face_image = np.full((640, 640, 3), 255, dtype=np.uint8)
    cv2.circle(face_image, (320, 320), 180, (210, 210, 210), -1)
    start = time.perf_counter()
    faces = face.detect(face_image)
    infer_elapsed = time.perf_counter() - start
    print('face_provider=', face.provider, flush=True)
    print('face_load_seconds=', round(load_elapsed, 3), flush=True)
    print('face_inference_seconds=', round(infer_elapsed, 3), flush=True)
    print('face_detections_on_synthetic=', len(faces), flush=True)
    face_ok = face.provider == 'cann'
except Exception as exc:
    print(f'face_cann_error={type(exc).__name__}: {exc}', flush=True)

cpu_text = ' '.join(ocr_results.get('cpu') or []).upper()
npu_text = ' '.join(ocr_results.get('npu') or []).upper()
cpu_ok = 'QATAR' in cpu_text or 'WORLD' in cpu_text
npu_ok = 'QATAR' in npu_text or 'WORLD' in npu_text
print(f'OCR_CPU_CORRECT={int(cpu_ok)}', flush=True)
print(f'OCR_CANN_CORRECT={int(npu_ok)}', flush=True)
print(f'FACE_CANN_INITIALIZED={int(face_ok)}', flush=True)
if cpu_ok and npu_ok and face_ok:
    print('CANN_SMOKE_RESULT=PASS', flush=True)
else:
    print('CANN_SMOKE_RESULT=FAIL', flush=True)
    raise SystemExit(2)
PY
  "
