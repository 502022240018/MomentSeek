# syntax=docker/dockerfile:1.7
ARG ASCEND_RUNTIME_IMAGE=momentseek-mvp:ascend-runtime
FROM node:20-alpine AS frontend-build
WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Override this with a portable CANN/torch_npu ARM64 base image on another server.
FROM ${ASCEND_RUNTIME_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    OPENBLAS_NUM_THREADS=8 \
    OPENBLAS_DEFAULT_NUM_THREADS=8 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    BLIS_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false
RUN if command -v ffmpeg >/dev/null 2>&1; then \
        echo "ffmpeg already present in Ascend runtime"; \
    elif command -v apt-get >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y ffmpeg mesa-libGL glib2 \
        && dnf clean all; \
    else \
        echo "No supported package manager found for ffmpeg installation" >&2; \
        exit 1; \
    fi
WORKDIR /app/backend
COPY vendor-wheels/ /tmp/vendor-wheels/
COPY backend/requirements-ascend.txt ./
COPY backend/requirements-ascend-preserve.txt ./
COPY backend/requirements-milvus-client.txt ./
COPY backend/requirements-speaker.txt ./
COPY backend/constraints-ascend.txt ./
RUN (cd /tmp/vendor-wheels && sha256sum -c SHA256SUMS) \
    && pip install --no-index --no-deps --find-links=/tmp/vendor-wheels \
        -r requirements-milvus-client.txt \
    && pip install --no-index --no-deps /tmp/vendor-wheels/insightface-1.0.1-py3-none-any.whl \
    && pip install -c constraints-ascend.txt -r requirements-ascend.txt \
    && pip install --no-deps -r requirements-ascend-preserve.txt \
    && python3 -c "from importlib.metadata import version; import cv2; assert version('torch') == '2.9.0'; assert version('torch-npu') == '2.9.0.post1'; assert version('torchaudio') == '2.9.0'; assert cv2.__version__ == '4.11.0'" \
    && rm -rf /tmp/vendor-wheels
COPY backend/ ./
COPY deploy/orchestration/ ./deploy/orchestration/
COPY deploy/models/ascend-prod.models.json /app/deploy/models/ascend-prod.models.json
COPY scripts/verify_models.py /app/scripts/verify_models.py
COPY --from=frontend-build /src/frontend/dist ./app/static
RUN mkdir -p /app/runtime /app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import os, urllib.request; port=os.environ.get('APP_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${APP_PORT:-8000}\" --workers 1"]
