# Optional Video Color Grading

MomentSeek and VideoColorGrading are separate containers. They exchange task
metadata over the private Compose network and media files through the shared
`/app/runtime` mount. VideoColorGrading keeps its own PostgreSQL database.

## Prepare the upstream source

Check out the `api` branch of `Hi-WenXin/VideoColorGrading`, then apply the
minimal output-root patch:

```bash
git -C /home/wenxin/VideoColorGrading apply \
  /path/to/MomentSeek/deploy/color-grading/VideoColorGrading-vcg-output-root.patch
```

The patch only makes the existing output root configurable through
`VCG_OUTPUT_ROOT`. MomentSeek sets it to
`/app/runtime/color_grading/upstream`.

## Configure

Copy the normal MomentSeek environment and append the values from
`deploy/env/color-grading.ascend.example`. Confirm that:

- `HOST_RUNTIME_DIR` is writable and has sufficient free space.
- `VCG_SOURCE_DIR` points to the upstream `api` branch.
- `VCG_PACKAGES_DIR` contains the local PyTorch wheel required by `uv.lock`.
- `VCG_PRETRAINED_MODELS_DIR` contains all four model groups.
- `VCG_POSTGRES_PASSWORD` is changed for customer deployments.
- the selected Ascend device is not concurrently overloaded by indexing.

## Start

MomentSeek without color grading:

```bash
docker compose -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d
```

MomentSeek with color grading:

```bash
docker compose \
  -f compose.yml \
  -f compose.server.yml \
  -f compose.ascend.yml \
  -f compose.color-grading.yml \
  --profile color-grading \
  up -d
```

The grading API and its PostgreSQL port are not published to the customer
network. MomentSeek reaches the API at `http://video-color-grading:8000`.

## Verify

```bash
curl http://127.0.0.1:${APP_PORT:-8000}/api/color-grading/status
```

An enabled and ready deployment returns `enabled=true` and `available=true`.
Disabling the optional Compose file leaves all retrieval functions available.

## Shared Ascend production server

The shared-server platform container uses host networking rather than the
normal Compose application service. Run the grading API in its own bridge
network, publish it only on the host loopback address, and enable the adapter
when invoking the versioned platform deploy script:

```bash
COLOR_GRADING_ENABLED=true \
COLOR_GRADING_BASE_URL=http://127.0.0.1:21098 \
NPU_ID=5 APP_PORT=8000 \
  bash scripts/deploy_ascend_shared_server.sh
```

The grading Compose service should bind `127.0.0.1:21098:8000`, mount the same
host runtime at `/app/runtime`, and set:

```text
VCG_OUTPUT_ROOT=/app/runtime/color_grading/upstream
```

Its PostgreSQL port does not need to be public. If a host diagnostic mapping is
retained, bind that mapping to `127.0.0.1` as well.
