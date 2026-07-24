# Milvus deployment and NPZ backfill

Milvus is the primary vector store. SQLite remains the catalog and metadata
store; NPZ files are retained as recovery artifacts and per-channel fallback.
Do not remove NPZ files as part of this migration.

## Shared-server isolation

Use a unique Compose project, network, runtime directory, and host ports. This
prevents the deployment from replacing another developer's containers:

```bash
export COMPOSE_PROJECT_NAME=momentseek-29154-milvus
export MOMENTSEEK_NETWORK_NAME=momentseek-29154-milvus-net
export HOST_RUNTIME_DIR=/absolute/path/to/29154/runtime
export HOST_MODEL_DIR=/absolute/path/to/shared/read-only/models
export APP_PORT=18001
export MINIO_CONSOLE_PORT=19001
export MILVUS_GRPC_PORT=19531
export MILVUS_HEALTH_PORT=19091
```

The backend connects to `milvus:19530` inside the private Compose network, so
the configurable host-side Milvus port does not change `MILVUS_PORT`.

## Preflight and deployment

Render the complete Compose configuration before starting anything:

```bash
docker compose --env-file .env \
  -f compose.yml -f compose.milvus.yml -f compose.server.yml config
docker compose --env-file .env \
  -f compose.yml -f compose.milvus.yml -f compose.server.yml up -d
docker compose --env-file .env \
  -f compose.yml -f compose.milvus.yml -f compose.server.yml ps
```

Wait for etcd, MinIO, and Milvus to become healthy. Then inspect the schema:

```bash
docker compose exec app python -m scripts.migrate_milvus_schema --dry-run
```

If the dry-run reports incompatible collections, migration drops only those
collections. Confirm the retained NPZ directory is complete before running:

```bash
docker compose exec app python -m scripts.migrate_milvus_schema --yes
```

## Full historical backfill

Run without `--resume` for the first full migration. Each video/modality is
made idempotent by deleting its old rows, inserting the NPZ contents, flushing,
and verifying the persisted row count:

```bash
docker compose exec app python -m scripts.backfill_milvus \
  --asset-version historical-v1
```

If interrupted, repeat with `--resume`. Resume markers are version-scoped and
their Milvus row counts are rechecked before a pair is skipped:

```bash
docker compose exec app python -m scripts.backfill_milvus \
  --asset-version historical-v1 --resume
```

The command exits with status 2 if any pair fails. Do not start formal testing
until it exits successfully.

## Validation and cleanup

Check backend health, run a known query through each indexed modality, and
temporarily inspect logs for `Milvus coverage gap`. A coverage-gap warning
means the request remained available through NPZ fallback, but the migration
is not complete.

Retry any vector cleanup tasks left by earlier video deletions:

```bash
docker compose exec app python -m scripts.retry_milvus_cleanup --dry-run
docker compose exec app python -m scripts.retry_milvus_cleanup
```

Keep `MILVUS_FALLBACK_ENABLED=true` during initial formal testing. Disable it
only for a deliberate fail-closed test after coverage and failure recovery
have been verified.
