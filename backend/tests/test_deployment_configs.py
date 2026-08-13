from pathlib import Path

from app.settings import Settings


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _parse_env(relative_path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _read(relative_path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def test_ascend_compose_separates_host_card_from_container_npu_id():
    compose = _read("compose.ascend.yml")
    staging = _parse_env("deploy/env/staging.ascend.example")
    prod = _parse_env("deploy/env/prod.ascend.example")

    assert "HOST_NPU_DEVICE_ID" in compose
    assert "${NPU_DEVICE_ID" not in compose
    assert "HOST_NPU_DEVICE_ID:-" not in compose
    assert "HOST_NPU_DEVICE_ID:?" in compose
    assert "ASCEND_RT_VISIBLE_DEVICES:" not in compose
    assert staging["HOST_NPU_DEVICE_ID"] == staging["ASCEND_VISIBLE_DEVICES"]
    assert staging["HOST_NPU_DEVICE_ID"] == staging["ASCEND_RT_VISIBLE_DEVICES"]
    assert staging["NPU_DEVICE_ID"] == "0"
    assert prod["HOST_NPU_DEVICE_ID"] == prod["ASCEND_VISIBLE_DEVICES"]
    assert prod["HOST_NPU_DEVICE_ID"] == prod["ASCEND_RT_VISIBLE_DEVICES"]
    assert prod["NPU_DEVICE_ID"] == "0"


def test_compose_runtime_and_model_mounts_are_host_configurable():
    compose = _read("compose.yml")
    staging = _parse_env("deploy/env/staging.ascend.example")
    prod = _parse_env("deploy/env/prod.ascend.example")

    assert "${APP_PORT:-8000}:8000" in compose
    assert "APP_PORT:-8300" not in compose
    assert "${HOST_RUNTIME_DIR:-./runtime}:/app/runtime" in compose
    assert "${HOST_MODEL_DIR:-./models}:/app/models" in compose
    assert staging["HOST_RUNTIME_DIR"] == "/opt/momentseek/runtime"
    assert staging["HOST_MODEL_DIR"] == "/opt/momentseek/models"
    assert prod["HOST_RUNTIME_DIR"] == "/opt/momentseek/runtime"
    assert prod["HOST_MODEL_DIR"] == "/opt/momentseek/models"


def test_cuda_docker_profile_mounts_migrated_runtime_and_enables_gpu():
    compose = _read("compose.cuda.yml")
    dockerfile = _read("Dockerfile.cuda")
    dev_cuda = _parse_env("deploy/env/dev.cuda.example")
    gitignore = _read(".gitignore")
    migration_doc = _read("docs/LOCAL_GPU_MIGRATION.md")

    assert "dockerfile: Dockerfile.cuda" in compose
    assert "APT_MIRROR:" in compose
    assert "APT_SECURITY_MIRROR:" in compose
    assert "ARG APT_MIRROR" in dockerfile
    assert "Acquire::Retries" in dockerfile
    assert "PIP_INDEX_URL:" in compose
    assert "ARG PIP_INDEX_URL" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "--default-timeout=120" in dockerfile
    assert 'cpus: "${APP_CPUS:-12}"' in compose
    assert 'mem_limit: "${APP_MEMORY_LIMIT:-12g}"' in compose
    assert "NPU_ENABLED: \"false\"" in compose
    assert "CUDA_ENABLED: \"true\"" in compose
    assert "APP_DATA_DIR: /app/runtime" in compose
    assert "${HOST_RUNTIME_DIR:-./runtime-server}:/app/runtime" in compose
    assert "capabilities: [gpu]" in compose
    assert "requirements-cuda.txt" in dockerfile
    assert dev_cuda["ENV_PROFILE"] == "dev.cuda"
    assert dev_cuda["CUDA_ENABLED"] == "true"
    assert dev_cuda["NPU_ENABLED"] == "false"
    assert dev_cuda["APP_CPUS"] == "12"
    assert dev_cuda["APP_MEMORY_LIMIT"] == "12g"
    assert dev_cuda["VISUAL_MODEL"] == "siglip2-so400m-384"
    assert dev_cuda["VISUAL_HF_CACHE_DIR"] == "/app/runtime/hf_cache"
    assert "VISUAL_HF_CACHE_DIR: \"${VISUAL_HF_CACHE_DIR:-/app/runtime/hf_cache}\"" in compose
    assert dev_cuda["HOST_RUNTIME_DIR"] == "./runtime-server"
    assert dev_cuda["HOST_MODEL_DIR"] == "./models"
    assert "runtime-server/" in gitignore
    assert "runtime-server" in migration_doc
    assert "compose.cuda.yml" in migration_doc


def test_docker_cpu_docs_use_dev_cpu_port():
    dev_cpu = _parse_env("deploy/env/dev.cpu.example")
    expected_health = f"http://127.0.0.1:{dev_cpu['APP_PORT']}/api/health"

    assert expected_health in _read("DEPLOY.md")
    assert "http://127.0.0.1:8300/api/health" not in _read("DEPLOY.md")


def test_bootstrap_writes_dev_lock_to_runtime():
    assert "--lock runtime/dev-models.lock.json" in _read("scripts/bootstrap_dev.sh")
    assert '"runtime/dev-models.lock.json"' in _read("scripts/bootstrap_dev.ps1")


def test_root_env_example_is_safe_dev_cpu_profile():
    root_env = _parse_env(".env.example")
    dev_cpu = _parse_env("deploy/env/dev.cpu.example")

    assert root_env["ENV_PROFILE"] == "dev.cpu"
    assert root_env["APP_PORT"] == dev_cpu["APP_PORT"]
    assert root_env["APP_PUBLIC_URL"] == dev_cpu["APP_PUBLIC_URL"]
    assert root_env["APP_DATA_DIR"] == "runtime"
    assert root_env["APP_MODEL_DIR"] == "models"
    assert root_env["NPU_ENABLED"] == "false"
    assert "NPU_DEVICE_ID" not in root_env
    assert "HOST_NPU_DEVICE_ID" not in root_env


def test_runtime_defaults_match_safe_dev_cpu_profile():
    settings = Settings(_env_file=None)

    assert settings.app_public_url == "http://127.0.0.1:8000"
    assert settings.npu_enabled is False
    assert settings.npu_device_id == 0


def test_resource_scripts_use_host_npu_device_id_not_container_npu_id():
    check_resource = _read("scripts/check_resource.sh")
    verify_release = _read("scripts/verify_model_release.sh")

    assert "load_env_file" in check_resource
    assert "${APP_PORT:-8000}" in check_resource
    assert "load_env_file" in verify_release
    assert "HOST_NPU_DEVICE_ID" in verify_release
    assert "${NPU_DEVICE_ID" not in verify_release
    assert "HOST_NPU_DEVICE_ID:-" not in verify_release
    assert "ASCEND_VISIBLE_DEVICES:-" not in verify_release


def test_ascend_profiles_cap_cpu_inference_thread_pools():
    variables = (
        "OPENBLAS_NUM_THREADS",
        "OPENBLAS_DEFAULT_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    dockerfile = _read("Dockerfile.ascend")
    compose = _read("compose.ascend.yml")
    deploy_script = _read("scripts/deploy_ascend_shared_server.sh")

    for relative_path in (
        "deploy/env/staging.ascend.example",
        "deploy/env/prod.ascend.example",
    ):
        profile = _parse_env(relative_path)
        for variable in variables:
            assert profile[variable] == "8"
        assert profile["TOKENIZERS_PARALLELISM"] == "false"

    for variable in variables:
        assert f"{variable}=8" in dockerfile
        assert f"{variable}: \"${{{variable}:-8}}\"" in compose
        assert f'-e {variable}="$CPU_THREAD_LIMIT"' in deploy_script
    assert "TOKENIZERS_PARALLELISM=false" in dockerfile
    assert 'TOKENIZERS_PARALLELISM: "${TOKENIZERS_PARALLELISM:-false}"' in compose
    assert deploy_script.count("TOKENIZERS_PARALLELISM=false") >= 1
    assert 'CPU_THREAD_LIMIT="${CPU_THREAD_LIMIT:-8}"' in deploy_script


def test_shared_ascend_deploy_honors_port_milvus_and_required_models():
    dockerfile = _read("Dockerfile.ascend")
    deploy_script = _read("scripts/deploy_ascend_shared_server.sh")

    assert 'CMD ["sh", "-c"' in dockerfile
    assert '--port \\"${APP_PORT:-8000}\\"' in dockerfile
    assert 'CMD ["uvicorn"' not in dockerfile

    for variable in (
        "MILVUS_ENABLED",
        "MILVUS_HOST",
        "MILVUS_PORT",
        "MILVUS_READ_ENABLED",
        "MILVUS_WRITE_ENABLED",
        "MILVUS_FALLBACK_ENABLED",
    ):
        assert f'-e {variable}="${variable}"' in deploy_script
    assert "socket.create_connection" in deploy_script
    assert "milvus_preflight=PASS" in deploy_script

    preflight = deploy_script.index(
        "Verify required production models before quiescing the current platform"
    )
    quiesce = deploy_script.index('docker rename "$CONTAINER_NAME" "$ROLLBACK_NAME"')
    runtime_verify = deploy_script.index(
        "Verify required model inventory in the replacement container"
    )
    remove_rollback = deploy_script.index('docker rm "$ROLLBACK_NAME"')
    assert preflight < quiesce
    assert runtime_verify < remove_rollback
    assert "verify_models.py \\\n  --manifest /app/deploy/models/ascend-prod.models.json || true" not in deploy_script


def test_production_ascend_uses_isolated_resident_workers():
    prod = _parse_env("deploy/env/prod.ascend.example")
    staging = _parse_env("deploy/env/staging.ascend.example")
    compose = _read("compose.ascend.yml")
    deploy_script = _read("scripts/deploy_ascend_shared_server.sh")

    assert prod["INDEXER_MODE"] == "daemon"
    assert "MODEL_IDLE_POLICY" not in prod
    assert "MODEL_IDLE_POLICY" not in staging
    assert prod["NPU_WORKER_MODE"] == "isolated"
    assert staging["NPU_WORKER_MODE"] == "legacy"
    assert 'NPU_WORKER_MODE: "${NPU_WORKER_MODE:-isolated}"' in compose
    assert "-e NPU_WORKER_MODE=isolated" in deploy_script
    assert 'CONTAINER_CPU_LIMIT="${CONTAINER_CPU_LIMIT:-24}"' in deploy_script
    assert 'CONTAINER_PID_LIMIT="${CONTAINER_PID_LIMIT:-2048}"' in deploy_script
    assert '--cpus "$CONTAINER_CPU_LIMIT"' in deploy_script
    assert '--pids-limit "$CONTAINER_PID_LIMIT"' in deploy_script
    assert '-e FACE_ORT_INTRA_OP_THREADS="$CPU_THREAD_LIMIT"' in deploy_script
    assert "-e FACE_ORT_INTER_OP_THREADS=1" in deploy_script
