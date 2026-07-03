from pathlib import Path


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

    assert "${HOST_RUNTIME_DIR:-./runtime}:/app/runtime" in compose
    assert "${HOST_MODEL_DIR:-./models}:/app/models" in compose
    assert staging["HOST_RUNTIME_DIR"] == "/opt/momentseek/runtime"
    assert staging["HOST_MODEL_DIR"] == "/opt/momentseek/models"
    assert prod["HOST_RUNTIME_DIR"] == "/opt/momentseek/runtime"
    assert prod["HOST_MODEL_DIR"] == "/opt/momentseek/models"


def test_docker_cpu_docs_use_dev_cpu_port():
    dev_cpu = _parse_env("deploy/env/dev.cpu.example")
    expected_health = f"http://127.0.0.1:{dev_cpu['APP_PORT']}/api/health"

    assert expected_health in _read("DEPLOY.md")
    assert "http://127.0.0.1:8300/api/health" not in _read("DEPLOY.md")


def test_bootstrap_writes_dev_lock_to_runtime():
    assert "--lock runtime/dev-models.lock.json" in _read("scripts/bootstrap_dev.sh")
    assert '"runtime/dev-models.lock.json"' in _read("scripts/bootstrap_dev.ps1")
