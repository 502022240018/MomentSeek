import json
import subprocess
import sys

from fastapi.testclient import TestClient

from app.settings import Settings


def test_spawn_indexer_daemon_passes_profile_environment(monkeypatch, tmp_path):
    import app.main as main

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        indexer_mode="daemon",
        env_profile="staging.ascend",
        release_manifest_path=tmp_path / "release.json",
        npu_enabled=True,
        npu_device_id=0,
        ascend_visible_devices="2",
        ascend_rt_visible_devices="2",
        visual_model="siglip2-so400m-384",
        visual_hf_cache_dir=tmp_path / "models" / "hf-cache",
    )
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setenv("EXISTING_CONTAINER_ENV", "kept")
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        kwargs["stdout"].close()
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    main._spawn_indexer_daemon()

    environment = captured["env"]
    assert captured["args"][:3] == [sys.executable, "-m", "app.indexer_daemon"]
    assert environment["EXISTING_CONTAINER_ENV"] == "kept"
    assert environment["ENV_PROFILE"] == "staging.ascend"
    assert environment["RELEASE_MANIFEST_PATH"] == str((tmp_path / "release.json").resolve())
    assert environment["NPU_ENABLED"] == "true"
    assert environment["NPU_DEVICE_ID"] == "0"
    assert environment["ASCEND_VISIBLE_DEVICES"] == "2"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert environment["VISUAL_MODEL"] == "siglip2-so400m-384"
    assert environment["VISUAL_HF_CACHE_DIR"] == str((tmp_path / "models" / "hf-cache").resolve())


def test_health_endpoint_serializes_release_manifest_metadata(monkeypatch, tmp_path):
    import app.main as main

    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps({
            "release_id": "2026-07-03-health",
            "git_commit": "abc123",
            "env_profile": "staging.ascend",
            "models": {"manifest": "deploy/models/ascend-prod.models.json"},
            "image": {"ascend": "momentseek-mvp:ascend-abc123"},
        }),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        env_profile="staging.ascend",
        release_manifest_path=manifest,
        npu_enabled=True,
        npu_device_id=0,
    )
    monkeypatch.setattr(main, "settings", settings)

    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["env_profile"] == "staging.ascend"
    assert body["release_id"] == "2026-07-03-health"
    assert body["git_commit"] == "abc123"
    assert body["image_tag"] == "momentseek-mvp:ascend-abc123"
    assert body["model_manifest"] == "deploy/models/ascend-prod.models.json"
    assert body["npu_enabled"] is True
    assert body["npu_device_id"] == 0
