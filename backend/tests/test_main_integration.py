import json
import subprocess
import sys

from fastapi.testclient import TestClient

from app.api import job_routes
from app.platform import context
from app.catalog.db import Catalog
from app.core.settings import Settings


def test_folder_api_deletion_keeps_asset_and_its_index(monkeypatch, tmp_path):
    import app.main as main

    settings = Settings(_env_file=None, app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models", indexer_mode="process_exit")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video-1.mp4"
    video_path.write_bytes(b"video")
    index_path = settings.index_dir / "video-1" / "visual.npz"
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b"index")
    catalog.create_video({"id": "video-1", "name": "demo.mp4", "file_path": str(video_path), "duration": 1, "fps": 25, "width": 640, "height": 480, "status": "ready"})
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setattr(context, "catalog", catalog)

    with TestClient(main.app) as client:
        folder = client.post("/api/folders", json={"name": "Campaign"})
        assert folder.status_code == 201
        folder_id = folder.json()["id"]
        assert client.post("/api/videos/folders", json={"video_ids": ["video-1"], "folder_ids": [folder_id], "operation": "add"}).status_code == 200
        assert client.delete(f"/api/folders/{folder_id}").json()["released_video_count"] == 1
        assert client.get("/api/videos").json()[0]["folder_ids"] == []

    assert video_path.exists()
    assert index_path.exists()


def test_spawn_indexer_daemon_passes_profile_environment(monkeypatch, tmp_path):

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
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setenv("EXISTING_CONTAINER_ENV", "kept")
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        kwargs["stdout"].close()
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    context._spawn_indexer_daemon()

    environment = captured["env"]
    assert captured["args"][:3] == [sys.executable, "-m", "app.execution.indexer_daemon"]
    assert environment["EXISTING_CONTAINER_ENV"] == "kept"
    assert environment["ENV_PROFILE"] == "staging.ascend"
    assert environment["RELEASE_MANIFEST_PATH"] == str((tmp_path / "release.json").resolve())
    assert environment["NPU_ENABLED"] == "true"
    assert environment["NPU_DEVICE_ID"] == "0"
    assert environment["ASCEND_VISIBLE_DEVICES"] == "2"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert environment["VISUAL_MODEL"] == "siglip2-so400m-384"
    assert environment["VISUAL_HF_CACHE_DIR"] == str((tmp_path / "models" / "hf-cache").resolve())


def test_spawn_indexer_daemon_strips_inherited_npu_environment_when_disabled(monkeypatch, tmp_path):

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        indexer_mode="daemon",
        npu_enabled=False,
    )
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setenv("NPU_DEVICE_ID", "7")
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7")
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        kwargs["stdout"].close()
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    context._spawn_indexer_daemon()

    environment = captured["env"]
    assert environment["NPU_ENABLED"] == "false"
    assert "NPU_DEVICE_ID" not in environment
    assert "ASCEND_VISIBLE_DEVICES" not in environment
    assert "ASCEND_RT_VISIBLE_DEVICES" not in environment


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
    monkeypatch.setattr(context, "settings", settings)

    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["env_profile"] == "staging.ascend"
    assert body["release_id"] == "2026-07-03-health"
    assert body["model_idle_policy"] == settings.model_idle_policy
    assert body["indexer_mode"] == settings.indexer_mode
    assert body["npu_worker_mode"] == settings.npu_worker_mode
    assert body["orchestration_enabled"] is False
    assert body["orchestration_profile"] == settings.orchestration_profile
    assert body["git_commit"] == "abc123"
    assert body["image_tag"] == "momentseek-mvp:ascend-abc123"
    assert body["model_manifest"] == "deploy/models/ascend-prod.models.json"
    assert body["npu_enabled"] is True
    assert body["npu_device_id"] == 0


def test_orchestration_profiles_endpoint_uses_runtime_orchestrator(monkeypatch):
    import app.main as main

    expected = {
        "enabled": True,
        "default_profile": "qwen35-unified",
        "profiles": [{"name": "qwen35-unified"}],
    }

    class FakeOrchestrator:
        def profiles(self):
            return expected

    monkeypatch.setattr(context, "search_orchestrator", FakeOrchestrator())

    with TestClient(main.app) as client:
        response = client.get("/api/orchestration/profiles")

    assert response.status_code == 200
    assert response.json() == expected


def test_create_index_job_queues_only_requested_modalities(monkeypatch, tmp_path):
    import app.main as main

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        indexer_mode="process_exit",
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video-1.mp4"
    video_path.write_bytes(b"fake")
    catalog.create_video({
        "id": "video-1",
        "name": "demo.mp4",
        "file_path": str(video_path),
        "duration": 10.0,
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["face", "visual"])
    launched: list[str] = []
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setattr(context, "catalog", catalog)
    monkeypatch.setattr(context, "launch_job", lambda job_id: launched.append(job_id))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/videos/video-1/index",
            json={
                "modalities": ["asr", "ocr", "asr"],
                "asr_model": "turbo",
                "asr_language": "auto",
                "ocr_sample_fps": 1.0,
            },
        )

    assert response.status_code == 202
    job = response.json()
    assert job["modalities"] == ["asr", "ocr"]
    assert job["options"] == {
        "ocr_sample_fps": 1.0,
        "asr_model": "turbo",
        "asr_language": "auto",
    }
    assert launched == [job["id"]]
    assert catalog.get_video("video-1")["indexed_modalities"] == ["face", "visual"]


def test_create_index_job_rejects_disabled_milvus(monkeypatch, tmp_path):
    import app.main as main

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        milvus_enabled=False,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video-1.mp4"
    video_path.write_bytes(b"fake")
    catalog.create_video({
        "id": "video-1", "name": "demo.mp4", "file_path": str(video_path),
        "duration": 10.0, "fps": 25.0, "width": 1920, "height": 1080, "status": "ready",
    })
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setattr(context, "catalog", catalog)

    with TestClient(main.app) as client:
        response = client.post("/api/videos/video-1/index", json={"modalities": ["visual"]})

    assert response.status_code == 503
    assert "MILVUS_ENABLED=true" in response.json()["detail"]
    assert catalog.list_jobs("video-1") == []


def _create_cancellable_job(catalog, tmp_path, *, status="queued"):
    video_path = tmp_path / "cancel.mp4"
    video_path.write_bytes(b"fake")
    catalog.create_video({
        "id": "cancel-video", "name": "cancel.mp4", "file_path": str(video_path),
        "duration": 10.0, "fps": 25.0, "width": 1280, "height": 720,
        "status": "indexing" if status == "running" else "uploaded",
    })
    return catalog.create_job({
        "id": "cancel-job", "video_id": "cancel-video", "status": status,
        "stage": "visual" if status == "running" else "queued", "progress": 0.2,
        "modalities": ["visual"], "options": {},
    })


def test_cancel_queued_subprocess_job_preserves_other_jobs(monkeypatch, tmp_path):

    settings = Settings(
        _env_file=None, app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models",
        indexer_mode="subprocess",
    )
    catalog = Catalog(settings.db_path)
    _create_cancellable_job(catalog, tmp_path, status="queued")
    terminated = []
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setattr(context, "catalog", catalog)
    monkeypatch.setattr(context, "_terminate_process_group", lambda pid, expected_job_id=None: terminated.append((pid, expected_job_id)))

    response = job_routes.cancel_job("cancel-job")

    assert response["status"] == "cancelled"
    assert response["stage"] == "cancelled"
    assert response["error"] == "用户取消任务"
    assert terminated == [(None, "cancel-job")]
    assert catalog.get_video("cancel-video")["status"] == "uploaded"


def test_cancel_running_daemon_job_restarts_queue_consumer(monkeypatch, tmp_path):

    settings = Settings(
        _env_file=None, app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models",
        indexer_mode="daemon",
    )
    catalog = Catalog(settings.db_path)
    _create_cancellable_job(catalog, tmp_path, status="running")
    restarted = []
    monkeypatch.setattr(context, "settings", settings)
    monkeypatch.setattr(context, "catalog", catalog)
    monkeypatch.setattr(context, "_restart_indexer_daemon", lambda: restarted.append(True))

    response = job_routes.cancel_job("cancel-job")

    assert response["status"] == "cancelled"
    assert restarted == [True]
    assert catalog.get_video("cancel-video")["status"] == "uploaded"


def test_stage_runner_uses_asr_engine_job_option(monkeypatch, tmp_path):
    import app.indexing.modalities.asr.asr as asr
    import app.execution.stage_runner as stage_runner

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        asr_engine="funasr",
        asr_model="turbo",
        asr_language="auto",
        asr_semantic_enabled=False,
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")
    video = catalog.create_video({
        "id": "video-1",
        "name": "demo.mp4",
        "file_path": str(video_path),
        "duration": 1.0,
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "status": "ready",
    })
    job = catalog.create_job({
        "id": "job-1",
        "video_id": video["id"],
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": ["asr"],
        "options": {"asr_engine": "faster-whisper", "asr_model": "turbo", "asr_language": "auto"},
    })
    captured = {}

    def fake_build_asr_index(**kwargs):
        captured.update(kwargs)
        return {
            "engine": kwargs["engine"],
            "model": kwargs["model_name"],
            "requested_language": kwargs["language"],
            "detected_language": "en",
            "language": "en",
            "task": "transcribe",
            "raw_items": 0,
            "retrieval_chunks": 0,
            "chunk_builder_stats": {},
            "text_profile": {},
            "decode_status": "empty",
            "semantic_status": "disabled",
        }

    monkeypatch.setattr(stage_runner, "get_settings", lambda: settings)
    monkeypatch.setattr(asr, "build_asr_index", fake_build_asr_index)
    import app.indexing.stage_executor as stage_executor
    monkeypatch.setattr(stage_executor, "_setup_milvus_context", lambda *args: None)

    result = stage_runner.run("asr", job["id"])

    assert result["engine"] == "faster-whisper"
    assert captured["engine"] == "faster-whisper"
    assert captured["model_name"] == "turbo"
    assert captured["language"] == "auto"
