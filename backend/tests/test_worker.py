from types import SimpleNamespace

from app.db import Catalog
from app.settings import Settings
from app.worker import subprocess_environment, worker_environment


def test_worker_environment_uses_absolute_runtime_paths(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")

    environment = worker_environment(settings)

    assert environment["APP_DATA_DIR"] == str((tmp_path / "runtime").resolve())
    assert environment["APP_MODEL_DIR"] == str((tmp_path / "models").resolve())
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["NPU_ENABLED"] == "false"
    assert "NPU_DEVICE_ID" not in environment


def test_subprocess_environment_strips_inherited_npu_vars_when_disabled(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        npu_enabled=False,
    )
    base_environment = {
        "KEEP_ME": "yes",
        "NPU_DEVICE_ID": "7",
        "ASCEND_VISIBLE_DEVICES": "7",
        "ASCEND_RT_VISIBLE_DEVICES": "7",
        "ASCEND_DEVICE_ID": "7",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }

    environment = subprocess_environment(settings, base_environment)

    assert environment["KEEP_ME"] == "yes"
    assert environment["NPU_ENABLED"] == "false"
    assert "NPU_DEVICE_ID" not in environment
    assert "ASCEND_VISIBLE_DEVICES" not in environment
    assert "ASCEND_RT_VISIBLE_DEVICES" not in environment
    assert "ASCEND_DEVICE_ID" not in environment
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD" not in environment


def test_worker_environment_propagates_indexing_profile_settings(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        env_profile="staging.ascend",
        release_manifest_path=tmp_path / "release.json",
        npu_enabled=True,
        npu_device_id=0,
        cuda_enabled=False,
        ascend_visible_devices="2",
        ascend_rt_visible_devices="2",
        torch_device_backend_autoload="0",
        visual_model="siglip2-so400m-384",
        visual_hf_cache_dir=tmp_path / "models" / "hf-cache",
        face_provider="cann",
        asr_model="small",
        asr_semantic_local_files_only=True,
        ocr_device="auto",
        ocr_version="PP-OCRv6",
        npu_worker_mode="isolated",
    )

    environment = worker_environment(settings)

    assert environment["ENV_PROFILE"] == "staging.ascend"
    assert environment["RELEASE_MANIFEST_PATH"] == str((tmp_path / "release.json").resolve())
    assert environment["NPU_ENABLED"] == "true"
    assert environment["NPU_DEVICE_ID"] == "0"
    assert environment["ASCEND_VISIBLE_DEVICES"] == "2"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert environment["VISUAL_MODEL"] == "siglip2-so400m-384"
    assert environment["VISUAL_HF_CACHE_DIR"] == str((tmp_path / "models" / "hf-cache").resolve())
    assert environment["FACE_PROVIDER"] == "cann"
    assert environment["ASR_MODEL"] == "small"
    assert environment["ASR_SEMANTIC_LOCAL_FILES_ONLY"] == "true"
    assert environment["OCR_DEVICE"] == "auto"
    assert environment["OCR_VERSION"] == "PP-OCRv6"
    assert environment["NPU_WORKER_MODE"] == "isolated"


def test_selective_rebuild_preserves_existing_modalities(monkeypatch, tmp_path):
    import app.worker as worker

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
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
    job = catalog.create_job({
        "id": "job-1",
        "video_id": "video-1",
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": ["asr"],
        "options": {"asr_language": "auto"},
    })
    monkeypatch.setattr(worker, "get_settings", lambda: settings)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"chunks": 3}\n',
            stderr="",
        ),
    )

    worker.execute_job(job["id"])

    updated_video = catalog.get_video("video-1")
    updated_job = catalog.get_job(job["id"])
    assert updated_video["indexed_modalities"] == ["asr", "face", "visual"]
    assert updated_video["status"] == "ready"
    assert updated_job["status"] == "completed"
    assert updated_job["modalities"] == ["asr"]
