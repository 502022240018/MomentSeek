from app.settings import Settings
from app.worker import worker_environment


def test_worker_environment_uses_absolute_runtime_paths(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")

    environment = worker_environment(settings)

    assert environment["APP_DATA_DIR"] == str((tmp_path / "runtime").resolve())
    assert environment["APP_MODEL_DIR"] == str((tmp_path / "models").resolve())
    assert environment["PYTHONIOENCODING"] == "utf-8"


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
        ocr_version="PP-OCRv4",
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
    assert environment["OCR_VERSION"] == "PP-OCRv4"
