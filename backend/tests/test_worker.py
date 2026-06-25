from app.settings import Settings
from app.worker import worker_environment


def test_worker_environment_uses_absolute_runtime_paths(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")

    environment = worker_environment(settings)

    assert environment["APP_DATA_DIR"] == str((tmp_path / "runtime").resolve())
    assert environment["APP_MODEL_DIR"] == str((tmp_path / "models").resolve())
    assert environment["PYTHONIOENCODING"] == "utf-8"
