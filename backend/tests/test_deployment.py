import json

from app.deployment import build_deployment_info, load_release_manifest
from app.settings import Settings


DEPLOYMENT_ENV_VARS = (
    "ENV_PROFILE",
    "RELEASE_ID",
    "GIT_COMMIT",
    "IMAGE_TAG",
    "MODEL_MANIFEST",
    "RELEASE_MANIFEST_PATH",
)


def _clear_deployment_env(monkeypatch):
    for name in DEPLOYMENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_build_deployment_info_prefers_explicit_settings(tmp_path, monkeypatch):
    _clear_deployment_env(monkeypatch)
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        env_profile="dev.cuda",
        release_id="dev-local",
        git_commit="abc123",
        image_tag="momentseek:dev",
        model_manifest="deploy/models/dev-full.models.json",
    )

    info = build_deployment_info(settings)

    assert info["env_profile"] == "dev.cuda"
    assert info["release_id"] == "dev-local"
    assert info["git_commit"] == "abc123"
    assert info["image_tag"] == "momentseek:dev"
    assert info["model_manifest"] == "deploy/models/dev-full.models.json"


def test_load_release_manifest_reads_json(tmp_path):
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({
        "release_id": "2026-07-03-abc123",
        "git_commit": "abc123",
        "env_profile": "staging.ascend",
        "models": {"manifest": "deploy/models/ascend-prod.models.json"},
        "image": {"ascend": "momentseek:ascend-abc123"},
    }), encoding="utf-8")

    data = load_release_manifest(manifest)

    assert data["release_id"] == "2026-07-03-abc123"
    assert data["git_commit"] == "abc123"
    assert data["env_profile"] == "staging.ascend"


def test_build_deployment_info_uses_release_manifest_fallbacks(tmp_path, monkeypatch):
    _clear_deployment_env(monkeypatch)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({
        "release_id": "2026-07-03-prod123",
        "git_commit": "prod123",
        "env_profile": "prod.ascend",
        "models": {"manifest": "deploy/models/ascend-prod.models.json"},
        "image": {
            "ascend": "momentseek:ascend-prod123",
            "cuda": "momentseek:cuda-prod123",
        },
    }), encoding="utf-8")
    settings = Settings(_env_file=None, release_manifest_path=manifest)

    info = build_deployment_info(settings)

    assert info["env_profile"] == "prod.ascend"
    assert info["release_id"] == "2026-07-03-prod123"
    assert info["git_commit"] == "prod123"
    assert info["image_tag"] == "momentseek:ascend-prod123"
    assert info["model_manifest"] == "deploy/models/ascend-prod.models.json"


def test_load_release_manifest_ignores_malformed_or_non_object_json(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    non_object = tmp_path / "non-object.json"
    non_object.write_text(json.dumps(["release"]), encoding="utf-8")

    assert load_release_manifest(malformed) == {}
    assert load_release_manifest(non_object) == {}


def test_build_deployment_info_sanitizes_numeric_manifest_values(tmp_path, monkeypatch):
    _clear_deployment_env(monkeypatch)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({
        "release_id": 20260703,
        "git_commit": 123456,
        "env_profile": "prod.cuda",
        "models": {"manifest": 789},
        "image": {"cuda": 456},
    }), encoding="utf-8")
    settings = Settings(_env_file=None, release_manifest_path=manifest)

    info = build_deployment_info(settings)

    assert info == {
        "env_profile": "prod.cuda",
        "release_id": "20260703",
        "git_commit": "123456",
        "image_tag": "456",
        "model_manifest": "789",
    }
    assert all(value is None or isinstance(value, str) for value in info.values())
