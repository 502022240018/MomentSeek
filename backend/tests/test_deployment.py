import json

from app.deployment import build_deployment_info, load_release_manifest
from app.settings import Settings


def test_build_deployment_info_prefers_explicit_settings(tmp_path):
    settings = Settings(
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
