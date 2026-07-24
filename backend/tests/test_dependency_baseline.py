import json
from pathlib import Path

from scripts.verify_models import verify_non_hf_target


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _requirement_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in _read(path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_ascend_deploy_uses_checked_in_build_definition_only():
    script = _read("scripts/deploy_ascend_shared_server.sh")

    assert 'cp "$SOURCE_DIR/Dockerfile.ascend" "$BUILD_DIR/Dockerfile"' in script
    assert "requirements-server.txt" not in script
    assert "constraints-server.txt" not in script
    assert "cat >\"$BUILD_DIR/Dockerfile\"" not in script


def test_ascend_dependency_layers_preserve_vendor_torch_stack():
    dockerfile = _read("Dockerfile.ascend")
    constraints = _read("backend/constraints-ascend.txt")
    preserve = _read("backend/requirements-ascend-preserve.txt")
    preserve_lines = _requirement_lines("backend/requirements-ascend-preserve.txt")

    assert "torch==2.9.0" in constraints
    assert "torch-npu==2.9.0.post1" in constraints
    assert "pip install --no-deps -r requirements-ascend-preserve.txt" in dockerfile
    assert "open_clip_torch==3.3.0" in preserve
    assert "funasr==1.3.9" in _read("backend/requirements-ascend.txt")
    assert "opencv-python==4.11.0.86" in constraints
    assert not any(line.startswith("torch==") for line in preserve_lines)
    assert "cv2.__version__ == '4.11.0'" in dockerfile


def test_speaker_runtime_excludes_research_only_dependency_chain():
    speaker = _requirement_lines("backend/requirements-speaker.txt")
    runtime = _read("backend/app/indexing/speaker_3dspeaker_runtime.py")

    for package in ("pyannote", "fastcluster", "umap-learn", "hdbscan"):
        assert not any(package in line.casefold() for line in speaker)
    assert "from sklearn.cluster import HDBSCAN" in _read(
        "backend/app/indexing/speaker.py"
    )
    assert "torch.vmap(self.feature_extractor)(batch)" in runtime
    assert "features.to(self.device)" in runtime
    assert "pyannote" not in runtime.split('"""', 2)[-1]


def test_milvus_server_and_client_are_on_matching_26_line():
    compose = _read("compose.milvus.yml")
    client = _read("backend/requirements-milvus-client.txt")
    client_lines = _requirement_lines("backend/requirements-milvus-client.txt")

    assert "milvusdb/milvus:v2.6.20" in compose
    assert "pymilvus==2.6.16" in client
    assert "orjson==3.11.9" in client
    assert "cachetools==5.5.2" in client
    assert not any("milvus-lite" in line for line in client_lines)


def test_production_manifest_requires_speaker_source_and_models():
    manifest = json.loads(_read("deploy/models/ascend-prod.models.json"))
    entries = {item["name"]: item for item in manifest["models"]}

    assert entries["speaker_3dspeaker_source"]["id"] == "065629c313ea"
    assert entries["speaker_3dspeaker_source"]["required"] is True
    assert entries["speaker_campplus_zh_en"]["required"] is True
    assert entries["speaker_fsmn_vad"]["required"] is True
    assert "/snapshots/v1.0.0" in entries["speaker_campplus_zh_en"]["target"]
    assert "/snapshots/v2.0.4" in entries["speaker_fsmn_vad"]["target"]


def test_source_asset_verification_checks_file_and_revision(tmp_path):
    source = tmp_path / "3D-Speaker"
    script = source / "speakerlab" / "bin" / "infer_diarization.py"
    script.parent.mkdir(parents=True)
    script.write_text("# pinned upstream source\n", encoding="utf-8")
    (source / ".momentseek-revision").write_text("065629c313ea\n", encoding="utf-8")

    assert verify_non_hf_target("source", source, "065629c313ea") is True
    assert verify_non_hf_target("source", source, "different") is False
