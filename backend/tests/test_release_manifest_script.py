import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_write_release_manifest_separates_api_and_search_smoke(tmp_path):
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text(
        json.dumps({"schema_version": 1, "name": "test", "allow_download": False, "models": []}),
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div>MomentSeek</div>", encoding="utf-8")
    out = tmp_path / "release.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "write_release_manifest.py"),
            "--env-profile",
            "staging.ascend",
            "--model-manifest",
            str(model_manifest),
            "--frontend-dist",
            str(dist),
            "--out",
            str(out),
            "--allow-dirty",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(out.read_text(encoding="utf-8"))
    verification = manifest["verification"]
    assert verification["api_smoke"] == "required"
    assert verification["search_smoke"] == "required"
    assert "smoke_search" not in verification
