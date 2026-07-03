import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_bootstrap_fixture(tmp_path: Path, script_name: str) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    script = root / "scripts" / script_name
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / script_name, script)

    _write_text(root / "deploy" / "env" / "staging.ascend.example", "ENV_PROFILE=staging.ascend\n")
    _write_text(root / "deploy" / "env" / "prod.ascend.example", "ENV_PROFILE=prod.ascend\n")
    _write_text(root / "deploy" / "env" / "dev.cpu.example", "ENV_PROFILE=dev.cpu\n")
    _write_text(root / "deploy" / "models" / "dev-full.models.json", "{}\n")
    _write_text(root / "deploy" / "models" / "ascend-prod.models.json", "{}\n")
    _write_text(root / "frontend" / "package.json", "{}\n")
    return root, script


def _fake_tool_path(tmp_path: Path) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("python", "npm"):
        tool = bin_dir / name
        _write_text(tool, "#!/usr/bin/env bash\nexit 0\n")
        tool.chmod(0o755)
        _write_text(bin_dir / f"{name}.cmd", "@echo off\r\nexit /b 0\r\n")
    return str(bin_dir)


def test_bash_bootstrap_rejects_ascend_profiles_before_touching_env(tmp_path):
    bash = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"
    if not Path(bash).exists():
        pytest.skip("bash is not available")
    root, script = _prepare_bootstrap_fixture(tmp_path, "bootstrap_dev.sh")
    env = os.environ.copy()
    env["PATH"] = _fake_tool_path(tmp_path) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [bash, str(script), "staging.ascend"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "bootstrap_dev only supports dev.cpu and dev.cuda" in result.stderr
    assert not (root / ".env").exists()


def test_powershell_bootstrap_rejects_ascend_profiles_before_touching_env(tmp_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is not available")
    root, script = _prepare_bootstrap_fixture(tmp_path, "bootstrap_dev.ps1")
    env = os.environ.copy()
    env["PATH"] = _fake_tool_path(tmp_path) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Profile",
            "prod.ascend",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "bootstrap_dev only supports dev.cpu and dev.cuda" in (result.stderr + result.stdout)
    assert not (root / ".env").exists()
