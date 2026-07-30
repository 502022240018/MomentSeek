from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CONFIG_NAMES = {
    "config.json",
    "configuration.json",
    "modules.json",
    "open_clip_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
}
WEIGHT_SUFFIXES = {".bin", ".onnx", ".om", ".pt", ".safetensors"}


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取模型清单 {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), list):
        raise ValueError("模型清单必须包含 schema_version=1 和 models 数组")
    return payload


def has_non_empty_file(root: Path, suffixes: set[str] | None = None) -> bool:
    if not root.is_dir():
        return False
    try:
        return any(
            item.is_file()
            and item.stat().st_size > 0
            and (suffixes is None or item.suffix.lower() in suffixes)
            for item in root.rglob("*")
        )
    except OSError:
        return False


def valid_hf_snapshot(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False
    has_config = any((snapshot / name).is_file() for name in CONFIG_NAMES)
    return has_config and has_non_empty_file(snapshot, {".bin", ".safetensors"})


def verify_huggingface(root: Path, model_id: str) -> tuple[bool, str]:
    repo = f"models--{model_id.replace('/', '--')}"
    for repo_dir in (root / "hub" / repo, root / repo):
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        ref = repo_dir / "refs" / "main"
        if ref.is_file():
            target = snapshots / ref.read_text(encoding="utf-8").strip()
            if valid_hf_snapshot(target):
                return True, str(target)
        for snapshot in sorted(snapshots.iterdir()):
            if valid_hf_snapshot(snapshot):
                return True, str(snapshot)
    return False, str(root)


def verify_entry(model_root: Path, entry: dict[str, Any]) -> tuple[bool, str]:
    target = model_root / str(entry["target"])
    kind = str(entry["kind"])
    model_id = str(entry["id"])
    if kind == "huggingface":
        return verify_huggingface(target, model_id)
    if kind == "source":
        ok = (target / "speakerlab").is_dir() and has_non_empty_file(target, {".py"})
        return ok, str(target)
    if kind == "directory":
        return has_non_empty_file(target, WEIGHT_SUFFIXES), str(target)
    raise ValueError(f"不支持的模型类型: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 MomentSeek 本地模型目录")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    missing_required = False
    for entry in manifest["models"]:
        try:
            ok, resolved = verify_entry(args.model_dir, entry)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] 无效模型条目: {exc}", file=sys.stderr)
            return 2
        required = bool(entry.get("required", False))
        status = "OK" if ok else ("ERROR" if required else "WARN")
        print(f"[{status}] {entry['name']}: {resolved}")
        missing_required = missing_required or (required and not ok)
        results.append(
            {
                "name": entry["name"],
                "id": entry["id"],
                "required": required,
                "verified": ok,
                "resolved_path": resolved,
            }
        )

    if args.lock and not missing_required:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "manifest_name": manifest.get("name"),
                    "models": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] 已写入模型锁: {args.lock}")

    if missing_required:
        print("[ERROR] 缺少必需模型，禁止继续部署。", file=sys.stderr)
        return 1
    if args.strict and any(not item["verified"] for item in results):
        print("[ERROR] strict 模式下存在未通过的可选模型。", file=sys.stderr)
        return 1
    print("[OK] 模型校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

