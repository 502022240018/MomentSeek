#!/usr/bin/env python3
"""Compile exact, observed RapidOCR tensor shapes into Ascend OM artifacts.

This is a conversion feasibility experiment, not the final product shape policy.
It reads ocr-shape-profile.json, compiles each unique observed shape separately,
and records every command/result in a manifest for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("det", "cls", "rec")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_model_path(raw_path: str, model_root: Path) -> Path:
    path = Path(raw_path).resolve()
    root = model_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"profile model path escapes model root: {path}") from exc
    if not path.is_file() or path.suffix.casefold() != ".onnx":
        raise FileNotFoundError(f"profile ONNX model is missing: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=Path("/app/runtime/ocr-shape-profile.json"))
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/app/models/rapidocr/ascend/910b4-cann9-profile"),
    )
    parser.add_argument("--soc-version", default="Ascend910B4")
    parser.add_argument("--precision-mode", default="must_keep_origin_dtype")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    atc = shutil.which("atc")
    if not atc:
        raise SystemExit("ATC is not available in PATH")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "purpose": "exact-shape PP-OCRv6 Small OM conversion feasibility",
        "product_ready": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": str(args.profile),
        "profile_sha256": _sha256(args.profile),
        "soc_version": args.soc_version,
        "precision_mode": args.precision_mode,
        "atc": atc,
        "cann_home": os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME"),
        "artifacts": [],
    }

    for stage in STAGES:
        raw_model_path = profile["runtime_models"][stage]["session"].get("_model_path")
        if not raw_model_path:
            raise ValueError(f"profile does not expose runtime model path for {stage}")
        model_path = _safe_model_path(raw_model_path, args.model_root)
        unique_shapes = profile["tensor_shapes"][stage]["unique_shapes"]
        for item in unique_shapes:
            shape = [int(value) for value in item["shape"]]
            if len(shape) != 4 or any(value <= 0 for value in shape):
                raise ValueError(f"invalid observed shape for {stage}: {shape}")
            shape_slug = "x".join(str(value) for value in shape)
            output_prefix = args.output_dir / stage / f"{model_path.stem}-{shape_slug}"
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            om_path = output_prefix.with_suffix(".om")
            log_path = output_prefix.with_suffix(".atc.log")
            command = [
                atc,
                f"--model={model_path}",
                "--framework=5",
                f"--output={output_prefix}",
                "--input_format=NCHW",
                f"--input_shape=x:{','.join(str(value) for value in shape)}",
                f"--soc_version={args.soc_version}",
                f"--precision_mode={args.precision_mode}",
                "--log=error",
            ]
            started = time.perf_counter()
            if om_path.is_file() and not args.force:
                returncode = 0
                output = "SKIPPED: existing artifact"
                skipped = True
            else:
                process = subprocess.run(command, text=True, capture_output=True, check=False)
                returncode = process.returncode
                output = "\n".join(part for part in (process.stdout, process.stderr) if part).strip()
                log_path.write_text(output + "\n", encoding="utf-8")
                skipped = False
            artifact = {
                "stage": stage,
                "source_model": str(model_path),
                "source_sha256": _sha256(model_path),
                "observed_count": int(item.get("count") or 0),
                "input_shape": shape,
                "command": command,
                "returncode": returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "skipped": skipped,
                "om_path": str(om_path),
                "om_exists": om_path.is_file(),
                "om_bytes": om_path.stat().st_size if om_path.is_file() else 0,
                "om_sha256": _sha256(om_path) if om_path.is_file() else None,
                "log_path": str(log_path),
                "log_tail": output[-3000:],
            }
            manifest["artifacts"].append(artifact)
            print(
                f"stage={stage} shape={shape_slug} rc={returncode} "
                f"seconds={artifact['elapsed_seconds']} om={artifact['om_exists']}",
                flush=True,
            )

    manifest["success"] = all(item["returncode"] == 0 and item["om_exists"] for item in manifest["artifacts"])
    manifest_path = args.output_dir / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "success": manifest["success"],
        "artifact_count": len(manifest["artifacts"]),
        "manifest": str(manifest_path),
    }, ensure_ascii=False))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
