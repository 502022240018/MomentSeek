#!/usr/bin/env python3
"""Read-only preflight for the PP-OCRv6 Small ONNX -> Ascend OM experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import onnx


MODEL_NAMES = (
    "PP-OCRv6_det_small.onnx",
    "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
    "PP-OCRv6_rec_small.onnx",
)


def _shape(value_info) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_value:
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(str(dimension.dim_param))
        else:
            result.append("dynamic")
    return result


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: str) -> dict:
    path = shutil.which(command)
    if not path:
        return {"available": False}
    process = subprocess.run([path, "--version"], text=True, capture_output=True, timeout=20, check=False)
    return {
        "available": True,
        "path": path,
        "returncode": process.returncode,
        "output": (process.stdout or process.stderr).strip()[-1000:],
    }


def inspect_model(path: Path) -> dict:
    payload = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return payload
    payload.update({
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    initializers = {item.name for item in model.graph.initializer}
    payload["inputs"] = [
        {"name": item.name, "shape": _shape(item)}
        for item in model.graph.input
        if item.name not in initializers
    ]
    payload["outputs"] = [{"name": item.name, "shape": _shape(item)} for item in model.graph.output]
    payload["opset"] = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    payload["valid"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    models = [inspect_model(args.model_root / name) for name in MODEL_NAMES]
    report = {
        "goal": "reuse PP-OCRv6 Small and replace only its Ascend inference runtime",
        "model_root": str(args.model_root),
        "models": models,
        "tools": {"atc": _command_version("atc"), "npu_smi": _command_version("npu-smi")},
        "packages": {
            name: _package_version(name)
            for name in ("onnx", "onnxruntime-cann", "paddlepaddle", "paddle-custom-npu", "paddlex", "paddleocr")
        },
        "ready_for_shape_design": all(item.get("valid") for item in models),
        "ready_for_local_atc_conversion": all(item.get("valid") for item in models)
        and _command_version("atc")["available"],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ready_for_shape_design"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
