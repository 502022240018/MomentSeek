#!/usr/bin/env python3
"""Compare one static PP-OCR ONNX model with its Ascend OM artifact.

The same deterministic float32 tensor is sent to ONNX Runtime CPU and pyACL.
CPU output metadata is used to interpret the OM byte buffers, avoiding assumptions
about model-specific output layouts in this first execution-layer validation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import acl
import numpy as np
import onnxruntime as ort


ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(name: str, ret: int) -> None:
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{name} failed: ret={ret}")


class StaticOmSession:
    def __init__(self, model_path: Path, device_id: int, manage_runtime: bool = True):
        self.model_path = model_path
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers: list[tuple[int, int, int]] = []
        self.output_buffers: list[tuple[int, int, int]] = []
        self.manage_runtime = manage_runtime

        if self.manage_runtime:
            _check("acl.init", acl.init())
        try:
            _check("acl.rt.set_device", acl.rt.set_device(device_id))
            self.context, ret = acl.rt.create_context(device_id)
            _check("acl.rt.create_context", ret)
            self.model_id, ret = acl.mdl.load_from_file(str(model_path))
            _check("acl.mdl.load_from_file", ret)
            self.desc = acl.mdl.create_desc()
            _check("acl.mdl.get_desc", acl.mdl.get_desc(self.desc, self.model_id))
        except Exception:
            self.close()
            raise

    def _dataset(self, arrays: list[np.ndarray], output: bool):
        dataset = acl.mdl.create_dataset()
        records = []
        try:
            for array in arrays:
                size = int(array.nbytes)
                device_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
                _check("acl.rt.malloc", ret)
                data_buffer = acl.create_data_buffer(device_ptr, size)
                _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
                _check("acl.mdl.add_dataset_buffer", ret)
                records.append((device_ptr, size, data_buffer))
                if not output:
                    host_ptr = acl.util.numpy_to_ptr(array)
                    _check(
                        "acl.rt.memcpy(H2D)",
                        acl.rt.memcpy(
                            device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE
                        ),
                    )
            return dataset, records
        except Exception:
            self._free_dataset(dataset, records)
            raise

    @staticmethod
    def _free_dataset(dataset, records: list[tuple[int, int, int]]) -> None:
        for device_ptr, _, data_buffer in reversed(records):
            if data_buffer:
                acl.destroy_data_buffer(data_buffer)
            if device_ptr:
                acl.rt.free(device_ptr)
        if dataset:
            acl.mdl.destroy_dataset(dataset)

    def infer(self, inputs: list[np.ndarray], output_templates: list[np.ndarray]):
        if len(inputs) != acl.mdl.get_num_inputs(self.desc):
            raise ValueError("input count does not match OM model")
        if len(output_templates) != acl.mdl.get_num_outputs(self.desc):
            raise ValueError("output count does not match OM model")
        for index, array in enumerate(inputs):
            expected = int(acl.mdl.get_input_size_by_index(self.desc, index))
            if array.nbytes != expected:
                raise ValueError(
                    f"input {index} byte size mismatch: numpy={array.nbytes} om={expected}"
                )
        for index, array in enumerate(output_templates):
            expected = int(acl.mdl.get_output_size_by_index(self.desc, index))
            if array.nbytes != expected:
                raise ValueError(
                    f"output {index} byte size mismatch: cpu={array.nbytes} om={expected}"
                )

        self.input_dataset, self.input_buffers = self._dataset(inputs, output=False)
        self.output_dataset, self.output_buffers = self._dataset(output_templates, output=True)
        started = time.perf_counter()
        _check(
            "acl.mdl.execute",
            acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
        )
        elapsed = time.perf_counter() - started
        outputs = []
        for template, (device_ptr, size, _) in zip(output_templates, self.output_buffers):
            output = np.empty(template.shape, dtype=template.dtype)
            host_ptr = acl.util.numpy_to_ptr(output)
            _check(
                "acl.rt.memcpy(D2H)",
                acl.rt.memcpy(host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST),
            )
            outputs.append(output)
        self._free_dataset(self.input_dataset, self.input_buffers)
        self._free_dataset(self.output_dataset, self.output_buffers)
        self.input_dataset = self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        return outputs, elapsed

    def close(self) -> None:
        if self.input_dataset:
            self._free_dataset(self.input_dataset, self.input_buffers)
            self.input_dataset = None
        if self.output_dataset:
            self._free_dataset(self.output_dataset, self.output_buffers)
            self.output_dataset = None
        if self.desc:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        if self.context:
            acl.rt.destroy_context(self.context)
            self.context = None
        if self.manage_runtime:
            try:
                acl.rt.reset_device(self.device_id)
            except Exception:
                pass
            try:
                acl.finalize()
            except Exception:
                pass


class DynamicDimsOmSession(StaticOmSession):
    """Run a multi-gear dynamic-dims OM with one real tensor input."""

    DYNAMIC_INPUT_NAME = "ascend_mbatch_shape_data"

    def infer(self, inputs: list[np.ndarray], output_templates: list[np.ndarray]):
        if len(inputs) != 1:
            raise ValueError("dynamic-dims experiment expects one real model input")
        if len(output_templates) != acl.mdl.get_num_outputs(self.desc):
            raise ValueError("output count does not match OM model")

        dynamic_index, ret = acl.mdl.get_input_index_by_name(
            self.desc, self.DYNAMIC_INPUT_NAME
        )
        _check("acl.mdl.get_input_index_by_name", ret)
        if acl.mdl.get_num_inputs(self.desc) != 2:
            raise ValueError("dynamic-dims OM must expose data plus metadata inputs")

        tensor = np.ascontiguousarray(inputs[0])
        metadata_size = int(acl.mdl.get_input_size_by_index(self.desc, dynamic_index))
        metadata = np.zeros(metadata_size, dtype=np.uint8)
        model_inputs = [None, None]
        model_inputs[1 - dynamic_index] = tensor
        model_inputs[dynamic_index] = metadata
        self.input_dataset, self.input_buffers = self._dataset(model_inputs, output=False)

        output_buffers = []
        for index, template in enumerate(output_templates):
            maximum_size = int(acl.mdl.get_output_size_by_index(self.desc, index))
            if template.nbytes > maximum_size:
                raise ValueError(
                    f"output {index} exceeds dynamic OM maximum: "
                    f"cpu={template.nbytes} om_max={maximum_size}"
                )
            output_buffers.append(np.empty(maximum_size, dtype=np.uint8))
        self.output_dataset, self.output_buffers = self._dataset(
            output_buffers, output=True
        )

        dims = {
            "name": "",
            "dimCount": tensor.ndim,
            "dims": [int(value) for value in tensor.shape],
        }
        _check(
            "acl.mdl.set_input_dynamic_dims",
            acl.mdl.set_input_dynamic_dims(
                self.model_id, self.input_dataset, dynamic_index, dims
            ),
        )
        started = time.perf_counter()
        _check(
            "acl.mdl.execute",
            acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
        )
        elapsed = time.perf_counter() - started

        outputs = []
        for template, (device_ptr, _, _) in zip(output_templates, self.output_buffers):
            output = np.empty(template.shape, dtype=template.dtype)
            host_ptr = acl.util.numpy_to_ptr(output)
            _check(
                "acl.rt.memcpy(D2H)",
                acl.rt.memcpy(
                    host_ptr,
                    output.nbytes,
                    device_ptr,
                    output.nbytes,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
            )
            outputs.append(output)
        self._free_dataset(self.input_dataset, self.input_buffers)
        self._free_dataset(self.output_dataset, self.output_buffers)
        self.input_dataset = self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        return outputs, elapsed


def _metrics(cpu: np.ndarray, npu: np.ndarray) -> dict:
    cpu64 = cpu.astype(np.float64, copy=False).reshape(-1)
    npu64 = npu.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(cpu64 - npu64)
    denominator = np.maximum(np.abs(cpu64), 1e-8)
    norm_product = np.linalg.norm(cpu64) * np.linalg.norm(npu64)
    cosine = float(np.dot(cpu64, npu64) / norm_product) if norm_product else 1.0
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "max_abs_error": float(delta.max(initial=0)),
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "p99_abs_error": float(np.percentile(delta, 99)) if delta.size else 0.0,
        "max_relative_error": float((delta / denominator).max(initial=0)),
        "cosine_similarity": cosine,
        "cpu_min": float(cpu64.min()) if cpu64.size else 0.0,
        "cpu_max": float(cpu64.max()) if cpu64.size else 0.0,
        "npu_min": float(npu64.min()) if npu64.size else 0.0,
        "npu_max": float(npu64.max()) if npu64.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dynamic-dims", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tensor = rng.normal(0, 1, size=args.shape).astype(np.float32)
    cpu_session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    input_name = cpu_session.get_inputs()[0].name
    cpu_started = time.perf_counter()
    cpu_outputs = cpu_session.run(None, {input_name: tensor})
    cpu_seconds = time.perf_counter() - cpu_started

    session_type = DynamicDimsOmSession if args.dynamic_dims else StaticOmSession
    om_session = session_type(args.om, args.device_id)
    try:
        for _ in range(max(0, args.warmup)):
            om_session.infer([tensor], cpu_outputs)
        run_seconds = []
        npu_outputs = None
        for _ in range(max(1, args.runs)):
            npu_outputs, elapsed = om_session.infer([tensor], cpu_outputs)
            run_seconds.append(elapsed)
    finally:
        om_session.close()

    assert npu_outputs is not None
    report = {
        "onnx": str(args.onnx),
        "om": str(args.om),
        "input_shape": args.shape,
        "seed": args.seed,
        "cpu_seconds": round(cpu_seconds, 6),
        "npu_execute_seconds": [round(value, 6) for value in run_seconds],
        "outputs": [
            _metrics(cpu, npu) for cpu, npu in zip(cpu_outputs, npu_outputs)
        ],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
