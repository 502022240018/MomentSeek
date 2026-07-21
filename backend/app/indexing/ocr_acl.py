from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np


ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_REPEAT_INITIALIZE = 100002
_SHAPE_RE = re.compile(r"-(\d+)x(\d+)x(\d+)x(\d+)\.om$")
_REC_DYNAMIC_WIDTH_GEARS = (
    320, 384, 448, 512, 576, 640, 704, 768, 812, 832, 896, 960, 1024,
    1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1920, 2048,
)


def _check(name: str, ret: int) -> None:
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{name} failed: ret={ret}")


def _shape_from_name(path: Path) -> tuple[int, int, int, int] | None:
    match = _SHAPE_RE.search(path.name)
    return tuple(int(value) for value in match.groups()) if match else None


def _choose_rec_width(width: int) -> int:
    target = next((value for value in _REC_DYNAMIC_WIDTH_GEARS if value >= width), None)
    if target is None:
        raise ValueError(
            f"OCR Rec input width={width} exceeds maximum gear {_REC_DYNAMIC_WIDTH_GEARS[-1]}"
        )
    return target


def _limit_rec_tensor_width(tensor: np.ndarray) -> tuple[np.ndarray, int]:
    max_width = _REC_DYNAMIC_WIDTH_GEARS[-1]
    if tensor.shape[-1] <= max_width:
        return tensor, 0
    import cv2

    resized = np.stack([
        cv2.resize(
            np.transpose(sample, (1, 2, 0)),
            (max_width, int(tensor.shape[2])),
            interpolation=cv2.INTER_AREA,
        ).transpose(2, 0, 1)
        for sample in tensor
    ]).astype(tensor.dtype, copy=False)
    return np.ascontiguousarray(resized), int(tensor.shape[0])


def _choose_covering_shape(
    shapes: dict[tuple[int, int, int, int], Path],
    requested: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], Path]:
    candidates = [
        (shape, path)
        for shape, path in shapes.items()
        if all(actual >= wanted for actual, wanted in zip(shape, requested))
        and shape[1] == requested[1]
    ]
    if not candidates:
        raise ValueError(f"没有可覆盖输入 {requested} 的 OM 档位")
    return min(candidates, key=lambda item: np.prod(item[0]))


class _AclRuntime:
    def __init__(self, device_id: int):
        try:
            import acl
        except ImportError as exc:
            raise RuntimeError("OCR ACL 后端需要容器内提供 pyACL") from exc
        self.acl = acl
        self.device_id = int(device_id)
        self.lock = threading.RLock()
        self.closed = False
        init_ret = acl.init()
        if init_ret not in (ACL_SUCCESS, ACL_ERROR_REPEAT_INITIALIZE):
            _check("acl.init", init_ret)
        self.owns_acl_runtime = init_ret == ACL_SUCCESS
        try:
            _check("acl.rt.set_device", acl.rt.set_device(self.device_id))
            self.context, ret = acl.rt.create_context(self.device_id)
            _check("acl.rt.create_context", ret)
        except Exception:
            if self.owns_acl_runtime:
                try:
                    acl.rt.reset_device(self.device_id)
                finally:
                    acl.finalize()
            raise

    def activate(self) -> None:
        setter = getattr(self.acl.rt, "set_context", None)
        if setter is not None:
            _check("acl.rt.set_context", setter(self.context))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.acl.rt.destroy_context(self.context)
        finally:
            if self.owns_acl_runtime:
                try:
                    self.acl.rt.reset_device(self.device_id)
                finally:
                    self.acl.finalize()


class _AclOmModel:
    DYNAMIC_INPUT_NAME = "ascend_mbatch_shape_data"

    def __init__(self, runtime: _AclRuntime, path: Path, *, dynamic_dims: bool = False):
        if not path.is_file():
            raise FileNotFoundError(f"OCR OM 模型缺失: {path}")
        self.runtime = runtime
        self.path = path
        self.dynamic_dims = dynamic_dims
        self.closed = False
        acl = runtime.acl
        with runtime.lock:
            runtime.activate()
            self.model_id, ret = acl.mdl.load_from_file(str(path))
            _check("acl.mdl.load_from_file", ret)
            self.desc = acl.mdl.create_desc()
            _check("acl.mdl.get_desc", acl.mdl.get_desc(self.desc, self.model_id))

    def _dataset(self, arrays: list[np.ndarray], *, copy_to_device: bool):
        acl = self.runtime.acl
        dataset = acl.mdl.create_dataset()
        records = []
        try:
            for array in arrays:
                array = np.ascontiguousarray(array)
                size = int(array.nbytes)
                device_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
                _check("acl.rt.malloc", ret)
                buffer = acl.create_data_buffer(device_ptr, size)
                _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
                _check("acl.mdl.add_dataset_buffer", ret)
                records.append((device_ptr, size, buffer))
                if copy_to_device:
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

    def _free_dataset(self, dataset, records) -> None:
        acl = self.runtime.acl
        for device_ptr, _, buffer in reversed(records):
            if buffer:
                acl.destroy_data_buffer(buffer)
            if device_ptr:
                acl.rt.free(device_ptr)
        if dataset:
            acl.mdl.destroy_dataset(dataset)

    def infer(self, tensor: np.ndarray, output_templates: list[np.ndarray]) -> list[np.ndarray]:
        acl = self.runtime.acl
        tensor = np.ascontiguousarray(tensor)
        with self.runtime.lock:
            self.runtime.activate()
            inputs: list[np.ndarray]
            dynamic_index = None
            if self.dynamic_dims:
                dynamic_index, ret = acl.mdl.get_input_index_by_name(
                    self.desc, self.DYNAMIC_INPUT_NAME
                )
                _check("acl.mdl.get_input_index_by_name", ret)
                metadata_size = int(
                    acl.mdl.get_input_size_by_index(self.desc, dynamic_index)
                )
                inputs = [tensor, np.zeros(metadata_size, dtype=np.uint8)]
                if dynamic_index == 0:
                    inputs.reverse()
            else:
                inputs = [tensor]

            input_dataset, input_records = self._dataset(inputs, copy_to_device=True)
            output_storage = []
            for index, template in enumerate(output_templates):
                maximum = int(acl.mdl.get_output_size_by_index(self.desc, index))
                if template.nbytes > maximum:
                    raise ValueError(
                        f"OM 输出缓冲区不足: model={self.path}, "
                        f"required={template.nbytes}, maximum={maximum}"
                    )
                output_storage.append(np.empty(maximum, dtype=np.uint8))
            output_dataset, output_records = self._dataset(
                output_storage, copy_to_device=False
            )
            try:
                if dynamic_index is not None:
                    dims = {
                        "name": "",
                        "dimCount": tensor.ndim,
                        "dims": [int(value) for value in tensor.shape],
                    }
                    _check(
                        "acl.mdl.set_input_dynamic_dims",
                        acl.mdl.set_input_dynamic_dims(
                            self.model_id, input_dataset, dynamic_index, dims
                        ),
                    )
                _check(
                    "acl.mdl.execute",
                    acl.mdl.execute(self.model_id, input_dataset, output_dataset),
                )
                outputs = []
                for template, (device_ptr, _, _) in zip(
                    output_templates, output_records
                ):
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
                return outputs
            finally:
                self._free_dataset(input_dataset, input_records)
                self._free_dataset(output_dataset, output_records)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        acl = self.runtime.acl
        with self.runtime.lock:
            self.runtime.activate()
            acl.mdl.destroy_desc(self.desc)
            acl.mdl.unload(self.model_id)


class _AclOrtSession:
    def __init__(self, metadata_session, stage: str, owner: "RapidOCRAclBackend"):
        self.metadata_session = metadata_session
        self.stage = stage
        self.owner = owner

    def run(self, _output_names, input_feed: dict[str, np.ndarray]):
        if len(input_feed) != 1:
            raise ValueError(f"OCR {self.stage} 仅支持一个模型输入")
        tensor = np.ascontiguousarray(next(iter(input_feed.values())))
        return self.owner.infer_stage(self.stage, tensor)

    def get_providers(self):
        return ["ACLExecutionProvider"]

    def __getattr__(self, name: str):
        return getattr(self.metadata_session, name)


class RapidOCRAclBackend:
    engine = "rapidocr_acl"
    device = "npu"
    providers = {stage: ["ACLExecutionProvider"] for stage in ("det", "cls", "rec")}

    def __init__(
        self,
        *,
        device_id: int,
        model_root: str | Path,
        om_root: str | Path,
        ocr_version: str,
        det_lang: str,
        rec_lang: str,
        model_type: str,
        npu_self_test: bool,
    ):
        from app.indexing.ocr import _load_ocr

        self.model_root = Path(model_root)
        self.om_root = Path(om_root)
        if not self.om_root.is_dir():
            raise FileNotFoundError(f"OCR ACL OM 目录缺失: {self.om_root}")
        self.ocr, _ = _load_ocr(
            "cpu",
            device_id,
            self.model_root,
            ocr_version=ocr_version,
            det_lang=det_lang,
            rec_lang=rec_lang,
            model_type=model_type,
            npu_self_test=False,
            # These CPU sessions only provide RapidOCR preprocessing and
            # postprocessing metadata before their inference calls are
            # replaced by ACL OM sessions below. ORT's default creates a
            # host-sized pool per model and can exhaust the container PID
            # limit on large shared servers.
            ort_intra_op_threads=1,
            ort_inter_op_threads=1,
        )
        self.runtime = _AclRuntime(device_id)
        self.models: dict[Path, _AclOmModel] = {}
        self.det_shapes = self._static_shapes("det", "PP-OCRv6_det_small")
        self.cls_shapes = self._static_shapes(
            "cls", "ch_ppocr_mobile_v2.0_cls_mobile"
        )
        self.rec_path = (
            self.om_root
            / "rec-dynamic-width-b5"
            / "PP-OCRv6_rec_small-b5-dynamic-width-1600.om"
        )
        self.rec_wide_path = (
            self.om_root
            / "rec-dynamic-width-b5"
            / "PP-OCRv6_rec_small-b5-dynamic-width-2048.om"
        )
        self.rec_resized_inputs = 0
        self.rec_max_input_width = 0
        if (
            not self.det_shapes
            or not self.cls_shapes
            or not self.rec_path.is_file()
            or not self.rec_wide_path.is_file()
        ):
            self.close()
            raise FileNotFoundError(
                "OCR ACL 模型不完整，需要 Det/Cls 静态档位和 Rec 动态宽度 OM"
            )

        for stage, attr in {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}.items():
            wrapper = getattr(self.ocr, attr).session
            wrapper.session = _AclOrtSession(wrapper.session, stage, self)
        if npu_self_test:
            self._self_test()

    def _self_test(self) -> None:
        import cv2

        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "QATAR WORLD CUP",
            (80, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        output = self.ocr(image, text_score=0.1, box_thresh=0.1)
        text = " ".join(getattr(output, "txts", None) or [])
        if "QATAR" not in text.upper() and "WORLD" not in text.upper():
            raise RuntimeError(f"OCR ACL NPU 自检失败: output={text!r}")

    def _static_shapes(self, stage: str, stem: str):
        result = {}
        for path in (self.om_root / stage).glob(f"{stem}-*.om"):
            shape = _shape_from_name(path)
            if shape is not None:
                result[shape] = path
        return result

    def _model(self, path: Path, *, dynamic_dims: bool = False) -> _AclOmModel:
        model = self.models.get(path)
        if model is None:
            model = _AclOmModel(self.runtime, path, dynamic_dims=dynamic_dims)
            self.models[path] = model
        return model

    def infer_stage(self, stage: str, tensor: np.ndarray) -> list[np.ndarray]:
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != 4:
            raise ValueError(f"OCR {stage} 输入必须为四维张量: {shape}")
        if stage == "det":
            target, path = _choose_covering_shape(self.det_shapes, shape)
            padded = np.zeros(target, dtype=tensor.dtype)
            padded[tuple(slice(0, value) for value in shape)] = tensor
            output = self._model(path).infer(
                padded, [np.empty((target[0], 1, target[2], target[3]), np.float32)]
            )[0]
            return [output[:, :, : shape[2], : shape[3]]]
        if stage == "cls":
            target, path = _choose_covering_shape(self.cls_shapes, shape)
            padded = np.zeros(target, dtype=tensor.dtype)
            padded[: shape[0]] = tensor
            output = self._model(path).infer(
                padded, [np.empty((target[0], 2), np.float32)]
            )[0]
            return [output[: shape[0]]]
        if stage == "rec":
            return [self._infer_rec(tensor)]
        raise ValueError(f"未知 OCR ACL stage: {stage}")

    def _infer_rec(self, tensor: np.ndarray) -> np.ndarray:
        self.rec_max_input_width = max(self.rec_max_input_width, int(tensor.shape[-1]))
        tensor, resized_inputs = _limit_rec_tensor_width(tensor)
        self.rec_resized_inputs += resized_inputs
        batch, channels, height, width = tensor.shape
        target_width = _choose_rec_width(width)
        model_path = self.rec_wide_path if target_width > 1600 else self.rec_path
        model = self._model(model_path, dynamic_dims=True)
        outputs = []
        for offset in range(0, batch, 5):
            chunk = tensor[offset : offset + 5]
            padded = np.zeros((5, channels, height, target_width), dtype=tensor.dtype)
            padded[: len(chunk), :, :, :width] = chunk
            output = model.infer(
                padded,
                [np.empty((5, target_width // 8, 18710), dtype=np.float32)],
            )[0]
            outputs.append(output[: len(chunk), : width // 8])
        return np.concatenate(outputs, axis=0)

    def __call__(self, frame: np.ndarray):
        return self.ocr(frame)

    def close(self) -> None:
        for model in reversed(list(getattr(self, "models", {}).values())):
            model.close()
        if hasattr(self, "runtime"):
            self.runtime.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
