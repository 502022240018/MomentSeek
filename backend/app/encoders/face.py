"""InsightFace/ArcFace face encoder shared by indexing and retrieval."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.indexing.common import normalize


def _has_non_empty_onnx(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() and item.stat().st_size > 0 and item.suffix.lower() == ".onnx" for item in path.rglob("*"))


def _resolve_insightface_root(root: str | None, model_name: str) -> Path:
    root_path = Path(root or "~/.insightface").expanduser()
    canonical_model_dir = root_path / "models" / model_name
    if _has_non_empty_onnx(canonical_model_dir):
        return root_path

    models_parent_dir = root_path / model_name
    if _has_non_empty_onnx(models_parent_dir) and root_path.name == "models":
        return root_path.parent

    if _has_non_empty_onnx(root_path) and root_path.name == model_name and root_path.parent.name == "models":
        return root_path.parent.parent

    raise FileNotFoundError(f"本地 InsightFace 模型缺失: expected {canonical_model_dir}")


class FaceEncoder:
    def __init__(
        self,
        model_name: str,
        provider: str = "cpu",
        device_id: int = 0,
        root: str | None = None,
        ort_intra_op_threads: int = 8,
        ort_inter_op_threads: int = 1,
    ):
        face_root = _resolve_insightface_root(root, model_name)
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider == "cann" and "CANNExecutionProvider" not in available:
            raise RuntimeError(
                "Face 已配置为 CANN NPU，但 onnxruntime 未提供 CANNExecutionProvider；"
                f"available_providers={available}。为避免产品环境静默回落 CPU，任务已终止。"
            )
        from insightface.app import FaceAnalysis

        if provider == "cann":
            providers = [("CANNExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
            ctx_id = device_id
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        analysis_options = {
            "name": model_name,
            "providers": providers,
            "root": str(face_root),
            "allowed_modules": ["detection", "recognition"],
        }
        if provider != "cann":
            # The Ascend ORT build currently fails CANN graph initialization
            # when an explicit SessionOptions object is supplied. Bound the
            # CPU query encoder, but leave CANN session creation to the EP.
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = max(1, int(ort_intra_op_threads))
            session_options.inter_op_num_threads = max(1, int(ort_inter_op_threads))
            analysis_options["sess_options"] = session_options
        self.app = FaceAnalysis(**analysis_options)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.provider = provider

    def detect(self, frame_bgr: np.ndarray):
        return self.app.get(frame_bgr)

    def encode_reference(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise OSError(f"无法读取参考图: {path}")
        faces = self.detect(image)
        if not faces:
            raise ValueError("参考图中未检测到人脸")
        face = max(faces, key=lambda item: float(np.prod(item.bbox[2:] - item.bbox[:2])))
        return normalize(face.normed_embedding)

