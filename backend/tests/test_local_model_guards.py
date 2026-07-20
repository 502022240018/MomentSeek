from pathlib import Path
import sys

import pytest

from app.indexing.faces import FaceEncoder
from app.indexing.ocr import _load_ocr, create_ocr_backend


def test_rapidocr_requires_local_model_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="本地 RapidOCR 模型缺失"):
        _load_ocr("cpu", 0, tmp_path, npu_self_test=False)


def test_unknown_ocr_backend_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match="尚未启用的 OCR backend"):
        create_ocr_backend(
            "ppocr_om",
            device="npu",
            device_id=0,
            model_root=tmp_path,
            ocr_version="PP-OCRv6",
            det_lang="ch",
            rec_lang="ch",
            model_type="small",
            npu_self_test=True,
        )


def test_acl_ocr_requires_npu_device(tmp_path):
    with pytest.raises(ValueError, match="仅支持 device=npu"):
        create_ocr_backend(
            "rapidocr_acl",
            device="cpu",
            device_id=0,
            model_root=tmp_path,
            ocr_version="PP-OCRv6",
            det_lang="ch",
            rec_lang="ch",
            model_type="small",
            npu_self_test=False,
        )


def test_face_encoder_requires_local_model_files(monkeypatch, tmp_path):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt())

    with pytest.raises(FileNotFoundError, match="本地 InsightFace 模型缺失"):
        FaceEncoder("buffalo_l", "cpu", 0, str(tmp_path))


def test_face_encoder_accepts_local_model_files(monkeypatch, tmp_path):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, name, providers, root):
            self.name = name
            self.providers = providers
            self.root = root

        def prepare(self, ctx_id, det_size):
            self.ctx_id = ctx_id
            self.det_size = det_size

    class FakeInsightfaceApp:
        FaceAnalysis = FakeFaceAnalysis

    model_dir = tmp_path / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"weights")

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt())
    monkeypatch.setitem(sys.modules, "insightface.app", FakeInsightfaceApp())

    encoder = FaceEncoder("buffalo_l", "cpu", 0, str(tmp_path))

    assert encoder.provider == "cpu"
    assert isinstance(encoder.app, FakeFaceAnalysis)
    assert Path(encoder.app.root) == tmp_path


def test_face_encoder_cann_does_not_silently_fall_back_to_cpu(monkeypatch, tmp_path):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    model_dir = tmp_path / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt())

    with pytest.raises(RuntimeError, match="CANNExecutionProvider"):
        FaceEncoder("buffalo_l", "cann", 0, str(tmp_path))
