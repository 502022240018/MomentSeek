#!/usr/bin/env python3
"""Compare RapidOCR CPU detection with an OM-backed detector on one real frame."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.indexing.ocr import _load_ocr
from compare_ppocr_onnx_om import StaticOmSession, _metrics


class CaptureSession:
    def __init__(self, session):
        self.session = session
        self.feed = None
        self.outputs = None

    def run(self, *args, **kwargs):
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        self.feed = {name: np.ascontiguousarray(value) for name, value in feed.items()}
        self.outputs = self.session.run(*args, **kwargs)
        return self.outputs

    def __getattr__(self, name):
        return getattr(self.session, name)


class OmReplaySession:
    def __init__(self, metadata_session, om_session: StaticOmSession, templates):
        self.metadata_session = metadata_session
        self.om_session = om_session
        self.templates = templates
        self.execute_seconds = []
        self.last_outputs = None

    def run(self, *args, **kwargs):
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        inputs = [np.ascontiguousarray(value) for value in feed.values()]
        outputs, elapsed = self.om_session.infer(inputs, self.templates)
        self.execute_seconds.append(elapsed)
        self.last_outputs = outputs
        return outputs

    def __getattr__(self, name):
        return getattr(self.metadata_session, name)


def _decode_frame(path: Path, timestamp: float, decode_height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame at {timestamp}s: {path}")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    if decode_height > 0 and height > decode_height:
        output_width = max(2, int(round(width * decode_height / height / 2) * 2))
        frame = cv2.resize(frame, (output_width, decode_height), interpolation=cv2.INTER_LINEAR)
    return frame


def _result_payload(result) -> dict:
    boxes = getattr(result, "boxes", None)
    txts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    return {
        "texts": [str(value) for value in txts],
        "scores": [float(value) for value in scores],
        "boxes": np.asarray(boxes).tolist() if boxes is not None else [],
    }


def _final_metrics(cpu: dict, om: dict) -> dict:
    cpu_boxes = np.asarray(cpu["boxes"], dtype=np.float64)
    om_boxes = np.asarray(om["boxes"], dtype=np.float64)
    comparable_boxes = cpu_boxes.shape == om_boxes.shape and cpu_boxes.size > 0
    return {
        "cpu_detection_count": len(cpu["texts"]),
        "om_detection_count": len(om["texts"]),
        "texts_exact_match": cpu["texts"] == om["texts"],
        "cpu_texts": cpu["texts"],
        "om_texts": om["texts"],
        "box_shapes_match": list(cpu_boxes.shape) == list(om_boxes.shape),
        "box_max_abs_error": (
            float(np.abs(cpu_boxes - om_boxes).max()) if comparable_boxes else None
        ),
        "score_max_abs_error": (
            float(np.max(np.abs(np.asarray(cpu["scores"]) - np.asarray(om["scores"]))))
            if len(cpu["scores"]) == len(om["scores"]) and cpu["scores"] else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timestamp", type=float, default=0.0)
    parser.add_argument("--decode-height", type=int, default=720)
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--om-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = _decode_frame(args.video, args.timestamp, args.decode_height)
    ocr, providers = _load_ocr("cpu", 0, args.model_root, npu_self_test=False)
    det_wrapper = ocr.text_det.session
    original_session = det_wrapper.session
    capture = CaptureSession(original_session)
    det_wrapper.session = capture
    cpu_started = time.perf_counter()
    cpu_result = ocr(frame)
    cpu_pipeline_seconds = time.perf_counter() - cpu_started
    if not capture.feed or capture.outputs is None:
        raise RuntimeError("RapidOCR detector input was not captured")

    input_tensor = next(iter(capture.feed.values()))
    shape_slug = "x".join(str(value) for value in input_tensor.shape)
    om_path = args.om_root / "det" / f"PP-OCRv6_det_small-{shape_slug}.om"
    if not om_path.is_file():
        raise FileNotFoundError(f"no exact-shape detector OM for {input_tensor.shape}: {om_path}")

    om_session = StaticOmSession(om_path, args.device_id)
    replay = OmReplaySession(original_session, om_session, capture.outputs)
    det_wrapper.session = replay
    try:
        om_started = time.perf_counter()
        om_result = ocr(frame)
        om_pipeline_seconds = time.perf_counter() - om_started
    finally:
        det_wrapper.session = original_session
        om_session.close()

    cpu_payload = _result_payload(cpu_result)
    om_payload = _result_payload(om_result)
    report = {
        "video": str(args.video),
        "timestamp": args.timestamp,
        "frame_shape": list(frame.shape),
        "det_input_shape": list(input_tensor.shape),
        "om": str(om_path),
        "providers": providers,
        "cpu_pipeline_seconds": round(cpu_pipeline_seconds, 6),
        "om_pipeline_seconds": round(om_pipeline_seconds, 6),
        "om_det_execute_seconds": [round(value, 6) for value in replay.execute_seconds],
        "raw_det_output": _metrics(capture.outputs[0], replay.last_outputs[0]),
        "final_result": _final_metrics(cpu_payload, om_payload),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
