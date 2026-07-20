#!/usr/bin/env python3
"""Replace Det/Cls/Rec ORT sessions with exact-shape OM on one real frame."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.indexing.ocr import _load_ocr
from compare_ppocr_onnx_om import StaticOmSession, _metrics


STAGE_ATTRS = {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}
MODEL_STEMS = {
    "det": "PP-OCRv6_det_small",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile",
    "rec": "PP-OCRv6_rec_small",
}


class CaptureSession:
    def __init__(self, session, stage: str):
        self.session = session
        self.stage = stage
        self.calls: list[dict] = []

    def run(self, *args, **kwargs):
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        captured_feed = {
            name: np.ascontiguousarray(value) for name, value in (feed or {}).items()
        }
        outputs = self.session.run(*args, **kwargs)
        self.calls.append({"feed": captured_feed, "outputs": outputs})
        return outputs

    def __getattr__(self, name):
        return getattr(self.session, name)


class ExactShapeOmSession:
    def __init__(self, metadata_session, stage: str, calls: list[dict], om_root: Path, device_id: int):
        self.metadata_session = metadata_session
        self.stage = stage
        self.calls = calls
        self.om_root = om_root
        self.device_id = device_id
        self.call_index = 0
        self.records: list[dict] = []

    def run(self, *args, **kwargs):
        if self.call_index >= len(self.calls):
            raise RuntimeError(f"{self.stage} OM replay received more calls than CPU baseline")
        baseline = self.calls[self.call_index]
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        inputs = [np.ascontiguousarray(value) for value in (feed or {}).values()]
        if len(inputs) != 1:
            raise ValueError(f"{self.stage} expected one model input, got {len(inputs)}")
        shape = list(inputs[0].shape)
        shape_slug = "x".join(str(value) for value in shape)
        om_path = self.om_root / self.stage / f"{MODEL_STEMS[self.stage]}-{shape_slug}.om"
        if not om_path.is_file():
            raise FileNotFoundError(
                f"missing exact-shape {self.stage} OM for {shape}: {om_path}"
            )

        total_started = time.perf_counter()
        session = StaticOmSession(om_path, self.device_id)
        try:
            outputs, execute_seconds = session.infer(inputs, baseline["outputs"])
        finally:
            session.close()
        self.records.append({
            "shape": shape,
            "om": str(om_path),
            "execute_seconds": execute_seconds,
            "load_execute_release_seconds": time.perf_counter() - total_started,
            "outputs": [
                _metrics(cpu, npu)
                for cpu, npu in zip(baseline["outputs"], outputs)
            ],
        })
        self.call_index += 1
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


def _payload(result) -> dict:
    boxes = getattr(result, "boxes", None)
    return {
        "texts": [str(value) for value in (getattr(result, "txts", None) or [])],
        "scores": [float(value) for value in (getattr(result, "scores", None) or [])],
        "boxes": np.asarray(boxes).tolist() if boxes is not None else [],
    }


def _final_metrics(cpu: dict, om: dict) -> dict:
    cpu_boxes = np.asarray(cpu["boxes"], dtype=np.float64)
    om_boxes = np.asarray(om["boxes"], dtype=np.float64)
    boxes_match = cpu_boxes.shape == om_boxes.shape
    scores_match = len(cpu["scores"]) == len(om["scores"])
    return {
        "cpu_detection_count": len(cpu["texts"]),
        "om_detection_count": len(om["texts"]),
        "texts_exact_match": cpu["texts"] == om["texts"],
        "cpu_texts": cpu["texts"],
        "om_texts": om["texts"],
        "box_shapes_match": boxes_match,
        "box_max_abs_error": (
            float(np.abs(cpu_boxes - om_boxes).max())
            if boxes_match and cpu_boxes.size else None
        ),
        "score_max_abs_error": (
            float(np.abs(np.asarray(cpu["scores"]) - np.asarray(om["scores"])).max())
            if scores_match and cpu["scores"] else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timestamp", type=float, default=10.0)
    parser.add_argument("--decode-height", type=int, default=720)
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--om-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = _decode_frame(args.video, args.timestamp, args.decode_height)
    ocr, providers = _load_ocr("cpu", 0, args.model_root, npu_self_test=False)
    wrappers = {
        stage: getattr(ocr, attr).session for stage, attr in STAGE_ATTRS.items()
    }
    originals = {stage: wrapper.session for stage, wrapper in wrappers.items()}
    captures = {
        stage: CaptureSession(originals[stage], stage) for stage in STAGE_ATTRS
    }
    for stage, wrapper in wrappers.items():
        wrapper.session = captures[stage]

    cpu_started = time.perf_counter()
    cpu_result = ocr(frame)
    cpu_pipeline_seconds = time.perf_counter() - cpu_started
    if not captures["det"].calls:
        raise RuntimeError("detector was not called during CPU baseline")

    replays = {
        stage: ExactShapeOmSession(
            originals[stage], stage, captures[stage].calls, args.om_root, args.device_id
        )
        for stage in STAGE_ATTRS if captures[stage].calls
    }
    for stage, wrapper in wrappers.items():
        wrapper.session = replays.get(stage, originals[stage])
    try:
        om_started = time.perf_counter()
        om_result = ocr(frame)
        om_pipeline_seconds = time.perf_counter() - om_started
    finally:
        for stage, wrapper in wrappers.items():
            wrapper.session = originals[stage]

    cpu_payload = _payload(cpu_result)
    om_payload = _payload(om_result)
    report = {
        "video": str(args.video),
        "timestamp": args.timestamp,
        "frame_shape": list(frame.shape),
        "providers": providers,
        "cpu_pipeline_seconds": round(cpu_pipeline_seconds, 6),
        "om_pipeline_seconds_with_per_stage_load": round(om_pipeline_seconds, 6),
        "stages": {
            stage: {"call_count": len(captures[stage].calls), "calls": replay.records}
            for stage, replay in replays.items()
        },
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
