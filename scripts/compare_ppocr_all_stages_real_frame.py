#!/usr/bin/env python3
"""Replace Det/Cls/Rec ORT sessions with exact-shape OM on one real frame."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import acl
import numpy as np

from app.indexing.ocr import _load_ocr
from compare_ppocr_onnx_om import DynamicDimsOmSession, StaticOmSession, _metrics


STAGE_ATTRS = {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}
MODEL_STEMS = {
    "det": "PP-OCRv6_det_small",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile",
    "rec": "PP-OCRv6_rec_small",
}
REC_DYNAMIC_WIDTH_GEARS = (320, 384, 448, 512, 576, 640, 704, 768, 812, 832, 896, 960, 1024)
REC_DYNAMIC_BATCH = 5


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
    def __init__(
        self,
        metadata_session,
        stage: str,
        calls: list[dict],
        om_root: Path,
        device_id: int,
        rec_dynamic_om: Path | None = None,
    ):
        self.metadata_session = metadata_session
        self.stage = stage
        self.calls = calls
        self.om_root = om_root
        self.device_id = device_id
        self.rec_dynamic_om = rec_dynamic_om
        self.call_index = 0
        self.records: list[dict] = []

    def _find_padded_cls(self, shape: list[int]) -> tuple[Path, int] | None:
        if self.stage != "cls" or len(shape) != 4:
            return None
        suffix = "x".join(str(value) for value in shape[1:])
        candidates: list[tuple[int, Path]] = []
        for path in (self.om_root / "cls").glob(
            f"{MODEL_STEMS['cls']}-*x{suffix}.om"
        ):
            batch_text = path.name.removeprefix(
                f"{MODEL_STEMS['cls']}-"
            ).split("x", 1)[0]
            if batch_text.isdigit() and int(batch_text) >= shape[0]:
                candidates.append((int(batch_text), path))
        return min(candidates, key=lambda item: item[0]) if candidates else None

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
        session_type = StaticOmSession
        mode = "exact_shape"
        padded_batch = None
        padded_width = None
        if not om_path.is_file():
            padded_cls = self._find_padded_cls(shape)
            if padded_cls is not None:
                padded_batch, om_path = padded_cls
                mode = "fixed_batch_padding"
        if (
            not om_path.is_file()
            and self.stage == "rec"
            and self.rec_dynamic_om is not None
            and self.rec_dynamic_om.is_file()
            and shape[0] <= REC_DYNAMIC_BATCH
        ):
            padded_width = next(
                (width for width in REC_DYNAMIC_WIDTH_GEARS if width >= shape[3]),
                None,
            )
            if padded_width is None:
                raise ValueError(
                    f"rec width {shape[3]} exceeds largest dynamic gear "
                    f"{REC_DYNAMIC_WIDTH_GEARS[-1]}"
                )
            om_path = self.rec_dynamic_om
            session_type = DynamicDimsOmSession
            padded_batch = REC_DYNAMIC_BATCH
            mode = "dynamic_width_fixed_batch_padding"
        if not om_path.is_file():
            raise FileNotFoundError(
                f"missing exact-shape {self.stage} OM for {shape}: {om_path}"
            )

        inference_inputs = inputs
        inference_templates = baseline["outputs"]
        if padded_batch is not None and self.stage == "cls":
            padded_input = np.zeros(
                (padded_batch, *inputs[0].shape[1:]), dtype=inputs[0].dtype
            )
            padded_input[: shape[0]] = inputs[0]
            inference_inputs = [padded_input]
            inference_templates = [
                np.empty((padded_batch, *output.shape[1:]), dtype=output.dtype)
                for output in baseline["outputs"]
            ]
        elif padded_batch is not None and self.stage == "rec":
            padded_input = np.zeros(
                (padded_batch, shape[1], shape[2], padded_width),
                dtype=inputs[0].dtype,
            )
            padded_input[: shape[0], :, :, : shape[3]] = inputs[0]
            inference_inputs = [padded_input]
            input_name = next(iter((feed or {}).keys()))
            output_names = args[0] if args else None
            inference_templates = self.metadata_session.run(
                output_names, {input_name: padded_input}
            )

        total_started = time.perf_counter()
        session = session_type(om_path, self.device_id, manage_runtime=False)
        try:
            outputs, execute_seconds = session.infer(
                inference_inputs, inference_templates
            )
        finally:
            session.close()
        if padded_batch is not None:
            outputs = [
                output[tuple(slice(0, size) for size in baseline_output.shape)]
                for output, baseline_output in zip(outputs, baseline["outputs"])
            ]
        self.records.append({
            "shape": shape,
            "om": str(om_path),
            "mode": mode,
            "padded_batch": padded_batch,
            "padded_width": padded_width,
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
    parser.add_argument("--rec-dynamic-om", type=Path)
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
            originals[stage], stage, captures[stage].calls, args.om_root,
            args.device_id, args.rec_dynamic_om
        )
        for stage in STAGE_ATTRS if captures[stage].calls
    }
    for stage, wrapper in wrappers.items():
        wrapper.session = replays.get(stage, originals[stage])
    acl_initialized = False
    try:
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: ret={ret}")
        acl_initialized = True
        om_started = time.perf_counter()
        om_result = ocr(frame)
        om_pipeline_seconds = time.perf_counter() - om_started
    finally:
        for stage, wrapper in wrappers.items():
            wrapper.session = originals[stage]
        if acl_initialized:
            try:
                acl.rt.reset_device(args.device_id)
            finally:
                acl.finalize()

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
