#!/usr/bin/env python3
"""Profile the exact tensor shapes RapidOCR feeds into PP-OCR models.

This is a read-only CPU baseline tool. It samples real uploaded videos and wraps
the already-created ONNX Runtime sessions, so recorded shapes include RapidOCR's
actual resize, crop, orientation and recognition preprocessing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.indexing.ocr import _load_ocr
from app.media import probe_video


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class RecordingSession:
    def __init__(self, session, stage: str, records: list[dict]):
        self._session = session
        self._stage = stage
        self._records = records

    def run(self, *args, **kwargs):
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        for name, value in (feed or {}).items():
            array = np.asarray(value)
            self._records.append({
                "stage": self._stage,
                "name": str(name),
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
            })
        return self._session.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _wrap_sessions(ocr, records: list[dict]) -> None:
    for stage, attr in {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}.items():
        owner = getattr(ocr, attr)
        wrapper = getattr(owner, "session")
        wrapper.session = RecordingSession(wrapper.session, stage, records)


def _sample_frame(path: Path, timestamp: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _summary(records: list[dict]) -> dict:
    result = {}
    for stage in ("det", "cls", "rec"):
        shapes = [tuple(item["shape"]) for item in records if item["stage"] == stage]
        counts = Counter(shapes)
        stage_result = {
            "calls": len(shapes),
            "unique_shapes": [
                {"shape": list(shape), "count": count}
                for shape, count in counts.most_common()
            ],
        }
        if shapes and all(len(shape) == 4 for shape in shapes):
            widths = np.asarray([shape[3] for shape in shapes], dtype=np.int32)
            heights = np.asarray([shape[2] for shape in shapes], dtype=np.int32)
            stage_result["height_percentiles"] = {
                key: int(np.percentile(heights, percentile))
                for key, percentile in (("min", 0), ("p50", 50), ("p90", 90), ("p95", 95), ("max", 100))
            }
            stage_result["width_percentiles"] = {
                key: int(np.percentile(widths, percentile))
                for key, percentile in (("min", 0), ("p50", 50), ("p90", 90), ("p95", 95), ("max", 100))
            }
        result[stage] = stage_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=Path("/app/runtime/uploads"))
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--output", type=Path, default=Path("/app/runtime/ocr-shape-profile.json"))
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--max-videos", type=int, default=12)
    args = parser.parse_args()

    videos = sorted(
        path for path in args.video_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )[: max(1, args.max_videos)]
    if not videos:
        raise SystemExit(f"no videos found under {args.video_root}")

    ocr, providers = _load_ocr("cpu", 0, args.model_root, npu_self_test=False)
    records: list[dict] = []
    _wrap_sessions(ocr, records)
    sampled = []
    for video in videos:
        info = probe_video(video)
        frame_count = max(1, args.frames_per_video)
        timestamps = np.linspace(0, max(0.0, info.duration - 0.1), frame_count)
        successful = 0
        errors = []
        for timestamp in timestamps:
            frame = _sample_frame(video, float(timestamp))
            if frame is None:
                continue
            try:
                ocr(frame)
                successful += 1
            except Exception as exc:
                if len(errors) < 3:
                    errors.append(str(exc))
        sampled.append({
            "path": str(video),
            "duration": round(info.duration, 3),
            "source_size": [info.width, info.height],
            "sampled_frames": successful,
            "errors": errors,
        })

    report = {
        "providers": providers,
        "videos": sampled,
        "tensor_shapes": _summary(records),
        "raw_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "video_count": len(sampled),
        "sampled_frames": sum(item["sampled_frames"] for item in sampled),
        "tensor_shapes": report["tensor_shapes"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
