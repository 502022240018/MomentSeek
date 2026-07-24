#!/usr/bin/env python3
"""Generate a stage-by-stage CPU ONNX versus Ascend ACL OCR diagnostic report."""
from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.indexing.ocr import _load_ocr
from app.indexing.ocr_acl import RapidOCRAclBackend


STAGE_ATTRS = {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}
MODES = {
    "cpu": set(),
    "det_om": {"det"},
    "cls_om": {"cls"},
    "rec_om": {"rec"},
    "all_om": {"det", "cls", "rec"},
}


class CaptureSession:
    def __init__(self, session):
        self.session = session
        self.calls: list[dict[str, Any]] = []

    def run(self, *args, **kwargs):
        feed = args[1] if len(args) >= 2 else kwargs.get("input_feed", {})
        outputs = self.session.run(*args, **kwargs)
        self.calls.append({
            "input_shapes": {
                name: list(np.asarray(value).shape) for name, value in (feed or {}).items()
            },
            "outputs": [np.asarray(value).copy() for value in outputs],
        })
        return outputs

    def __getattr__(self, name):
        return getattr(self.session, name)


def _decode_frame(path: Path, timestamp: float, decode_height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"无法解码 {timestamp:.3f}s: {path}")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    if decode_height > 0 and height > decode_height:
        output_width = max(2, int(round(width * decode_height / height / 2) * 2))
        frame = cv2.resize(frame, (output_width, decode_height), interpolation=cv2.INTER_LINEAR)
    return frame


def _payload(result) -> dict[str, Any]:
    texts = [str(value) for value in (getattr(result, "txts", None) or [])]
    scores = [float(value) for value in (getattr(result, "scores", None) or [])]
    raw_boxes = getattr(result, "boxes", None)
    boxes = np.asarray(raw_boxes, dtype=np.float32) if raw_boxes is not None else np.empty((0, 4, 2), np.float32)
    return {
        "texts": texts,
        "scores": scores,
        "boxes": boxes.tolist(),
    }


def _numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": reference.shape == candidate.shape,
    }
    if reference.shape != candidate.shape:
        return result
    left = reference.astype(np.float64, copy=False).reshape(-1)
    right = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(left - right)
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    result.update({
        "max_abs_error": float(delta.max(initial=0)),
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "p99_abs_error": float(np.percentile(delta, 99)) if delta.size else 0.0,
        "cosine_similarity": float(np.dot(left, right) / norm) if norm else 1.0,
    })
    return result


def _capture_metrics(reference: dict[str, CaptureSession], candidate: dict[str, CaptureSession]):
    report = {}
    for stage in STAGE_ATTRS:
        left_calls = reference[stage].calls
        right_calls = candidate[stage].calls
        calls = []
        for index in range(max(len(left_calls), len(right_calls))):
            if index >= len(left_calls) or index >= len(right_calls):
                calls.append({"call_missing": True})
                continue
            left = left_calls[index]
            right = right_calls[index]
            outputs = [
                _numeric_metrics(a, b)
                for a, b in zip(left["outputs"], right["outputs"])
            ]
            calls.append({
                "reference_input_shapes": left["input_shapes"],
                "candidate_input_shapes": right["input_shapes"],
                "outputs": outputs,
            })
        report[stage] = {
            "reference_calls": len(left_calls),
            "candidate_calls": len(right_calls),
            "calls": calls,
        }
    return report


def _text_diff(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    left = Counter(reference)
    right = Counter(candidate)
    return {
        "exact_order_match": reference == candidate,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "missing_from_candidate": list((left - right).elements()),
        "added_by_candidate": list((right - left).elements()),
    }


def _save_visuals(
    frame: np.ndarray,
    payload: dict[str, Any],
    output_dir: Path,
    case_slug: str,
    mode: str,
) -> dict[str, Any]:
    mode_dir = output_dir / "assets" / case_slug / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    overlay = frame.copy()
    items = []
    height, width = frame.shape[:2]
    for index, text_value in enumerate(payload["texts"]):
        if index >= len(payload["boxes"]):
            break
        box = np.asarray(payload["boxes"][index], dtype=np.float32)
        if box.shape != (4, 2):
            continue
        points = np.rint(box).astype(np.int32)
        cv2.polylines(overlay, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
        anchor = tuple(points[np.argmin(points[:, 1])])
        cv2.putText(
            overlay, str(index), anchor, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2, cv2.LINE_AA,
        )
        x0 = max(0, int(np.floor(box[:, 0].min())))
        y0 = max(0, int(np.floor(box[:, 1].min())))
        x1 = min(width, int(np.ceil(box[:, 0].max())) + 1)
        y1 = min(height, int(np.ceil(box[:, 1].max())) + 1)
        crop_rel = None
        if x1 > x0 and y1 > y0:
            crop_path = mode_dir / f"crop-{index:03d}.jpg"
            cv2.imwrite(str(crop_path), frame[y0:y1, x0:x1])
            crop_rel = crop_path.relative_to(output_dir).as_posix()
        items.append({
            "index": index,
            "text": text_value,
            "score": payload["scores"][index] if index < len(payload["scores"]) else None,
            "box": box.tolist(),
            "crop": crop_rel,
        })
    overlay_path = mode_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "overlay": overlay_path.relative_to(output_dir).as_posix(),
        "items": items,
    }


def _run_mode(ocr, cpu_sessions, acl_sessions, frame, om_stages):
    wrappers = {stage: getattr(ocr, attr).session for stage, attr in STAGE_ATTRS.items()}
    captures = {}
    for stage, wrapper in wrappers.items():
        base = acl_sessions[stage] if stage in om_stages else cpu_sessions[stage]
        captures[stage] = CaptureSession(base)
        wrapper.session = captures[stage]
    started = time.perf_counter()
    try:
        result = ocr(frame)
    finally:
        for stage, wrapper in wrappers.items():
            wrapper.session = cpu_sessions[stage]
    return result, captures, time.perf_counter() - started


def _html_report(report: dict[str, Any]) -> str:
    style = """
body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;color:#222}
.case{border-top:3px solid #333;margin-top:34px;padding-top:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.panel{border:1px solid #bbb;border-radius:8px;padding:10px;background:#fafafa}.panel img.overlay{width:100%;height:auto}.item{display:flex;gap:8px;align-items:center;margin:6px 0;padding:4px;background:white}.item img{width:120px;max-height:62px;object-fit:contain;background:#ddd}.bad{color:#b00020;font-weight:bold}.ok{color:#087f23;font-weight:bold}code{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}
"""
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>MomentSeek OCR ACL diagnostic</title><style>{style}</style>",
        "<h1>MomentSeek OCR ACL stage diagnostic</h1>",
        f"<p>Video: <code>{html.escape(report['video'])}</code></p>",
        "<p>Modes: CPU; only Det OM; only Cls OM; only Rec OM; all OM.</p>",
    ]
    for case in report["cases"]:
        parts.append(f"<section class='case'><h2>{case['timestamp']:.3f}s</h2><div class='grid'>")
        for mode in MODES:
            value = case["modes"][mode]
            if value.get("error"):
                parts.append(
                    f"<div class='panel'><h3>{html.escape(mode)} "
                    f"<span class='bad'>ERROR</span></h3><pre>{html.escape(value['error'])}</pre></div>"
                )
                continue
            diff = value.get("text_diff")
            css = "ok" if mode == "cpu" or (diff and diff["exact_order_match"]) else "bad"
            label = "baseline" if mode == "cpu" else ("exact" if diff["exact_order_match"] else "different")
            parts.append(
                f"<div class='panel'><h3>{html.escape(mode)} "
                f"<span class='{css}'>{label}</span></h3>"
                f"<p>{value['elapsed_seconds']:.4f}s; boxes={len(value['payload']['texts'])}</p>"
                f"<img class='overlay' src='{html.escape(value['visuals']['overlay'])}'>"
            )
            if diff and not diff["exact_order_match"]:
                parts.append(
                    "<p class='bad'>Missing: " + html.escape(str(diff["missing_from_candidate"]))
                    + "<br>Added: " + html.escape(str(diff["added_by_candidate"])) + "</p>"
                )
            for item in value["visuals"]["items"]:
                image_tag = f"<img src='{html.escape(item['crop'])}'>" if item["crop"] else ""
                score = "" if item["score"] is None else f"{item['score']:.3f}"
                parts.append(
                    f"<div class='item'>{image_tag}<div>#{item['index']} "
                    f"<b>{html.escape(item['text'])}</b><br>{score}</div></div>"
                )
            parts.append("</div>")
        parts.append("</div></section>")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timestamps", type=float, nargs="+", required=True)
    parser.add_argument("--decode-height", type=int, default=720)
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--om-root", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_ocr, cpu_providers = _load_ocr(
        "cpu", 0, args.model_root, npu_self_test=False
    )
    acl_backend = RapidOCRAclBackend(
        device_id=args.device_id,
        model_root=args.model_root,
        om_root=args.om_root,
        ocr_version="PP-OCRv6",
        det_lang="ch",
        rec_lang="ch",
        model_type="small",
        npu_self_test=False,
    )
    cpu_sessions = {
        stage: getattr(cpu_ocr, attr).session.session
        for stage, attr in STAGE_ATTRS.items()
    }
    acl_sessions = {
        stage: getattr(acl_backend.ocr, attr).session.session
        for stage, attr in STAGE_ATTRS.items()
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "video": str(args.video),
        "decode_height": args.decode_height,
        "cpu_providers": cpu_providers,
        "acl_providers": acl_backend.providers,
        "cases": [],
    }
    try:
        for timestamp in args.timestamps:
            frame = _decode_frame(args.video, timestamp, args.decode_height)
            case_slug = f"t-{timestamp:010.3f}".replace(".", "_")
            frame_path = args.output_dir / "assets" / case_slug / "frame.jpg"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame)
            case = {
                "timestamp": timestamp,
                "frame_shape": list(frame.shape),
                "frame": frame_path.relative_to(args.output_dir).as_posix(),
                "modes": {},
            }
            baseline_captures = None
            baseline_payload = None
            for mode, stages in MODES.items():
                try:
                    result, captures, elapsed = _run_mode(
                        cpu_ocr, cpu_sessions, acl_sessions, frame, stages
                    )
                except Exception as exc:
                    if mode == "cpu":
                        raise
                    case["modes"][mode] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
                    continue
                payload = _payload(result)
                value = {
                    "elapsed_seconds": elapsed,
                    "payload": payload,
                    "visuals": _save_visuals(
                        frame, payload, args.output_dir, case_slug, mode
                    ),
                }
                if mode == "cpu":
                    baseline_captures = captures
                    baseline_payload = payload
                else:
                    value["text_diff"] = _text_diff(
                        baseline_payload["texts"], payload["texts"]
                    )
                    value["stage_numeric_comparison"] = _capture_metrics(
                        baseline_captures, captures
                    )
                case["modes"][mode] = value
            report["cases"].append(case)
            print(f"case={timestamp:.3f}s complete", flush=True)
    finally:
        acl_backend.close()

    exact_counts = {
        mode: sum(
            1 for case in report["cases"]
            if mode != "cpu"
            and case["modes"][mode].get("text_diff", {}).get("exact_order_match")
        )
        for mode in MODES if mode != "cpu"
    }
    report["summary"] = {
        "case_count": len(report["cases"]),
        "exact_text_case_counts": exact_counts,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.html").write_text(
        _html_report(report), encoding="utf-8"
    )
    lines = [
        "MomentSeek OCR ACL stage diagnostic",
        f"video={args.video}",
        f"cases={len(report['cases'])}",
    ]
    for mode, count in exact_counts.items():
        lines.append(f"{mode}: exact_text_cases={count}/{len(report['cases'])}")
    for case in report["cases"]:
        differences = [
            mode for mode in MODES if mode != "cpu"
            and not case["modes"][mode].get("text_diff", {}).get("exact_order_match")
        ]
        lines.append(f"{case['timestamp']:.3f}s different_modes={','.join(differences) or 'none'}")
        for mode in differences:
            if case["modes"][mode].get("error"):
                lines.append(f"  {mode} error={case['modes'][mode]['error']}")
                continue
            diff = case["modes"][mode]["text_diff"]
            lines.append(f"  {mode} missing={diff['missing_from_candidate']}")
            lines.append(f"  {mode} added={diff['added_by_candidate']}")
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"OCR_ACL_DIAGNOSTIC_OK=1 output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
