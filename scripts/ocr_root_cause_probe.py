#!/usr/bin/env python3
"""Run one isolated probe from the OCR/OpenBLAS root-cause matrix.

The shell suite starts a fresh container for every case.  This program keeps
the intentionally combined cases in one Python process so global CANN, ACL,
PyTorch, and BLAS state is part of the experiment rather than hidden by a
subprocess boundary.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CASES = ("semantic_only", "ocr_only", "ocr_semantic", "face_ocr_semantic")
THREAD_ENV_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_DEFAULT_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


def _read_status(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[key] = value.strip()
    except OSError:
        pass
    return result


def _process_tree(root_pid: int) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        status = _read_status(pid)
        if not status:
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except OSError:
            command = ""
        records[pid] = {
            "ppid": int(status.get("PPid", "0") or 0),
            "threads": int(status.get("Threads", "0") or 0),
            "rss_kb": int((status.get("VmRSS", "0").split() or ["0"])[0]),
            "command": command,
        }

    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, value in records.items():
            if value["ppid"] in selected and pid not in selected:
                selected.add(pid)
                changed = True
    values = [records[pid] for pid in selected if pid in records]
    commands = Counter()
    for value in values:
        command = value["command"]
        if "forkserver" in command:
            commands["python:forkserver"] += 1
        elif "resource_tracker" in command:
            commands["python:resource_tracker"] += 1
        elif "ffmpeg" in command:
            commands["ffmpeg"] += 1
        elif "ocr_root_cause_probe" in command:
            commands["probe"] += 1
        elif command:
            commands[command[:120]] += 1
        else:
            commands["unknown"] += 1
    return {
        "processes": len(values),
        "threads": sum(value["threads"] for value in values),
        "rss_mb": round(sum(value["rss_kb"] for value in values) / 1024, 3),
        "commands": dict(commands.most_common(12)),
    }


def _threadpools() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


class Recorder:
    def __init__(self, output: Path, case: str, monitor_interval: float):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.report_path = output / "report.json"
        self.monitor_path = output / "process-monitor.jsonl"
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "case": case,
            "pid": self.pid,
            "started_at_epoch": time.time(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
            "phases": [],
            "success": False,
        }
        self.monitor_interval = monitor_interval
        self._write()
        self.monitor = threading.Thread(target=self._monitor, name="probe-resource-monitor", daemon=True)
        self.monitor.start()

    def _write(self) -> None:
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.report_path)

    def snapshot(self, label: str) -> dict[str, Any]:
        status = _read_status(self.pid)
        value = {
            "epoch": time.time(),
            "label": label,
            "self_threads": int(status.get("Threads", "0") or 0),
            "self_rss_mb": round(int((status.get("VmRSS", "0").split() or ["0"])[0]) / 1024, 3),
            "wait_channel": Path(f"/proc/{self.pid}/wchan").read_text(errors="replace").strip()
            if Path(f"/proc/{self.pid}/wchan").exists() else None,
            "tree": _process_tree(self.pid),
        }
        return value

    def phase_start(self, name: str) -> float:
        print(f"PHASE_START={name}", flush=True)
        self.report["phases"].append({"name": name, "status": "running", "start": self.snapshot(f"{name}:start")})
        self._write()
        return time.perf_counter()

    def phase_end(self, name: str, started: float, **metrics: Any) -> None:
        phase = next(item for item in reversed(self.report["phases"]) if item["name"] == name)
        phase.update({
            "status": "completed",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "end": self.snapshot(f"{name}:end"),
            **metrics,
        })
        print(f"PHASE_END={name} seconds={phase['elapsed_seconds']}", flush=True)
        self._write()

    def fail(self, exc: BaseException) -> None:
        for phase in reversed(self.report["phases"]):
            if phase.get("status") == "running":
                phase["status"] = "failed"
                phase["end"] = self.snapshot(f"{phase['name']}:failed")
                break
        self.report["error"] = f"{type(exc).__name__}: {exc}"
        self.report["traceback"] = traceback.format_exc()
        self.report["final_snapshot"] = self.snapshot("failed")
        self._write()

    def finish(self) -> None:
        self.report["success"] = True
        self.report["finished_at_epoch"] = time.time()
        self.report["threadpools"] = _threadpools()
        self.report["final_snapshot"] = self.snapshot("completed")
        self._write()

    def close(self) -> None:
        self.stop_event.set()
        self.monitor.join(timeout=max(2.0, self.monitor_interval * 2))

    def _monitor(self) -> None:
        while not self.stop_event.is_set():
            try:
                value = self.snapshot("periodic")
                with self.monitor_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self.stop_event.wait(self.monitor_interval)


def _synthetic_texts(count: int) -> list[str]:
    templates = (
        "电视剧字幕样本 第{index}句 人物正在讨论今天发生的事情",
        "MomentSeek OCR semantic diagnostic sample number {index}",
        "画面文字包含地点 时间 人物和事件编号{index}",
        "多语言字幕 prueba de reconocimiento numero {index}",
    )
    return [templates[index % len(templates)].format(index=index) for index in range(count)]


def _expanded_texts(values: list[str], count: int) -> list[str]:
    cleaned = [value.strip() for value in values if value.strip()]
    if not cleaned:
        return _synthetic_texts(count)
    return [f"{cleaned[index % len(cleaned)]} #{index}" for index in range(count)]


def _decode_first_frame(path: Path, decode_height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"无法解码视频首帧: {path}")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    if decode_height > 0 and height > decode_height:
        target_width = max(2, int(round(width * decode_height / height / 2) * 2))
        frame = cv2.resize(frame, (target_width, decode_height), interpolation=cv2.INTER_LINEAR)
    return frame


def _run_face(args: argparse.Namespace, recorder: Recorder):
    from app.indexing.faces import FaceEncoder

    started = recorder.phase_start("face_cann_load_and_infer")
    encoder = FaceEncoder("buffalo_l", "cann", args.device_id, str(args.face_root))
    frame = _decode_first_frame(args.video, args.decode_height)
    detections = encoder.detect(frame)
    recorder.phase_end(
        "face_cann_load_and_infer", started,
        detections=len(detections), provider=encoder.provider,
    )
    return encoder


def _run_ocr(args: argparse.Namespace, recorder: Recorder):
    from app.indexing.ocr_acl import RapidOCRAclBackend
    from app.media import read_frames

    started = recorder.phase_start("ocr_acl_load")
    backend = RapidOCRAclBackend(
        device_id=args.device_id,
        model_root=args.ocr_model_root,
        om_root=args.ocr_om_root,
        ocr_version="PP-OCRv6",
        det_lang="ch",
        rec_lang="ch",
        model_type="small",
        npu_self_test=False,
    )
    recorder.phase_end("ocr_acl_load", started, providers=backend.providers)

    texts: list[str] = []
    decoded = 0
    hit_frames = 0
    started = recorder.phase_start("ocr_acl_frames")
    iterator = read_frames(args.video, args.sample_fps, out_height=args.decode_height, prefer_ffmpeg=True)
    try:
        for timestamp, frame in iterator:
            output = backend(frame)
            decoded += 1
            frame_texts = [str(value).strip() for value in (getattr(output, "txts", None) or []) if str(value).strip()]
            if frame_texts:
                hit_frames += 1
                texts.extend(frame_texts)
            if decoded % args.progress_every == 0:
                print(
                    f"OCR_PROGRESS frames={decoded}/{args.max_frames} timestamp={timestamp:.3f} "
                    f"texts={len(texts)}",
                    flush=True,
                )
            if decoded >= args.max_frames:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    recorder.phase_end(
        "ocr_acl_frames", started,
        decoded_frames=decoded, hit_frames=hit_frames, raw_texts=len(texts),
    )
    return backend, texts


def _run_semantic(args: argparse.Namespace, recorder: Recorder, texts: list[str]) -> np.ndarray:
    from app.indexing.text_semantic import TextEmbeddingEncoder

    values = _expanded_texts(texts, args.semantic_text_count)
    started = recorder.phase_start("semantic_model_load")
    encoder = TextEmbeddingEncoder(
        args.semantic_model,
        args.semantic_model_root,
        device="cpu",
        local_files_only=True,
    )
    recorder.phase_end("semantic_model_load", started, text_count=len(values))
    started = recorder.phase_start("semantic_encode")
    embeddings = encoder.encode(values, batch_size=args.semantic_batch_size)
    recorder.phase_end(
        "semantic_encode", started,
        text_count=len(values), embedding_shape=list(embeddings.shape),
    )
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ocr-model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument("--ocr-om-root", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, default=Path("/app/models/insightface"))
    parser.add_argument("--semantic-model-root", type=Path, default=Path("/app/models/text-embeddings"))
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--semantic-text-count", type=int, default=2000)
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--decode-height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--stack-dump-seconds", type=float, default=120.0)
    args = parser.parse_args()

    if args.semantic_text_count <= 0 or args.max_frames <= 0 or args.sample_fps <= 0:
        parser.error("semantic-text-count, max-frames and sample-fps must be positive")
    if not args.video.is_file():
        parser.error(f"video is missing: {args.video}")

    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(args.stack_dump_seconds, repeat=True)
    recorder = Recorder(args.output_dir, args.case, args.monitor_interval)
    acl_backend = None
    face_encoder = None
    try:
        if args.case == "semantic_only":
            _run_semantic(args, recorder, _synthetic_texts(args.semantic_text_count))
        else:
            if args.case == "face_ocr_semantic":
                face_encoder = _run_face(args, recorder)
            acl_backend, texts = _run_ocr(args, recorder)
            if args.case in {"ocr_semantic", "face_ocr_semantic"}:
                _run_semantic(args, recorder, texts)
        recorder.finish()
        print(f"ROOT_CAUSE_CASE_OK=1 case={args.case}", flush=True)
        return 0
    except BaseException as exc:
        recorder.fail(exc)
        traceback.print_exc()
        print(f"ROOT_CAUSE_CASE_OK=0 case={args.case} error={type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        # Keep Face alive for the whole combined case, then let process exit own
        # its CANN EP teardown. ACL has an explicit deterministic close path.
        _ = face_encoder
        if acl_backend is not None:
            try:
                acl_backend.close()
            except Exception:
                traceback.print_exc()
        faulthandler.cancel_dump_traceback_later()
        recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
