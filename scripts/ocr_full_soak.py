#!/usr/bin/env python3
"""Run the production OCR index path on a complete video in an isolated container."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/app/models/rapidocr"))
    parser.add_argument(
        "--om-root",
        type=Path,
        default=Path("/app/models/rapidocr/ascend/910b4-cann9-profile"),
    )
    parser.add_argument("--semantic-model-root", type=Path, default=Path("/app/models/text-embeddings"))
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--decode-height", type=int, default=720)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    if args.sample_fps <= 0 or args.decode_height <= 0 or args.semantic_batch_size <= 0:
        parser.error("sample-fps, decode-height and semantic-batch-size must be positive")
    if not args.video.is_file():
        parser.error(f"video is missing: {args.video}")

    from app.indexing.ocr import build_ocr_index
    from app.indexing.ocr_acl import RapidOCRAclBackend

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "success": False,
        "video": str(args.video),
        "video_size_bytes": args.video.stat().st_size,
        "sample_fps": args.sample_fps,
        "decode_height": args.decode_height,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OPENBLAS_DEFAULT_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    backend = None
    try:
        phase_started = time.perf_counter()
        backend = RapidOCRAclBackend(
            device_id=args.device_id,
            model_root=args.model_root,
            om_root=args.om_root,
            ocr_version="PP-OCRv6",
            det_lang="ch",
            rec_lang="ch",
            model_type="small",
            npu_self_test=True,
        )
        backend_load_elapsed = time.perf_counter() - phase_started
        print(f"SOAK_PHASE backend_load seconds={backend_load_elapsed:.3f}", flush=True)

        index_result = build_ocr_index(
            video_path=args.video,
            output_path=args.output_dir / "ocr.npz",
            working_dir=args.output_dir / "work",
            sample_fps=args.sample_fps,
            decode_height=args.decode_height,
            min_confidence=args.min_confidence,
            device="npu",
            device_id=args.device_id,
            model_root=args.model_root,
            npu_self_test=False,
            semantic_enabled=True,
            semantic_model=args.semantic_model,
            semantic_device="cpu",
            semantic_model_dir=args.semantic_model_root,
            semantic_batch_size=args.semantic_batch_size,
            semantic_local_files_only=True,
            engine="rapidocr_acl",
            acl_model_dir=args.om_root,
            backend=backend,
        )
        report.update(index_result)
        report.update({
            "success": True,
            "backend_load_elapsed_seconds": round(backend_load_elapsed, 3),
            "soak_total_elapsed_seconds": round(time.perf_counter() - started, 3),
            "output_size_bytes": (args.output_dir / "ocr.npz").stat().st_size,
        })
        print("SOAK_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        report.update({
            "error": f"{type(exc).__name__}: {exc}",
            "soak_total_elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        traceback.print_exc()
        return 1
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        close = getattr(backend, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
