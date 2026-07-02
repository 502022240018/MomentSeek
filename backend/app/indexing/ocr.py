from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.indexing.common import atomic_save_json
from app.indexing.text_semantic import build_text_semantic_index, resolve_text_embedding_device
from app.media import read_frames, save_thumbnail


def _rapidocr_params(
    device: str,
    device_id: int,
    model_root: str | Path,
    ocr_version: str = "PP-OCRv4",
    det_lang: str = "en",
    rec_lang: str = "en",
    model_type: str = "mobile",
) -> dict[str, Any]:
    from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion

    version_map = {item.value.casefold(): item for item in OCRVersion}
    det_lang_map = {item.value.casefold(): item for item in LangDet}
    rec_lang_map = {item.value.casefold(): item for item in LangRec}
    model_type_map = {item.value.casefold(): item for item in ModelType}
    version_value = version_map.get(ocr_version.casefold())
    det_lang_value = det_lang_map.get(det_lang.casefold())
    rec_lang_value = rec_lang_map.get(rec_lang.casefold())
    model_type_value = model_type_map.get(model_type.casefold())
    if version_value is None:
        raise ValueError(f"不支持的 OCR version: {ocr_version}")
    if det_lang_value is None:
        raise ValueError(f"不支持的 OCR det_lang: {det_lang}")
    if rec_lang_value is None:
        raise ValueError(f"不支持的 OCR rec_lang: {rec_lang}")
    if model_type_value is None:
        raise ValueError(f"不支持的 OCR model_type: {model_type}")

    params: dict[str, Any] = {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Det.ocr_version": version_value,
        "Det.lang_type": det_lang_value,
        "Det.model_type": model_type_value,
        "Rec.ocr_version": version_value,
        "Rec.lang_type": rec_lang_value,
        "Rec.model_type": model_type_value,
        "Global.model_root_dir": str(model_root),
        "Global.log_level": "warning",
    }
    if device == "npu":
        params.update({
            "EngineConfig.onnxruntime.use_cann": True,
            "EngineConfig.onnxruntime.cann_ep_cfg.device_id": int(device_id),
        })
    return params


def _session_providers(ocr) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for name, attr in {"det": "text_det", "cls": "text_cls", "rec": "text_rec"}.items():
        session = getattr(getattr(ocr, attr), "session", None)
        ort_session = getattr(session, "session", None)
        providers[name] = list(ort_session.get_providers()) if ort_session is not None else []
    return providers


def _run_npu_self_test(ocr) -> None:
    import cv2

    image = np.full((240, 720, 3), 255, dtype=np.uint8)
    cv2.putText(image, "QATAR WORLD CUP", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    output = ocr(image, text_score=0.1, box_thresh=0.1)
    text = " ".join(getattr(output, "txts", None) or [])
    if "QATAR" not in text.upper() and "WORLD" not in text.upper():
        raise RuntimeError(f"OCR NPU 自检失败：CANN Provider 已加载，但合成文字未被正确识别，output={text!r}")


def _load_ocr(
    device: str,
    device_id: int,
    model_root: str | Path,
    ocr_version: str = "PP-OCRv4",
    det_lang: str = "en",
    rec_lang: str = "en",
    model_type: str = "mobile",
    npu_self_test: bool = True,
):
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR 依赖 rapidocr 未安装；请安装 rapidocr 后重试") from exc

    Path(model_root).mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR(params=_rapidocr_params(device, device_id, model_root, ocr_version, det_lang, rec_lang, model_type))
    providers = _session_providers(ocr)
    if device == "npu":
        missing = [name for name, values in providers.items() if "CANNExecutionProvider" not in values]
        if missing:
            raise RuntimeError(
                "OCR 已配置为 NPU，但 RapidOCR 未使用 CANNExecutionProvider；"
                f"missing={missing}, providers={providers}"
            )
        if npu_self_test:
            _run_npu_self_test(ocr)
    return ocr, providers


def _ocr_items(output, min_confidence: float) -> list[dict]:
    texts = getattr(output, "txts", None) or []
    scores = getattr(output, "scores", None) or []
    boxes = getattr(output, "boxes", None)
    items = []
    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) else 0.0
        clean_text = str(text).strip()
        if not clean_text or score < min_confidence:
            continue
        box = boxes[index].tolist() if boxes is not None and index < len(boxes) else None
        items.append({"text": clean_text, "score": round(score, 4), "box": box})
    return items


def build_ocr_index(
    video_path: str | Path,
    output_path: str | Path,
    thumbnail_dir: str | Path,
    working_dir: str | Path,
    sample_fps: float = 1.0,
    decode_height: int = 720,
    min_confidence: float = 0.5,
    device: str = "npu",
    device_id: int = 0,
    model_root: str | Path = "models/rapidocr",
    ocr_version: str = "PP-OCRv4",
    det_lang: str = "en",
    rec_lang: str = "en",
    model_type: str = "mobile",
    npu_self_test: bool = True,
    prefer_ffmpeg: bool = True,
    semantic_enabled: bool = True,
    semantic_output_path: str | Path | None = None,
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    semantic_device: str = "cpu",
    semantic_model_dir: str | Path = "models/text-embeddings",
    semantic_batch_size: int = 32,
    semantic_local_files_only: bool = True,
) -> dict:
    if sample_fps <= 0:
        raise ValueError("ocr_sample_fps 必须大于 0")
    started = time.perf_counter()
    thumbnail_dir = Path(thumbnail_dir)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    ocr, providers = _load_ocr(
        device,
        device_id,
        model_root,
        ocr_version=ocr_version,
        det_lang=det_lang,
        rec_lang=rec_lang,
        model_type=model_type,
        npu_self_test=npu_self_test,
    )

    chunks: list[dict] = []
    interval = 1.0 / sample_fps
    ocr_elapsed = 0.0
    decoded_frames = 0
    hit_frames = 0
    for frame_index, (timestamp, frame) in enumerate(
        read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg)
    ):
        decoded_frames += 1
        frame_start = time.perf_counter()
        output = ocr(frame)
        ocr_elapsed += time.perf_counter() - frame_start
        items = _ocr_items(output, min_confidence)
        if not items:
            continue
        hit_frames += 1
        thumbnail_name = f"ocr_{frame_index:06d}.jpg"
        save_thumbnail(frame, thumbnail_dir / thumbnail_name)
        text = " ".join(item["text"] for item in items)
        chunks.append({
            "start_time": round(float(timestamp), 3),
            "end_time": round(float(timestamp + interval), 3),
            "text": text,
            "items": items,
            "thumbnail": thumbnail_name,
            "score": round(max(item["score"] for item in items), 4),
        })

    payload = {
        "schema_version": 1,
        "engine": "rapidocr",
        "device": device,
        "providers": providers,
        "ocr_version": ocr_version,
        "det_lang": det_lang,
        "rec_lang": rec_lang,
        "model_type": model_type,
        "sample_fps": sample_fps,
        "decode_height": decode_height,
        "min_confidence": min_confidence,
        "chunks": chunks,
    }
    atomic_save_json(output_path, payload)

    result = {
        "engine": "rapidocr",
        "device": device,
        "providers": providers,
        "ocr_version": ocr_version,
        "det_lang": det_lang,
        "rec_lang": rec_lang,
        "model_type": model_type,
        "frames": decoded_frames,
        "hit_frames": hit_frames,
        "chunks": len(chunks),
        "ocr_elapsed_seconds": round(ocr_elapsed, 3),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if semantic_enabled and semantic_output_path is not None:
        try:
            resolved_device = resolve_text_embedding_device(semantic_device)
            semantic_result = build_text_semantic_index(
                chunks=chunks,
                output_path=semantic_output_path,
                model_name=semantic_model,
                model_dir=semantic_model_dir,
                device=resolved_device,
                batch_size=semantic_batch_size,
                local_files_only=semantic_local_files_only,
            )
            result.update({f"semantic_{key}": value for key, value in semantic_result.items()})
        except Exception as exc:
            Path(semantic_output_path).unlink(missing_ok=True)
            result["semantic_status"] = "unavailable"
            result["semantic_error"] = str(exc)
    return result
