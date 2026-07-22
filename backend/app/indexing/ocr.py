from __future__ import annotations

import re
import unicodedata
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.indexing.common import atomic_save_npz
from app.indexing.text_semantic import build_text_semantic_arrays, resolve_text_embedding_device
from app.media import read_frames


def _rapidocr_params(
    device: str,
    device_id: int,
    model_root: str | Path,
    ocr_version: str = "PP-OCRv6",
    det_lang: str = "ch",
    rec_lang: str = "ch",
    model_type: str = "small",
    ort_intra_op_threads: int = 8,
    ort_inter_op_threads: int = 1,
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

    if ort_intra_op_threads < 1 or ort_inter_op_threads < 1:
        raise ValueError("RapidOCR ONNX Runtime thread limits must be positive")

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
        "EngineConfig.onnxruntime.intra_op_num_threads": int(ort_intra_op_threads),
        "EngineConfig.onnxruntime.inter_op_num_threads": int(ort_inter_op_threads),
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


def _has_local_rapidocr_assets(model_root: str | Path) -> bool:
    root = Path(model_root).expanduser()
    if not root.is_dir():
        return False
    return any(
        path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in {".onnx", ".bin"}
        for path in root.rglob("*")
    )


def _load_ocr(
    device: str,
    device_id: int,
    model_root: str | Path,
    ocr_version: str = "PP-OCRv6",
    det_lang: str = "ch",
    rec_lang: str = "ch",
    model_type: str = "small",
    npu_self_test: bool = True,
    ort_intra_op_threads: int = 8,
    ort_inter_op_threads: int = 1,
):
    if not _has_local_rapidocr_assets(model_root):
        raise FileNotFoundError(f"本地 RapidOCR 模型缺失: {model_root}")

    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR 依赖 rapidocr 未安装；请安装 rapidocr 后重试") from exc

    ocr = RapidOCR(params=_rapidocr_params(
        device,
        device_id,
        model_root,
        ocr_version,
        det_lang,
        rec_lang,
        model_type,
        ort_intra_op_threads=ort_intra_op_threads,
        ort_inter_op_threads=ort_inter_op_threads,
    ))
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


class OCRBackend(Protocol):
    """Stable boundary between OCR indexing and a device-specific runtime."""

    engine: str
    device: str
    providers: dict[str, list[str]]

    def __call__(self, frame: np.ndarray): ...


class RapidOCRBackend:
    engine = "rapidocr"

    def __init__(
        self,
        *,
        device: str,
        device_id: int,
        model_root: str | Path,
        ocr_version: str,
        det_lang: str,
        rec_lang: str,
        model_type: str,
        npu_self_test: bool,
    ):
        self.device = device
        self.ocr, self.providers = _load_ocr(
            device,
            device_id,
            model_root,
            ocr_version=ocr_version,
            det_lang=det_lang,
            rec_lang=rec_lang,
            model_type=model_type,
            npu_self_test=npu_self_test,
        )

    def __call__(self, frame: np.ndarray):
        return self.ocr(frame)


def create_ocr_backend(
    engine: str,
    *,
    device: str,
    device_id: int,
    model_root: str | Path,
    ocr_version: str,
    det_lang: str,
    rec_lang: str,
    model_type: str,
    npu_self_test: bool,
    acl_model_dir: str | Path | None = None,
) -> OCRBackend:
    normalized = str(engine or "rapidocr").replace("-", "_").casefold()
    if normalized == "rapidocr_acl":
        if device != "npu":
            raise ValueError("rapidocr_acl 仅支持 device=npu")
        from app.indexing.ocr_acl import RapidOCRAclBackend

        om_root = Path(acl_model_dir) if acl_model_dir else Path(model_root) / "ascend"
        return RapidOCRAclBackend(
            device_id=device_id,
            model_root=model_root,
            om_root=om_root,
            ocr_version=ocr_version,
            det_lang=det_lang,
            rec_lang=rec_lang,
            model_type=model_type,
            npu_self_test=npu_self_test,
        )
    if normalized != "rapidocr":
        raise ValueError(f"尚未启用的 OCR backend: {engine}")
    return RapidOCRBackend(
        device=device,
        device_id=device_id,
        model_root=model_root,
        ocr_version=ocr_version,
        det_lang=det_lang,
        rec_lang=rec_lang,
        model_type=model_type,
        npu_self_test=npu_self_test,
    )


_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_REPEAT_CHAR_RE = re.compile(r"^(.)\1{3,}$", re.UNICODE)


def _clean_ocr_text(text: Any) -> str:
    """
    用于 OCR 入库前的轻量清洗。

    注意：
    - 这里不做激进纠错，避免误伤真实文本。
    - NFKC 可以把全角英数、全角符号归一化，利于检索和去重。
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_mostly_punctuation(text: str) -> bool:
    return bool(text) and bool(_PUNCT_RE.match(text))


def _has_alnum_or_cjk(text: str) -> bool:
    """
    判断文本里是否至少包含：
    - 中文/CJK
    - 英文字母
    - 数字

    纯符号、纯分隔线、孤立装饰字符一般不应进入 OCR 索引。
    """
    for ch in text:
        if ch.isalnum():
            return True
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF      # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF   # CJK Extension A
            or 0x3040 <= code <= 0x30FF   # Hiragana/Katakana
            or 0xAC00 <= code <= 0xD7AF   # Hangul
        ):
            return True
    return False


def _box_array(box: Any) -> np.ndarray | None:
    if box is None:
        return None
    values = np.asarray(box, dtype=np.float32)
    if values.shape != (4, 2):
        return None
    if not np.isfinite(values).all():
        return None
    return values


def _box_bounds(box: Any) -> tuple[float, float, float, float] | None:
    values = _box_array(box)
    if values is None:
        return None
    xs = values[:, 0]
    ys = values[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _box_area(box: Any) -> float:
    """
    使用多边形面积，比简单 width*height 更适合倾斜文字框。
    """
    values = _box_array(box)
    if values is None:
        return 0.0
    x = values[:, 0]
    y = values[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _box_area_ratio(box: Any, frame_shape: tuple[int, int] | None) -> float | None:
    if frame_shape is None:
        return None
    height, width = frame_shape
    frame_area = max(float(width * height), 1.0)
    return _box_area(box) / frame_area


def _is_basic_ocr_noise(text: str, score: float, compact: str) -> bool:
    return (
        not text
        or score <= 0
        or _is_mostly_punctuation(text)
        or not _has_alnum_or_cjk(text)
        or (len(compact) <= 1 and score < 0.85)
        or bool(_REPEAT_CHAR_RE.match(compact))
        or (compact.isdigit() and len(compact) <= 2 and score < 0.8)
    )


def _is_invalid_ocr_geometry(
    bounds: tuple[float, float, float, float],
    box: Any,
    frame_shape: tuple[int, int] | None,
    score: float,
) -> bool:
    left, top, right, bottom = bounds
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    if width < 2 or height < 2:
        return True
    aspect = width / max(height, 1.0)
    if aspect > 40 or aspect < 0.025:
        return True
    area_ratio = _box_area_ratio(box, frame_shape)
    return area_ratio is not None and area_ratio < 0.00001 and score < 0.9


def _is_low_quality_ocr_item(
    text: str,
    score: float,
    box: Any,
    frame_shape: tuple[int, int] | None,
) -> bool:
    """
    OCR item 级别过滤。

    设计原则：
    - 对低置信度短文本更严格；
    - 对中文单字不要一刀切，因为视频标题/按钮里可能确实有单字；
    - 几何过滤只过滤明显异常框，避免误伤小字幕。
    """
    compact = text.replace(" ", "")
    if _is_basic_ocr_noise(text, score, compact):
        return True
    bounds = _box_bounds(box)
    return bounds is not None and _is_invalid_ocr_geometry(bounds, box, frame_shape, score)


def _sort_ocr_items_reading_order(items: list[dict]) -> list[dict]:
    """
    按阅读顺序排序 OCR items。

    简单的 y/x 排序在同一行文字有轻微倾斜或检测框高度不同的时候容易错位。
    这里先按中心 y 聚类成行，再在行内按 left x 排序。
    """
    if len(items) <= 1:
        return items

    sortable: list[tuple[dict, tuple[float, float, float, float]]] = []
    fallback: list[dict] = []

    for item in items:
        bounds = _box_bounds(item.get("box"))
        if bounds is None:
            fallback.append(item)
        else:
            sortable.append((item, bounds))

    if not sortable:
        return items

    heights = [max(bottom - top, 1.0) for _, (_, top, _, bottom) in sortable]
    median_height = float(np.median(heights)) if heights else 16.0
    row_threshold = max(8.0, median_height * 0.6)

    sortable.sort(key=lambda pair: ((pair[1][1] + pair[1][3]) / 2.0, pair[1][0]))

    rows: list[list[tuple[dict, tuple[float, float, float, float]]]] = []
    row_centers: list[float] = []

    for item, bounds in sortable:
        left, top, right, bottom = bounds
        center_y = (top + bottom) / 2.0

        placed = False
        for row_index, row_center in enumerate(row_centers):
            if abs(center_y - row_center) <= row_threshold:
                rows[row_index].append((item, bounds))
                # 轻量更新该行中心
                row_centers[row_index] = (
                    row_centers[row_index] * (len(rows[row_index]) - 1) + center_y
                ) / len(rows[row_index])
                placed = True
                break

        if not placed:
            rows.append([(item, bounds)])
            row_centers.append(center_y)

    ordered: list[dict] = []
    rows_with_centers = list(zip(row_centers, rows))
    rows_with_centers.sort(key=lambda pair: pair[0])

    for _, row in rows_with_centers:
        row.sort(key=lambda pair: pair[1][0])
        ordered.extend(item for item, _ in row)

    # 没有 box 的 item 放最后，避免破坏有位置信息的阅读顺序。
    ordered.extend(fallback)
    return ordered


def _ocr_items(
    output,
    min_confidence: float,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[list[dict], int]:
    texts = getattr(output, "txts", None) or []
    scores = getattr(output, "scores", None) or []
    boxes = getattr(output, "boxes", None)

    items: list[dict] = []
    filtered_items = 0

    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) else 0.0
        clean_text = _clean_ocr_text(text)

        box = None
        if boxes is not None and index < len(boxes):
            box_values = _box_array(boxes[index])
            if box_values is not None:
                box = box_values.tolist()

        if score < min_confidence:
            filtered_items += 1
            continue

        if _is_low_quality_ocr_item(clean_text, score, box, frame_shape):
            filtered_items += 1
            continue

        items.append({
            "text": clean_text,
            "score": round(score, 4),
            "box": box,
        })

    items = _sort_ocr_items_reading_order(items)
    return items, filtered_items


def _chunk_frame_times_ms(chunks: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            int(round(float(chunk.get("frame_time", chunk.get("start_time", 0))) * 1000))
            for chunk in chunks
        ],
        dtype=np.int32,
    )


def _embedding_chunk_indices_to_frame_times(
    chunks: list[dict],
    embedding_chunk_indices: np.ndarray,
    embedding_count: int,
) -> np.ndarray:
    """
    build_text_semantic_arrays 返回的是 embedding_id -> chunk_id。
    OCR 新 schema 里要保存 embedding_id -> frame_ms，所以这里做一次映射。
    """
    if embedding_count <= 0:
        return np.empty((0,), dtype=np.int32)

    frame_times_ms = _chunk_frame_times_ms(chunks)
    source_indices = np.asarray(embedding_chunk_indices, dtype=np.int32).reshape((-1,))

    if len(source_indices) == embedding_count:
        if np.all((source_indices >= 0) & (source_indices < len(frame_times_ms))):
            return frame_times_ms[source_indices].astype(np.int32)

    # 兜底：如果 embedding 数量和 chunk 数量一一对应，但没有可靠 indices，
    # 则按顺序映射。正常情况下不会走到这里。
    if embedding_count <= len(frame_times_ms):
        return frame_times_ms[:embedding_count].astype(np.int32)

    raise ValueError(
        f"OCR semantic embedding 数量异常：embeddings={embedding_count}, chunks={len(chunks)}"
    )


def _run_ocr_frame_loop(
    video_path: str | Path,
    backend: OCRBackend,
    *,
    sample_fps: float,
    decode_height: int,
    min_confidence: float,
    prefer_ffmpeg: bool,
) -> tuple[list[dict], dict]:
    chunks: list[dict] = []
    interval = 1.0 / sample_fps
    stats = {
        "ocr_elapsed": 0.0,
        "decoded_frames": 0,
        "hit_frames": 0,
        "ocr_failed_frames": 0,
        "ocr_filtered_items": 0,
        "ocr_error_samples": [],
    }
    started = time.perf_counter()
    frames = read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg)
    for frame_index, (timestamp, frame) in enumerate(frames):
        stats["decoded_frames"] += 1
        frame_start = time.perf_counter()
        try:
            output = backend(frame)
        except Exception as exc:
            stats["ocr_elapsed"] += time.perf_counter() - frame_start
            stats["ocr_failed_frames"] += 1
            if len(stats["ocr_error_samples"]) < 5:
                stats["ocr_error_samples"].append({
                    "frame_index": int(frame_index),
                    "timestamp": round(float(timestamp), 3),
                    "error": str(exc),
                })
            continue
        stats["ocr_elapsed"] += time.perf_counter() - frame_start
        items, filtered_items = _ocr_items(output, min_confidence, frame_shape=frame.shape[:2])
        stats["ocr_filtered_items"] += filtered_items
        if not items:
            continue
        stats["hit_frames"] += 1
        chunks.append({
            "start_time": round(float(timestamp), 3),
            "end_time": round(float(timestamp + interval), 3),
            "frame_time": round(float(timestamp), 3),
            "text": " ".join(item["text"] for item in items),
            "items": items,
            "score": round(max(item["score"] for item in items), 4),
            "frame_shape": frame.shape[:2],
        })
    stats["frame_loop_elapsed"] = time.perf_counter() - started
    return chunks, stats


def _build_ocr_semantic_result(
    chunks: list[dict],
    *,
    enabled: bool,
    output_path: str | Path | None,
    model_name: str,
    model_dir: str | Path,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not enabled:
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)
        return (
            np.empty((0, 0), dtype=np.float16),
            np.empty((0,), dtype=np.int32),
            {"semantic_chunks": 0, "semantic_status": "disabled"},
        )
    try:
        raw_result = build_text_semantic_arrays(
            chunks=chunks,
            model_name=model_name,
            model_dir=model_dir,
            device=resolve_text_embedding_device(device),
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
        embeddings = np.asarray(raw_result["embeddings"], dtype=np.float16)
        indices = np.asarray(raw_result["embedding_chunk_indices"], dtype=np.int32)
        result = {key: value for key, value in raw_result.items() if key not in {"embeddings", "embedding_chunk_indices"}}
        result["semantic_chunks"] = int(embeddings.shape[0])
        return embeddings, indices, result
    except Exception as exc:
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)
        return (
            np.empty((0, 0), dtype=np.float16),
            np.empty((0,), dtype=np.int32),
            {
                "semantic_chunks": 0,
                "semantic_model": model_name,
                "semantic_status": "unavailable",
                "semantic_error": str(exc),
            },
        )


def _ocr_index_result(
    backend: OCRBackend,
    chunks: list[dict],
    stats: dict,
    *,
    ocr_version: str,
    det_lang: str,
    rec_lang: str,
    model_type: str,
    backend_init_elapsed: float,
) -> dict:
    frame_loop_elapsed = float(stats["frame_loop_elapsed"])
    ocr_elapsed = float(stats["ocr_elapsed"])
    return {
        "engine": backend.engine,
        "device": backend.device,
        "providers": backend.providers,
        "ocr_version": ocr_version,
        "det_lang": det_lang,
        "rec_lang": rec_lang,
        "model_type": model_type,
        "frames": stats["decoded_frames"],
        "hit_frames": stats["hit_frames"],
        "chunks": len(chunks),
        "ocr_failed_frames": stats["ocr_failed_frames"],
        "ocr_filtered_items": stats["ocr_filtered_items"],
        "ocr_rec_resized_inputs": int(getattr(backend, "rec_resized_inputs", 0)),
        "ocr_rec_max_input_width": int(getattr(backend, "rec_max_input_width", 0)),
        "ocr_error_samples": stats["ocr_error_samples"],
        "backend_init_elapsed_seconds": round(backend_init_elapsed, 3),
        "frame_loop_elapsed_seconds": round(frame_loop_elapsed, 3),
        "decode_postprocess_elapsed_seconds": round(max(0.0, frame_loop_elapsed - ocr_elapsed), 3),
        "ocr_elapsed_seconds": round(ocr_elapsed, 3),
        "schema_version": 3,
        "decode_status": "complete" if stats["decoded_frames"] else "empty",
    }



def build_ocr_index(
    video_path: str | Path,
    output_path: str | Path,
    working_dir: str | Path,
    sample_fps: float = 1.0,
    decode_height: int = 720,
    min_confidence: float = 0.5,
    device: str = "npu",
    device_id: int = 0,
    model_root: str | Path = "models/rapidocr",
    ocr_version: str = "PP-OCRv6",
    det_lang: str = "ch",
    rec_lang: str = "ch",
    model_type: str = "small",
    npu_self_test: bool = True,
    prefer_ffmpeg: bool = True,
    semantic_enabled: bool = True,
    semantic_output_path: str | Path | None = None,
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    semantic_device: str = "cpu",
    semantic_model_dir: str | Path = "models/text-embeddings",
    semantic_batch_size: int = 32,
    semantic_local_files_only: bool = True,
    engine: str = "rapidocr",
    acl_model_dir: str | Path | None = None,
    backend: OCRBackend | None = None,
) -> dict:
    if sample_fps <= 0:
        raise ValueError("ocr_sample_fps 必须大于 0")
    started = time.perf_counter()
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    backend_init_started = time.perf_counter()
    if backend is None:
        backend = create_ocr_backend(
            engine,
            device=device,
            device_id=device_id,
            model_root=model_root,
            ocr_version=ocr_version,
            det_lang=det_lang,
            rec_lang=rec_lang,
            model_type=model_type,
            npu_self_test=npu_self_test,
            acl_model_dir=acl_model_dir,
        )
    backend_init_elapsed = time.perf_counter() - backend_init_started

    chunks, frame_stats = _run_ocr_frame_loop(
        video_path,
        backend,
        sample_fps=sample_fps,
        decode_height=decode_height,
        min_confidence=min_confidence,
        prefer_ffmpeg=prefer_ffmpeg,
    )
    result = _ocr_index_result(
        backend,
        chunks,
        frame_stats,
        ocr_version=ocr_version,
        det_lang=det_lang,
        rec_lang=rec_lang,
        model_type=model_type,
        backend_init_elapsed=backend_init_elapsed,
    )
    semantic_started = time.perf_counter()
    embeddings, embedding_frame_indices, semantic_result = _build_ocr_semantic_result(
        chunks,
        enabled=semantic_enabled,
        output_path=semantic_output_path,
        model_name=semantic_model,
        model_dir=semantic_model_dir,
        device=semantic_device,
        batch_size=semantic_batch_size,
        local_files_only=semantic_local_files_only,
    )
    semantic_elapsed = time.perf_counter() - semantic_started
    save_started = time.perf_counter()
    _save_ocr_npz(
        output_path,
        chunks,
        embeddings,
        embedding_frame_indices,
    )
    save_elapsed = time.perf_counter() - save_started

    result.update(semantic_result)
    result.update({
        "semantic_elapsed_seconds": round(semantic_elapsed, 3),
        "index_save_elapsed_seconds": round(save_elapsed, 3),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    return result


def _normalized_box(box, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    values = np.asarray(box if box is not None else np.zeros((4, 2)), dtype=np.float32)
    if values.shape != (4, 2):
        values = np.zeros((4, 2), dtype=np.float32)
    scale = np.asarray([max(width, 1), max(height, 1)], dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0)


def _save_ocr_npz(
    output_path: str | Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_frame_indices: np.ndarray,
) -> None:
    box_frame_indices: list[int] = []
    box_texts: list[str] = []
    box_scores: list[float] = []
    boxes: list[np.ndarray] = []

    frame_times_ms: list[int] = []
    frame_windows_ms: list[list[int]] = []
    for frame_index, chunk in enumerate(chunks):
        frame_shape = tuple(chunk.get("frame_shape") or (1, 1))
        frame_time_ms = int(round(float(chunk.get("frame_time", chunk.get("start_time", 0))) * 1000))
        frame_times_ms.append(frame_time_ms)
        frame_windows_ms.append([
            int(round(float(chunk.get("start_time", 0)) * 1000)),
            int(round(float(chunk.get("end_time", 0)) * 1000)),
        ])

        for item in chunk.get("items", []):
            box_frame_indices.append(frame_index)
            box_texts.append(str(item.get("text", "")).strip())
            box_scores.append(float(item.get("score", 0.0)))
            boxes.append(_normalized_box(item.get("box"), frame_shape))
    atomic_save_npz(
        output_path,
        frame_times_ms=np.asarray(frame_times_ms, dtype=np.int32),
        frame_windows_ms=np.asarray(frame_windows_ms, dtype=np.int32).reshape((-1, 2)),
        embeddings=np.asarray(embeddings, dtype=np.float16),
        embedding_frame_indices=np.asarray(embedding_frame_indices, dtype=np.int32).reshape((-1,)),
        box_frame_indices=np.asarray(box_frame_indices, dtype=np.int32),
        box_texts=np.asarray(box_texts, dtype="U"),
        box_scores=np.asarray(box_scores, dtype=np.float32),
        boxes=np.stack(boxes).astype(np.float32) if boxes else np.empty((0, 4, 2), dtype=np.float32),
    )
