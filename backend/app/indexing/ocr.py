from __future__ import annotations

import re
import unicodedata
import time
from pathlib import Path
from typing import Any

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
):
    if not _has_local_rapidocr_assets(model_root):
        raise FileNotFoundError(f"本地 RapidOCR 模型缺失: {model_root}")

    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("OCR 依赖 rapidocr 未安装；请安装 rapidocr 后重试") from exc

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
    if not text:
        return True

    if score <= 0:
        return True

    # 纯符号、纯标点、装饰线
    if _is_mostly_punctuation(text):
        return True

    # 没有任何字母、数字、CJK 字符，通常是 OCR 噪声
    if not _has_alnum_or_cjk(text):
        return True

    compact = text.replace(" ", "")

    # 低置信度单字符，噪声概率高。
    # 但高置信度单个中文/数字/字母仍保留。
    if len(compact) <= 1 and score < 0.85:
        return True

    # 重复字符，例如 "----"、"||||"、"oooo"
    if _REPEAT_CHAR_RE.match(compact):
        return True

    # 很短的纯数字，且置信度不高，常见于页码/角标/误检。
    # 高置信度保留，避免误删比分、年份、价格等。
    if compact.isdigit() and len(compact) <= 2 and score < 0.8:
        return True

    bounds = _box_bounds(box)
    if bounds is not None:
        left, top, right, bottom = bounds
        width = max(right - left, 0.0)
        height = max(bottom - top, 0.0)

        # 宽高明显无效
        if width < 2 or height < 2:
            return True

        # 极端长宽比，通常是分割线、边框、误检。
        aspect = width / max(height, 1.0)
        if aspect > 40 or aspect < 0.025:
            return True

        # 面积极小 + 低置信度，基本没有检索价值。
        area_ratio = _box_area_ratio(box, frame_shape)
        if area_ratio is not None and area_ratio < 0.00001 and score < 0.9:
            return True

    return False


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
) -> dict:
    if sample_fps <= 0:
        raise ValueError("ocr_sample_fps 必须大于 0")
    started = time.perf_counter()
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
    ocr_failed_frames = 0
    ocr_filtered_items = 0
    ocr_error_samples: list[dict] = []

    for frame_index, (timestamp, frame) in enumerate(
        read_frames(video_path, sample_fps, out_height=decode_height, prefer_ffmpeg=prefer_ffmpeg)
    ):
        decoded_frames += 1
        frame_start = time.perf_counter()
        try:
            output = ocr(frame)
        except Exception as exc:
            ocr_elapsed += time.perf_counter() - frame_start
            ocr_failed_frames += 1

            # 只保留少量样例，避免 result 过大。
            if len(ocr_error_samples) < 5:
                ocr_error_samples.append({
                    "frame_index": int(frame_index),
                    "timestamp": round(float(timestamp), 3),
                    "error": str(exc),
                })
            continue

        ocr_elapsed += time.perf_counter() - frame_start

        items, filtered_items = _ocr_items(
            output,
            min_confidence,
            frame_shape=frame.shape[:2],
        )
        ocr_filtered_items += filtered_items

        if not items:
            continue
        hit_frames += 1
        text = " ".join(item["text"] for item in items)
        chunks.append({
            "start_time": round(float(timestamp), 3),
            "end_time": round(float(timestamp + interval), 3),
            "frame_time": round(float(timestamp), 3),
            "text": text,
            "items": items,
            "score": round(max(item["score"] for item in items), 4),
            "frame_shape": frame.shape[:2],
        })

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
        "ocr_failed_frames": ocr_failed_frames,
        "ocr_filtered_items": ocr_filtered_items,
        "ocr_error_samples": ocr_error_samples,
        "ocr_elapsed_seconds": round(ocr_elapsed, 3),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "schema_version": 4,
        "decode_status": "complete" if decoded_frames else "empty",
    }
    semantic_result: dict
    embeddings: np.ndarray
    embedding_frame_times_ms: np.ndarray
    if not semantic_enabled:
        if semantic_output_path is not None:
            Path(semantic_output_path).unlink(missing_ok=True)
        embeddings = np.empty((0, 0), dtype=np.float16)
        embedding_frame_times_ms = np.empty((0,), dtype=np.int32)
        semantic_result = {
            "semantic_chunks": 0,
            "semantic_status": "disabled",
        }
    else:
        try:
            resolved_device = resolve_text_embedding_device(semantic_device)
            raw_semantic_result = build_text_semantic_arrays(
                chunks=chunks,
                model_name=semantic_model,
                model_dir=semantic_model_dir,
                device=resolved_device,
                batch_size=semantic_batch_size,
                local_files_only=semantic_local_files_only,
            )

            embeddings = np.asarray(raw_semantic_result["embeddings"], dtype=np.float16)
            raw_embedding_chunk_indices = np.asarray(
                raw_semantic_result["embedding_chunk_indices"],
                dtype=np.int32,
            )

            embedding_frame_times_ms = _embedding_chunk_indices_to_frame_times(
                chunks,
                raw_embedding_chunk_indices,
                embedding_count=int(embeddings.shape[0]),
            )

            semantic_result = {
                key: value for key, value in raw_semantic_result.items()
                if key not in {"embeddings", "embedding_chunk_indices"}
            }
            semantic_result["semantic_chunks"] = int(embeddings.shape[0])

        except Exception as exc:
            if semantic_output_path is not None:
                Path(semantic_output_path).unlink(missing_ok=True)
            embeddings = np.empty((0, 0), dtype=np.float16)
            embedding_frame_times_ms = np.empty((0,), dtype=np.int32)
            semantic_result = {
                "semantic_chunks": 0,
                "semantic_model": semantic_model,
                "semantic_status": "unavailable",
                "semantic_error": str(exc),
            }
    _save_ocr_npz(
        output_path,
        chunks,
        embeddings,
        embedding_frame_times_ms,
    )

    result.update(semantic_result)
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
    embedding_frame_times_ms: np.ndarray,
) -> None:
    box_chunk_indices: list[int] = []
    box_texts: list[str] = []
    box_scores: list[float] = []
    boxes: list[np.ndarray] = []

    for chunk in chunks:
        frame_shape = tuple(chunk.get("frame_shape") or (1, 1))
        frame_time_ms = int(round(float(chunk.get("frame_time", chunk.get("start_time", 0))) * 1000))

        for item in chunk.get("items", []):
            # 新 schema：box_chunk_indices 不再保存 chunk_id，而是保存该 box 所在帧的 frame_ms。
            box_chunk_indices.append(frame_time_ms)
            box_texts.append(str(item.get("text", "")).strip())
            box_scores.append(float(item.get("score", 0.0)))
            boxes.append(_normalized_box(item.get("box"), frame_shape))
    atomic_save_npz(
        output_path,
        embeddings=np.asarray(embeddings, dtype=np.float16),
        # 新 schema：embedding_chunk_indices 不再保存 chunk_id，而是保存该 embedding 对应帧的 frame_ms。
        embedding_chunk_indices=np.asarray(embedding_frame_times_ms, dtype=np.int32).reshape((-1,)),
        box_chunk_indices=np.asarray(box_chunk_indices, dtype=np.int32),
        box_texts=np.asarray(box_texts, dtype="U"),
        box_scores=np.asarray(box_scores, dtype=np.float32),
        boxes=np.stack(boxes).astype(np.float32) if boxes else np.empty((0, 4, 2), dtype=np.float32),
    )
