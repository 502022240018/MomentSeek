from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path

import numpy as np

from app.db import Catalog
from app.indexing.common import normalize
from app.indexing.manifest import require_channel_manifest
from app.indexing.asr_text import normalize_search_text
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    video_id: str
    start_time: float
    end_time: float
    score: float
    modality: str
    evidence: str | None = None
    raw_score: float | None = None
    robust_z: float | None = None
    percentile: float | None = None
    decision: str = "hit"
    above_threshold: bool = True
    distribution_reliable: bool | None = None
    distribution_median: float | None = None
    distribution_mad: float | None = None
    best_time: float | None = None
    visual_top1: float | None = None
    visual_top3: float | None = None
    visual_mean: float | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    semantic_cosine: float | None = None
    unit_type: str | None = None
    unit_id: int | None = None
    best_ms: int | None = None
    text: str | None = None
    features: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    video_id: str
    video_name: str
    start_time: float
    end_time: float
    score: float
    modalities: list[str]
    thumbnail_url: str | None
    media_url: str
    clip_url: str
    decision: str
    above_threshold: bool = True
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["start_time"] = round(self.start_time, 3)
        value["end_time"] = round(self.end_time, 3)
        value["score"] = round(self.score, 4)
        return value


def normalize_text(text: str) -> str:
    return normalize_search_text(text)


def lexical_score(query: str, text: str) -> float:
    query_value, text_value = normalize_text(query), normalize_text(text)
    if not query_value or not text_value:
        return 0
    if query_value in text_value:
        return 1
    size = 2 if len(query_value) > 1 else 1
    query_grams = {query_value[index:index + size] for index in range(max(1, len(query_value) - size + 1))}
    text_grams = {text_value[index:index + size] for index in range(max(1, len(text_value) - size + 1))}
    coverage = len(query_grams & text_grams) / max(1, len(query_grams))
    return float(coverage)


def robust_distribution(scores: np.ndarray) -> dict:
    """Return per-query/per-video robust z-scores and empirical percentiles."""
    values = np.asarray(scores, dtype=np.float32)
    if not len(values):
        return {
            "z_scores": np.empty(0, np.float32), "percentiles": np.empty(0, np.float32),
            "median": 0.0, "mad": 0.0, "reliable": False,
        }
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-6:
        z_scores = 0.67448975 * (values - median) / mad
    else:
        standard_deviation = float(values.std())
        z_scores = (values - float(values.mean())) / standard_deviation if standard_deviation > 1e-6 else np.zeros_like(values)
    z_scores = np.clip(z_scores, -8, 8).astype(np.float32)
    ordered = np.sort(values)
    # Ties receive the same upper empirical percentile.
    percentiles = np.asarray(
        [np.searchsorted(ordered, value, side="right") / len(values) for value in values],
        dtype=np.float32,
    )
    return {
        "z_scores": z_scores,
        "percentiles": percentiles,
        "median": median,
        "mad": mad,
        "reliable": bool(len(values) >= 8 and (mad > 1e-6 or float(values.std()) > 1e-6)),
    }


def face_confidence(cosine: float) -> float:
    """Map an ArcFace (buffalo_l) cosine to a calibrated [0,1] confidence.

    Face cosine is absolutely meaningful (distance to a reference identity), unlike
    CLIP text-image scores. Raw cosines for true matches cluster around 0.45-0.7,
    so a logistic centred at 0.45 lifts a strong match to ~1.0 — putting it on the
    same scale as the visual empirical percentile, which is what the fusion step
    weighs. Without this, a cosine=0.6 face hit (raw 0.6) would lose to a visual
    percentile=0.98 hit even though both are strong.
    """
    return float(1.0 / (1.0 + np.exp(-12.0 * (cosine - 0.45))))


def asr_semantic_confidence(cosine: float) -> float:
    """Map normalized text embedding cosine to a useful [0,1] ASR score."""
    return float(1.0 / (1.0 + np.exp(-10.0 * (cosine - 0.35))))


def visual_confidence(cosine: float) -> float:
    """Map visual raw cosine to a cross-video ranking score."""
    return float(np.clip((cosine + 1.0) / 2.0, 0, 1))


def _seconds(ms: int | float) -> float:
    return float(ms) / 1000.0


def _decode_text_array(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def _semantic_arrays(data) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "embeddings" not in data.files or "embedding_chunk_indices" not in data.files:
        return None, None
    embeddings = data["embeddings"]
    indices = data["embedding_chunk_indices"].astype(np.int32)
    if embeddings.ndim != 2 or not len(embeddings) or not len(indices):
        return None, None
    return np.asarray(embeddings, dtype=np.float32), indices


def _visual_index_arrays(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    required = {"frame_embeddings", "frame_times_ms", "segment_frame_offsets"}
    if not required.issubset(set(data.files)):
        raise ValueError("visual v3 索引缺少必要数组，请重跑 visual 索引")
    embeddings = np.asarray(data["frame_embeddings"], dtype=np.float32)
    frame_times_ms = data["frame_times_ms"].astype(np.int32)
    offsets = data["segment_frame_offsets"].astype(np.int32)
    if embeddings.ndim != 2 or len(embeddings) != len(frame_times_ms):
        raise ValueError("visual v3 索引数组长度不一致，请重跑 visual 索引")
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(frame_times_ms) or np.any(np.diff(offsets) < 0):
        raise ValueError("visual v3 segment_frame_offsets 无效，请重跑 visual 索引")
    segment_times_ms = None
    if "segment_times_ms" in data.files:
        segment_times_ms = data["segment_times_ms"].astype(np.int32)
        if segment_times_ms.shape != (len(offsets) - 1, 2):
            raise ValueError("visual v3 segment_times_ms 无效，请重跑 visual 索引")
        if np.any(segment_times_ms[:, 1] < segment_times_ms[:, 0]):
            raise ValueError("visual v3 segment_times_ms 时间范围无效，请重跑 visual 索引")
    return embeddings, frame_times_ms, offsets, segment_times_ms


def _visual_segment_scores(
    frame_scores: np.ndarray,
    frame_times_ms: np.ndarray,
    offsets: np.ndarray,
) -> tuple[list[int], np.ndarray, list[float], list[float], list[list[float]], list[int]]:
    score_values = np.asarray(frame_scores, dtype=np.float32)
    if score_values.ndim == 1:
        score_values = score_values.reshape(-1, 1)
    if score_values.ndim != 2 or score_values.shape[0] != len(frame_times_ms):
        raise ValueError("visual frame score shape does not match the index")
    segment_ids: list[int] = []
    raw_scores: list[float] = []
    top3_scores: list[float] = []
    mean_scores: list[float] = []
    subquery_scores: list[list[float]] = []
    best_times_ms: list[int] = []
    for segment_id in range(len(offsets) - 1):
        start, end = int(offsets[segment_id]), int(offsets[segment_id + 1])
        if start == end:
            continue
        bucket_scores = score_values[start:end]
        per_query_top = np.max(bucket_scores, axis=0)
        if score_values.shape[1] == 1:
            aggregate_score = float(per_query_top[0])
            frame_aggregate = bucket_scores[:, 0]
        else:
            aggregate_score = float(0.65 * np.mean(per_query_top) + 0.35 * np.min(per_query_top))
            frame_aggregate = 0.65 * np.mean(bucket_scores, axis=1) + 0.35 * np.min(
                bucket_scores, axis=1
            )
        order = np.argsort(frame_aggregate)[::-1]
        top_values = frame_aggregate[order]
        segment_ids.append(segment_id)
        raw_scores.append(aggregate_score)
        top3_scores.append(float(np.mean(top_values[:min(3, len(top_values))])))
        mean_scores.append(float(np.mean(frame_aggregate)))
        subquery_scores.append([float(value) for value in per_query_top])
        best_times_ms.append(int(frame_times_ms[start + int(order[0])]))
    return (
        segment_ids,
        np.asarray(raw_scores, dtype=np.float32),
        top3_scores,
        mean_scores,
        subquery_scores,
        best_times_ms,
    )


def _visual_decision(
    profile: str,
    reliable: bool,
    local_index: int,
    fallback_indices: set[int],
    raw_score: float,
    ranking_score: float,
    percentile: float,
    z_score: float,
    sample_count: int,
) -> tuple[str, bool, str]:
    if not reliable:
        decision, above = ("fallback", True) if local_index in fallback_indices else ("weak", False)
        detail = (
            f"visual score={raw_score:.3f} · rank_score={ranking_score:.3f}"
            f" · distribution fallback (n={sample_count})"
        )
        return decision, above, detail
    if z_score >= 2.0 or percentile >= 0.975:
        decision, above = "strong", True
    elif percentile >= 0.80:
        qualifies = not (
            (profile == "balanced" and not (z_score >= 1.0 or percentile >= 0.90))
            or profile == "precision"
        )
        decision, above = (("fuzzy", True) if qualifies else ("weak", False))
    else:
        decision, above = "weak", False
    detail = (
        f"visual score={raw_score:.3f} · rank_score={ranking_score:.3f}"
        f" · percentile={percentile * 100:.1f}% · robust_z={z_score:.2f}"
    )
    return decision, above, detail


def _visual_segment_bounds(
    segment_id: int,
    segment_times_ms: np.ndarray | None,
    segment_ms: int,
    duration_ms: int,
) -> tuple[int, int, str]:
    if segment_times_ms is not None:
        start_ms, end_ms = [int(value) for value in segment_times_ms[segment_id]]
        return start_ms, end_ms, "explicit"
    start_ms = segment_id * segment_ms
    end_ms = min((segment_id + 1) * segment_ms, duration_ms or (segment_id + 1) * segment_ms)
    return start_ms, end_ms, "fixed"


def _visual_candidates(
    data,
    query: np.ndarray,
    video_id: str,
    duration_ms: int,
    segment_ms: int,
    profile: str = "balanced",
    limit: int = 72,
    segment_strategy: str = "fixed",
) -> list[Candidate]:
    frame_embeddings, frame_times_ms, offsets, segment_times_ms = _visual_index_arrays(data)
    if not len(frame_embeddings):
        return []
    query_values = np.asarray(query, dtype=np.float32)
    if query_values.ndim == 1:
        query_values = query_values.reshape(1, -1)
    if query_values.ndim != 2 or query_values.shape[1] != frame_embeddings.shape[1]:
        raise ValueError("visual query embedding shape does not match the index")
    query_values = np.stack([normalize(value) for value in query_values])
    (
        segment_ids,
        raw_values,
        top3_scores,
        mean_scores,
        subquery_scores,
        best_times_ms,
    ) = _visual_segment_scores(
        frame_embeddings @ query_values.T, frame_times_ms, offsets
    )
    if not len(raw_values):
        return []
    distribution = robust_distribution(raw_values)
    z_scores = distribution["z_scores"]
    percentiles = distribution["percentiles"]
    reliable = distribution["reliable"]
    raw_order = np.argsort(raw_values)[::-1]
    fallback_counts = {"recall": 3, "balanced": 2, "precision": 1}
    fallback_indices = set(int(index) for index in raw_order[:min(len(raw_order), fallback_counts[profile])])
    candidates = []
    cap = 500 if profile == "recall" else limit
    for local_index in raw_order[:cap]:
        local_index = int(local_index)
        segment_id = int(segment_ids[local_index])
        raw_score = float(raw_values[local_index])
        z_score = float(z_scores[local_index])
        percentile = float(percentiles[local_index])
        ranking_score = visual_confidence(raw_score)
        decision, above, detail = _visual_decision(
            profile,
            reliable,
            local_index,
            fallback_indices,
            raw_score,
            ranking_score,
            percentile,
            z_score,
            len(raw_values),
        )

        top3 = float(top3_scores[local_index])
        mean = float(mean_scores[local_index])
        best_ms = int(best_times_ms[local_index])
        detail += f" · best_frame={best_ms / 1000:.2f}s · top1={raw_score:.3f} · top3={top3:.3f} · mean={mean:.3f}"
        if query_values.shape[0] > 1:
            detail += " · subqueries=" + ",".join(
                f"{value:.3f}" for value in subquery_scores[local_index]
            )
        start_ms, end_ms, time_source = _visual_segment_bounds(
            segment_id, segment_times_ms, segment_ms, duration_ms
        )
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=ranking_score,
            modality="visual",
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=raw_score,
            robust_z=z_score,
            percentile=percentile,
            decision=decision,
            above_threshold=above,
            distribution_reliable=reliable,
            distribution_median=distribution["median"],
            distribution_mad=distribution["mad"],
            best_time=_seconds(best_ms),
            visual_top1=raw_score,
            visual_top3=top3,
            visual_mean=mean,
            unit_type="segment",
            unit_id=segment_id,
            best_ms=best_ms,
            features={
                "visual_top1": raw_score,
                "visual_top3": top3,
                "visual_mean": mean,
                "visual_rank_score": ranking_score,
                "visual_subquery_scores": subquery_scores[local_index],
                "visual_subquery_count": int(query_values.shape[0]),
                "percentile": percentile,
                "robust_z": z_score,
                "segment_time_source": time_source,
                "segment_strategy": segment_strategy,
            },
        ))
    return candidates


def _face_candidates(data, query: np.ndarray, video_id: str, limit: int, threshold: float = 0.35) -> list[Candidate]:
    if "embeddings" not in data.files or "track_times_ms" not in data.files:
        raise ValueError("face v3 索引缺少必要数组，请重跑 face 索引")
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    times = data["track_times_ms"].astype(np.int32)
    if embeddings.ndim != 2 or times.shape != (len(embeddings), 3):
        raise ValueError("face v3 索引数组长度不一致，请重跑 face 索引")
    if not len(embeddings):
        return []
    scores = embeddings @ normalize(query)
    candidates: list[Candidate] = []
    for index in np.argsort(scores)[::-1]:
        if len(candidates) >= limit:
            break
        index = int(index)
        cosine = float(scores[index])
        above = cosine >= threshold
        confidence = face_confidence(cosine)
        start_ms, end_ms, best_ms = [int(value) for value in times[index]]
        detail = f"face cosine={cosine:.3f} · confidence={confidence * 100:.1f}%"
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=confidence,
            modality="face",
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=cosine,
            decision="absolute_hit" if above else "weak",
            above_threshold=above,
            best_time=_seconds(best_ms),
            unit_type="track",
            unit_id=index,
            best_ms=best_ms,
            features={"face_cosine": cosine},
        ))
    return candidates


def _semantic_chunk_scores(
    chunk_count: int,
    semantic_embeddings: np.ndarray | None,
    embedding_chunk_indices: np.ndarray | None,
    semantic_query: np.ndarray | None,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    scores = np.zeros(chunk_count, dtype=np.float32)
    cosines_by_chunk = np.full(chunk_count, np.nan, dtype=np.float32)
    if (
        semantic_embeddings is None
        or embedding_chunk_indices is None
        or semantic_query is None
        or not len(semantic_embeddings)
        or not len(embedding_chunk_indices)
    ):
        return scores, cosines_by_chunk, set()
    cosines = semantic_embeddings @ normalize(semantic_query)
    percentiles = robust_distribution(cosines)["percentiles"]
    for local_index, chunk_index in enumerate(embedding_chunk_indices):
        chunk_index = int(chunk_index)
        if not 0 <= chunk_index < chunk_count:
            continue
        cosine = float(cosines[local_index])
        percentile = float(percentiles[local_index]) if len(percentiles) else 0.0
        cosines_by_chunk[chunk_index] = cosine
        scores[chunk_index] = 0.7 * asr_semantic_confidence(cosine) + 0.3 * percentile
    order = np.argsort(scores)[::-1]
    return scores, cosines_by_chunk, set(int(index) for index in order[:min(len(order), limit)])


def _text_candidate_decision(lexical: float, semantic: float) -> tuple[str, bool]:
    semantic_hit = semantic >= 0.55
    lexical_hit = lexical >= 0.25
    if semantic_hit and lexical_hit:
        return "semantic_lexical_hit", True
    if semantic_hit:
        return "semantic_hit", True
    if lexical_hit:
        return "lexical_hit", True
    return "weak", False


def _text_candidate_evidence(detail: str, lexical: float, semantic: float, cosine: float | None) -> str:
    metrics = [f"lexical={lexical:.3f}"]
    if cosine is None:
        metrics.append("semantic=unavailable")
    else:
        metrics.extend((f"semantic={semantic:.3f}", f"semantic_cosine={cosine:.3f}"))
    return f"{detail} · {' · '.join(metrics)}"


def _asr_candidates(
    chunks: list[dict],
    query_text: str,
    video_id: str,
    limit: int,
    modality: str = "asr",
    semantic_embeddings: np.ndarray | None = None,
    embedding_chunk_indices: np.ndarray | None = None,
    semantic_query: np.ndarray | None = None,
) -> list[Candidate]:
    if not chunks:
        return []

    lexical_scores = np.asarray([lexical_score(query_text, chunk.get("text", "")) for chunk in chunks], dtype=np.float32)
    semantic_scores, semantic_cosines, semantic_top_indices = _semantic_chunk_scores(
        len(chunks), semantic_embeddings, embedding_chunk_indices, semantic_query, limit
    )

    combined_scores = np.maximum(lexical_scores, 0.65 * semantic_scores + 0.35 * lexical_scores)
    candidate_indices = [
        int(index) for index in np.argsort(combined_scores)[::-1]
        if lexical_scores[index] > 0 or index in semantic_top_indices
    ][:limit]

    candidates: list[Candidate] = []
    for index in candidate_indices:
        chunk = chunks[index]
        lexical = float(lexical_scores[index])
        semantic = float(semantic_scores[index])
        semantic_cosine = None if np.isnan(semantic_cosines[index]) else float(semantic_cosines[index])
        score = float(combined_scores[index])
        decision, above = _text_candidate_decision(lexical, semantic)

        detail = str(chunk.get("text", "")).strip()
        display_features = {}

        if modality == "ocr":
            detail, display_features = _ocr_display_text(query_text, chunk)

        evidence = _text_candidate_evidence(detail, lexical, semantic, semantic_cosine)
        start_ms = int(chunk.get("start_ms", round(float(chunk.get("start_time", 0)) * 1000)))
        end_ms = int(chunk.get("end_ms", round(float(chunk.get("end_time", 0)) * 1000)))
        # OCR chunks carry the sampled frame timestamp; ASR is audio-only so we fall
        # back to the chunk start. Either way the thumbnail is fetched on demand.
        best_ms = int(chunk.get("frame_ms", start_ms))
        features = {
            "lexical_score": lexical,
            "semantic_score": semantic if semantic_cosine is not None else None,
            "semantic_cosine": semantic_cosine,
        }
        features.update(display_features)
        if "score" in chunk:
            features[f"{modality}_score"] = float(chunk["score"])
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=score,
            modality=modality,
            evidence=evidence if above else evidence + " · 低于阈值",
            raw_score=score,
            decision=decision,
            above_threshold=above,
            lexical_score=lexical,
            semantic_score=semantic if semantic_cosine is not None else None,
            semantic_cosine=semantic_cosine,
            best_time=_seconds(best_ms),
            unit_type="chunk",
            unit_id=int(chunk.get("chunk_id", index)),
            best_ms=best_ms,
            text=detail,
            features=features,
        ))
    return candidates


def _asr_chunks_from_npz(data) -> list[dict]:
    if "chunk_times_ms" not in data.files or "texts" not in data.files:
        raise ValueError("asr v3 索引缺少必要数组，请重跑 ASR 索引")
    times = data["chunk_times_ms"].astype(np.int32)
    texts = _decode_text_array(data["texts"])
    if times.shape != (len(texts), 2):
        raise ValueError("asr v3 chunk_times_ms/texts 长度不一致，请重跑 ASR 索引")
    return [
        {
            "chunk_id": index,
            "start_ms": int(row[0]),
            "end_ms": int(row[1]),
            "text": texts[index],
        }
        for index, row in enumerate(times)
    ]


def _limit_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max(0, max_chars - 3)].rstrip() + "..."


def _ocr_display_text(
    query_text: str,
    chunk: dict,
    max_boxes: int = 3,
    max_chars: int = 120,
    max_frame_chars: int = 500,
) -> tuple[str, dict]:
    """
    为 OCR evidence 选择更适合展示的文本。

    检索仍然基于整帧 OCR 文本；
    展示优先显示和 query 词面最相关的 box 文本；
    如果找不到明确相关 box，则回退到整帧 OCR 文本。
    """
    frame_text = str(chunk.get("text", "")).strip()

    box_texts = [
        str(value).strip()
        for value in chunk.get("ocr_box_texts", [])
        if str(value).strip()
    ]

    box_scores = [
        float(value)
        for value in chunk.get("ocr_box_scores", [])
    ]

    frame_context = _limit_text(frame_text, max_frame_chars)

    if not box_texts:
        return _limit_text(frame_text, max_chars), {
            "ocr_display_mode": "frame_text",
            "ocr_frame_text": frame_context,
            "ocr_box_count": 0,
        }

    scored: list[tuple[float, float, float, int, str]] = []

    for index, text in enumerate(box_texts):
        box_lexical = lexical_score(query_text, text)
        box_confidence = box_scores[index] if index < len(box_scores) else 0.0

        # 展示排序：query 相关性为主，OCR 置信度为辅。
        display_score = 0.85 * float(box_lexical) + 0.15 * float(box_confidence)

        scored.append((
            display_score,
            float(box_lexical),
            float(box_confidence),
            index,
            text,
        ))

    scored.sort(reverse=True)

    selected: list[str] = []
    selected_box_scores: list[float] = []
    selected_box_lexical_scores: list[float] = []

    for _display_score, box_lexical, box_confidence, _index, text in scored:
        # 只展示和 query 有词面相关性的 box。
        # 如果是纯语义命中但没有明确 box 词面命中，则回退整帧文本。
        if box_lexical <= 0:
            continue

        selected.append(text)
        selected_box_scores.append(box_confidence)
        selected_box_lexical_scores.append(box_lexical)

        if len(selected) >= max_boxes:
            break

    if not selected:
        return _limit_text(frame_text, max_chars), {
            "ocr_display_mode": "frame_text",
            "ocr_frame_text": frame_context,
            "ocr_box_count": len(box_texts),
        }

    display_text = _limit_text(" / ".join(selected), max_chars)

    return display_text, {
        "ocr_display_mode": "matched_boxes",
        "ocr_frame_text": frame_context,
        "ocr_box_count": len(box_texts),
        "ocr_matched_box_texts": selected,
        "ocr_matched_box_scores": selected_box_scores,
        "ocr_matched_box_lexical_scores": selected_box_lexical_scores,
    }



def _ocr_chunks_from_npz(data) -> list[dict]:
    """Load the frame-native OCR v3 layout."""
    required = {"frame_times_ms", "frame_windows_ms", "box_frame_indices", "box_texts", "box_scores", "boxes"}
    if not required.issubset(set(data.files)):
        raise ValueError("ocr v3 索引缺少帧级数组，请重跑 OCR 索引")
    frame_times = data["frame_times_ms"].astype(np.int32)
    frame_windows = data["frame_windows_ms"].astype(np.int32)
    box_frame_indices = data["box_frame_indices"].astype(np.int32)
    box_texts = _decode_text_array(data["box_texts"])
    box_scores = np.asarray(data["box_scores"], dtype=np.float32)
    boxes = np.asarray(data["boxes"], dtype=np.float32)
    if frame_times.ndim != 1:
        raise ValueError("ocr v3 frame_times_ms 必须是一维数组，请重跑 OCR 索引")
    if frame_windows.ndim != 2 or frame_windows.shape != (len(frame_times), 2):
        raise ValueError("ocr v3 frame_windows_ms 必须是 [num_frames, 2]，请重跑 OCR 索引")
    if not (len(box_frame_indices) == len(box_texts) == len(box_scores) == len(boxes)):
        raise ValueError("ocr v3 box 数组长度不一致，请重跑 OCR 索引")
    if len(box_frame_indices) and np.any((box_frame_indices < 0) | (box_frame_indices >= len(frame_times))):
        raise ValueError("ocr v3 box_frame_indices 越界，请重跑 OCR 索引")
    chunks: list[dict] = []
    for frame_index, frame_ms in enumerate(frame_times):
        indices = np.flatnonzero(box_frame_indices == frame_index)
        box_text_values = [box_texts[int(index)].strip() for index in indices if box_texts[int(index)].strip()]
        box_score_values = [float(box_scores[int(index)]) for index in indices if box_texts[int(index)].strip()]
        chunks.append({
            "chunk_id": frame_index,
            "start_ms": int(frame_windows[frame_index, 0]),
            "end_ms": int(frame_windows[frame_index, 1]),
            "frame_ms": int(frame_ms),
            "text": " ".join(box_text_values),
            "ocr_box_texts": box_text_values,
            "ocr_box_scores": box_score_values,
            "score": max(box_score_values) if box_score_values else 0.0,
        })
    return chunks


def _remap_embedding_frame_times_to_chunk_indices(
    chunks: list[dict],
    embedding_frame_times_ms: np.ndarray | None,
) -> np.ndarray | None:
    """
    OCR 新 schema 的 embedding_chunk_indices 保存的是 frame_ms。
    但 _asr_candidates 内部需要 embedding_id -> chunk_id。
    所以这里把 frame_ms 映射成当前重建 chunks 的 chunk_id。
    """
    if embedding_frame_times_ms is None:
        return None

    frame_to_chunk_id = {
        int(chunk["frame_ms"]): int(chunk["chunk_id"])
        for chunk in chunks
        if "frame_ms" in chunk and "chunk_id" in chunk
    }

    values = np.asarray(embedding_frame_times_ms, dtype=np.int32).reshape((-1,))
    return np.asarray(
        [frame_to_chunk_id.get(int(frame_ms), -1) for frame_ms in values],
        dtype=np.int32,
    )


def _channel_manifest_for(video: dict, index_dir: Path, channel: str) -> tuple[dict, dict, Path]:
    manifest, channel_manifest = require_channel_manifest(index_dir, str(video.get("name") or video["id"]), channel)
    default_files = {"visual": "visual.npz", "face": "face.npz", "asr": "asr.npz", "ocr": "ocr.npz"}
    file_name = str(channel_manifest.get("file") or default_files[channel])
    index_file = index_dir / file_name
    if not index_file.exists():
        raise ValueError(f"视频 {video.get('name') or video['id']} 缺少 {channel} v3 索引文件，请重跑该通道")
    return manifest, channel_manifest, index_file


def _round_optional(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _serialized_features(features: dict) -> dict:
    serialized = {}
    for key, value in features.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            serialized[key] = round(value, 4)
        else:
            serialized[key] = value
    return serialized


def _serialize_evidence(item: Candidate) -> dict:
    return {
        "modality": item.modality,
        "score": round(item.score, 4),
        "raw_score": _round_optional(item.raw_score, 4),
        "robust_z": _round_optional(item.robust_z, 3),
        "percentile": _round_optional(item.percentile, 4),
        "decision": item.decision,
        "distribution_reliable": item.distribution_reliable,
        "distribution_median": _round_optional(item.distribution_median, 4),
        "distribution_mad": _round_optional(item.distribution_mad, 4),
        "best_time": _round_optional(item.best_time, 3),
        "visual_top1": _round_optional(item.visual_top1, 4),
        "visual_top3": _round_optional(item.visual_top3, 4),
        "visual_mean": _round_optional(item.visual_mean, 4),
        "lexical_score": _round_optional(item.lexical_score, 4),
        "semantic_score": _round_optional(item.semantic_score, 4),
        "semantic_cosine": _round_optional(item.semantic_cosine, 4),
        "unit_type": item.unit_type,
        "unit_id": item.unit_id,
        "best_ms": item.best_ms,
        "text": item.text,
        "features": _serialized_features(item.features),
        "detail": item.evidence,
    }

_OCR_ONLY_MERGE_GAP_SECONDS = 0.35
_OCR_MERGE_MIN_SCORE_RATIO = 0.70
_OCR_MERGE_MAX_SCORE_DROP = 0.25

def _ocr_scores_compatible(group: list[Candidate], candidate: Candidate) -> bool:
    """
    OCR-only 合并时，避免高分命中被明显低分命中拖长。

    规则：
    - candidate 必须是 above_threshold；
    - group 里必须已有 above_threshold 的 OCR 命中；
    - candidate 分数不能比 group 里最佳 OCR 命中低太多。
    """
    if candidate.modality != "ocr" or not candidate.above_threshold:
        return False

    group_scores = [
        float(item.score)
        for item in group
        if item.modality == "ocr" and item.above_threshold
    ]
    if not group_scores:
        return False

    best_score = max(group_scores)

    if candidate.score >= best_score:
        return True

    threshold = max(
        best_score * _OCR_MERGE_MIN_SCORE_RATIO,
        best_score - _OCR_MERGE_MAX_SCORE_DROP,
    )
    return float(candidate.score) >= threshold


def _should_merge_ocr_only(
    group: list[Candidate],
    candidate: Candidate,
) -> bool:
    """
    OCR-only 结果使用更严格的帧级合并策略。

    只允许：
    - OCR 与 OCR 合并；
    - 都是 above-threshold；
    - 时间窗口重叠或几乎相邻；
    - 分数不能差太多。

    不设置最大合并时长：
    如果同一段 OCR 文本持续稳定出现很久，它应该保留为一个连续命中片段。
    """
    if candidate.modality != "ocr":
        return False
    if any(item.modality != "ocr" for item in group):
        return False

    if not candidate.above_threshold:
        return False
    if not any(item.above_threshold for item in group):
        return False

    group_end = max(item.end_time for item in group)

    # 只允许重叠或几乎直接相邻。
    # 例如 frame_window_ms=800 且 1fps 时：
    # 10.0s -> 9.6-10.4
    # 11.0s -> 10.6-11.4
    # gap=0.2，可以合并。
    gap = candidate.start_time - group_end
    if gap > _OCR_ONLY_MERGE_GAP_SECONDS:
        return False

    if not _ocr_scores_compatible(group, candidate):
        return False

    return True


def _should_merge(group: list[Candidate], candidate: Candidate, gap: float, max_duration: float) -> bool:
    if group[0].video_id != candidate.video_id:
        return False

    group_modalities = {item.modality for item in group}

    # OCR-only 使用更严格的帧级合并规则。
    # 不再使用全局 merge_gap=2，避免弱命中拖长结果时间段。
    if candidate.modality == "ocr" and group_modalities == {"ocr"}:
        return _should_merge_ocr_only(group, candidate)

    group_start = min(item.start_time for item in group)
    group_end = max(item.end_time for item in group)
    merged_start = min(group_start, candidate.start_time)
    merged_end = max(group_end, candidate.end_time)
    if merged_end - merged_start > max_duration:
        return False

    overlaps = candidate.start_time < group_end and candidate.end_time > group_start
    near = candidate.start_time <= group_end + gap

    # Visual buckets are already the display granularity. Do not chain adjacent
    # visual-only hits into a full-video result; merge them only when another
    # modality anchors the same moment, or when intervals genuinely overlap.
    if candidate.modality == "visual" and group_modalities == {"visual"}:
        return overlaps
    if candidate.modality == "visual" or group_modalities == {"visual"}:
        return overlaps or (near and bool(group_modalities - {"visual"}))
    return near


def _groups(candidates: list[Candidate], gap: float, max_duration: float = 15) -> list[list[Candidate]]:
    groups: list[list[Candidate]] = []
    for candidate in sorted(candidates, key=lambda item: (item.video_id, item.start_time, item.end_time)):
        if groups and _should_merge(groups[-1], candidate, gap, max_duration):
            groups[-1].append(candidate)
            continue
        groups.append([candidate])
    return groups


_ASR_LEXICAL_RESERVE_POOL_SIZE = 50
_ASR_LEXICAL_RESERVE_MIN_SCORE = 0.50
_ASR_LEXICAL_RESERVE_INITIAL_PRIMARY = 3
_ASR_LEXICAL_RESERVE_PRIMARY_RUN = 8


def _asr_result_lexical_score(result: SearchResult) -> float:
    return max(
        (
            float(item.get("lexical_score") or 0.0)
            for item in result.evidence
            if item.get("modality") == "asr"
        ),
        default=0.0,
    )


def _reserve_asr_lexical_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    """Reserve sparse result slots for strong lexical hits without rewriting confidence scores."""
    above = [result for result in results if result.above_threshold]
    below = [result for result in results if not result.above_threshold]
    pool_size = max(_ASR_LEXICAL_RESERVE_POOL_SIZE, limit)
    primary = above[:pool_size]
    lexical = sorted(
        (
            result
            for result in above
            if _asr_result_lexical_score(result) >= _ASR_LEXICAL_RESERVE_MIN_SCORE
        ),
        key=lambda result: (_asr_result_lexical_score(result), result.score),
        reverse=True,
    )[:pool_size]
    if not lexical:
        return results

    reranked: list[SearchResult] = []
    emitted: set[int] = set()
    primary_position = 0
    lexical_position = 0

    def emit(result: SearchResult) -> bool:
        identity = id(result)
        if identity in emitted:
            return False
        emitted.add(identity)
        reranked.append(result)
        return True

    while (
        primary_position < len(primary)
        and primary_position < _ASR_LEXICAL_RESERVE_INITIAL_PRIMARY
    ):
        emit(primary[primary_position])
        primary_position += 1

    while primary_position < len(primary) or lexical_position < len(lexical):
        while lexical_position < len(lexical):
            candidate = lexical[lexical_position]
            lexical_position += 1
            if emit(candidate):
                break

        taken = 0
        while (
            primary_position < len(primary)
            and taken < _ASR_LEXICAL_RESERVE_PRIMARY_RUN
        ):
            candidate = primary[primary_position]
            primary_position += 1
            if emit(candidate):
                taken += 1

    remaining_above = [result for result in above if id(result) not in emitted]
    return reranked + remaining_above + below


def _ocr_semantic_arrays(data) -> tuple[np.ndarray | None, np.ndarray | None]:
    embeddings = data["embeddings"].astype(np.float32) if "embeddings" in data.files else None
    if embeddings is not None and (embeddings.ndim != 2 or not embeddings.shape[0] or not embeddings.shape[1]):
        embeddings = None
    indices = (
        data["embedding_frame_indices"].astype(np.int32)
        if "embedding_frame_indices" in data.files
        else None
    )
    if embeddings is not None and indices is None:
        raise ValueError("ocr v3 索引缺少 embedding_frame_indices，请重跑 OCR 索引")
    if embeddings is not None and len(indices) != embeddings.shape[0]:
        raise ValueError("ocr v3 semantic 数组长度不一致，请重跑 OCR 索引")
    return embeddings, indices


def _fuse_candidate_groups(
    candidates: list[Candidate],
    videos: list[dict],
    merge_gap: float,
    max_result_seconds: float,
) -> list[SearchResult]:
    names = {video["id"]: video["name"] for video in videos}
    weights = {"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}
    results = []
    for group in _groups(candidates, merge_gap, max_result_seconds):
        best_by_modality = {}
        for item in group:
            best_by_modality[item.modality] = max(best_by_modality.get(item.modality, -1), item.score)
        denominator = sum(weights.get(name, 1) for name in best_by_modality)
        score = sum(weights.get(name, 1) * value for name, value in best_by_modality.items()) / denominator
        ranked = sorted(group, key=lambda value: value.score, reverse=True)
        best_ms = next((item.best_ms for item in ranked if item.best_ms is not None), None)
        video_id = group[0].video_id
        group_start = min(item.start_time for item in group)
        if best_ms is None:
            best_ms = max(0, round(group_start * 1000))
        group_decisions = {item.decision for item in group}
        decision = next(
            (
                name
                for name in (
                    "strong", "fuzzy", "fallback", "absolute_hit", "semantic_lexical_hit",
                    "semantic_hit", "lexical_hit", "weak",
                )
                if name in group_decisions
            ),
            "hit",
        )
        start_time = min(item.start_time for item in group)
        end_time = max(item.end_time for item in group)
        results.append(SearchResult(
            video_id=video_id,
            video_name=names.get(video_id, video_id),
            start_time=start_time,
            end_time=end_time,
            score=score,
            modalities=sorted(best_by_modality),
            thumbnail_url=f"/api/videos/{video_id}/frame?time={best_ms / 1000:.3f}",
            media_url=f"/api/videos/{video_id}/media",
            clip_url=f"/api/videos/{video_id}/clip?start={start_time:.3f}&end={end_time:.3f}",
            decision=decision,
            above_threshold=any(item.above_threshold for item in group),
            evidence=[_serialize_evidence(item) for item in group],
        ))
    results.sort(key=lambda item: (item.above_threshold, item.score), reverse=True)
    return results


class SearchEngine:
    def __init__(self, settings: Settings, catalog: Catalog):
        self.settings = settings
        self.catalog = catalog
        self._clip_encoders = {}
        self._face_encoder = None
        self._text_encoders = {}

    def _clip(self, visual_model: str | None = None):
        from app.indexing.visual import ClipEncoder, normalize_visual_model, resolve_device

        model_key = normalize_visual_model(visual_model or self.settings.visual_model)
        if model_key not in self._clip_encoders:
            device = resolve_device(self.settings.npu_enabled, self.settings.npu_device_id, self.settings.cuda_enabled)
            self._clip_encoders[model_key] = ClipEncoder(
                self.settings.clip_model,
                self.settings.clip_pretrained,
                device,
                visual_model=model_key,
                model_cache_dir=str(self.settings.resolve_path(self.settings.visual_hf_cache_dir)),
            )
        return self._clip_encoders[model_key]

    def _face(self):
        if self._face_encoder is None:
            from app.indexing.faces import FaceEncoder

            self._face_encoder = FaceEncoder(
                self.settings.face_model,
                "cpu",
                0,
                str(self.settings.app_model_dir / "insightface"),
                self.settings.face_ort_intra_op_threads,
                self.settings.face_ort_inter_op_threads,
            )
        return self._face_encoder

    def _encode_asr_query(self, text: str, model_name: str) -> np.ndarray:
        from app.indexing.text_semantic import TextEmbeddingEncoder, resolve_text_embedding_device

        device = resolve_text_embedding_device(self.settings.asr_semantic_device, self.settings.cuda_enabled)
        key = (model_name, device)
        if key not in self._text_encoders:
            self._text_encoders[key] = TextEmbeddingEncoder(
                model_name,
                self.settings.app_model_dir / "text-embeddings",
                device,
                local_files_only=self.settings.asr_semantic_local_files_only,
            )
        return self._text_encoders[key].encode([text], batch_size=1)[0]

    def _selected_videos(self, video_ids: list[str] | None) -> list[dict]:
        videos = self.catalog.list_videos()
        if not video_ids:
            return videos
        allowed = set(video_ids)
        return [video for video in videos if video["id"] in allowed]

    def _resolve_face_query(self, text: str | None, image_path: str | None) -> np.ndarray | None:
        if image_path:
            return self._face().encode_reference(image_path)
        if not text:
            return None
        entity = self.catalog.find_entity_in_text(text)
        if entity and entity.get("embedding_path") and Path(entity["embedding_path"]).exists():
            return np.load(entity["embedding_path"])["embedding"]
        return None

    def _visual_for_video(
        self,
        video: dict,
        text: str | None,
        image_path: str | None,
        alpha: float,
        profile: str,
        limit: int,
        visual_queries: dict[str, np.ndarray],
        visual_subqueries: list[str] | None,
    ) -> list[Candidate]:
        index_dir = self.settings.index_dir / video["id"]
        manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "visual")
        with np.load(index_file, allow_pickle=False) as data:
            visual_model = str(channel_manifest.get("model_key") or self.settings.visual_model)
            if visual_model not in visual_queries:
                query_texts: list[str | None] = (
                    list(dict.fromkeys(visual_subqueries or []))
                    if text and visual_subqueries
                    else [text]
                )
                visual_queries[visual_model] = np.stack([
                    self._clip(visual_model).encode_query(query_text, image_path, alpha)
                    for query_text in query_texts
                ])
            return _visual_candidates(
                data,
                visual_queries[visual_model],
                video["id"],
                int(manifest.get("duration_ms") or round(float(video.get("duration") or 0) * 1000)),
                int(manifest.get("segment_ms") or round(float(self.settings.visual_segment_seconds) * 1000)),
                profile,
                limit,
                str(channel_manifest.get("segment_strategy") or "fixed"),
            )

    def _face_for_video(self, video: dict, face_query: np.ndarray, limit: int) -> list[Candidate]:
        index_dir = self.settings.index_dir / video["id"]
        _manifest, _channel_manifest, index_file = _channel_manifest_for(video, index_dir, "face")
        with np.load(index_file, allow_pickle=False) as data:
            return _face_candidates(data, face_query, video["id"], limit, 0.35)

    def _semantic_query(self, text: str, channel_manifest: dict, embeddings: np.ndarray | None) -> np.ndarray | None:
        if embeddings is None:
            return None
        model_name = str(channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model)
        try:
            return self._encode_asr_query(text, model_name)
        except Exception:
            return None

    def _asr_for_video(self, video: dict, text: str, limit: int) -> list[Candidate]:
        index_dir = self.settings.index_dir / video["id"]
        _manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "asr")
        with np.load(index_file, allow_pickle=False) as data:
            embeddings, indices = _semantic_arrays(data)
            return _asr_candidates(
                _asr_chunks_from_npz(data),
                text,
                video["id"],
                limit,
                semantic_embeddings=embeddings,
                embedding_chunk_indices=indices,
                semantic_query=self._semantic_query(text, channel_manifest, embeddings),
            )

    def _ocr_for_video(self, video: dict, text: str, limit: int) -> list[Candidate]:
        index_dir = self.settings.index_dir / video["id"]
        _manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "ocr")
        if int(channel_manifest.get("schema_version") or 0) != 3:
            raise ValueError(f"视频 {video.get('name') or video['id']} 的 OCR 索引不是重构后的 v3，请重跑 OCR 索引")
        with np.load(index_file, allow_pickle=False) as data:
            embeddings, indices = _ocr_semantic_arrays(data)
            semantic_query = None
            if embeddings is not None:
                model_name = str(channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model)
                semantic_query = self._encode_asr_query(text, model_name)
            return _asr_candidates(
                _ocr_chunks_from_npz(data),
                text,
                video["id"],
                limit,
                modality="ocr",
                semantic_embeddings=embeddings,
                embedding_chunk_indices=indices,
                semantic_query=semantic_query,
            )

    def _candidates_for_video(
        self,
        video: dict,
        *,
        text: str | None,
        image_path: str | None,
        modalities: list[str],
        alpha: float,
        limit: int,
        visual_profile: str,
        visual_queries: dict[str, np.ndarray],
        face_query: np.ndarray | None,
        channel_limits: dict[str, int],
        visual_subqueries: list[str] | None,
    ) -> list[Candidate]:
        candidates = []
        indexed = set(video.get("indexed_modalities") or [])
        if "visual" in modalities and bool(text or image_path) and "visual" in indexed:
            candidates.extend(self._visual_for_video(
                video,
                text,
                image_path,
                alpha,
                visual_profile,
                channel_limits["visual"],
                visual_queries,
                visual_subqueries,
            ))
        if face_query is not None and "face" in indexed:
            candidates.extend(self._face_for_video(video, face_query, channel_limits["face"]))
        if "asr" in modalities and text and "asr" in indexed:
            candidates.extend(self._asr_for_video(video, text, channel_limits["asr"]))
        if "ocr" in modalities and text and "ocr" in indexed:
            candidates.extend(self._ocr_for_video(video, text, channel_limits["ocr"]))
        return candidates

    def _milvus_candidates_for_video(
        self,
        video: dict,
        *,
        text: str | None,
        image_path: str | None,
        modalities: list[str],
        alpha: float,
        visual_profile: str,
        visual_queries: dict[str, np.ndarray],
        face_query: np.ndarray | None,
        channel_limits: dict[str, int],
        visual_subqueries: list[str] | None,
    ) -> list[Candidate]:
        from app.indexing.milvus_client import get_milvus_client
        from app.indexing.milvus_search import (
            milvus_asr_candidates,
            milvus_face_candidates,
            milvus_ocr_candidates,
            milvus_visual_candidates,
        )

        client = get_milvus_client()
        video_id = video["id"]
        index_dir = self.settings.index_dir / video_id
        indexed = set(video.get("indexed_modalities") or [])
        candidates: list[Candidate] = []
        if "visual" in modalities and bool(text or image_path) and "visual" in indexed:
            manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "visual"
            )
            visual_model = str(channel_manifest.get("model_key") or self.settings.visual_model)
            if visual_model not in visual_queries:
                query_texts: list[str | None] = (
                    list(dict.fromkeys(visual_subqueries or []))
                    if text and visual_subqueries
                    else [text]
                )
                visual_queries[visual_model] = np.stack([
                    self._clip(visual_model).encode_query(query_text, image_path, alpha)
                    for query_text in query_texts
                ])
            candidates.extend(milvus_visual_candidates(
                client,
                video_id,
                visual_queries[visual_model],
                int(manifest.get("duration_ms") or round(float(video.get("duration") or 0) * 1000)),
                int(manifest.get("segment_ms") or round(float(self.settings.visual_segment_seconds) * 1000)),
                visual_profile,
                channel_limits["visual"],
            ))
        if face_query is not None and "face" in indexed:
            candidates.extend(milvus_face_candidates(
                client, video_id, face_query, channel_limits["face"], 0.35
            ))
        if "asr" in modalities and text and "asr" in indexed:
            _manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "asr"
            )
            model_name = str(
                channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model
            )
            semantic_query = None
            try:
                semantic_query = self._encode_asr_query(text, model_name)
            except Exception as exc:
                logger.warning("Milvus ASR semantic query unavailable: %s", exc)
            candidates.extend(milvus_asr_candidates(
                client, video_id, text, semantic_query, channel_limits["asr"]
            ))
        if "ocr" in modalities and text and "ocr" in indexed:
            _manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "ocr"
            )
            model_name = str(
                channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model
            )
            semantic_query = None
            try:
                semantic_query = self._encode_asr_query(text, model_name)
            except Exception as exc:
                logger.warning("Milvus OCR semantic query unavailable: %s", exc)
            candidates.extend(milvus_ocr_candidates(
                client, video_id, text, semantic_query, channel_limits["ocr"]
            ))
        return candidates

    def search(
        self,
        text: str | None,
        image_path: str | None,
        modalities: list[str],
        video_ids: list[str] | None = None,
        alpha: float = 0.5,
        limit: int = 24,
        merge_gap: float = 2,
        max_result_seconds: float = 15,
        visual_profile: str = "balanced",
        channel_limits: dict[str, int] | None = None,
        visual_subqueries: list[str] | None = None,
    ) -> list[dict]:
        if visual_profile not in {"recall", "balanced", "precision"}:
            raise ValueError("visual_profile 必须是 recall、balanced 或 precision")
        videos = self._selected_videos(video_ids)
        candidates: list[Candidate] = []
        requested_channel_limits = channel_limits or {}
        resolved_channel_limits = {
            name: max(1, int(requested_channel_limits.get(name, limit * 3)))
            for name in ("visual", "face", "asr", "ocr")
        }
        visual_queries: dict[str, np.ndarray] = {}
        face_query = self._resolve_face_query(text, image_path) if "face" in modalities else None
        from app.indexing.milvus_flags import (
            milvus_fallback_enabled,
            milvus_shadow_compare_enabled,
            should_use_milvus_for_video,
        )
        from app.indexing.milvus_search import shadow_compare_log

        for video in videos:
            use_milvus = should_use_milvus_for_video(video["id"])
            shadow = milvus_shadow_compare_enabled()
            npz_candidates: list[Candidate] | None = None
            if not use_milvus or shadow:
                npz_candidates = self._candidates_for_video(
                    video,
                    text=text,
                    image_path=image_path,
                    modalities=modalities,
                    alpha=alpha,
                    limit=limit,
                    visual_profile=visual_profile,
                    visual_queries=visual_queries,
                    face_query=face_query,
                    channel_limits=resolved_channel_limits,
                    visual_subqueries=visual_subqueries,
                )
            milvus_candidates: list[Candidate] | None = None
            if use_milvus or shadow:
                try:
                    milvus_candidates = self._milvus_candidates_for_video(
                        video,
                        text=text,
                        image_path=image_path,
                        modalities=modalities,
                        alpha=alpha,
                        visual_profile=visual_profile,
                        visual_queries=visual_queries,
                        face_query=face_query,
                        channel_limits=resolved_channel_limits,
                        visual_subqueries=visual_subqueries,
                    )
                except Exception as exc:
                    if use_milvus and not milvus_fallback_enabled():
                        raise
                    logger.warning(
                        "Milvus search failed for video=%s; using NPZ fallback: %s",
                        video["id"],
                        exc,
                    )
                    if npz_candidates is None:
                        npz_candidates = self._candidates_for_video(
                            video,
                            text=text,
                            image_path=image_path,
                            modalities=modalities,
                            alpha=alpha,
                            limit=limit,
                            visual_profile=visual_profile,
                            visual_queries=visual_queries,
                            face_query=face_query,
                            channel_limits=resolved_channel_limits,
                            visual_subqueries=visual_subqueries,
                        )
            if shadow and npz_candidates is not None and milvus_candidates is not None:
                for modality in modalities:
                    shadow_compare_log(
                        video["id"],
                        modality,
                        [item for item in npz_candidates if item.modality == modality],
                        [item for item in milvus_candidates if item.modality == modality],
                    )
            candidates.extend(
                milvus_candidates
                if use_milvus and milvus_candidates is not None
                else (npz_candidates or [])
            )
        results = _fuse_candidate_groups(candidates, videos, merge_gap, max_result_seconds)
        if set(modalities) == {"asr"} and text:
            results = _reserve_asr_lexical_results(results, limit)
        result_limit = 500 if visual_profile == "recall" else limit
        return [item.to_dict() for item in results[:result_limit]]
