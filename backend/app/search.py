from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.db import Catalog
from app.indexing.common import normalize
from app.indexing.manifest import require_channel_manifest
from app.indexing.asr_text import normalize_search_text
from app.settings import Settings


@dataclass
class Candidate:
    video_id: str
    start_time: float
    end_time: float
    score: float
    modality: str
    thumbnail: str | None = None
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
    required = {"frame_embeddings", "frame_times_ms", "segment_frame_offsets"}
    if not required.issubset(set(data.files)):
        raise ValueError("visual v3 索引缺少必要数组，请重跑 visual 索引")
    frame_embeddings = np.asarray(data["frame_embeddings"], dtype=np.float32)
    frame_times_ms = data["frame_times_ms"].astype(np.int32)
    offsets = data["segment_frame_offsets"].astype(np.int32)
    if frame_embeddings.ndim != 2 or len(frame_embeddings) != len(frame_times_ms):
        raise ValueError("visual v3 索引数组长度不一致，请重跑 visual 索引")
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(frame_times_ms) or np.any(np.diff(offsets) < 0):
        raise ValueError("visual v3 segment_frame_offsets 无效，请重跑 visual 索引")
    if not len(frame_embeddings):
        return []
    segment_times_ms = None
    if "segment_times_ms" in data.files:
        segment_times_ms = data["segment_times_ms"].astype(np.int32)
        if segment_times_ms.shape != (len(offsets) - 1, 2):
            raise ValueError("visual v3 segment_times_ms 无效，请重跑 visual 索引")
        if np.any(segment_times_ms[:, 1] < segment_times_ms[:, 0]):
            raise ValueError("visual v3 segment_times_ms 时间范围无效，请重跑 visual 索引")

    query = normalize(query)
    frame_scores = frame_embeddings @ query
    segment_ids: list[int] = []
    raw_scores: list[float] = []
    top3_scores: list[float] = []
    mean_scores: list[float] = []
    best_times_ms: list[int] = []
    for segment_id in range(len(offsets) - 1):
        start, end = int(offsets[segment_id]), int(offsets[segment_id + 1])
        if start == end:
            continue
        bucket_scores = frame_scores[start:end]
        order = np.argsort(bucket_scores)[::-1]
        top_values = bucket_scores[order]
        segment_ids.append(segment_id)
        raw_scores.append(float(top_values[0]))
        top3_scores.append(float(np.mean(top_values[:min(3, len(top_values))])))
        mean_scores.append(float(np.mean(bucket_scores)))
        best_times_ms.append(int(frame_times_ms[start + int(order[0])]))

    if not raw_scores:
        return []
    raw_values = np.asarray(raw_scores, dtype=np.float32)
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
        if reliable:
            if z_score >= 2.0 or percentile >= 0.975:
                decision, above = "strong", True
            elif percentile >= 0.80:
                qualifies = not (
                    (profile == "balanced" and not (z_score >= 1.0 or percentile >= 0.90))
                    or profile == "precision"
                )
                decision, above = ("fuzzy", True) if qualifies else ("weak", False)
            else:
                decision, above = "weak", False
            detail = (
                f"visual score={raw_score:.3f} · rank_score={ranking_score:.3f}"
                f" · percentile={percentile * 100:.1f}% · robust_z={z_score:.2f}"
            )
        else:
            if local_index in fallback_indices:
                decision, above = "fallback", True
            else:
                decision, above = "weak", False
            detail = (
                f"visual score={raw_score:.3f} · rank_score={ranking_score:.3f}"
                f" · distribution fallback (n={len(raw_values)})"
            )

        top3 = float(top3_scores[local_index])
        mean = float(mean_scores[local_index])
        best_ms = int(best_times_ms[local_index])
        detail += f" · best_frame={best_ms / 1000:.2f}s · top1={raw_score:.3f} · top3={top3:.3f} · mean={mean:.3f}"
        if segment_times_ms is not None:
            start_ms, end_ms = [int(value) for value in segment_times_ms[segment_id]]
            time_source = "explicit"
        else:
            start_ms = segment_id * segment_ms
            end_ms = min((segment_id + 1) * segment_ms, duration_ms or (segment_id + 1) * segment_ms)
            time_source = "fixed"
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=ranking_score,
            modality="visual",
            thumbnail=f"visual_{segment_id:06d}.jpg",
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
            thumbnail=f"face_{index:06d}.jpg",
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
    semantic_scores = np.zeros(len(chunks), dtype=np.float32)
    semantic_cosines = np.full(len(chunks), np.nan, dtype=np.float32)
    semantic_available = (
        semantic_embeddings is not None
        and embedding_chunk_indices is not None
        and semantic_query is not None
        and len(semantic_embeddings) > 0
        and len(embedding_chunk_indices) > 0
    )

    semantic_top_indices: set[int] = set()
    if semantic_available:
        cosines = semantic_embeddings @ normalize(semantic_query)
        distribution = robust_distribution(cosines)
        percentiles = distribution["percentiles"]
        for local_index, chunk_index in enumerate(embedding_chunk_indices):
            if 0 <= int(chunk_index) < len(chunks):
                cosine = float(cosines[local_index])
                percentile = float(percentiles[local_index]) if len(percentiles) else 0.0
                semantic_cosines[int(chunk_index)] = cosine
                semantic_scores[int(chunk_index)] = (
                    0.7 * asr_semantic_confidence(cosine)
                    + 0.3 * percentile
                )
        order = np.argsort(semantic_scores)[::-1]
        semantic_top_indices = set(int(index) for index in order[:min(len(order), limit)])

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
        semantic_hit = semantic >= 0.55
        lexical_hit = lexical >= 0.25
        above = semantic_hit or lexical_hit
        if semantic_hit and lexical_hit:
            decision = "semantic_lexical_hit"
        elif semantic_hit:
            decision = "semantic_hit"
        elif lexical_hit:
            decision = "lexical_hit"
        else:
            decision = "weak"

        detail = str(chunk.get("text", "")).strip()
        metrics = [f"lexical={lexical:.3f}"]
        if semantic_cosine is not None:
            metrics.append(f"semantic={semantic:.3f}")
            metrics.append(f"semantic_cosine={semantic_cosine:.3f}")
        else:
            metrics.append("semantic=unavailable")
        evidence = f"{detail} · {' · '.join(metrics)}"
        start_ms = int(chunk.get("start_ms", round(float(chunk.get("start_time", 0)) * 1000)))
        end_ms = int(chunk.get("end_ms", round(float(chunk.get("end_time", 0)) * 1000)))
        features = {
            "lexical_score": lexical,
            "semantic_score": semantic if semantic_cosine is not None else None,
            "semantic_cosine": semantic_cosine,
        }
        if "score" in chunk:
            features[f"{modality}_score"] = float(chunk["score"])
        candidates.append(Candidate(
            video_id=video_id,
            start_time=_seconds(start_ms),
            end_time=_seconds(end_ms),
            score=score,
            modality=modality,
            thumbnail=str(chunk.get("thumbnail") or "") or None,
            evidence=evidence if above else evidence + " · 低于阈值",
            raw_score=score,
            decision=decision,
            above_threshold=above,
            lexical_score=lexical,
            semantic_score=semantic if semantic_cosine is not None else None,
            semantic_cosine=semantic_cosine,
            unit_type="chunk",
            unit_id=int(chunk.get("chunk_id", index)),
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


def _ocr_chunks_from_npz(data) -> list[dict]:
    required = {"chunk_times_ms", "box_chunk_indices", "box_texts", "box_scores", "boxes"}
    if not required.issubset(set(data.files)):
        raise ValueError("ocr v3 索引缺少必要数组，请重跑 OCR 索引")
    times = data["chunk_times_ms"].astype(np.int32)
    box_chunk_indices = data["box_chunk_indices"].astype(np.int32)
    box_texts = _decode_text_array(data["box_texts"])
    box_scores = np.asarray(data["box_scores"], dtype=np.float32)
    if times.ndim != 2 or times.shape[1] != 3:
        raise ValueError("ocr v3 chunk_times_ms 必须是 [num_chunks, 3]，请重跑 OCR 索引")
    if not (len(box_chunk_indices) == len(box_texts) == len(box_scores)):
        raise ValueError("ocr v3 box 数组长度不一致，请重跑 OCR 索引")
    chunks: list[dict] = []
    for chunk_id, row in enumerate(times):
        indices = np.flatnonzero(box_chunk_indices == chunk_id)
        texts = [box_texts[int(index)] for index in indices if box_texts[int(index)].strip()]
        scores = box_scores[indices] if len(indices) else np.empty((0,), dtype=np.float32)
        chunks.append({
            "chunk_id": chunk_id,
            "start_ms": int(row[0]),
            "end_ms": int(row[1]),
            "frame_ms": int(row[2]),
            "text": " ".join(texts),
            "score": float(scores.max()) if len(scores) else 0.0,
            "thumbnail": f"ocr_{chunk_id:06d}.jpg",
        })
    return chunks


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


def _should_merge(group: list[Candidate], candidate: Candidate, gap: float, max_duration: float) -> bool:
    if group[0].video_id != candidate.video_id:
        return False
    group_start = min(item.start_time for item in group)
    group_end = max(item.end_time for item in group)
    merged_start = min(group_start, candidate.start_time)
    merged_end = max(group_end, candidate.end_time)
    if merged_end - merged_start > max_duration:
        return False

    group_modalities = {item.modality for item in group}
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
                self.settings.face_model, "cpu", 0, str(self.settings.app_model_dir / "insightface")
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
    ) -> list[dict]:
        if visual_profile not in {"recall", "balanced", "precision"}:
            raise ValueError("visual_profile 必须是 recall、balanced 或 precision")
        videos = self.catalog.list_videos()
        if video_ids:
            allowed = set(video_ids)
            videos = [video for video in videos if video["id"] in allowed]
        candidates: list[Candidate] = []

        visual_queries: dict[str, np.ndarray] = {}
        wants_visual = "visual" in modalities and bool(text or image_path)

        face_query = None
        if "face" in modalities:
            if image_path:
                face_query = self._face().encode_reference(image_path)
            elif text:
                entity = self.catalog.find_entity_in_text(text)
                if entity and entity.get("embedding_path") and Path(entity["embedding_path"]).exists():
                    face_query = np.load(entity["embedding_path"])["embedding"]

        for video in videos:
            index_dir = self.settings.index_dir / video["id"]
            indexed_modalities = set(video.get("indexed_modalities") or [])
            if wants_visual and "visual" in indexed_modalities:
                manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "visual")
                with np.load(index_file, allow_pickle=False) as data:
                    visual_model = str(channel_manifest.get("model_key") or self.settings.visual_model)
                    if visual_model not in visual_queries:
                        visual_queries[visual_model] = self._clip(visual_model).encode_query(text, image_path, alpha)
                    visual_query = visual_queries[visual_model]
                    candidates.extend(_visual_candidates(
                        data,
                        visual_query,
                        video["id"],
                        int(manifest.get("duration_ms") or round(float(video.get("duration") or 0) * 1000)),
                        int(manifest.get("segment_ms") or round(float(self.settings.visual_segment_seconds) * 1000)),
                        visual_profile,
                        limit * 3,
                        str(channel_manifest.get("segment_strategy") or "fixed"),
                    ))
            if face_query is not None and "face" in indexed_modalities:
                _manifest, _channel_manifest, index_file = _channel_manifest_for(video, index_dir, "face")
                with np.load(index_file, allow_pickle=False) as data:
                    candidates.extend(_face_candidates(data, face_query, video["id"], limit * 3, 0.35))
            if "asr" in modalities and text and "asr" in indexed_modalities:
                _manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "asr")
                with np.load(index_file, allow_pickle=False) as data:
                    semantic_embeddings, embedding_chunk_indices = _semantic_arrays(data)
                    semantic_query = None
                    if semantic_embeddings is not None:
                        model_name = str(channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model)
                        try:
                            semantic_query = self._encode_asr_query(text, model_name)
                        except Exception:
                            semantic_query = None
                    candidates.extend(_asr_candidates(
                        _asr_chunks_from_npz(data),
                        text,
                        video["id"],
                        limit * 3,
                        semantic_embeddings=semantic_embeddings,
                        embedding_chunk_indices=embedding_chunk_indices,
                        semantic_query=semantic_query,
                    ))
            if "ocr" in modalities and text and "ocr" in indexed_modalities:
                _manifest, channel_manifest, index_file = _channel_manifest_for(video, index_dir, "ocr")
                with np.load(index_file, allow_pickle=False) as data:
                    semantic_embeddings, embedding_chunk_indices = _semantic_arrays(data)
                    semantic_query = None
                    if semantic_embeddings is not None:
                        model_name = str(channel_manifest.get("semantic_model_key") or self.settings.asr_semantic_model)
                        try:
                            semantic_query = self._encode_asr_query(text, model_name)
                        except Exception:
                            semantic_query = None
                    candidates.extend(_asr_candidates(
                        _ocr_chunks_from_npz(data),
                        text,
                        video["id"],
                        limit * 3,
                        modality="ocr",
                        semantic_embeddings=semantic_embeddings,
                        embedding_chunk_indices=embedding_chunk_indices,
                        semantic_query=semantic_query,
                    ))

        names = {video["id"]: video["name"] for video in videos}
        # Each modality score is calibrated to a comparable [0,1] scale, so these
        # weights express modality priority, not scale fixes.
        weights = {"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}
        results = []
        for group in _groups(candidates, merge_gap, max_result_seconds):
            best_by_modality = {}
            for item in group:
                best_by_modality[item.modality] = max(best_by_modality.get(item.modality, -1), item.score)
            denominator = sum(weights.get(name, 1) for name in best_by_modality)
            score = sum(weights.get(name, 1) * value for name, value in best_by_modality.items()) / denominator
            best_thumbnail = next((item.thumbnail for item in sorted(group, key=lambda value: value.score, reverse=True) if item.thumbnail), None)
            video_id = group[0].video_id
            group_decisions = {item.decision for item in group}
            decision = next(
                (name for name in ("strong", "fuzzy", "fallback", "absolute_hit", "semantic_lexical_hit", "semantic_hit", "lexical_hit", "weak") if name in group_decisions),
                "hit",
            )
            group_above = any(item.above_threshold for item in group)
            results.append(SearchResult(
                video_id=video_id,
                video_name=names.get(video_id, video_id),
                start_time=min(item.start_time for item in group),
                end_time=max(item.end_time for item in group),
                score=score,
                modalities=sorted(best_by_modality),
                thumbnail_url=(f"/api/thumbnails/{video_id}/{best_thumbnail}" if best_thumbnail else None),
                media_url=f"/api/videos/{video_id}/media",
                clip_url=f"/api/videos/{video_id}/clip?start={min(item.start_time for item in group):.3f}&end={max(item.end_time for item in group):.3f}",
                decision=decision,
                above_threshold=group_above,
                evidence=[_serialize_evidence(item) for item in group],
            ))
        # Above-threshold results first (each block sorted by score), so the UI can
        # draw a single divider where matches start dropping below threshold.
        results.sort(key=lambda item: (item.above_threshold, item.score), reverse=True)
        result_limit = 500 if visual_profile == "recall" else limit
        return [item.to_dict() for item in results[:result_limit]]
