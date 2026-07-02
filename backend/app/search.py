from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.db import Catalog
from app.indexing.common import normalize
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


_CHINESE_VARIANT_FOLD = str.maketrans({
    "來": "来", "這": "这", "邊": "边", "們": "们", "癡": "痴", "魚": "鱼",
    "電": "电", "資": "资", "聲": "声", "語": "语", "話": "话", "說": "说",
    "聽": "听", "見": "见", "個": "个", "時": "时", "過": "过", "兩": "两",
    "錢": "钱", "買": "买", "賣": "卖", "樓": "楼", "請": "请", "沒": "没",
    "麼": "么", "麵": "面", "點": "点", "對": "对", "樣": "样", "裡": "里",
    "裏": "里", "後": "后", "會": "会", "給": "给", "還": "还", "為": "为",
    "嗎": "吗", "著": "着", "兒": "儿", "間": "间", "場": "场", "開": "开",
    "關": "关", "門": "门", "車": "车", "長": "长", "當": "当", "從": "从",
    "愛": "爱", "認": "认", "識": "识", "寫": "写", "讀": "读", "問": "问",
    "讓": "让", "應": "应", "該": "该", "經": "经", "難": "难", "離": "离",
    "實": "实", "現": "现", "發": "发", "國": "国", "頭": "头", "歲": "岁",
    "萬": "万", "與": "与", "誰": "谁", "妳": "你",
})


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_CHINESE_VARIANT_FOLD)
    return "".join(character for character in folded if character.isalnum())


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


def _top_vectors(data, query: np.ndarray, modality: str, video_id: str, limit: int, threshold: float) -> list[Candidate]:
    embeddings = data["embeddings"]
    if not len(embeddings):
        return []
    scores = embeddings @ normalize(query)
    candidates = []
    # Recall everything sorted by score; tag whether each clears the threshold
    # instead of hard-dropping, so the UI can surface borderline matches marked
    # as low-confidence rather than silently discarding them.
    for index in np.argsort(scores)[::-1]:
        if len(candidates) >= limit:
            break
        cosine = float(scores[index])
        above = cosine >= threshold
        confidence = face_confidence(cosine) if modality == "face" else cosine
        thumbnail = str(data["thumbnails"][index]) if "thumbnails" in data.files else ""
        detail = f"{modality} cosine={cosine:.3f} · confidence={confidence * 100:.1f}%"
        candidates.append(Candidate(
            video_id=video_id,
            start_time=float(data["start_times"][index]),
            end_time=float(data["end_times"][index]),
            score=confidence,
            modality=modality,
            thumbnail=thumbnail or None,
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=cosine,
            decision="absolute_hit" if above else "weak",
            above_threshold=above,
        ))
    return candidates


def _visual_candidates(
    data, query: np.ndarray, video_id: str, profile: str = "balanced", limit: int = 72
) -> list[Candidate]:
    embeddings = data["embeddings"]
    if not len(embeddings):
        return []
    query = normalize(query)
    mean_scores = embeddings @ query
    raw_scores = mean_scores
    top1_scores: np.ndarray | None = None
    top3_scores: np.ndarray | None = None
    best_times: np.ndarray | None = None

    # visual.npz schema v2 keeps frame-level CLIP embeddings in addition to the
    # segment mean. A segment can now be recalled by a short semantic peak
    # (top1/top3 frame MaxSim) instead of being diluted by averaging every frame in
    # the 5s bucket. Older indexes do not have these arrays and fall back to the
    # original segment-mean score.
    if {"frame_embeddings", "frame_segment_ids", "frame_times"}.issubset(set(data.files)):
        frame_scores = data["frame_embeddings"] @ query
        frame_segment_ids = data["frame_segment_ids"]
        frame_times = data["frame_times"]
        segment_ids = data["segment_ids"] if "segment_ids" in data.files else np.arange(len(embeddings), dtype=np.int32)
        top1_values = np.zeros(len(embeddings), dtype=np.float32)
        top3_values = np.zeros(len(embeddings), dtype=np.float32)
        best_time_values = np.full(len(embeddings), np.nan, dtype=np.float32)
        for segment_index, segment_id in enumerate(segment_ids):
            indices = np.flatnonzero(frame_segment_ids == segment_id)
            if not len(indices):
                top1_values[segment_index] = float(mean_scores[segment_index])
                top3_values[segment_index] = float(mean_scores[segment_index])
                continue
            values = frame_scores[indices]
            local_order = np.argsort(values)[::-1]
            ordered_values = values[local_order]
            top1_values[segment_index] = float(ordered_values[0])
            top3_values[segment_index] = float(np.mean(ordered_values[:min(3, len(ordered_values))]))
            best_time_values[segment_index] = float(frame_times[indices[local_order[0]]])
        top1_scores = top1_values
        top3_scores = top3_values
        best_times = best_time_values
        raw_scores = (0.65 * top1_scores) + (0.25 * top3_scores) + (0.10 * mean_scores)

    distribution = robust_distribution(raw_scores)
    z_scores = distribution["z_scores"]
    percentiles = distribution["percentiles"]
    reliable = distribution["reliable"]
    raw_order = np.argsort(raw_scores)[::-1]
    fallback_counts = {"recall": 3, "balanced": 2, "precision": 1}
    fallback_indices = set(int(index) for index in raw_order[:min(len(raw_order), fallback_counts[profile])])
    candidates = []
    cap = 500 if profile == "recall" else limit

    # Recall every bucket sorted by score (capped), tagging each as above/below the
    # profile threshold instead of dropping it, so borderline segments stay visible.
    for index in raw_order:
        if len(candidates) >= cap:
            break
        index = int(index)
        raw_score = float(raw_scores[index])
        z_score = float(z_scores[index])
        percentile = float(percentiles[index])
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
            ranking_score = percentile
            detail = f"visual score={raw_score:.3f} · percentile={percentile * 100:.1f}% · robust_z={z_score:.2f}"
        else:
            if index in fallback_indices:
                decision, above = "fallback", True
            else:
                decision, above = "weak", False
            ranking_score = float(np.clip((raw_score + 1.0) / 2.0, 0, 1))
            detail = f"visual score={raw_score:.3f} · distribution fallback (n={len(raw_scores)})"

        visual_top1 = float(top1_scores[index]) if top1_scores is not None else None
        visual_top3 = float(top3_scores[index]) if top3_scores is not None else None
        visual_mean = float(mean_scores[index])
        best_time = float(best_times[index]) if best_times is not None and not np.isnan(best_times[index]) else None
        if visual_top1 is not None and visual_top3 is not None:
            detail += (
                f" · best_frame={best_time:.2f}s" if best_time is not None else ""
            )
            detail += f" · top1={visual_top1:.3f} · top3={visual_top3:.3f} · mean={visual_mean:.3f}"
        else:
            detail += f" · mean={visual_mean:.3f} · legacy_index"

        thumbnail = str(data["thumbnails"][index]) if "thumbnails" in data.files else ""
        candidates.append(Candidate(
            video_id=video_id,
            start_time=float(data["start_times"][index]),
            end_time=float(data["end_times"][index]),
            score=ranking_score,
            modality="visual",
            thumbnail=thumbnail or None,
            evidence=detail if above else detail + " · 低于阈值",
            raw_score=raw_score,
            robust_z=z_score,
            percentile=percentile,
            decision=decision,
            above_threshold=above,
            distribution_reliable=reliable,
            distribution_median=distribution["median"],
            distribution_mad=distribution["mad"],
            best_time=best_time,
            visual_top1=visual_top1,
            visual_top3=visual_top3,
            visual_mean=visual_mean,
        ))
    return candidates


def _visual_model_key_from_index(data, default_model: str) -> str:
    from app.indexing.visual import normalize_visual_model

    if "visual_model" in data.files:
        return normalize_visual_model(str(data["visual_model"][0]))
    if "model" in data.files:
        label = str(data["model"][0])
        lowered = label.lower()
        if "siglip2" in lowered:
            return "siglip2-so400m-384"
        if "chinese" in lowered:
            return "chinese-clip-vit-b16"
        if "vit-b-16" in lowered or "vit-b/16" in lowered:
            return "openclip-vit-b16"
        if "vit-l-14" in lowered or "vit-l/14" in lowered:
            return "openclip-vit-l14"
        if "vit-b-32" in lowered or "vit-b/32" in lowered:
            return "openclip-vit-b32"
    return normalize_visual_model(default_model)


def _asr_candidates(
    chunks: list[dict],
    query_text: str,
    video_id: str,
    limit: int,
    modality: str = "asr",
    semantic_data=None,
    semantic_query: np.ndarray | None = None,
) -> list[Candidate]:
    if not chunks:
        return []

    lexical_scores = np.asarray([lexical_score(query_text, chunk.get("text", "")) for chunk in chunks], dtype=np.float32)
    semantic_scores = np.zeros(len(chunks), dtype=np.float32)
    semantic_cosines = np.full(len(chunks), np.nan, dtype=np.float32)
    semantic_available = (
        semantic_data is not None
        and semantic_query is not None
        and "embeddings" in semantic_data.files
        and "chunk_indices" in semantic_data.files
        and len(semantic_data["embeddings"]) > 0
    )

    semantic_top_indices: set[int] = set()
    if semantic_available:
        embeddings = semantic_data["embeddings"]
        chunk_indices = semantic_data["chunk_indices"].astype(np.int32)
        cosines = embeddings @ normalize(semantic_query)
        distribution = robust_distribution(cosines)
        percentiles = distribution["percentiles"]
        for local_index, chunk_index in enumerate(chunk_indices):
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
        candidates.append(Candidate(
            video_id=video_id,
            start_time=float(chunk.get("start_time", 0)),
            end_time=float(chunk.get("end_time", 0)),
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
        ))
    return candidates


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
            if wants_visual and (index_dir / "visual.npz").exists():
                with np.load(index_dir / "visual.npz", allow_pickle=False) as data:
                    visual_model = _visual_model_key_from_index(data, self.settings.visual_model)
                    if visual_model not in visual_queries:
                        visual_queries[visual_model] = self._clip(visual_model).encode_query(text, image_path, alpha)
                    visual_query = visual_queries[visual_model]
                    candidates.extend(_visual_candidates(data, visual_query, video["id"], visual_profile, limit * 3))
            if face_query is not None and (index_dir / "faces.npz").exists():
                with np.load(index_dir / "faces.npz", allow_pickle=False) as data:
                    candidates.extend(_top_vectors(data, face_query, "face", video["id"], limit * 3, 0.35))
            if "asr" in modalities and text and (index_dir / "asr.json").exists():
                payload = json.loads((index_dir / "asr.json").read_text(encoding="utf-8"))
                semantic_data = None
                semantic_query = None
                semantic_path = index_dir / "asr_semantic.npz"
                if semantic_path.exists():
                    semantic_data = np.load(semantic_path, allow_pickle=False)
                    try:
                        model_name = str(semantic_data["model"][0]) if "model" in semantic_data.files else self.settings.asr_semantic_model
                        semantic_query = self._encode_asr_query(text, model_name)
                    except Exception:
                        semantic_data.close()
                        semantic_data = None
                        semantic_query = None
                try:
                    candidates.extend(_asr_candidates(
                        payload.get("chunks", []),
                        text,
                        video["id"],
                        limit * 3,
                        semantic_data=semantic_data,
                        semantic_query=semantic_query,
                    ))
                finally:
                    if semantic_data is not None:
                        semantic_data.close()
            if "ocr" in modalities and text and (index_dir / "ocr.json").exists():
                payload = json.loads((index_dir / "ocr.json").read_text(encoding="utf-8"))
                semantic_data = None
                semantic_query = None
                semantic_path = index_dir / "ocr_semantic.npz"
                if semantic_path.exists():
                    semantic_data = np.load(semantic_path, allow_pickle=False)
                    try:
                        model_name = str(semantic_data["model"][0]) if "model" in semantic_data.files else self.settings.asr_semantic_model
                        semantic_query = self._encode_asr_query(text, model_name)
                    except Exception:
                        semantic_data.close()
                        semantic_data = None
                        semantic_query = None
                try:
                    candidates.extend(_asr_candidates(
                        payload.get("chunks", []),
                        text,
                        video["id"],
                        limit * 3,
                        modality="ocr",
                        semantic_data=semantic_data,
                        semantic_query=semantic_query,
                    ))
                finally:
                    if semantic_data is not None:
                        semantic_data.close()

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
                evidence=[{
                    "modality": item.modality,
                    "score": round(item.score, 4),
                    "raw_score": round(item.raw_score, 4) if item.raw_score is not None else None,
                    "robust_z": round(item.robust_z, 3) if item.robust_z is not None else None,
                    "percentile": round(item.percentile, 4) if item.percentile is not None else None,
                    "decision": item.decision,
                    "distribution_reliable": item.distribution_reliable,
                    "distribution_median": round(item.distribution_median, 4) if item.distribution_median is not None else None,
                    "distribution_mad": round(item.distribution_mad, 4) if item.distribution_mad is not None else None,
                    "best_time": round(item.best_time, 3) if item.best_time is not None else None,
                    "visual_top1": round(item.visual_top1, 4) if item.visual_top1 is not None else None,
                    "visual_top3": round(item.visual_top3, 4) if item.visual_top3 is not None else None,
                    "visual_mean": round(item.visual_mean, 4) if item.visual_mean is not None else None,
                    "lexical_score": round(item.lexical_score, 4) if item.lexical_score is not None else None,
                    "semantic_score": round(item.semantic_score, 4) if item.semantic_score is not None else None,
                    "semantic_cosine": round(item.semantic_cosine, 4) if item.semantic_cosine is not None else None,
                    "detail": item.evidence,
                } for item in group],
            ))
        # Above-threshold results first (each block sorted by score), so the UI can
        # draw a single divider where matches start dropping below threshold.
        results.sort(key=lambda item: (item.above_threshold, item.score), reverse=True)
        result_limit = 500 if visual_profile == "recall" else limit
        return [item.to_dict() for item in results[:result_limit]]
