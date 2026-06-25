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
    raw_scores = embeddings @ normalize(query)
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
            detail = f"visual cosine={raw_score:.3f} · percentile={percentile * 100:.1f}% · robust_z={z_score:.2f}"
        else:
            if index in fallback_indices:
                decision, above = "fallback", True
            else:
                decision, above = "weak", False
            ranking_score = float(np.clip((raw_score + 1.0) / 2.0, 0, 1))
            detail = f"visual cosine={raw_score:.3f} · distribution fallback (n={len(raw_scores)})"

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
        self._clip_encoder = None
        self._face_encoder = None

    def _clip(self):
        if self._clip_encoder is None:
            from app.indexing.visual import ClipEncoder, resolve_device

            device = resolve_device(self.settings.npu_enabled, self.settings.npu_device_id, self.settings.cuda_enabled)
            self._clip_encoder = ClipEncoder(self.settings.clip_model, self.settings.clip_pretrained, device)
        return self._clip_encoder

    def _face(self):
        if self._face_encoder is None:
            from app.indexing.faces import FaceEncoder

            self._face_encoder = FaceEncoder(
                self.settings.face_model, "cpu", 0, str(self.settings.app_model_dir / "insightface")
            )
        return self._face_encoder

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

        visual_query = None
        if "visual" in modalities and (text or image_path):
            visual_query = self._clip().encode_query(text, image_path, alpha)

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
            if visual_query is not None and (index_dir / "visual.npz").exists():
                with np.load(index_dir / "visual.npz", allow_pickle=False) as data:
                    candidates.extend(_visual_candidates(data, visual_query, video["id"], visual_profile, limit * 3))
            if face_query is not None and (index_dir / "faces.npz").exists():
                with np.load(index_dir / "faces.npz", allow_pickle=False) as data:
                    candidates.extend(_top_vectors(data, face_query, "face", video["id"], limit * 3, 0.35))
            if "asr" in modalities and text and (index_dir / "asr.json").exists():
                payload = json.loads((index_dir / "asr.json").read_text(encoding="utf-8"))
                scored = sorted(
                    ((lexical_score(text, chunk["text"]), chunk) for chunk in payload.get("chunks", [])),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                for score, chunk in scored[:limit * 3]:
                    if score <= 0:
                        continue
                    above = score >= 0.25
                    candidates.append(Candidate(
                        video["id"], float(chunk["start_time"]), float(chunk["end_time"]), score,
                        "asr", evidence=chunk["text"], raw_score=score,
                        decision="lexical_hit" if above else "weak", above_threshold=above,
                    ))

        names = {video["id"]: video["name"] for video in videos}
        # Each modality score is now calibrated to a comparable [0,1] scale —
        # visual=empirical percentile, face=logistic-calibrated cosine, asr=lexical
        # coverage — so these weights express modality priority, not scale fixes.
        weights = {"face": 0.55, "visual": 0.30, "asr": 0.15}
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
                (name for name in ("strong", "fuzzy", "fallback", "absolute_hit", "lexical_hit", "weak") if name in group_decisions),
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
                    "detail": item.evidence,
                } for item in group],
            ))
        # Above-threshold results first (each block sorted by score), so the UI can
        # draw a single divider where matches start dropping below threshold.
        results.sort(key=lambda item: (item.above_threshold, item.score), reverse=True)
        result_limit = 500 if visual_profile == "recall" else limit
        return [item.to_dict() for item in results[:result_limit]]
