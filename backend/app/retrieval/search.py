from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import logging
import threading

import numpy as np

from app.catalog.db import Catalog
from app.retrieval.retrieval_metrics import RetrievalProfiler
from app.core.settings import Settings

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
    decision: str = "hit"
    above_threshold: bool = True
    best_time: float | None = None
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


def face_confidence(cosine: float) -> float:
    """Map an ArcFace (buffalo_l) cosine to a calibrated [0,1] confidence.

    Face cosine is absolutely meaningful (distance to a reference identity), unlike
    CLIP text-image scores. Raw cosines for true matches cluster around 0.45-0.7,
    so a logistic centred at 0.45 lifts a strong match to ~1.0 — putting it on the
    same bounded scale as visual confidence, which is what the fusion step weighs.
    Without this, a cosine=0.6 face hit would be systematically underweighted.
    """
    return float(1.0 / (1.0 + np.exp(-12.0 * (cosine - 0.45))))


def visual_confidence(cosine: float) -> float:
    """Map visual raw cosine to a cross-video ranking score."""
    return float(np.clip((cosine + 1.0) / 2.0, 0, 1))


def _seconds(ms: int | float) -> float:
    return float(ms) / 1000.0


def _channel_publication_for(video: dict, channel: str) -> dict:
    publication = (video.get("index_publications") or {}).get(channel)
    if not isinstance(publication, dict) or publication.get("status") != "ready":
        raise ValueError(
            f"视频 {video.get('name') or video['id']} 的 {channel} 索引尚未发布"
        )
    return publication


def _published_asset_version(publication: dict, video_name: str, channel: str) -> str:
    """Return the only Milvus version that online retrieval may read."""
    value = publication.get("asset_version")
    if value is None or not str(value).strip():
        raise ValueError(
            f"视频 {video_name} 的 {channel} 索引尚未发布到 Milvus，请重跑该通道"
        )
    return str(value)


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
        "decision": item.decision,
        "best_time": _round_optional(item.best_time, 3),
        "unit_type": item.unit_type,
        "unit_id": item.unit_id,
        "best_ms": item.best_ms,
        "text": item.text,
        "features": _serialized_features(item.features),
        "detail": item.evidence,
    }

_OCR_ONLY_MERGE_GAP_SECONDS = 0.35
_OCR_MERGE_MIN_SCORE_RATIO = 0.90  # 至少保留 90% 的最佳分数
_OCR_MERGE_MAX_SCORE_DROP = 0.10   # 绝对分数差不超过 0.10

# Face-only 合并的 cosine 相似度带宽。face track 是"同一人连续出现"的语义单元，
# 时间相邻的两条 track 未必是同一个人。只有当两条 track 的 cosine（raw_score）
# 落在同一带宽内时才合并，避免把目标人脸片段和非目标人脸片段拼成一个长片段
# （否则显示分取组内最高分、evidence 却混入低分非目标项，导致分数/文字不符）。
_FACE_MERGE_MAX_COSINE_DROP = 0.15


def _face_scores_compatible(group: list[Candidate], candidate: Candidate) -> bool:
    """Face-only 合并时，避免不同人脸（cosine 差距大）被拼进同一片段。

    规则：
    - candidate 必须是 face 模态；
    - group 里必须已有 face 命中；
    - candidate 的 cosine（raw_score）不能比 group 内最佳 face cosine 低太多，
      也不能高太多——对称带宽，保证同组 track 属于同一相似度层级。

    raw_score 缺失时（理论上 face 恒有）退化为仅按时间合并，返回 True。
    组内无可用 cosine 时同样退化（对称处理，与 docstring 承诺一致）。
    """
    if candidate.modality != "face":
        return False

    # 检查组内是否有 face 命中
    if not any(item.modality == "face" for item in group):
        return False

    group_cosines = [
        float(item.raw_score)
        for item in group
        if item.modality == "face" and item.raw_score is not None
    ]
    # 组内无可用 cosine 或候选无 cosine → 退化为纯时间合并
    if not group_cosines or candidate.raw_score is None:
        return True

    cand_cosine = float(candidate.raw_score)
    best_cosine = max(group_cosines)
    worst_cosine = min(group_cosines)
    # 对称带宽：candidate 与组内最强/最弱 face 都不能相差超过阈值，
    # 防止高分 track 被并入低分锚点组，或反之。
    return (
        cand_cosine >= best_cosine - _FACE_MERGE_MAX_COSINE_DROP
        and cand_cosine <= worst_cosine + _FACE_MERGE_MAX_COSINE_DROP
    )



def _apply_global_threshold(candidates: list[Candidate], modality: str) -> None:
    """对指定模态的候选应用全局动态阈值。

    规则：
    - 阈值 = max(0.10, 全局最高分 * 0.3)
    - 低于阈值的候选标记 above_threshold=False 并在 evidence 添加 "· 低于阈值"
    """
    modality_candidates = [c for c in candidates if c.modality == modality]
    if not modality_candidates:
        return

    global_top_score = max(float(c.score) for c in modality_candidates)
    global_threshold = max(0.10, global_top_score * 0.3)

    for candidate in modality_candidates:
        candidate.above_threshold = float(candidate.score) >= global_threshold
        if not candidate.above_threshold and " · 低于阈值" not in (candidate.evidence or ""):
            candidate.evidence = (candidate.evidence or "") + " · 低于阈值"


def _temporal_gap(group: list[Candidate], candidate: Candidate) -> float:
    """Return the non-negative time gap between a group and a candidate."""
    group_start = min(item.start_time for item in group)
    group_end = max(item.end_time for item in group)
    return max(
        candidate.start_time - group_end,
        group_start - candidate.end_time,
        0.0,
    )


def _should_merge(group: list[Candidate], candidate: Candidate, gap: float, max_duration: float) -> bool:
    """判断候选是否应合并到组内。

    通用规则：
    - 同一视频
    - 合并后总时长不超过 max_duration
    - 按模态分支判断时间/分数兼容性

    关键修复（2026-08-11）：
    - 改用双向间隙判断（候选在组前/后都正确计算间隙），修复单向 `near` 导致的
      "候选早于组也判为相邻"的问题。
    """
    if group[0].video_id != candidate.video_id:
        return False

    group_modalities = {item.modality for item in group}

    group_start = min(item.start_time for item in group)
    group_end = max(item.end_time for item in group)
    merged_start = min(group_start, candidate.start_time)
    merged_end = max(group_end, candidate.end_time)
    if merged_end - merged_start > max_duration:
        return False

    overlaps = candidate.start_time < group_end and candidate.end_time > group_start

    # 双向间隙判断：候选在组前（group_start - candidate.end_time）或
    # 组后（candidate.start_time - group_end），取两者中的正值（无间隙时为0）
    gap_between = _temporal_gap(group, candidate)
    near = gap_between <= gap

    # Face-only 合并须额外满足 cosine 带宽约束。face track 是"同一人连续出现"
    # 的语义单元；仅凭时间相邻（gap≤2s）就合并，会把目标人脸和时间上恰好邻近的
    # 非目标人脸拼成一个长片段，进而显示分取组内最高分、evidence 混入低分非目标项，
    # 造成"分数 99% 但明细却是 cosine=-0.01 · 低于阈值"的不一致。
    if candidate.modality == "face" and group_modalities == {"face"}:
        return near and _face_scores_compatible(group, candidate)

    # Visual buckets are already the display granularity. Do not chain adjacent
    # visual-only hits into a full-video result; merge them only when another
    # modality anchors the same moment, or when intervals genuinely overlap.
    if candidate.modality == "visual" and group_modalities == {"visual"}:
        return overlaps
    if candidate.modality == "visual" or group_modalities == {"visual"}:
        return overlaps or (near and bool(group_modalities - {"visual"}))
    return near


def _groups_ocr_score_first(candidates: list[Candidate], max_duration: float = 15) -> list[list[Candidate]]:
    """
    OCR 专用聚合：从高分帧开始向两边扩展。

    算法：
    1. 按分数降序选种子（未聚合的最高分）
    2. 从种子向时间两边扩展，基于种子分数判断是否合并
    3. 扩展时始终保证组时间跨度不超过 max_duration
    4. 标记已聚合的帧，避免重复处理
    5. 重复直到所有帧都处理完
    """
    if not candidates:
        return []

    # 按 video_id 分组处理
    by_video: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_video.setdefault(candidate.video_id, []).append(candidate)

    all_groups: list[list[Candidate]] = []

    for video_id, video_candidates in by_video.items():
        # 按时间排序（用于向两边扩展）
        time_sorted = sorted(video_candidates, key=lambda c: (c.start_time, c.end_time))
        # 按分数降序（用于选种子）
        score_sorted = sorted(video_candidates, key=lambda c: -float(c.score))

        # 预建索引映射，避免 O(n²)
        candidate_to_idx = {id(c): i for i, c in enumerate(time_sorted)}

        used_ids = set()  # 记录已聚合的候选

        for seed in score_sorted:
            seed_id = id(seed)
            if seed_id in used_ids:
                continue  # 已被其他组聚合，跳过

            # 以 seed 为核心建新组
            group = [seed]
            used_ids.add(seed_id)

            # 计算分数阈值（基于种子分数，不会滑坡）
            seed_score = float(seed.score)
            score_threshold = max(
                seed_score * _OCR_MERGE_MIN_SCORE_RATIO,
                seed_score - _OCR_MERGE_MAX_SCORE_DROP,
            )

            seed_idx = candidate_to_idx[seed_id]

            # === 向左扩展（时间更早的帧）===
            for i in range(seed_idx - 1, -1, -1):
                candidate = time_sorted[i]
                cand_id = id(candidate)

                if cand_id in used_ids:
                    continue  # 已被聚合，跳过但继续尝试更早的帧

                # 检查分数兼容性
                if float(candidate.score) < score_threshold:
                    break  # 分数不够，停止向左扩展

                # 检查时间 gap（candidate 在左边，检查它的 end_time 和 group 最早的 start_time）
                group_start = min(c.start_time for c in group)
                gap_to_group = group_start - candidate.end_time

                if gap_to_group > _OCR_ONLY_MERGE_GAP_SECONDS:
                    break  # 时间太远，停止向左扩展

                merged_start = min(candidate.start_time, group_start)
                merged_end = max(candidate.end_time, max(c.end_time for c in group))
                if merged_end - merged_start > max_duration:
                    break

                # 满足条件，加入组
                group.append(candidate)
                used_ids.add(cand_id)

            # === 向右扩展（时间更晚的帧）===
            for i in range(seed_idx + 1, len(time_sorted)):
                candidate = time_sorted[i]
                cand_id = id(candidate)

                if cand_id in used_ids:
                    continue  # 已被聚合，跳过但继续尝试更晚的帧

                # 检查分数兼容性
                if float(candidate.score) < score_threshold:
                    break  # 分数不够，停止向右扩展

                # 检查时间 gap（candidate 在右边，检查 group 最晚的 end_time 和它的 start_time）
                group_end = max(c.end_time for c in group)
                gap_to_group = candidate.start_time - group_end

                if gap_to_group > _OCR_ONLY_MERGE_GAP_SECONDS:
                    break  # 时间太远，停止向右扩展

                merged_start = min(candidate.start_time, min(c.start_time for c in group))
                merged_end = max(candidate.end_time, group_end)
                if merged_end - merged_start > max_duration:
                    break

                # 满足条件，加入组
                group.append(candidate)
                used_ids.add(cand_id)

            all_groups.append(group)

    return all_groups


def _groups(candidates: list[Candidate], gap: float, max_duration: float = 15) -> list[list[Candidate]]:
    """将候选聚合为组。

    算法：
    1. OCR 使用分数优先聚合（_groups_ocr_score_first），从高分种子向两边扩展
    2. 非 OCR 候选遍历所有现存组，选择时间间隔最小的可合并组
    3. 最终按时间排序保证展示稳定

    关键修复（2026-08-11）：
    - 非 OCR 候选改为"遍历所有组择优"，修复混合模态下候选合并到错误 OCR 组的问题。
      旧逻辑只比对 groups[-1]，但 OCR 组已按分数（非时间）顺序插入，导致时间匹配的
      早期组被跳过。
    """
    ocr_candidates = [c for c in candidates if c.modality == "ocr"]
    non_ocr_candidates = [c for c in candidates if c.modality != "ocr"]

    groups: list[list[Candidate]] = []

    # OCR 使用分数优先聚合
    if ocr_candidates:
        if not non_ocr_candidates:
            # 纯 OCR，直接使用新算法
            return _groups_ocr_score_first(ocr_candidates, max_duration)
        else:
            # 混合模态，OCR 先聚合，再和其他模态合并
            ocr_groups = _groups_ocr_score_first(ocr_candidates, max_duration)
            groups.extend(ocr_groups)

    # 非 OCR 候选遍历所有组，优先并入时间上最近的组。
    # 时间间隔相同时，min() 保留现有组顺序作为稳定 tie-breaker。
    for candidate in sorted(non_ocr_candidates, key=lambda item: (item.video_id, item.start_time, item.end_time)):
        target_group = min(
            (g for g in groups if _should_merge(g, candidate, gap, max_duration)),
            key=lambda g: _temporal_gap(g, candidate),
            default=None,
        )
        if target_group is not None:
            target_group.append(candidate)
        else:
            groups.append([candidate])

    # 按时间排序保证展示稳定（OCR 组按分数插入，非 OCR 合并后可能乱序）
    return sorted(groups, key=lambda g: (g[0].video_id, min(item.start_time for item in g)))


def _fuse_candidate_groups(
    candidates: list[Candidate],
    videos: list[dict],
    merge_gap: float,
    max_result_seconds: float,
    primary_modality: str | None = None,
) -> list[SearchResult]:
    """Fuse overlapping candidates and optionally prefer one requested modality.

    Threshold status remains the primary ordering boundary. Within one tier,
    results containing ``primary_modality`` are ordered before auxiliary-only
    candidates. Among primary-backed results, the fused score ranks candidates
    so corroborating evidence can affect order without promoting an
    auxiliary-only result above primary evidence.
    """
    names = {video["id"]: video["name"] for video in videos}
    weights = {"face": 0.55, "visual": 0.30, "ocr": 0.20, "asr": 0.15}
    results: list[SearchResult] = []
    primary_scores: dict[int, float] = {}
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
        result = SearchResult(
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
        )
        results.append(result)
        if primary_modality in best_by_modality:
            primary_scores[id(result)] = best_by_modality[primary_modality]

    if primary_modality is None:
        results.sort(
            key=lambda item: (item.above_threshold, item.score, -item.start_time),
            reverse=True,
        )
    else:
        results.sort(
            key=lambda item: (
                item.above_threshold,
                primary_modality in item.modalities,
                item.score,
                primary_scores.get(id(item), item.score),
                -item.start_time,
            ),
            reverse=True,
        )
    return results


class SearchEngine:
    def __init__(self, settings: Settings, catalog: Catalog):
        self.settings = settings
        self.catalog = catalog
        self._clip_encoders = {}
        self._face_encoder = None
        self._text_encoders = {}
        self._encoder_lock = threading.RLock()
        self._visual_encode_lock = threading.Lock()
        self._text_encode_lock = threading.Lock()
        self._prewarm_status: dict = {
            "status": "disabled" if not settings.search_prewarm_enabled else "pending",
            "resident": True,
        }

    def _clip(
        self,
        visual_model: str | None = None,
        profiler: RetrievalProfiler | None = None,
    ):
        from app.encoders.visual import ClipEncoder, normalize_visual_model, resolve_device

        model_key = normalize_visual_model(visual_model or self.settings.visual_model)
        if model_key not in self._clip_encoders:
            with self._encoder_lock:
                if model_key not in self._clip_encoders:
                    span = (
                        profiler.span("model_load", "visual")
                        if profiler
                        else nullcontext()
                    )
                    with span:
                        device = resolve_device(
                            self.settings.npu_enabled,
                            self.settings.npu_device_id,
                            self.settings.cuda_enabled,
                        )
                        self._clip_encoders[model_key] = ClipEncoder(
                            self.settings.clip_model,
                            self.settings.clip_pretrained,
                            device,
                            visual_model=model_key,
                            model_cache_dir=str(
                                self.settings.resolve_path(
                                    self.settings.visual_hf_cache_dir
                                )
                            ),
                        )
        return self._clip_encoders[model_key]

    def _face(self):
        if self._face_encoder is None:
            with self._encoder_lock:
                if self._face_encoder is None:
                    from app.encoders.face import FaceEncoder

                    # Match the index-side provider/device while serialising the
                    # expensive first model load across concurrent requests.
                    self._face_encoder = FaceEncoder(
                        self.settings.face_model,
                        self.settings.face_provider,
                        self.settings.npu_device_id,
                        str(self.settings.app_model_dir / "insightface"),
                        self.settings.face_ort_intra_op_threads,
                        self.settings.face_ort_inter_op_threads,
                    )
        return self._face_encoder

    def _encode_asr_query(
        self,
        text: str,
        model_name: str,
        profiler: RetrievalProfiler | None = None,
    ) -> np.ndarray:
        from app.encoders.text import TextEmbeddingEncoder, resolve_text_embedding_device

        device = resolve_text_embedding_device(self.settings.asr_semantic_device, self.settings.cuda_enabled)
        key = (model_name, device)
        if key not in self._text_encoders:
            with self._encoder_lock:
                if key not in self._text_encoders:
                    span = (
                        profiler.span("model_load", "asr")
                        if profiler
                        else nullcontext()
                    )
                    with span:
                        self._text_encoders[key] = TextEmbeddingEncoder(
                            model_name,
                            self.settings.app_model_dir / "text-embeddings",
                            device,
                            local_files_only=self.settings.asr_semantic_local_files_only,
                        )
        span = (
            profiler.span("query_encode", "asr")
            if profiler
            else nullcontext()
        )
        with self._text_encode_lock, span:
            return self._text_encoders[key].encode([text], batch_size=1)[0]

    def _encode_visual_queries(
        self,
        visual_model: str,
        query_texts: list[str | None],
        image_path: str | None,
        alpha: float,
        profiler: RetrievalProfiler | None,
    ) -> np.ndarray:
        encoder = (
            self._clip(visual_model, profiler)
            if profiler
            else self._clip(visual_model)
        )
        span = (
            profiler.span("query_encode", "visual")
            if profiler
            else nullcontext()
        )
        with self._visual_encode_lock, span:
            texts = [value for value in query_texts if value]
            if hasattr(encoder, "encode_queries"):
                return encoder.encode_queries(texts, image_path, alpha)
            return np.stack([
                encoder.encode_query(value, image_path, alpha)
                for value in texts
            ])

    def prewarm(self) -> dict:
        profiler = RetrievalProfiler()
        visual_models, text_models, publication_errors = self._indexed_query_models()
        model_errors: list[dict[str, str]] = []
        for model_key in sorted(visual_models):
            try:
                visual = self._clip(model_key, profiler)
                with self._visual_encode_lock, profiler.span(
                    "query_encode", "visual"
                ):
                    visual.encode_queries(["MomentSeek warmup"], None)
            except Exception as exc:
                model_errors.append({
                    "kind": "visual",
                    "model": model_key,
                    "error": str(exc),
                })
        for model_name in sorted(text_models):
            try:
                self._encode_asr_query(
                    "MomentSeek warmup",
                    model_name,
                    profiler,
                )
            except Exception as exc:
                model_errors.append({
                    "kind": "text",
                    "model": model_name,
                    "error": str(exc),
                })

        errors = [*publication_errors, *model_errors]
        status = {
            "status": "error" if errors else "ready",
            "resident": True,
            "requested_visual_models": sorted(visual_models),
            "requested_text_models": sorted(text_models),
            "errors": errors,
            **profiler.snapshot(),
        }
        with self._encoder_lock:
            self._prewarm_status = status
        if errors and self.settings.search_prewarm_required:
            summary = "; ".join(
                f"{item.get('kind')}:{item.get('model', item.get('video_id', 'unknown'))}:"
                f" {item['error']}"
                for item in errors
            )
            raise RuntimeError(f"Query-model prewarm failed: {summary}")
        return self.query_model_status()

    def _indexed_query_models(
        self,
    ) -> tuple[set[str], set[str], list[dict[str, str]]]:
        """Return deduplicated model keys used by currently searchable indexes."""
        visual_models = {self.settings.visual_model}
        text_models: set[str] = set()
        errors: list[dict[str, str]] = []
        for video in self._selected_videos(None):
            indexed = set(video.get("indexed_modalities") or [])
            for channel in sorted(indexed & {"visual", "asr", "ocr"}):
                try:
                    channel_publication = _channel_publication_for(video, channel)
                except ValueError as exc:
                    errors.append({
                        "kind": "publication",
                        "video_id": str(video["id"]),
                        "model": channel,
                        "error": str(exc),
                    })
                    continue
                if channel == "visual":
                    visual_models.add(
                        str(
                            channel_publication.get("model_key")
                            or self.settings.visual_model
                        )
                    )
                else:
                    model_name = self._semantic_model_for_channel(
                        channel_publication
                    )
                    if model_name is not None:
                        text_models.add(model_name)
        return visual_models, text_models, errors

    def query_model_status(self) -> dict:
        with self._encoder_lock:
            status = dict(self._prewarm_status)
            visual_models = sorted(self._clip_encoders)
            text_models = sorted({
                model for model, _device in self._text_encoders
            })
        return {
            **status,
            "visual_models": visual_models,
            "text_models": text_models,
        }

    def _get_milvus_client(self):
        """Resolve the live client behind a unit-testable connection boundary."""
        from app.vector_store.milvus.milvus_client import (
            ensure_milvus_reachable,
            get_milvus_client,
        )

        ensure_milvus_reachable()
        return get_milvus_client()

    def close(self) -> None:
        with self._encoder_lock:
            self._clip_encoders.clear()
            self._text_encoders.clear()
            self._face_encoder = None

    def _selected_videos(self, video_ids: list[str] | None) -> list[dict]:
        videos = self.catalog.list_videos()
        # An empty folder resolves to ``[]`` and must not silently expand to all
        # assets. ``None`` alone means the established all-video scope.
        if video_ids is None:
            return videos
        allowed = set(video_ids)
        return [video for video in videos if video["id"] in allowed]

    def _resolve_face_query(
        self,
        text: str | None,
        image_path: str | None,
        *,
        optional: bool = False,
    ) -> np.ndarray | None:
        if image_path:
            try:
                return self._face().encode_reference(image_path)
            except ValueError as exc:
                if optional and str(exc) == "参考图中未检测到人脸":
                    logger.info(
                        "Reference image has no face; skipping optional face channel"
                    )
                    return None
                raise
        if not text:
            return None
        entity = self.catalog.find_entity_in_text(text)
        if entity:
            try:
                from app.vector_store.milvus.milvus_client import get_milvus_client

                entity_id = str(entity["id"]).replace("\\", "\\\\").replace('"', '\\"')
                rows = get_milvus_client().collection("entity_face_samples").query(
                    expr=f'entity_id == "{entity_id}"', output_fields=["embedding"], limit=1024
                )
                if rows:
                    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
                    prototype = np.mean(vectors, axis=0)
                    return prototype / max(float(np.linalg.norm(prototype)), 1e-12)
            except Exception:
                logger.exception("Milvus entity face sample lookup failed")
                raise
        return None

    def _semantic_query(
        self,
        text: str,
        channel_publication: dict,
        embeddings: np.ndarray | None,
        semantic_queries: dict[str, np.ndarray | None],
        profiler: RetrievalProfiler | None,
    ) -> np.ndarray | None:
        if embeddings is None:
            return None
        model_name = self._semantic_model_for_channel(channel_publication)
        if model_name is None:
            return None
        if model_name not in semantic_queries:
            try:
                semantic_queries[model_name] = (
                    self._encode_asr_query(text, model_name, profiler)
                    if profiler
                    else self._encode_asr_query(text, model_name)
                )
            except Exception:
                semantic_queries[model_name] = None
        return semantic_queries[model_name]

    def _semantic_model_for_channel(
        self,
        channel_publication: dict,
    ) -> str | None:
        """Return the model only when this indexed channel has semantic data."""
        semantic_status = str(
            channel_publication.get("semantic_status") or ""
        ).strip().casefold()
        if semantic_status != "complete":
            return None
        return str(
            channel_publication.get("semantic_model_key")
            or self.settings.asr_semantic_model
        )

    def _milvus_candidates_for_video(
        self,
        video: dict,
        *,
        text: str | None,
        modalities: list[str],
        visual_profile: str,
        visual_queries: dict[str, np.ndarray],
        face_query: np.ndarray | None,
        channel_limits: dict[str, int],
        semantic_queries: dict[str, np.ndarray | None],
        profiler: RetrievalProfiler | None,
        client=None,
    ) -> list[Candidate]:
        from app.vector_store.milvus.milvus_search import (
            milvus_asr_candidates_hybrid,
            milvus_face_candidates,
            milvus_ocr_candidates_hybrid,
            milvus_visual_candidates,
        )

        if client is None:
            client = self._get_milvus_client()
        video_id = video["id"]
        indexed = set(video.get("indexed_modalities") or [])
        candidates: list[Candidate] = []
        if "visual" in modalities and "visual" in indexed:
            channel_publication = _channel_publication_for(video, "visual")
            asset_version = _published_asset_version(
                channel_publication, str(video.get("name") or video_id), "visual"
            )
            visual_model = str(channel_publication.get("model_key") or self.settings.visual_model)
            if visual_model not in visual_queries:
                raise RuntimeError(
                    f"visual query vector was not prepared for model={visual_model}"
                )
            candidates.extend(milvus_visual_candidates(
                client,
                video_id,
                visual_queries[visual_model],
                asset_version,
                profile=visual_profile,
                limit=channel_limits["visual"],
                profiler=profiler,
            ))
        if "face" in modalities and face_query is not None and "face" in indexed:
            channel_publication = _channel_publication_for(video, "face")
            candidates.extend(milvus_face_candidates(
                client,
                video_id,
                face_query,
                _published_asset_version(
                    channel_publication, str(video.get("name") or video_id), "face"
                ),
                channel_limits["face"],
                None,  # threshold=None → settings.face_identity_threshold
                profiler,
            ))
        if "asr" in modalities and text and "asr" in indexed:
            channel_publication = _channel_publication_for(video, "asr")
            model_name = self._semantic_model_for_channel(channel_publication)
            semantic_query = (
                semantic_queries.get(model_name)
                if model_name is not None
                else None
            )
            candidates.extend(milvus_asr_candidates_hybrid(
                client,
                video_id,
                _published_asset_version(
                    channel_publication, str(video.get("name") or video_id), "asr"
                ),
                text,
                semantic_query,
                channel_limits["asr"],
                profiler,
            ))
        if "ocr" in modalities and text and "ocr" in indexed:
            channel_publication = _channel_publication_for(video, "ocr")
            model_name = self._semantic_model_for_channel(channel_publication)
            semantic_query = (
                semantic_queries.get(model_name)
                if model_name is not None
                else None
            )
            candidates.extend(milvus_ocr_candidates_hybrid(
                client,
                video_id,
                _published_asset_version(
                    channel_publication, str(video.get("name") or video_id), "ocr"
                ),
                text,
                semantic_query,
                channel_limits["ocr"],
                profiler,
            ))
        return candidates

    @staticmethod
    def _requested_indexed_modalities(
        video: dict,
        *,
        text: str | None,
        image_path: str | None,
        modalities: list[str],
        face_query: np.ndarray | None,
    ) -> set[str]:
        """Return channels that should produce candidates from Milvus."""
        indexed = set(video.get("indexed_modalities") or [])
        requested: set[str] = set()
        if "visual" in modalities and bool(text or image_path) and "visual" in indexed:
            requested.add("visual")
        if face_query is not None and "face" in indexed:
            requested.add("face")
        if text:
            requested.update({"asr", "ocr"} & set(modalities) & indexed)
        return requested

    def _prepare_query_vectors(
        self,
        videos: list[dict],
        *,
        text: str | None,
        image_path: str | None,
        alpha: float,
        visual_subqueries: list[str] | None,
        requested_by_video: dict[str, set[str]],
        visual_queries: dict[str, np.ndarray],
        semantic_queries: dict[str, np.ndarray | None],
        profiler: RetrievalProfiler | None,
    ) -> None:
        """Encode all publication-selected query models before candidate scoring."""
        query_texts: list[str | None] = (
            list(dict.fromkeys(visual_subqueries or []))
            if text and visual_subqueries
            else [text]
        )
        for video in videos:
            requested = requested_by_video[video["id"]]
            if "visual" in requested:
                channel_publication = _channel_publication_for(video, "visual")
                model_key = str(
                    channel_publication.get("model_key")
                    or self.settings.visual_model
                )
                if model_key not in visual_queries:
                    visual_queries[model_key] = self._encode_visual_queries(
                        model_key,
                        query_texts,
                        image_path,
                        alpha,
                        profiler,
                    )
            if text:
                for channel in sorted(requested & {"asr", "ocr"}):
                    channel_publication = _channel_publication_for(video, channel)
                    if self._semantic_model_for_channel(
                        channel_publication
                    ) is not None:
                        self._semantic_query(
                            text,
                            channel_publication,
                            np.empty((1, 1), dtype=np.float32),
                            semantic_queries,
                            profiler,
                        )

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
        profiler: RetrievalProfiler | None = None,
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
        semantic_queries: dict[str, np.ndarray | None] = {}
        face_span = (
            profiler.span("query_encode", "face")
            if profiler and "face" in modalities
            else nullcontext()
        )
        with face_span:
            face_query = (
                self._resolve_face_query(
                    text,
                    image_path,
                    optional=any(channel != "face" for channel in modalities),
                )
                if "face" in modalities
                else None
            )
        requested_by_video = {
            video["id"]: self._requested_indexed_modalities(
                video,
                text=text,
                image_path=image_path,
                modalities=modalities,
                face_query=face_query,
            )
            for video in videos
        }
        self._prepare_query_vectors(
            videos,
            text=text,
            image_path=image_path,
            alpha=alpha,
            visual_subqueries=visual_subqueries,
            requested_by_video=requested_by_video,
            visual_queries=visual_queries,
            semantic_queries=semantic_queries,
            profiler=profiler,
        )
        milvus_video_ids = [
            video["id"]
            for video in videos
            if requested_by_video[video["id"]]
        ]
        milvus_client = None
        if milvus_video_ids:
            milvus_client = self._get_milvus_client()

        batch_size = self.settings.milvus_search_video_batch_size
        for batch_offset in range(0, len(videos), batch_size):
            batch_videos = videos[batch_offset:batch_offset + batch_size]
            for video in batch_videos:
                video_id = video["id"]
                requested_modalities = requested_by_video[video_id]
                for modality in sorted(requested_modalities):
                    scoring_span = (
                        profiler.span("local_processing", f"{modality}_scoring")
                        if profiler and modality != "face"
                        else nullcontext()
                    )
                    with scoring_span:
                        modality_candidates = self._milvus_candidates_for_video(
                            video,
                            text=text,
                            modalities=[modality],
                            visual_profile=visual_profile,
                            visual_queries=visual_queries,
                            face_query=face_query,
                            channel_limits=resolved_channel_limits,
                            semantic_queries=semantic_queries,
                            profiler=profiler,
                            client=milvus_client,
                        )
                    candidates.extend(
                        item for item in modality_candidates if item.modality == modality
                    )
        # Apply global dynamic threshold to OCR and ASR candidates
        _apply_global_threshold(candidates, "ocr")
        _apply_global_threshold(candidates, "asr")

        fusion_span = (
            profiler.span("local_processing", "fusion")
            if profiler
            else nullcontext()
        )
        with fusion_span:
            results = _fuse_candidate_groups(
                candidates,
                videos,
                merge_gap,
                max_result_seconds,
                primary_modality=(
                    "visual"
                    if self.settings.search_visual_priority_enabled
                    and "visual" in modalities
                    else None
                ),
            )
        result_limit = 500 if visual_profile == "recall" else limit
        return [item.to_dict() for item in results[:result_limit]]
