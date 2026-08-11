from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import threading

import numpy as np

from app.catalog.db import Catalog
from app.indexing.common import normalize
from app.indexing.manifest import require_channel_manifest
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


def visual_confidence(cosine: float) -> float:
    """Map visual raw cosine to a cross-video ranking score."""
    return float(np.clip((cosine + 1.0) / 2.0, 0, 1))


def _seconds(ms: int | float) -> float:
    return float(ms) / 1000.0


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


def _channel_manifest_for(video: dict, index_dir: Path, channel: str) -> tuple[dict, dict, Path]:
    manifest, channel_manifest = require_channel_manifest(index_dir, str(video.get("name") or video["id"]), channel)
    file_name = str(channel_manifest.get("file") or "")
    index_file = index_dir / file_name
    # The manifest selects model/version metadata.  Its NPZ file is an offline
    # recovery artifact and must never gate the online Milvus read path.
    return manifest, channel_manifest, index_file


def _published_asset_version(channel_manifest: dict, video_name: str, channel: str) -> str:
    """Return the only Milvus version that online retrieval may read."""
    value = channel_manifest.get("milvus_asset_version")
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


def _ocr_scores_compatible(group: list[Candidate], candidate: Candidate) -> bool:
    """OCR-only 合并时，避免高分命中被明显低分命中拖长。

    规则：
    - candidate 必须是 OCR 模态；
    - group 里必须已有 OCR 命中；
    - candidate 分数不能比 group 里最佳 OCR 命中低太多。

    注意：不再检查 above_threshold，聚合只基于分数差异。

    【已被 _groups_ocr_score_first 替代，保留以备将来独立 OCR 合并场景】
    """
    if candidate.modality != "ocr":
        return False

    group_scores = [
        float(item.score)
        for item in group
        if item.modality == "ocr"
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
    """OCR-only 结果使用更严格的帧级合并策略。

    只允许：
    - OCR 与 OCR 合并；
    - 时间窗口重叠或几乎相邻；
    - 分数不能差太多。

    注意：不再检查 above_threshold，聚合只基于分数和时间。
    above_threshold 只影响最终展示，不影响聚合逻辑。

    不设置最大合并时长：
    如果同一段 OCR 文本持续稳定出现很久，它应该保留为一个连续命中片段。

    【已被 _groups_ocr_score_first 替代，保留以备将来独立 OCR 合并场景】
    """
    if candidate.modality != "ocr":
        return False
    if any(item.modality != "ocr" for item in group):
        return False

    group_end = max(item.end_time for item in group)

    gap = candidate.start_time - group_end
    if gap > _OCR_ONLY_MERGE_GAP_SECONDS:
        return False

    if not _ocr_scores_compatible(group, candidate):
        return False

    return True


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

    # OCR-only 使用更严格的帧级合并规则（已死代码，保留以备将来）
    if candidate.modality == "ocr" and group_modalities == {"ocr"}:
        return _should_merge_ocr_only(group, candidate)

    group_start = min(item.start_time for item in group)
    group_end = max(item.end_time for item in group)
    merged_start = min(group_start, candidate.start_time)
    merged_end = max(group_end, candidate.end_time)
    if merged_end - merged_start > max_duration:
        return False

    overlaps = candidate.start_time < group_end and candidate.end_time > group_start

    # 双向间隙判断：候选在组前（group_start - candidate.end_time）或
    # 组后（candidate.start_time - group_end），取两者中的正值（无间隙时为0）
    gap_between = max(
        candidate.start_time - group_end,  # 候选在组后的间隙
        group_start - candidate.end_time,  # 候选在组前的间隙
        0.0,                                # 重叠时间隙为0
    )
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


def _groups_ocr_score_first(candidates: list[Candidate]) -> list[list[Candidate]]:
    """
    OCR 专用聚合：从高分帧开始向两边扩展。

    算法：
    1. 按分数降序选种子（未聚合的最高分）
    2. 从种子向时间两边扩展，基于种子分数判断是否合并
    3. 标记已聚合的帧，避免重复处理
    4. 重复直到所有帧都处理完
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

                # 满足条件，加入组
                group.append(candidate)
                used_ids.add(cand_id)

            all_groups.append(group)

    return all_groups


def _groups(candidates: list[Candidate], gap: float, max_duration: float = 15) -> list[list[Candidate]]:
    """将候选聚合为组。

    算法：
    1. OCR 使用分数优先聚合（_groups_ocr_score_first），从高分种子向两边扩展
    2. 非 OCR 候选遍历所有现存组，找到第一个可合并的组（而非只看 groups[-1]）
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
            return _groups_ocr_score_first(ocr_candidates)
        else:
            # 混合模态，OCR 先聚合，再和其他模态合并
            ocr_groups = _groups_ocr_score_first(ocr_candidates)
            groups.extend(ocr_groups)

    # 非 OCR 候选遍历所有组择优（修复只看 groups[-1] 的问题）
    for candidate in sorted(non_ocr_candidates, key=lambda item: (item.video_id, item.start_time, item.end_time)):
        target_group = next(
            (g for g in groups if _should_merge(g, candidate, gap, max_duration)),
            None,
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
    candidates; the modality's own score resolves ties among those results.
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
        results.sort(key=lambda item: (item.above_threshold, item.score), reverse=True)
    else:
        results.sort(
            key=lambda item: (
                item.above_threshold,
                primary_modality in item.modalities,
                primary_scores.get(id(item), item.score),
                item.score,
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
            from app.encoders.face import FaceEncoder

            # Provider/device wired from settings (was hardcoded cpu/0), matching
            # the index side (stage_executor.py) so query-side reference-image
            # encoding can use the same NPU/GPU. face_provider defaults to "cpu",
            # so behaviour is unchanged unless explicitly configured otherwise.
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
        visual_models, text_models, manifest_errors = self._indexed_query_models()
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

        errors = [*manifest_errors, *model_errors]
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
            index_dir = self.settings.index_dir / video["id"]
            for channel in sorted(indexed & {"visual", "asr", "ocr"}):
                try:
                    _manifest, channel_manifest, _index_file = (
                        _channel_manifest_for(video, index_dir, channel)
                    )
                except (OSError, ValueError) as exc:
                    errors.append({
                        "kind": "manifest",
                        "video_id": str(video["id"]),
                        "model": channel,
                        "error": str(exc),
                    })
                    continue
                if channel == "visual":
                    visual_models.add(
                        str(
                            channel_manifest.get("model_key")
                            or self.settings.visual_model
                        )
                    )
                else:
                    model_name = self._semantic_model_for_channel(
                        channel_manifest
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

    def _query_rows_for_videos(
        self,
        client,
        modality: str,
        video_ids: list[str],
        asset_versions: dict[str, str],
        output_fields: list[str],
        profiler: RetrievalProfiler | None,
    ) -> dict[str, list[dict]]:
        """Resolve one bulk query behind a unit-testable data boundary."""
        from app.vector_store.milvus.milvus_search import query_rows_for_videos

        return query_rows_for_videos(
            client,
            modality,
            video_ids,
            asset_versions,
            output_fields,
            profiler,
        )

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

    def _resolve_face_query(self, text: str | None, image_path: str | None) -> np.ndarray | None:
        if image_path:
            return self._face().encode_reference(image_path)
        if not text:
            return None
        entity = self.catalog.find_entity_in_text(text)
        if entity and entity.get("embedding_path") and Path(entity["embedding_path"]).exists():
            return np.load(entity["embedding_path"])["embedding"]
        return None

    def _semantic_query(
        self,
        text: str,
        channel_manifest: dict,
        embeddings: np.ndarray | None,
        semantic_queries: dict[str, np.ndarray | None],
        profiler: RetrievalProfiler | None,
    ) -> np.ndarray | None:
        if embeddings is None:
            return None
        model_name = self._semantic_model_for_channel(channel_manifest)
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
        channel_manifest: dict,
    ) -> str | None:
        """Return the model only when this indexed channel has semantic data."""
        semantic_status = str(
            channel_manifest.get("semantic_status") or ""
        ).strip().casefold()
        if semantic_status != "complete":
            return None
        return str(
            channel_manifest.get("semantic_model_key")
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
        prefetched_rows: dict[str, list[dict]] | None = None,
    ) -> list[Candidate]:
        from app.vector_store.milvus.milvus_search import (
            milvus_asr_candidates_hybrid,
            milvus_face_candidates,
            milvus_ocr_candidates_hybrid,
            milvus_visual_candidates,
        )

        if client is None:
            client = self._get_milvus_client()
        prefetched_rows = prefetched_rows or {}
        video_id = video["id"]
        index_dir = self.settings.index_dir / video_id
        indexed = set(video.get("indexed_modalities") or [])
        candidates: list[Candidate] = []
        if "visual" in modalities and "visual" in indexed:
            manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "visual"
            )
            asset_version = _published_asset_version(
                channel_manifest, str(video.get("name") or video_id), "visual"
            )
            visual_model = str(channel_manifest.get("model_key") or self.settings.visual_model)
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
            _manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "face"
            )
            candidates.extend(milvus_face_candidates(
                client,
                video_id,
                face_query,
                _published_asset_version(
                    channel_manifest, str(video.get("name") or video_id), "face"
                ),
                channel_limits["face"],
                None,  # threshold=None → settings.face_identity_threshold
                profiler,
            ))
        if "asr" in modalities and text and "asr" in indexed:
            _manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "asr"
            )
            model_name = self._semantic_model_for_channel(channel_manifest)
            semantic_query = (
                semantic_queries.get(model_name)
                if model_name is not None
                else None
            )
            candidates.extend(milvus_asr_candidates_hybrid(
                client,
                video_id,
                _published_asset_version(
                    channel_manifest, str(video.get("name") or video_id), "asr"
                ),
                text,
                semantic_query,
                channel_limits["asr"],
                profiler,
            ))
        if "ocr" in modalities and text and "ocr" in indexed:
            _manifest, channel_manifest, _index_file = _channel_manifest_for(
                video, index_dir, "ocr"
            )
            model_name = self._semantic_model_for_channel(channel_manifest)
            semantic_query = (
                semantic_queries.get(model_name)
                if model_name is not None
                else None
            )
            candidates.extend(milvus_ocr_candidates_hybrid(
                client,
                video_id,
                _published_asset_version(
                    channel_manifest, str(video.get("name") or video_id), "ocr"
                ),
                text,
                semantic_query,
                channel_limits["ocr"],
                profiler,
                rows=prefetched_rows.get("ocr"),
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
        """Encode all manifest-selected query models before candidate scoring."""
        query_texts: list[str | None] = (
            list(dict.fromkeys(visual_subqueries or []))
            if text and visual_subqueries
            else [text]
        )
        for video in videos:
            requested = requested_by_video[video["id"]]
            index_dir = self.settings.index_dir / video["id"]
            if "visual" in requested:
                _manifest, channel_manifest, _index_file = _channel_manifest_for(
                    video, index_dir, "visual"
                )
                model_key = str(
                    channel_manifest.get("model_key")
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
                    _manifest, channel_manifest, _index_file = (
                        _channel_manifest_for(video, index_dir, channel)
                    )
                    if self._semantic_model_for_channel(
                        channel_manifest
                    ) is not None:
                        self._semantic_query(
                            text,
                            channel_manifest,
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
                self._resolve_face_query(text, image_path)
                if "face" in modalities
                else None
            )
        from app.vector_store.milvus.milvus_search import BULK_QUERY_FIELDS

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
        milvus_video_id_set = set(milvus_video_ids)
        milvus_client = None
        if milvus_video_ids:
            milvus_client = self._get_milvus_client()

        batch_size = self.settings.milvus_search_video_batch_size
        for batch_offset in range(0, len(videos), batch_size):
            batch_videos = videos[batch_offset:batch_offset + batch_size]
            batch_video_ids = [
                video["id"]
                for video in batch_videos
                if video["id"] in milvus_video_id_set
            ]
            prefetched_rows: dict[tuple[str, str], list[dict]] = {}
            modality_rows: dict[str, list[dict]] = {}
            if milvus_client is not None and batch_video_ids:
                for modality, output_fields in BULK_QUERY_FIELDS.items():
                    eligible_ids = [
                        video_id
                        for video_id in batch_video_ids
                        if modality in requested_by_video[video_id]
                    ]
                    if not eligible_ids:
                        continue
                    asset_versions = {}
                    for video in batch_videos:
                        video_id = video["id"]
                        if video_id not in eligible_ids:
                            continue
                        _manifest, channel_manifest, _index_file = _channel_manifest_for(
                            video, self.settings.index_dir / video_id, modality
                        )
                        asset_versions[video_id] = _published_asset_version(
                            channel_manifest,
                            str(video.get("name") or video_id),
                            modality,
                        )
                    modality_rows = self._query_rows_for_videos(
                        milvus_client,
                        modality,
                        eligible_ids,
                        asset_versions,
                        output_fields,
                        profiler,
                    )
                    for video_id in eligible_ids:
                        prefetched_rows[(video_id, modality)] = modality_rows.get(
                            video_id, []
                        )

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
                            prefetched_rows={
                                modality: prefetched_rows.get((video_id, modality), [])
                            }
                            if modality in BULK_QUERY_FIELDS
                            else None,
                        )
                    candidates.extend(
                        item for item in modality_candidates if item.modality == modality
                    )
            # Raw embeddings for this batch become unreachable before the next
            # Milvus query, bounding peak memory by video batch size.
            prefetched_rows.clear()
            modality_rows.clear()

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
