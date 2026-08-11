"""测试 Face 聚合的分数兼容性检查（修复时间相邻但 cosine 差距大的误合并）。"""
import pytest

from app.retrieval.search import (
    Candidate,
    _face_scores_compatible,
    _groups,
    _should_merge,
)


def test_face_scores_compatible_same_cosine():
    """组内 cosine 0.72，候选 0.72 → 兼容"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.98,
        modality="face",
        raw_score=0.72,
    )
    assert _face_scores_compatible(group, candidate) is True


def test_face_scores_compatible_small_drop():
    """组内最佳 0.72，候选 0.65 → drop=0.07 < 0.15 → 兼容"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.92,
        modality="face",
        raw_score=0.65,
    )
    assert _face_scores_compatible(group, candidate) is True


def test_face_scores_compatible_large_drop_rejected():
    """组内最佳 0.72，候选 -0.01 → drop=0.73 > 0.15 → 不兼容"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.004,
        modality="face",
        raw_score=-0.01,
        above_threshold=False,
    )
    assert _face_scores_compatible(group, candidate) is False


def test_face_scores_compatible_symmetric_bandwidth():
    """对称带宽：组内 [0.60, 0.72]，候选 0.58 → 与最弱差 0.02，与最强差 0.14 → 兼容"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        ),
        Candidate(
            video_id="v1",
            start_time=11.2,
            end_time=12.0,
            score=0.95,
            modality="face",
            raw_score=0.60,
        ),
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=12.5,
        end_time=13.5,
        score=0.93,
        modality="face",
        raw_score=0.58,
    )
    # 0.58 >= 0.72 - 0.15 = 0.57 ✓, 0.58 <= 0.60 + 0.15 = 0.75 ✓
    assert _face_scores_compatible(group, candidate) is True


def test_face_scores_compatible_non_face_candidate():
    """候选不是 face → 返回 False"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.85,
        modality="ocr",
    )
    assert _face_scores_compatible(group, candidate) is False


def test_face_scores_compatible_no_raw_score():
    """候选缺少 raw_score → 退化为 True（兜底）"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.98,
        modality="face",
        raw_score=None,
    )
    assert _face_scores_compatible(group, candidate) is True


def test_should_merge_face_only_rejects_low_cosine():
    """Face-only 场景：时间相邻但 cosine 差距大 → 拒绝合并"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,  # gap=0.5 < 2，时间满足
        end_time=12.5,
        score=0.004,
        modality="face",
        raw_score=-0.01,  # cosine 差距 0.73 > 0.15，分数不满足
        above_threshold=False,
    )
    # 旧逻辑：只要 near=True 就合并 → 会合并
    # 新逻辑：near=True 且 _face_scores_compatible=True 才合并 → 拒绝
    assert _should_merge(group, candidate, gap=2.0, max_duration=15.0) is False


def test_should_merge_face_only_accepts_compatible_cosine():
    """Face-only 场景：时间相邻且 cosine 兼容 → 合并"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        )
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,  # gap=0.5 < 2
        end_time=12.5,
        score=0.95,
        modality="face",
        raw_score=0.68,  # drop=0.04 < 0.15
    )
    assert _should_merge(group, candidate, gap=2.0, max_duration=15.0) is True


def test_should_merge_face_mixed_modality_no_score_check():
    """混合模态（face + ocr）→ 不走 face-only 分支，不检查 cosine 兼容性"""
    group = [
        Candidate(
            video_id="v1",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
        ),
        Candidate(
            video_id="v1",
            start_time=10.5,
            end_time=11.2,
            score=0.88,
            modality="ocr",
        ),
    ]
    candidate = Candidate(
        video_id="v1",
        start_time=11.5,
        end_time=12.5,
        score=0.004,
        modality="face",
        raw_score=-0.01,  # 即使 cosine 很低
        above_threshold=False,
    )
    # 混合模态场景，只要 near=True 就合并（OCR 锚定了片段，face 作为辅助）
    assert _should_merge(group, candidate, gap=2.0, max_duration=15.0) is True


def test_regression_user_bug_scenario():
    """回归测试：用户报告的真实场景 - 99% 和 0.4% 被合并"""
    # 第一个候选：目标人脸，cosine=0.72，confidence=99%
    group = [
        Candidate(
            video_id="test_video",
            start_time=10.0,
            end_time=11.0,
            score=0.99,
            modality="face",
            raw_score=0.72,
            above_threshold=True,
            decision="absolute_hit",
        )
    ]
    # 第二个候选：非目标人脸，cosine=-0.01，confidence=0.4%
    low_score_candidate = Candidate(
        video_id="test_video",
        start_time=11.5,  # 时间相邻（gap=0.5s < 2s）
        end_time=12.5,
        score=0.004,
        modality="face",
        raw_score=-0.01,
        above_threshold=False,
        decision="weak",
    )

    # 旧逻辑会合并（只检查时间），新逻辑应拒绝（cosine 差距 0.73 > 0.15）
    should_merge = _should_merge(group, low_score_candidate, gap=2.0, max_duration=15.0)
    assert should_merge is False, (
        "修复前：时间相邻的高分和低分 face 被合并，导致片段显示 99% 但 evidence 含 0.4% 项。"
        "修复后：cosine 差距超过 0.15 应拒绝合并。"
    )


def test_groups_prefers_overlapping_ocr_group_over_higher_score_nearby_group():
    """A mixed-modality candidate joins the closest compatible group."""
    higher_score_later = Candidate(
        video_id="v1",
        start_time=9.1,
        end_time=10.0,
        score=0.95,
        modality="ocr",
    )
    lower_score_overlapping = Candidate(
        video_id="v1",
        start_time=5.0,
        end_time=7.4,
        score=0.80,
        modality="ocr",
    )
    face = Candidate(
        video_id="v1",
        start_time=7.0,
        end_time=7.4,
        score=0.90,
        modality="face",
        raw_score=0.70,
    )

    groups = _groups(
        [higher_score_later, lower_score_overlapping, face],
        gap=2.0,
        max_duration=15.0,
    )

    face_group = next(group for group in groups if face in group)
    assert lower_score_overlapping in face_group
    assert higher_score_later not in face_group
