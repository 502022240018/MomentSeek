#!/usr/bin/env python3
"""Visual模态ANN优化 - 端到端测试脚本

独立运行脚本，验证优化效果:
- 对比新旧版本的性能和准确性
- 输出详细的测试报告
- 支持指定测试视频ID

用法:
    python backend/scripts/test_visual_ann.py [--video-id VIDEO_ID] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.indexing.milvus_client import MilvusClient, get_milvus_client
from app.indexing.milvus_search import milvus_visual_candidates
from app.settings import get_settings


def find_test_video() -> str | None:
    """自动查找一个已索引的视频"""
    try:
        client = get_milvus_client()
        collection = client.collection("visual_embeddings")

        # 查询任意一个video_id
        results = collection.query(
            expr="",
            output_fields=["video_id"],
            limit=1,
        )

        if results:
            return results[0]["video_id"]
    except Exception as e:
        print(f"查找测试视频失败: {e}")

    return None


def generate_test_query() -> np.ndarray:
    """生成测试查询向量（归一化随机向量）"""
    vec = np.random.randn(1152).astype(np.float32)
    return vec / np.linalg.norm(vec)


def compare_versions(video_id: str, query: np.ndarray, limit: int = 20):
    """对比新旧版本"""
    settings = get_settings()
    client = get_milvus_client()

    print(f"\n{'='*70}")
    print(f"Visual模态优化测试")
    print(f"{'='*70}")
    print(f"视频ID: {video_id}")
    print(f"返回限制: {limit}")
    print(f"查询向量维度: {query.shape}")

    # ===== 测试1: 旧版本（全量query） =====
    print(f"\n{'-'*70}")
    print("测试1: 旧版本（全量query）")
    print(f"{'-'*70}")

    settings.visual_use_ann_search = False
    old_start = time.perf_counter()
    try:
        old_candidates = milvus_visual_candidates(
            client, video_id, query, limit=limit
        )
        old_time = time.perf_counter() - old_start
        old_success = True

        print(f"✓ 执行成功")
        print(f"  耗时: {old_time*1000:.1f}ms")
        print(f"  召回候选: {len(old_candidates)}")
        if old_candidates:
            print(f"  Top-1分数: {old_candidates[0].score:.4f}")
            print(f"  Top-1决策: {old_candidates[0].decision}")
            print(f"  Top-1 z-score: {old_candidates[0].features.get('z_score', 'N/A')}")
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        old_candidates = []
        old_time = 0
        old_success = False

    # ===== 测试2: 新版本（ANN + 采样） =====
    print(f"\n{'-'*70}")
    print("测试2: 新版本（ANN + 混合采样）")
    print(f"{'-'*70}")

    settings.visual_use_ann_search = True
    new_start = time.perf_counter()
    try:
        new_candidates = milvus_visual_candidates(
            client, video_id, query, limit=limit
        )
        new_time = time.perf_counter() - new_start
        new_success = True

        print(f"✓ 执行成功")
        print(f"  耗时: {new_time*1000:.1f}ms")
        print(f"  召回候选: {len(new_candidates)}")
        if new_candidates:
            print(f"  Top-1分数: {new_candidates[0].score:.4f}")
            print(f"  Top-1决策: {new_candidates[0].decision}")
            print(f"  Top-1 z-score: {new_candidates[0].features.get('z_score', 'N/A')}")
            print(f"  来源: {new_candidates[0].features.get('source', 'N/A')}")
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        new_candidates = []
        new_time = 0
        new_success = False

    # ===== 对比分析 =====
    if old_success and new_success:
        print(f"\n{'='*70}")
        print("对比分析")
        print(f"{'='*70}")

        # 性能提升
        if old_time > 0:
            speedup = old_time / new_time
            latency_reduction = (1 - 1/speedup) * 100
            print(f"\n📊 性能指标:")
            print(f"  旧版延迟: {old_time*1000:.1f}ms")
            print(f"  新版延迟: {new_time*1000:.1f}ms")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  延迟降低: {latency_reduction:.1f}%")

            if speedup >= 2.5:
                print(f"  ✓ 超越目标 (目标: 1.67x, 实际: {speedup:.2f}x)")
            elif speedup >= 1.67:
                print(f"  ✓ 达到目标 (目标: 1.67x, 实际: {speedup:.2f}x)")
            else:
                print(f"  ✗ 未达目标 (目标: 1.67x, 实际: {speedup:.2f}x)")

        # 准确性对比
        if old_candidates and new_candidates:
            print(f"\n🎯 准确性指标:")

            # Top-K Jaccard
            for k in [5, 10, 20]:
                k = min(k, len(old_candidates), len(new_candidates))
                old_segments = {c.unit_id for c in old_candidates[:k]}
                new_segments = {c.unit_id for c in new_candidates[:k]}

                intersection = len(old_segments & new_segments)
                union = len(old_segments | new_segments)
                jaccard = intersection / union if union > 0 else 0.0

                print(f"  Top-{k} Jaccard: {jaccard:.3f}", end="")
                if jaccard >= 0.85:
                    print(" ✓")
                elif jaccard >= 0.70:
                    print(" ⚠")
                else:
                    print(" ✗")

            # 分数相关性
            old_scores = [c.score for c in old_candidates[:min(20, len(old_candidates))]]
            new_scores = [c.score for c in new_candidates[:min(20, len(new_candidates))]]

            if len(old_scores) > 1 and len(new_scores) > 1:
                # Spearman秩相关
                from scipy.stats import spearmanr
                correlation, _ = spearmanr(old_scores, new_scores)
                print(f"  分数秩相关: {correlation:.3f}", end="")
                if correlation >= 0.8:
                    print(" ✓")
                else:
                    print(" ⚠")

        # 决策一致性
        if old_candidates and new_candidates:
            print(f"\n⚖️  决策一致性:")
            old_decisions = [c.decision for c in old_candidates[:10]]
            new_decisions = [c.decision for c in new_candidates[:10]]

            strong_old = old_decisions.count("strong")
            strong_new = new_decisions.count("strong")

            print(f"  旧版strong决策: {strong_old}/10")
            print(f"  新版strong决策: {strong_new}/10")
            print(f"  一致性: {sum(o==n for o,n in zip(old_decisions, new_decisions))}/10")

    # ===== 总结 =====
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")

    if old_success and new_success:
        if old_time > 0 and new_time < old_time:
            print("✓ 新版本在性能上有提升")
        else:
            print("✗ 性能未提升")

        if old_candidates and new_candidates:
            # 简单的pass/fail判断
            top10_old = {c.unit_id for c in old_candidates[:10]}
            top10_new = {c.unit_id for c in new_candidates[:10]}
            jaccard = len(top10_old & top10_new) / len(top10_old | top10_new) if top10_old | top10_new else 0

            if jaccard >= 0.85:
                print("✓ 准确性达标 (Top-10 Jaccard >= 0.85)")
            elif jaccard >= 0.70:
                print("⚠ 准确性可接受 (Top-10 Jaccard >= 0.70)")
            else:
                print("✗ 准确性不足 (Top-10 Jaccard < 0.70)")
    elif new_success:
        print("⚠ 仅新版本成功，无法对比")
    elif old_success:
        print("✗ 新版本执行失败")
    else:
        print("✗ 两个版本均失败")

    return old_success and new_success


def main():
    parser = argparse.ArgumentParser(description="Visual模态ANN优化测试")
    parser.add_argument(
        "--video-id",
        type=str,
        help="测试视频ID（不指定则自动查找）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="返回候选数量（默认20）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 获取测试视频ID
    video_id = args.video_id
    if not video_id:
        print("未指定视频ID，自动查找...")
        video_id = find_test_video()
        if not video_id:
            print("✗ 无法找到测试视频，请先索引至少一个视频")
            sys.exit(1)
        print(f"✓ 找到测试视频: {video_id}")

    # 生成测试查询
    query = generate_test_query()

    # 执行测试
    try:
        success = compare_versions(video_id, query, args.limit)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
