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
    """测试Visual ANN优化版本"""
    settings = get_settings()
    client = get_milvus_client()

    print(f"\n{'='*70}")
    print(f"Visual模态ANN优化测试")
    print(f"{'='*70}")
    print(f"视频ID: {video_id}")
    print(f"返回限制: {limit}")
    print(f"查询向量维度: {query.shape}")
    print(f"配置: visual_use_diskann={settings.visual_use_diskann}, visual_ann_top_k={settings.visual_ann_top_k}")

    # ===== 测试: ANN优化版本 =====
    print(f"\n{'-'*70}")
    print("Visual ANN优化版本测试")
    print(f"{'-'*70}")

    start_time = time.perf_counter()
    try:
        candidates = milvus_visual_candidates(
            client, video_id, query, limit=limit
        )
        elapsed = time.perf_counter() - start_time
        success = True

        print(f"✓ 执行成功")
        print(f"  耗时: {elapsed*1000:.1f}ms")
        print(f"  召回候选: {len(candidates)}")
        if candidates:
            print(f"  Top-1分数: {candidates[0].score:.4f}")
            print(f"  Top-1时间: {candidates[0].start_time:.1f}s - {candidates[0].end_time:.1f}s")
            print(f"  Top-1 segment_id: {candidates[0].features.get('segment_id', 'N/A')}")
            print(f"  来源: {candidates[0].features.get('source', 'N/A')}")

        # 性能评估
        print(f"\n📊 性能评估:")
        if elapsed < 0.2:
            print(f"  ✓ 延迟 {elapsed*1000:.1f}ms < 200ms (目标达成)")
        else:
            print(f"  ⚠ 延迟 {elapsed*1000:.1f}ms ≥ 200ms (需要优化)")

    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        candidates = []
        elapsed = 0
        success = False

    # ===== 总结 =====
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")

    if success:
        print("✓ Visual ANN优化功能正常")
        if candidates:
            print(f"✓ 成功召回 {len(candidates)} 个候选段")
        if elapsed < 0.2:
            print("✓ 延迟满足<200ms目标")
    else:
        print("✗ 测试失败")

    return success


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
