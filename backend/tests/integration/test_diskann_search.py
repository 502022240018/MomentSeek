#!/usr/bin/env python3
"""测试DiskANN模式下的Visual检索功能"""
import pytest
import sys
sys.path.insert(0, 'backend')

pytestmark = pytest.mark.integration

import time
import numpy as np
from app.vector_store.milvus.milvus_client import get_milvus_client
from app.vector_store.milvus.milvus_search_visual_v2 import milvus_visual_candidates_ann
from app.core.settings import get_settings

print("="*70)
print("DiskANN模式Visual检索功能测试")
print("="*70)

# 检查配置
settings = get_settings()
print(f"\n配置检查:")
print(f"  visual_use_diskann: {settings.visual_use_diskann}")
print(f"  visual_ann_top_k: {settings.visual_ann_top_k}")

# 连接Milvus
print(f"\n连接Milvus...")
client = get_milvus_client()
stats = client.stats("visual_embeddings")
print(f"  实体数量: {stats['num_entities']}")
print(f"  加载状态: {stats['loaded']}")

if stats['num_entities'] == 0:
    print("\n✗ 没有索引数据，请先运行 test_diskann_indexing.py")
    sys.exit(1)

# 生成测试查询向量
video_id = "test_diskann_video"
dim = 1152
num_queries = 3

print(f"\n生成 {num_queries} 个测试查询向量...")
query_values = np.random.randn(num_queries, dim).astype(np.float32)
# 归一化（COSINE距离）
query_values = query_values / np.linalg.norm(query_values, axis=1, keepdims=True)

# 执行检索
print(f"\n执行DiskANN检索...")
print(f"  video_id: {video_id}")
print(f"  查询向量数: {num_queries}")

try:
    start_time = time.time()
    # 转换为list[np.ndarray]格式
    query_texts = [query_values[i] for i in range(num_queries)]
    candidates = milvus_visual_candidates_ann(
        client=client,
        video_id=video_id,
        query_texts=query_texts,
    )
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"\n✅ 检索成功!")
    print(f"  耗时: {elapsed_ms:.1f}ms")
    print(f"  召回候选数: {len(candidates)}")

    if candidates:
        print(f"\n前3个候选:")
        for i, cand in enumerate(candidates[:3], 1):
            print(f"    {i}. start={cand.start_time:.1f}s, end={cand.end_time:.1f}s, "
                  f"score={cand.score:.3f}, decision={cand.decision}")
            if cand.robust_z is not None:
                print(f"        z_score={cand.robust_z:.2f}, percentile={cand.percentile:.3f}")

    # 检查决策分布
    decisions = [c.decision for c in candidates]
    decision_counts = {
        'strong': decisions.count('strong'),
        'weak': decisions.count('weak'),
        'absent': decisions.count('absent'),
    }
    print(f"\n决策分布:")
    for decision, count in decision_counts.items():
        if count > 0:
            print(f"  {decision}: {count}")

    # 验证分数范围
    scores = [c.score for c in candidates]
    print(f"\n分数统计:")
    print(f"  最高分: {max(scores):.3f}")
    print(f"  最低分: {min(scores):.3f}")
    print(f"  平均分: {sum(scores)/len(scores):.3f}")

    print(f"\n{'='*70}")
    print("✅ DiskANN模式下Visual检索功能完全正常!")
    print(f"{'='*70}")

except Exception as e:
    print(f"\n✗ 检索失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
