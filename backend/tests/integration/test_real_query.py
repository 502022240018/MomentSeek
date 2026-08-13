#!/usr/bin/env python3
"""测试真实查询的得分分布"""
import pytest
import sys
sys.path.insert(0, '/app/backend')

pytestmark = pytest.mark.integration

from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_search import milvus_visual_candidates
from app.settings import get_settings
import numpy as np

settings = get_settings()
client = get_milvus_client()

video_id = "4c7f80cff1374441ae19c8de1c7a0b66"

# 使用随机查询向量（模拟任意文本查询）
query = np.random.randn(1152).astype(np.float32)
query = query / (np.linalg.norm(query) + 1e-9)

print(f"测试视频: {video_id}")
print(f"查询向量norm: {np.linalg.norm(query):.4f}")
print()

# 测试旧版全量query
print("【旧版策略 - 全量Query】")
candidates = milvus_visual_candidates(
    client, video_id, [query],
    limit=20, profile="balanced"
)

print(f"返回候选数: {len(candidates)}")
if candidates:
    print("\nTop 5候选:")
    for i, c in enumerate(candidates[:5], 1):
        print(f"  {i}. {c.start_time:.1f}s-{c.end_time:.1f}s")
        print(f"     score={c.score:.4f}, decision={c.decision}")

    # 查看所有候选的得分分布
    scores = [c.score for c in candidates]
    print(f"\n得分统计:")
    print(f"  最高: {max(scores):.4f}")
    print(f"  最低: {min(scores):.4f}")
    print(f"  平均: {np.mean(scores):.4f}")
