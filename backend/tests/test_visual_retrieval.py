#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/backend')

import numpy as np
import time
from app.indexing.milvus_client import get_milvus_client
from app.indexing.milvus_search import milvus_visual_candidates, milvus_visual_candidates_ann
from app.settings import get_settings

video_id = "4c7f80cff1374441ae19c8de1c7a0b66"
duration_ms = int(59.833 * 1000)

# 创建查询向量
np.random.seed(42)
query = np.random.randn(1152).astype(np.float32)
query = query / np.linalg.norm(query)

settings = get_settings()
client = get_milvus_client()

print("=" * 70)
print("Visual检索优化测试")
print("=" * 70)
print(f"测试视频: {video_id}")
print(f"视频时长: {duration_ms}ms ({duration_ms/1000:.1f}s)")
print(f"查询向量维度: {query.shape}")
print()

# 测试1: 旧版全量Query
print("【测试1】旧版策略 - 全量Query + Python侧计算")
print("-" * 70)
start = time.time()
try:
    candidates_old = milvus_visual_candidates(
        client, video_id, query,
        duration_ms=duration_ms,
        limit=20
    )
    elapsed_old = time.time() - start
    print(f"✅ 执行成功")
    print(f"   耗时: {elapsed_old:.3f}s")
    print(f"   返回候选数: {len(candidates_old)}")
    if candidates_old:
        print(f"   Top 3 候选:")
        for i, c in enumerate(candidates_old[:3], 1):
            print(f"     {i}. {c.start_time:.1f}s-{c.end_time:.1f}s, score={c.score:.4f}, decision={c.decision}")
except Exception as e:
    print(f"❌ 执行失败: {e}")
    import traceback
    traceback.print_exc()
    elapsed_old = None
    candidates_old = []

print()

# 测试2: 新版ANN搜索 + 混合采样
print("【测试2】新版策略 - ANN搜索 + 混合采样")
print("-" * 70)
print(f"配置: ann_top_k={settings.visual_ann_top_k}, sample_size={settings.visual_sample_size}")
start = time.time()
try:
    candidates_new = milvus_visual_candidates_ann(
        client, video_id, query,
        limit=20
    )
    elapsed_new = time.time() - start
    print(f"✅ 执行成功")
    print(f"   耗时: {elapsed_new:.3f}s")
    print(f"   返回候选数: {len(candidates_new)}")
    if candidates_new:
        print(f"   Top 3 候选:")
        for i, c in enumerate(candidates_new[:3], 1):
            print(f"     {i}. {c.start_time:.1f}s-{c.end_time:.1f}s, score={c.score:.4f}, decision={c.decision}")
except Exception as e:
    print(f"❌ 执行失败: {e}")
    import traceback
    traceback.print_exc()
    elapsed_new = None
    candidates_new = []

print()

# 性能对比
if elapsed_old and elapsed_new:
    print("=" * 70)
    print("性能对比")
    print("=" * 70)
    speedup = elapsed_old / elapsed_new
    print(f"旧版耗时: {elapsed_old:.3f}s")
    print(f"新版耗时: {elapsed_new:.3f}s")
    print(f"加速比: {speedup:.2f}x")

    if speedup > 1.5:
        print(f"✅ 新版性能提升显著！")
    elif speedup > 1.0:
        print(f"✅ 新版略有提升")
    else:
        print(f"⚠️  新版未见明显提升（可能视频太短或数据量太小）")

print()
print("=" * 70)
print("测试完成")
print("=" * 70)
