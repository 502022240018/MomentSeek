#!/usr/bin/env python3
"""索引测试数据到DiskANN集合"""
import sys
sys.path.insert(0, 'backend')

import numpy as np
from app.indexing.milvus_client import get_milvus_client
from pymilvus import Collection

print("Step 1: Connecting to MilvusClient...")
client = get_milvus_client()
col = Collection("visual_embeddings")

# 检查当前实体数
print(f"  Current entities: {col.num_entities}")

# 生成测试数据
video_id = "test_diskann_video"
num_frames = 300
dim = 1152

print(f"\nStep 2: Generating {num_frames} test frames...")
data = []
for i in range(num_frames):
    # 生成随机向量并归一化（COSINE距离需要）
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)

    # 生成正确的主键
    from app.indexing.milvus_schema import visual_pk
    pk = visual_pk(video_id, "v1", i)

    data.append({
        "pk": pk,
        "video_id": video_id,
        "asset_version": "v1",
        "model_version": "siglip2-so400m-v1",
        "frame_idx": i,
        "timestamp_ms": i * 1000,
        "segment_id": i // 10,
        "segment_start_ms": (i // 10) * 10000,
        "segment_end_ms": ((i // 10) + 1) * 10000,
        "embedding": vec.tolist(),
    })

print("  ✓ Data generated")

print("\nStep 3: Inserting to Milvus with DiskANN index...")
try:
    result = col.insert(data)
    print(f"  Insert result: {result}")
    col.flush()
    print("  ✓ Data flushed")
except Exception as e:
    print(f"  ✗ Insert failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Verifying insertion...")
print(f"  Collection entities: {col.num_entities}")

if col.num_entities >= num_frames:
    print(f"\n✅ Successfully indexed {num_frames} frames with DiskANN!")
    print(f"  Test video_id: {video_id}")
else:
    print(f"\n⚠️  Expected {num_frames} entities, got {col.num_entities}")
