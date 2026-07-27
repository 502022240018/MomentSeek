#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/backend')

from app.indexing.milvus_client import get_milvus_client
from app.settings import get_settings
from app.model_pool import ModelPool
from app.indexing.visual import ClipEncoder, VisualConfig
import numpy as np

settings = get_settings()
client = get_milvus_client()
pool = ModelPool()

video_id = "4c7f80cff1374441ae19c8de1c7a0b66"

# 使用真实的Visual encoder
print("加载Visual encoder...")
visual_config = VisualConfig(
    model=settings.visual_model,
    hf_cache_dir=settings.visual_hf_cache_dir
)
encoder = pool.get("visual_encoder", lambda: ClipEncoder(visual_config, "npu:0", model_cache_dir=settings.app_model_dir))
query_text = "a person walking"
query = encoder.encode_text([query_text])[0]
query = query / (np.linalg.norm(query) + 1e-9)

print(f"查询视频: {video_id}")
print(f"查询向量: shape={query.shape}, norm={np.linalg.norm(query):.4f}")
print()

# 测试ANN search
from app.indexing.milvus_search import _ann_search

print("【ANN Search测试】")
print(f"limit=500")
results = _ann_search(
    client, "visual", video_id, query.tolist(),
    limit=500,
    output_fields=["frame_idx", "timestamp_ms", "segment_id", "embedding"],
)
print(f"返回结果数: {len(results)}")
if results:
    print(f"第一条: frame_idx={results[0]['frame_idx']}, timestamp_ms={results[0]['timestamp_ms']}")
    print(f"embedding存在: {'embedding' in results[0]}")
    if 'embedding' in results[0]:
        emb = np.array(results[0]['embedding'])
        print(f"embedding shape: {emb.shape}")
        score = float(np.dot(query, emb))
        print(f"手动计算得分: {score:.4f}")
