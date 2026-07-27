#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/backend')

from pymilvus import utility, connections, Collection
from app.settings import get_settings

settings = get_settings()
connections.connect(host=settings.milvus_host, port=settings.milvus_port)

collections = utility.list_collections()
print(f"Milvus collections: {collections}")

if "visual_embeddings" in collections:
    col = Collection("visual_embeddings")
    print(f"Visual collection total entities: {col.num_entities}")

    # 查询测试视频的数据
    video_id = "4c7f80cff1374441ae19c8de1c7a0b66"
    results = col.query(
        expr=f'video_id == "{video_id}"',
        output_fields=["frame_idx", "timestamp_ms", "segment_id"],
        limit=10
    )
    print(f"\nTest video '{video_id}' has frames (first 10):")
    for r in results:
        print(f"  frame {r['frame_idx']}: {r['timestamp_ms']}ms, segment={r['segment_id']}")

    print(f"\n✅ Milvus数据验证完成，可以进行检索测试")
