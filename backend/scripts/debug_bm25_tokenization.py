#!/usr/bin/env python3
"""
诊断 Milvus BM25 分词问题

检查：
1. Query分词结果
2. BM25分数计算
3. 混合检索权重影响
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.settings import Settings
from app.indexing.milvus_client import MilvusClient
from pymilvus import WeightedRanker
import numpy as np


def debug_bm25_search(query: str, video_id: str):
    """调试 BM25 分词和检索"""
    settings = Settings()
    client = MilvusClient(settings)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Video: {video_id}")
    print(f"{'='*60}\n")

    # 获取collection
    col = client.collection_for_name("ocr_embeddings")

    # 1. 纯BM25检索（词面权重=1.0）
    print("1️⃣ 纯BM25检索（只看词面匹配）")
    print("-" * 60)

    sparse_req = client.ann_search_request(
        data=[[]],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=20,
        expr=f'video_id == "{video_id}"'
    )

    # 构造一个空的dense请求
    dummy_embedding = np.zeros((1, 768), dtype=np.float32)
    dense_req = client.ann_search_request(
        data=dummy_embedding.tolist(),
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"search_list": 100}},
        limit=20,
        expr=f'video_id == "{video_id}"'
    )

    # 混合检索，词面权重=1.0（相当于纯BM25）
    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=WeightedRanker(0.0, 1.0),  # semantic=0, lexical=1
        limit=20,
        output_fields=["frame_idx", "frame_ms", "text", "avg_box_score"],
    )

    if results[0]:
        print(f"Top 10 纯BM25结果：")
        for i, hit in enumerate(results[0][:10]):
            text = hit.entity.get("text", "")
            frame_ms = hit.entity.get("frame_ms", 0)
            score = float(hit.score)

            # 检查是否完全匹配
            is_exact = query in text
            match_mark = "✓ 完全匹配" if is_exact else ""

            print(f"  [{i+1}] 分数={score:.4f} | 时间={frame_ms/1000:.1f}s | {match_mark}")
            print(f"      文本: {text[:80]}")
    else:
        print("❌ 没有结果")

    # 2. 混合检索（当前配置：词面0.8 + 语义0.2）
    print(f"\n2️⃣ 混合检索（词面权重={settings.ocr_lexical_weight}）")
    print("-" * 60)

    # 需要真实的embedding
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    query_embedding = model.encode([query])[0].astype(np.float32)

    dense_req_real = client.ann_search_request(
        data=[query_embedding.tolist()],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"search_list": settings.ocr_diskann_search_list}},
        limit=20,
        expr=f'video_id == "{video_id}"'
    )

    results_hybrid = col.hybrid_search(
        reqs=[dense_req_real, sparse_req],
        rerank=WeightedRanker(1 - settings.ocr_lexical_weight, settings.ocr_lexical_weight),
        limit=20,
        output_fields=["frame_idx", "frame_ms", "text", "avg_box_score"],
    )

    if results_hybrid[0]:
        print(f"Top 10 混合检索结果：")
        for i, hit in enumerate(results_hybrid[0][:10]):
            text = hit.entity.get("text", "")
            frame_ms = hit.entity.get("frame_ms", 0)
            score = float(hit.score)

            is_exact = query in text
            match_mark = "✓ 完全匹配" if is_exact else ""

            print(f"  [{i+1}] 分数={score:.4f} | 时间={frame_ms/1000:.1f}s | {match_mark}")
            print(f"      文本: {text[:80]}")
    else:
        print("❌ 没有结果")

    # 3. 分析完全匹配的分数分布
    print(f"\n3️⃣ 完全匹配的分数分析")
    print("-" * 60)

    exact_matches = [
        (float(hit.score), hit.entity.get("frame_ms", 0) / 1000)
        for hit in results_hybrid[0]
        if query in hit.entity.get("text", "")
    ]

    if exact_matches:
        exact_matches.sort(reverse=True)
        print(f"找到 {len(exact_matches)} 个完全匹配的帧")
        print(f"最高分: {exact_matches[0][0]:.4f} @ {exact_matches[0][1]:.1f}s")
        print(f"最低分: {exact_matches[-1][0]:.4f} @ {exact_matches[-1][1]:.1f}s")
        print(f"分数范围: {exact_matches[0][0] / exact_matches[-1][0]:.2f}x")

        if exact_matches[0][0] / exact_matches[-1][0] > 2:
            print("\n⚠️ 警告：相同词面的分数差异超过2倍！")
            print("可能原因：")
            print("  1. BM25的IDF惩罚：高频词分数低")
            print("  2. 语义向量差异：不同上下文的embedding不同")
            print("  3. OCR置信度影响：avg_box_score不同")
    else:
        print("❌ 没有找到完全匹配的帧")

    print("\n" + "="*60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python debug_bm25_tokenization.py <query> <video_id>")
        print("示例: python debug_bm25_tokenization.py '最低工资' 'video_001'")
        sys.exit(1)

    query = sys.argv[1]
    video_id = sys.argv[2]

    debug_bm25_search(query, video_id)
