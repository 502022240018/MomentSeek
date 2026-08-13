#!/usr/bin/env python3
"""
Milvus BM25混合检索诊断工具

测试目标：
1. 验证BM25 Function是否正常工作
2. 验证WeightedRanker的权重计算是否正确
3. 对比纯BM25、纯语义、混合检索的分数
4. 分析分数范围和归一化问题
"""
import sys
sys.path.insert(0, '/app/backend')

import numpy as np
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from app.settings import Settings
from app.indexing.milvus_client import MilvusClient

settings = Settings()

# 连接到Milvus
connections.connect(alias='default', host='localhost', port=19531)

print("="*80)
print("Milvus BM25混合检索诊断")
print("="*80)

# 检查OCR collection状态
try:
    col = Collection('ocr_embeddings')
    col.load()

    print(f"\n✓ Collection已加载")
    col.flush()
    num_entities = col.num_entities
    print(f"  记录数: {num_entities:,}")

    if num_entities == 0:
        print("\n❌ Collection为空，请先索引视频")
        sys.exit(1)

    # 检查schema
    schema = col.schema
    print(f"\n✓ Schema信息:")

    # 检查text字段
    text_field = None
    for field in schema.fields:
        if field.name == 'text':
            text_field = field
            break

    if text_field:
        print(f"  text字段:")
        print(f"    enable_analyzer: {text_field.params.get('enable_analyzer', False)}")
        analyzer_params = text_field.params.get('analyzer_params')
        if analyzer_params:
            print(f"    analyzer_params: {analyzer_params}")
        else:
            print(f"    analyzer_params: 未配置（使用默认）")

    # 检查Function
    if hasattr(schema, 'functions') and schema.functions:
        print(f"\n  Functions: {len(schema.functions)}个")
        for func in schema.functions:
            print(f"    - {func.name}")
            print(f"      type: {func.type}")
            print(f"      input: {func.input_field_names}")
            print(f"      output: {func.output_field_names}")
    else:
        print(f"\n  ❌ 没有Function（BM25可能未配置）")

    # 获取测试数据
    print(f"\n{'='*80}")
    print("步骤1: 获取测试数据")
    print("="*80)

    # 随机取一条记录作为测试
    sample = col.query(
        expr="",
        limit=1,
        output_fields=["video_id", "text", "frame_idx"]
    )

    if not sample:
        print("❌ 无法获取测试数据")
        sys.exit(1)

    test_video_id = sample[0]['video_id']
    test_text = sample[0]['text']

    print(f"测试视频ID: {test_video_id}")
    print(f"测试文本: {test_text[:50]}...")

    # 构造query（使用测试文本的一部分）
    query_text = test_text.split()[0] if test_text.split() else test_text[:5]
    print(f"Query文本: {query_text}")

    # 生成随机embedding作为query
    query_embedding = np.random.rand(384).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    print(f"\n{'='*80}")
    print("步骤2: 纯BM25检索（词面）")
    print("="*80)

    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=10,
        expr=f'video_id == "{test_video_id}"',
    )

    try:
        # 纯BM25检索
        results_bm25 = col.hybrid_search(
            reqs=[sparse_req],
            rerank=WeightedRanker(1.0),
            limit=10,
            output_fields=["text", "frame_idx"]
        )

        print(f"✓ BM25检索成功，返回 {len(results_bm25[0])} 条结果")
        print(f"\n前5条结果:")
        for i, hit in enumerate(results_bm25[0][:5], 1):
            print(f"  {i}. score={hit.score:.6f}, text={hit.entity.get('text', '')[:60]}")

        if results_bm25[0]:
            bm25_scores = [hit.score for hit in results_bm25[0]]
            print(f"\nBM25分数统计:")
            print(f"  最高: {max(bm25_scores):.6f}")
            print(f"  最低: {min(bm25_scores):.6f}")
            print(f"  平均: {np.mean(bm25_scores):.6f}")
            print(f"  标准差: {np.std(bm25_scores):.6f}")

    except Exception as e:
        print(f"❌ BM25检索失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("步骤3: 纯语义检索（DiskANN）")
    print("="*80)

    dense_req = AnnSearchRequest(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param={
            "metric_type": "IP",
            "params": {"search_list": 200},
        },
        limit=10,
        expr=f'video_id == "{test_video_id}" AND has_embedding == True',
    )

    try:
        # 纯语义检索
        results_semantic = col.hybrid_search(
            reqs=[dense_req],
            rerank=WeightedRanker(1.0),
            limit=10,
            output_fields=["text", "frame_idx"]
        )

        print(f"✓ 语义检索成功，返回 {len(results_semantic[0])} 条结果")
        print(f"\n前5条结果:")
        for i, hit in enumerate(results_semantic[0][:5], 1):
            print(f"  {i}. score={hit.score:.6f}, text={hit.entity.get('text', '')[:60]}")

        if results_semantic[0]:
            semantic_scores = [hit.score for hit in results_semantic[0]]
            print(f"\n语义分数统计:")
            print(f"  最高: {max(semantic_scores):.6f}")
            print(f"  最低: {min(semantic_scores):.6f}")
            print(f"  平均: {np.mean(semantic_scores):.6f}")
            print(f"  标准差: {np.std(semantic_scores):.6f}")

    except Exception as e:
        print(f"❌ 语义检索失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("步骤4: 混合检索（权重测试）")
    print("="*80)

    # 测试不同权重组合
    weight_configs = [
        (1.0, 0.0, "纯语义"),
        (0.8, 0.2, "语义80% + 词面20%"),
        (0.5, 0.5, "语义50% + 词面50%"),
        (0.2, 0.8, "语义20% + 词面80%"),
        (0.0, 1.0, "纯词面"),
    ]

    for semantic_w, lexical_w, desc in weight_configs:
        print(f"\n--- {desc} ---")
        print(f"WeightedRanker({semantic_w}, {lexical_w})")

        try:
            results_hybrid = col.hybrid_search(
                reqs=[dense_req, sparse_req],
                rerank=WeightedRanker(semantic_w, lexical_w),
                limit=5,
                output_fields=["text", "frame_idx"]
            )

            if results_hybrid[0]:
                top_score = results_hybrid[0][0].score
                print(f"  Top-1 score: {top_score:.6f}")
                print(f"  Top-1 text: {results_hybrid[0][0].entity.get('text', '')[:60]}")

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    print(f"\n{'='*80}")
    print("步骤5: 分析和诊断")
    print("="*80)

    # 检查是否存在BM25分数
    if 'results_bm25' in locals() and results_bm25[0]:
        print("\n✓ BM25检索正常工作")

        # 检查分数范围
        if max(bm25_scores) < 1.0:
            print("⚠️  BM25分数范围很小（< 1.0），可能已被归一化")
        elif max(bm25_scores) > 10.0:
            print("⚠️  BM25分数范围很大（> 10.0），可能未归一化")
        else:
            print("✓ BM25分数范围正常（1.0 ~ 10.0）")
    else:
        print("\n❌ BM25检索失败或无结果")

    if 'results_semantic' in locals() and results_semantic[0]:
        print("\n✓ 语义检索正常工作")

        # 检查语义分数范围
        if max(semantic_scores) > 1.0:
            print("⚠️  语义分数 > 1.0，不是标准余弦相似度范围")
        else:
            print("✓ 语义分数范围正常（0 ~ 1.0）")
    else:
        print("\n❌ 语义检索失败或无结果")

    # 比较分数范围
    if 'bm25_scores' in locals() and 'semantic_scores' in locals():
        bm25_range = max(bm25_scores) - min(bm25_scores)
        semantic_range = max(semantic_scores) - min(semantic_scores)

        print(f"\n分数范围对比:")
        print(f"  BM25范围: {bm25_range:.6f}")
        print(f"  语义范围: {semantic_range:.6f}")

        if abs(bm25_range - semantic_range) > 1.0:
            print(f"\n⚠️  两者分数范围差异很大！")
            print(f"     这可能导致WeightedRanker加权不符合预期")
            print(f"     建议检查Milvus是否对分数做了自动归一化")

    print(f"\n{'='*80}")
    print("步骤6: 当前配置")
    print("="*80)

    print(f"\n环境变量:")
    print(f"  OCR_LEXICAL_WEIGHT = {settings.ocr_lexical_weight}")
    print(f"  OCR_SEMANTIC_WEIGHT = {1.0 - settings.ocr_lexical_weight}")
    print(f"  OCR_HYBRID_RECALL_SIZE = {settings.ocr_hybrid_recall_size}")

    print(f"\n实际使用的权重:")
    print(f"  WeightedRanker({1.0 - settings.ocr_lexical_weight}, {settings.ocr_lexical_weight})")
    print(f"  即: 语义{(1.0 - settings.ocr_lexical_weight)*100:.0f}% + 词面{settings.ocr_lexical_weight*100:.0f}%")

except Exception as e:
    print(f"\n❌ 诊断过程出错: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print("诊断完成")
print("="*80)
