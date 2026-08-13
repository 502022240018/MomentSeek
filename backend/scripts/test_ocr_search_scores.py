#!/usr/bin/env python3
"""
测试OCR混合检索的原始分数

直接调用search API，查看：
1. Milvus返回的原始hybrid分数
2. 聚合后的最终分数
3. 是否有分数被错误计算或覆盖
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.settings import Settings
from app.catalog import Catalog
from app.search import SearchEngine
import asyncio


async def test_ocr_search(query: str):
    """测试OCR混合检索"""
    settings = Settings()
    catalog = Catalog(settings)
    engine = SearchEngine(settings, catalog)

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"配置: 词面权重={settings.ocr_lexical_weight}, 召回={settings.ocr_hybrid_recall_size}")
    print(f"{'='*70}\n")

    # 执行搜索
    results = await engine.search(
        query=query,
        limit=20,
        modalities=["ocr"],
    )

    if not results:
        print("❌ 没有结果")
        return

    print(f"✅ 找到 {len(results)} 个结果\n")

    # 分析每个结果
    for i, result in enumerate(results[:10]):
        print(f"{'─'*70}")
        print(f"结果 #{i+1}")
        print(f"  最终分数: {result.score:.4f} {'(高于阈值)' if result.above_threshold else '(低于阈值)'}")
        print(f"  时间段: {result.start_time:.1f}s - {result.end_time:.1f}s (时长 {result.end_time - result.start_time:.1f}s)")
        print(f"  视频: {result.video_name}")

        # 分析evidence
        print(f"\n  Evidence详情 (共{len(result.evidence)}个候选帧):")

        ocr_scores = []
        for j, evidence in enumerate(result.evidence):
            if evidence.get("modality") == "ocr":
                text = evidence.get("text", "")
                score = evidence.get("score", 0)
                ocr_scores.append(score)

                # 检查是否包含query
                is_match = query in text
                match_mark = "✓完全匹配" if is_match else ""

                print(f"    [{j+1}] hybrid={score:.4f} | {match_mark}")
                print(f"        文本: {text[:60]}")

        # 统计
        if ocr_scores:
            print(f"\n  分数统计:")
            print(f"    最高分: {max(ocr_scores):.4f}")
            print(f"    最低分: {min(ocr_scores):.4f}")
            print(f"    平均分: {sum(ocr_scores)/len(ocr_scores):.4f}")
            print(f"    候选数: {len(ocr_scores)}")

            # 检查最终分数是否等于最高分
            # 注意：纯OCR时，最终分数应该等于组内最高分
            max_candidate_score = max(ocr_scores)
            if abs(result.score - max_candidate_score) < 0.001:
                print(f"    ✅ 最终分数 = 组内最高分")
            else:
                print(f"    ⚠️ 最终分数({result.score:.4f}) ≠ 组内最高分({max_candidate_score:.4f})")
                print(f"       差异: {abs(result.score - max_candidate_score):.4f}")

    print(f"\n{'='*70}")

    # 重点分析：完全匹配query的帧的分数分布
    print(f"\n完全匹配 '{query}' 的所有帧:")
    print(f"{'─'*70}")

    all_exact_matches = []
    for result in results:
        for evidence in result.evidence:
            if evidence.get("modality") == "ocr":
                text = evidence.get("text", "")
                if query in text:
                    score = evidence.get("score", 0)
                    all_exact_matches.append({
                        "score": score,
                        "text": text,
                        "result_score": result.score,
                        "time": f"{result.start_time:.1f}s-{result.end_time:.1f}s"
                    })

    if all_exact_matches:
        # 按分数降序
        all_exact_matches.sort(key=lambda x: x["score"], reverse=True)

        print(f"找到 {len(all_exact_matches)} 个完全匹配的帧:\n")
        for i, match in enumerate(all_exact_matches[:15]):
            print(f"  [{i+1}] hybrid={match['score']:.4f} | 时间段={match['time']}")
            print(f"      文本: {match['text'][:60]}")

        print(f"\n分数分析:")
        scores = [m["score"] for m in all_exact_matches]
        print(f"  最高分: {max(scores):.4f}")
        print(f"  最低分: {min(scores):.4f}")
        print(f"  差异倍数: {max(scores) / min(scores):.2f}x")

        if max(scores) / min(scores) > 2:
            print(f"\n⚠️ 相同词面的分数差异超过2倍！")
            print(f"   可能原因:")
            print(f"   1. 语义向量差异：不同上下文导致embedding不同")
            print(f"   2. BM25的文档长度归一化：短文本vs长文本")
            print(f"   3. BM25的词频统计：该帧中query出现的次数不同")
    else:
        print(f"❌ 没有找到完全匹配 '{query}' 的帧")
        print(f"   可能原因:")
        print(f"   1. BM25分词问题：'{query}' 被拆分了")
        print(f"   2. OCR文本没有包含完整的query")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_ocr_search_scores.py <query>")
        print("示例: python test_ocr_search_scores.py '最低工资'")
        sys.exit(1)

    query = sys.argv[1]
    asyncio.run(test_ocr_search(query))
