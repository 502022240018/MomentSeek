"""Visual模态优化测试 - 对比旧版全量query和新版ANN+采样

测试目标:
1. 功能正确性 - Jaccard相似度 > 0.85
2. 性能提升 - 延迟降低60-80%
3. 网络传输 - 数据量降低90%+

运行方式:
    python -m pytest backend/tests/test_visual_optimization.py -v -s
"""
import time
from typing import Any

import numpy as np
import pytest

from app.indexing.milvus_client import MilvusClient
from app.indexing.milvus_search import milvus_visual_candidates
from app.settings import Settings, get_settings


@pytest.fixture
def test_video_id() -> str:
    """测试用视频ID - 需要已索引的视频"""
    # TODO: 替换为实际已索引的视频ID
    return "test_video_001"


@pytest.fixture
def test_query_embedding() -> np.ndarray:
    """测试查询向量 - 模拟SigLIP编码"""
    # 生成随机归一化向量
    vec = np.random.randn(1152).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestVisualOptimization:
    """Visual模态优化对比测试"""

    def test_correctness_comparison(
        self,
        test_video_id: str,
        test_query_embedding: np.ndarray,
    ):
        """测试1: 功能正确性对比 - Jaccard相似度"""
        settings = get_settings()
        client = MilvusClient()

        # 旧版本: 全量query
        settings.visual_use_ann_search = False
        old_start = time.perf_counter()
        old_candidates = milvus_visual_candidates(
            client, test_video_id, test_query_embedding, limit=20
        )
        old_time = time.perf_counter() - old_start

        # 新版本: ANN + 采样
        settings.visual_use_ann_search = True
        new_start = time.perf_counter()
        new_candidates = milvus_visual_candidates(
            client, test_video_id, test_query_embedding, limit=20
        )
        new_time = time.perf_counter() - new_start

        # 计算Jaccard相似度
        old_segments = {c.unit_id for c in old_candidates[:10]}
        new_segments = {c.unit_id for c in new_candidates[:10]}

        intersection = len(old_segments & new_segments)
        union = len(old_segments | new_segments)
        jaccard = intersection / union if union > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"功能正确性测试结果:")
        print(f"{'='*60}")
        print(f"旧版召回: {len(old_candidates)} 候选, 耗时: {old_time*1000:.1f}ms")
        print(f"新版召回: {len(new_candidates)} 候选, 耗时: {new_time*1000:.1f}ms")
        print(f"Top-10 Jaccard相似度: {jaccard:.3f}")
        print(f"延迟降低: {(1 - new_time/old_time)*100:.1f}%")

        # 断言
        assert jaccard >= 0.85, f"Jaccard相似度 {jaccard:.3f} 低于目标 0.85"
        assert new_time < old_time, "新版本应该更快"

    def test_performance_benchmark(
        self,
        test_video_id: str,
        test_query_embedding: np.ndarray,
    ):
        """测试2: 性能基准测试 - 多次运行取平均"""
        settings = get_settings()
        client = MilvusClient()
        n_runs = 5

        # 旧版本基准
        settings.visual_use_ann_search = False
        old_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            milvus_visual_candidates(
                client, test_video_id, test_query_embedding, limit=20
            )
            old_times.append(time.perf_counter() - start)

        # 新版本基准
        settings.visual_use_ann_search = True
        new_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            milvus_visual_candidates(
                client, test_video_id, test_query_embedding, limit=20
            )
            new_times.append(time.perf_counter() - start)

        old_avg = np.mean(old_times)
        old_std = np.std(old_times)
        new_avg = np.mean(new_times)
        new_std = np.std(new_times)
        speedup = old_avg / new_avg

        print(f"\n{'='*60}")
        print(f"性能基准测试结果 (n={n_runs}):")
        print(f"{'='*60}")
        print(f"旧版: {old_avg*1000:.1f}±{old_std*1000:.1f}ms")
        print(f"新版: {new_avg*1000:.1f}±{new_std*1000:.1f}ms")
        print(f"加速比: {speedup:.2f}x")
        print(f"延迟降低: {(1 - 1/speedup)*100:.1f}%")

        # 断言: 延迟至少降低40% (保守目标)
        assert speedup >= 1.67, f"加速比 {speedup:.2f}x 未达到目标 1.67x (40%降低)"

    def test_multi_query_support(
        self,
        test_video_id: str,
    ):
        """测试3: 多子查询支持"""
        settings = get_settings()
        settings.visual_use_ann_search = True
        client = MilvusClient()

        # 生成3个子查询
        queries = [
            np.random.randn(1152).astype(np.float32) for _ in range(3)
        ]
        queries = [q / np.linalg.norm(q) for q in queries]
        query_array = np.stack(queries)

        # 执行检索
        candidates = milvus_visual_candidates(
            client, test_video_id, query_array, limit=20
        )

        print(f"\n{'='*60}")
        print(f"多子查询测试结果:")
        print(f"{'='*60}")
        print(f"子查询数: {len(queries)}")
        print(f"召回候选: {len(candidates)}")

        assert len(candidates) > 0, "多子查询应该返回结果"

    def test_sampling_distribution(
        self,
        test_video_id: str,
        test_query_embedding: np.ndarray,
    ):
        """测试4: 采样分布估算准确性"""
        settings = get_settings()
        client = MilvusClient()

        # 旧版本: 获取真实分布
        settings.visual_use_ann_search = False
        old_candidates = milvus_visual_candidates(
            client, test_video_id, test_query_embedding, limit=50
        )

        # 新版本: 基于采样的分布
        settings.visual_use_ann_search = True
        new_candidates = milvus_visual_candidates(
            client, test_video_id, test_query_embedding, limit=50
        )

        # 对比top候选的z-score分布
        if old_candidates and new_candidates:
            old_z_scores = [c.features.get("z_score", 0) for c in old_candidates[:10]]
            new_z_scores = [c.features.get("z_score", 0) for c in new_candidates[:10]]

            old_mean_z = np.mean(old_z_scores)
            new_mean_z = np.mean(new_z_scores)
            z_score_error = abs(old_mean_z - new_mean_z) / (abs(old_mean_z) + 1e-6)

            print(f"\n{'='*60}")
            print(f"分布估算测试结果:")
            print(f"{'='*60}")
            print(f"旧版平均z-score: {old_mean_z:.3f}")
            print(f"新版平均z-score: {new_mean_z:.3f}")
            print(f"相对误差: {z_score_error*100:.1f}%")

            # 容忍30%的相对误差
            assert z_score_error < 0.3, f"z-score相对误差 {z_score_error:.2f} 过大"


@pytest.mark.skipif(
    not get_settings().milvus_enabled,
    reason="Milvus not enabled"
)
class TestVisualDiskANN:
    """DiskANN索引测试"""

    def test_diskann_index_creation(self):
        """测试5: DiskANN索引创建"""
        settings = get_settings()

        # 启用DiskANN
        original_use_diskann = settings.visual_use_diskann
        settings.visual_use_diskann = True

        try:
            client = MilvusClient()
            collection = client.collection("visual_embeddings")

            # 检查索引类型
            indexes = collection.indexes
            assert len(indexes) > 0, "应该有索引"

            # TODO: 验证索引类型为DISKANN
            print(f"\n索引信息: {indexes}")

        finally:
            settings.visual_use_diskann = original_use_diskann

    def test_diskann_search_performance(
        self,
        test_video_id: str,
        test_query_embedding: np.ndarray,
    ):
        """测试6: DiskANN检索性能"""
        settings = get_settings()
        settings.visual_use_ann_search = True

        # HNSW基准
        settings.visual_use_diskann = False
        client_hnsw = MilvusClient()
        hnsw_start = time.perf_counter()
        hnsw_candidates = milvus_visual_candidates(
            client_hnsw, test_video_id, test_query_embedding, limit=20
        )
        hnsw_time = time.perf_counter() - hnsw_start

        # DiskANN测试
        settings.visual_use_diskann = True
        client_diskann = MilvusClient()
        diskann_start = time.perf_counter()
        diskann_candidates = milvus_visual_candidates(
            client_diskann, test_video_id, test_query_embedding, limit=20
        )
        diskann_time = time.perf_counter() - diskann_start

        print(f"\n{'='*60}")
        print(f"DiskANN vs HNSW性能对比:")
        print(f"{'='*60}")
        print(f"HNSW: {hnsw_time*1000:.1f}ms, {len(hnsw_candidates)} 候选")
        print(f"DiskANN: {diskann_time*1000:.1f}ms, {len(diskann_candidates)} 候选")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
