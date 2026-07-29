#!/usr/bin/env python3
"""配置检查脚本 - 验证Visual优化相关配置

检查项:
1. Milvus连接状态
2. Visual collection状态
3. 索引类型（HNSW/DiskANN）
4. 优化开关配置
5. 性能参数配置

用法:
    python backend/scripts/check_visual_config.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.indexing.milvus_client import get_milvus_client
from app.settings import get_settings


def check_milvus_connection():
    """检查Milvus连接"""
    print(f"\n{'='*70}")
    print("1. Milvus连接检查")
    print(f"{'='*70}")

    settings = get_settings()
    print(f"  Host: {settings.milvus_host}")
    print(f"  Port: {settings.milvus_port}")
    print(f"  Enabled: {settings.milvus_enabled}")
    print(f"  Read Enabled: {settings.milvus_read_enabled}")
    print(f"  Write Enabled: {settings.milvus_write_enabled}")

    if not settings.milvus_enabled:
        print("  ✗ Milvus未启用")
        return False

    try:
        client = get_milvus_client()
        print("  ✓ 连接成功")
        return True
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return False


def check_visual_collection(client):
    """检查Visual collection状态"""
    print(f"\n{'='*70}")
    print("2. Visual Collection检查")
    print(f"{'='*70}")

    try:
        stats = client.stats("visual_embeddings")
        print(f"  Collection名称: {stats['name']}")
        print(f"  实体数量: {stats['num_entities']:,}")
        print(f"  加载状态: {'已加载' if stats['loaded'] else '未加载'}")

        if stats['num_entities'] == 0:
            print("  ⚠ 没有索引数据")
        elif stats['num_entities'] < 1000:
            print(f"  ⚠ 数据量较少 ({stats['num_entities']}条)")
        else:
            print("  ✓ 数据量正常")

        return stats['loaded']
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def check_index_type(client):
    """检查索引类型"""
    print(f"\n{'='*70}")
    print("3. 索引类型检查")
    print(f"{'='*70}")

    try:
        collection = client.collection("visual_embeddings")
        indexes = collection.indexes

        if not indexes:
            print("  ✗ 没有索引")
            return

        for idx in indexes:
            print(f"  字段: {idx.field_name}")
            print(f"  索引类型: {idx.params.get('index_type', 'N/A')}")
            print(f"  距离度量: {idx.params.get('metric_type', 'N/A')}")

            index_type = idx.params.get('index_type', '')
            if index_type == 'HNSW':
                print("  ✓ 使用HNSW索引（内存ANN）")
                params = idx.params.get('params', {})
                print(f"    M: {params.get('M', 'N/A')}")
                print(f"    efConstruction: {params.get('efConstruction', 'N/A')}")
            elif index_type == 'DISKANN':
                print("  ✓ 使用DiskANN索引（磁盘ANN）")
                params = idx.params.get('params', {})
                print(f"    search_list_size: {params.get('search_list_size', 'N/A')}")
            else:
                print(f"  ⚠ 未知索引类型: {index_type}")

    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_optimization_config():
    """检查优化开关配置"""
    print(f"\n{'='*70}")
    print("4. 优化配置检查")
    print(f"{'='*70}")

    settings = get_settings()

    print(f"  visual_use_diskann: {settings.visual_use_diskann}")
    if settings.visual_use_diskann:
        print("    ✓ 已启用DiskANN索引")
    else:
        print("    → 使用HNSW索引（默认）")

    print(f"  visual_ann_top_k: {settings.visual_ann_top_k}")


def check_performance_params():
    """检查性能参数"""
    print(f"\n{'='*70}")
    print("5. 性能参数检查")
    print(f"{'='*70}")

    settings = get_settings()

    print(f"  milvus_query_timeout_seconds: {settings.milvus_query_timeout_seconds}")
    print(f"  milvus_search_video_batch_size: {settings.milvus_search_video_batch_size}")
    print(f"  milvus_rollout_percent: {settings.milvus_rollout_percent}")

    if settings.milvus_rollout_percent < 100:
        print(f"    ⚠ 灰度发布中 ({settings.milvus_rollout_percent}%)")
    else:
        print("    ✓ 全量发布")


def check_sample_video_count(client):
    """检查有多少视频可用于测试"""
    print(f"\n{'='*70}")
    print("6. 测试视频检查")
    print(f"{'='*70}")

    try:
        collection = client.collection("visual_embeddings")

        # 查询不同video_id的数量
        results = collection.query(
            expr="",
            output_fields=["video_id"],
            limit=100,
        )

        if not results:
            print("  ✗ 没有索引数据")
            return

        video_ids = set(r["video_id"] for r in results)
        print(f"  已索引视频数: {len(video_ids)}")

        if len(video_ids) >= 1:
            print("  ✓ 可以运行测试")
            print(f"  示例视频ID: {list(video_ids)[0]}")
        else:
            print("  ✗ 需要至少索引1个视频")

    except Exception as e:
        print(f"  ✗ 检查失败: {e}")


def main():
    print(f"\n{'#'*70}")
    print(f"{'#'*20} Visual模态优化配置检查 {'#'*20}")
    print(f"{'#'*70}")

    # 1. 连接检查
    connection_ok = check_milvus_connection()

    if not connection_ok:
        print(f"\n{'='*70}")
        print("总结: Milvus连接失败，无法继续检查")
        print(f"{'='*70}")
        sys.exit(1)

    # 获取client
    client = get_milvus_client()

    # 2. Collection检查
    collection_ok = check_visual_collection(client)

    # 3. 索引类型
    check_index_type(client)

    # 4. 优化配置
    check_optimization_config()

    # 5. 性能参数
    check_performance_params()

    # 6. 测试视频
    check_sample_video_count(client)

    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    settings = get_settings()

    if collection_ok:
        print("✓ 配置正确，可以运行测试")
        print("\n推荐下一步:")
        print("  python backend/scripts/test_visual_ann.py")
    else:
        print("✗ 配置有问题，请检查上述输出")

    print()


if __name__ == "__main__":
    main()
