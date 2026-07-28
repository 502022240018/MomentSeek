#!/usr/bin/env python3
"""重建Visual索引 - 用于切换HNSW/DiskANN索引类型

使用场景：
1. 从HNSW切换到DiskANN
2. 从DiskANN切换到HNSW
3. 修复损坏的索引

警告：
- 会删除visual_embeddings collection的所有数据
- 需要重新索引所有视频
- 操作不可逆

使用方法：
    python backend/scripts/rebuild_visual_index.py --confirm

环境变量：
    VISUAL_USE_DISKANN=true/false - 目标索引类型
    MILVUS_HOST, MILVUS_PORT - Milvus连接配置
"""
import argparse
import sys
from pathlib import Path

# 添加backend目录到Python路径
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from pymilvus import connections, utility, Collection
from app.settings import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="重建Visual索引（删除collection并重新创建）"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认执行删除操作（必需）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示当前状态，不执行删除",
    )
    args = parser.parse_args()

    settings = get_settings()

    # 连接Milvus
    print(f"连接到Milvus: {settings.milvus_host}:{settings.milvus_port}")
    try:
        connections.connect(
            host=settings.milvus_host,
            port=settings.milvus_port,
            timeout=5,
        )
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1

    # 检查collection是否存在
    if not utility.has_collection("visual_embeddings"):
        print("✅ visual_embeddings collection不存在，可以直接创建")
        print(f"   配置的索引类型: {'DiskANN' if settings.visual_use_diskann else 'HNSW'}")
        print("\n下次启动容器时会自动创建新索引。")
        return 0

    # 获取当前索引信息
    col = Collection("visual_embeddings")
    num_entities = col.num_entities

    try:
        index_info = col.index()
        if index_info:
            current_index_type = index_info.params.get("index_type", "UNKNOWN")
            index_params = index_info.params
        else:
            current_index_type = "NONE"
            index_params = {}
    except Exception as e:
        print(f"⚠️  无法获取索引信息: {e}")
        current_index_type = "UNKNOWN"
        index_params = {}

    # 显示当前状态
    print("\n" + "="*60)
    print("当前状态:")
    print("="*60)
    print(f"Collection: visual_embeddings")
    print(f"实体数量: {num_entities:,} 个帧")
    print(f"当前索引: {current_index_type}")
    if index_params:
        print(f"索引参数: {index_params}")
    print()
    print(f"配置目标: {'DiskANN' if settings.visual_use_diskann else 'HNSW'}")
    print("="*60)

    # 检查是否需要重建
    target_index = "DISKANN" if settings.visual_use_diskann else "HNSW"
    if current_index_type == target_index:
        print(f"\n✅ 当前索引已经是 {target_index}，无需重建")
        return 0

    if current_index_type != target_index:
        print(f"\n⚠️  索引类型不匹配:")
        print(f"   当前: {current_index_type}")
        print(f"   目标: {target_index}")

    # Dry-run模式
    if args.dry_run:
        print("\n[Dry-run] 以上是当前状态，未执行任何操作。")
        print("\n要执行重建，运行:")
        print(f"  python {Path(__file__).name} --confirm")
        return 0

    # 确认删除
    if not args.confirm:
        print("\n❌ 需要 --confirm 参数来执行删除操作")
        print("\n这个操作会:")
        print("  1. 删除 visual_embeddings collection")
        print(f"  2. 丢失 {num_entities:,} 个帧的索引数据")
        print("  3. 需要重新索引所有视频")
        print("\n如果确认，运行:")
        print(f"  python {Path(__file__).name} --confirm")
        return 1

    # 最后确认
    print("\n" + "!"*60)
    print("⚠️  最后确认")
    print("!"*60)
    print(f"即将删除 visual_embeddings collection ({num_entities:,} 个帧)")
    print("此操作不可逆！")
    print()
    response = input("输入 'DELETE' 确认删除: ")

    if response != "DELETE":
        print("\n❌ 已取消")
        return 1

    # 执行删除
    print("\n正在删除 visual_embeddings...")
    try:
        utility.drop_collection("visual_embeddings")
        print("✅ 删除成功")
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return 1

    # 完成
    print("\n" + "="*60)
    print("✅ 索引重建准备完成")
    print("="*60)
    print(f"新索引类型: {target_index}")
    print()
    print("下一步:")
    print("  1. 重启容器（会自动创建新索引）")
    print("  2. 重新索引所有视频")
    print()
    print("索引命令示例:")
    print("  curl -X POST http://localhost:8100/api/jobs/index \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"video_id\": \"your_video_id\"}'")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
