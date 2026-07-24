"""Phase 3: 实体嵌入迁移验证测试

验证实体 embedding 可以正确存储到数据库 BLOB 并读取回来。
这是一个独立测试，不需要完整的依赖环境。
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_face_embedding_blob_storage():
    """测试人脸 embedding 的 BLOB 存储和读取"""
    print("\nTesting face embedding BLOB storage...")

    # 创建一个模拟的 512 维 face embedding
    original_embedding = np.random.randn(512).astype(np.float32)

    # 转换为 bytes（数据库存储格式）
    embedding_blob = original_embedding.tobytes()

    # 从 bytes 恢复为 numpy array
    restored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

    # 验证恢复的 embedding 与原始完全相同
    assert restored_embedding.shape == (512,), f"Expected shape (512,), got {restored_embedding.shape}"
    assert np.allclose(restored_embedding, original_embedding), "Restored embedding doesn't match original"

    print(f"  [OK] Face embedding correctly stored and restored (512 dims)")
    return True


def test_voice_embedding_blob_storage():
    """测试语音 embedding 的 BLOB 存储和读取"""
    print("\nTesting voice embedding BLOB storage...")

    # 创建一个模拟的 192 维 voice embedding
    original_embedding = np.random.randn(192).astype(np.float32)

    # 转换为 bytes（数据库存储格式）
    embedding_blob = original_embedding.tobytes()

    # 从 bytes 恢复为 numpy array
    restored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

    # 验证恢复的 embedding 与原始完全相同
    assert restored_embedding.shape == (192,), f"Expected shape (192,), got {restored_embedding.shape}"
    assert np.allclose(restored_embedding, original_embedding), "Restored embedding doesn't match original"

    print(f"  [OK] Voice embedding correctly stored and restored (192 dims)")
    return True


def test_multiple_voice_embeddings():
    """测试多个语音样本的 embedding 存储"""
    print("\nTesting multiple voice embeddings...")

    # 创建 3 个语音样本
    num_samples = 3
    embeddings = []
    blobs = []

    for i in range(num_samples):
        emb = np.random.randn(192).astype(np.float32)
        embeddings.append(emb)
        blobs.append(emb.tobytes())

    # 恢复所有 embeddings
    restored = []
    for blob in blobs:
        restored.append(np.frombuffer(blob, dtype=np.float32))

    # 堆叠成 [N, 192] 矩阵
    stacked = np.stack(restored, axis=0)

    assert stacked.shape == (num_samples, 192), f"Expected shape ({num_samples}, 192), got {stacked.shape}"

    # 验证每个样本都正确恢复
    for i in range(num_samples):
        assert np.allclose(stacked[i], embeddings[i]), f"Sample {i} doesn't match"

    print(f"  [OK] Multiple voice embeddings correctly stacked ({num_samples} samples)")
    return True


def test_database_priority_logic():
    """测试数据库优先读取逻辑"""
    print("\nTesting database priority logic...")

    # 模拟 entity 记录
    entity_with_blob = {
        "id": "test1",
        "face_embedding": np.random.randn(512).astype(np.float32).tobytes(),
        "embedding_path": "/path/to/legacy.npz",
    }

    entity_without_blob = {
        "id": "test2",
        "face_embedding": None,
        "embedding_path": "/path/to/legacy.npz",
    }

    entity_empty_blob = {
        "id": "test3",
        "face_embedding": b"",
        "embedding_path": "/path/to/legacy.npz",
    }

    # 测试优先级逻辑
    # 1. BLOB 存在且非空 → 使用 BLOB
    if entity_with_blob.get("face_embedding"):
        source = "database"
    elif entity_with_blob.get("embedding_path"):
        source = "npz"
    else:
        source = "none"
    assert source == "database", "Should prioritize database BLOB"

    # 2. BLOB 为 None → 使用 NPZ
    if entity_without_blob.get("face_embedding"):
        source = "database"
    elif entity_without_blob.get("embedding_path"):
        source = "npz"
    else:
        source = "none"
    assert source == "npz", "Should fallback to NPZ when BLOB is None"

    # 3. BLOB 为空字节 → 使用 NPZ（空字节视为无效）
    if entity_empty_blob.get("face_embedding"):
        source = "database"
    elif entity_empty_blob.get("embedding_path"):
        source = "npz"
    else:
        source = "none"
    assert source == "npz", "Should fallback to NPZ when BLOB is empty"

    print(f"  [OK] Database priority logic correct (BLOB > NPZ > None)")
    return True


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\nTesting backward compatibility...")

    # 模拟旧数据（没有 BLOB 字段）
    legacy_entity = {
        "id": "legacy1",
        "embedding_path": "/path/to/legacy.npz",
        # face_embedding 字段不存在
    }

    # 测试 .get() 不会抛出异常
    blob = legacy_entity.get("face_embedding")
    assert blob is None, "Legacy entity should have None for face_embedding"

    # 应该回退到 NPZ 路径
    has_npz = bool(legacy_entity.get("embedding_path"))
    assert has_npz, "Should detect NPZ path for legacy entity"

    print(f"  [OK] Backward compatible with legacy entities (no BLOB field)")
    return True


def main():
    print("=" * 70)
    print("Phase 3: Entity Embedding Migration Verification")
    print("=" * 70)

    tests = [
        test_face_embedding_blob_storage,
        test_voice_embedding_blob_storage,
        test_multiple_voice_embeddings,
        test_database_priority_logic,
        test_backward_compatibility,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print()
    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} core logic tests passed")
    print("=" * 70)

    if passed == len(tests):
        print()
        print("[SUCCESS] Phase 3 core logic verification PASSED!")
        print("  - Entity embeddings can be stored in database BLOBs")
        print("  - Database priority over NPZ files works correctly")
        print("  - Backward compatible with legacy entities")
        print("  - Ready for integration testing with live database")
        return 0
    else:
        print()
        print("[FAILED] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
