#!/usr/bin/env python3
"""测试Visual索引功能是否能正常使用NPU"""

import sys
import time

def test_npu_in_container():
    """在容器内测试NPU是否可用"""
    print("=== 测试NPU可用性 ===\n")

    try:
        import torch
        import torch_npu
        print(f"✅ torch版本: {torch.__version__}")
        print(f"✅ torch_npu版本: {torch_npu.__version__}")

        # 检查NPU设备数量
        device_count = torch.npu.device_count()
        print(f"✅ NPU设备数量: {device_count}")

        if device_count == 0:
            print("❌ 没有可用的NPU设备")
            return False

        # 尝试使用NPU 0（容器内的逻辑ID）
        device = torch.device("npu:0")
        print(f"✅ 使用设备: {device}")

        # 测试简单的张量操作
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y
        print(f"✅ 矩阵乘法测试通过")
        print(f"   结果形状: {z.shape}")

        # 清理
        del x, y, z
        torch.npu.empty_cache()

        print("\n✅ NPU功能正常！")
        return True

    except Exception as e:
        print(f"❌ NPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_npu_in_container()
    sys.exit(0 if success else 1)
