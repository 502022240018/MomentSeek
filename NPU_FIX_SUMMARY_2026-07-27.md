# NPU索引问题修复总结

**日期**: 2026-07-27  
**环境**: momentseek-0829 开发环境  
**状态**: ✅ 已修复并验证成功

---

## 问题描述

前端索引测试报错，NPU初始化失败：
```
RuntimeError: Initialize NPU function error: aclInit, error code is 507899
[ERROR] Resource_Busy(EL0005): The resources are busy.
drv devId is invalid, drv devId=0, retCode=0x7020010
```

---

## 根本原因分析

### 问题1: NPU设备号不匹配
- **现象**: 容器挂载了 `/dev/davinci6`，但实际应该使用 NPU 4
- **原因**: `deploy_0829.sh` 第21行硬编码了 `NPU_ID="${NPU_ID:-6}"`
- **影响**: 容器尝试使用被占用的NPU 6导致Resource_Busy错误

### 问题2: 环境变量冲突
- **现象**: torch_npu.device_count() 返回 0
- **原因**: 设置了 `ASCEND_VISIBLE_DEVICES=4` 和 `ASCEND_RT_VISIBLE_DEVICES=4`
- **影响**: ACL库期望找到逻辑设备4，但容器内只有一个物理设备，导致设备无法识别

### 问题3: 模型路径配置缺失
- **现象**: 模型文件查找失败 - `FileNotFoundError: /app/runtime/hf_cache`
- **原因**: 部署脚本未传递 `VISUAL_HF_CACHE_DIR` 环境变量
- **影响**: 代码默认使用 `runtime/hf_cache`，而实际模型在 `/app/models/hf-cache`

---

## 解决方案

### 修复1: 统一NPU设备配置
**文件**: `deploy_0829.sh` 第21行

**修改前**:
```bash
NPU_ID="${NPU_ID:-6}"
```

**修改后**:
```bash
NPU_ID="${HOST_NPU_DEVICE_ID:-4}"
```

**效果**: 从 `.env.0829` 读取正确的NPU ID（4）

### 修复2: 移除冲突的环境变量
**文件**: `deploy_0829.sh` 第224-226行

**修改前**:
```bash
-e NPU_DEVICE_ID=0 \
-e ASCEND_VISIBLE_DEVICES="$NPU_ID" \
-e ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" \
```

**修改后**:
```bash
-e NPU_DEVICE_ID=0 \
```

**效果**: 让ACL自动将挂载的 `/dev/davinci4` 映射为逻辑设备0

### 修复3: 添加模型路径配置
**文件**: `deploy_0829.sh` 第232行

**修改前**:
```bash
-e VISUAL_MODEL=siglip2-so400m-384 \
-e FACE_PROVIDER=cann \
```

**修改后**:
```bash
-e VISUAL_MODEL=siglip2-so400m-384 \
-e VISUAL_HF_CACHE_DIR=/app/models/hf-cache \
-e FACE_PROVIDER=cann \
```

**效果**: 指向正确的共享模型目录

---

## 验证结果

### NPU设备状态
```python
NPU可用: True
NPU数量: 1
当前设备: 0
```

### 容器设备挂载
```bash
crw-rw-rw- 1 root root 251,  4 Jul 27 01:53 /dev/davinci4  ✅
crw-rw-rw- 1 root root 251, 16 Jul 27 01:53 /dev/davinci_manager  ✅
```

### 环境变量配置
```bash
NPU_ENABLED=true
NPU_DEVICE_ID=0
VISUAL_HF_CACHE_DIR=/app/models/hf-cache  ✅
```

### 索引测试结果
```
视频: Video Project.mp4
- ID: 4c7f80cff1374441ae19c8de1c7a0b66
- 时长: 59.8秒
- 分辨率: 1920x1080 @ 30fps
- 状态: ready ✅
- 已索引模态: ['visual'] ✅

最新索引任务:
- ID: 25ee6543f4ee463abf1b001d9c9f5fea
- 状态: completed ✅
- 进度: 100.0% ✅
- 创建: 2026-07-27 01:55:16
- 完成: 2026-07-27 01:56:01
- 耗时: ~45秒
```

---

## 关键要点

### NPU设备映射逻辑
1. **物理设备**: 宿主机上的 `/dev/davinci{N}`（N=0-7）
2. **容器挂载**: 通过 `--device` 挂载到容器内
3. **逻辑映射**: 容器内看到的设备会被ACL自动映射为逻辑设备0
4. **配置原则**: 
   - 只挂载一个物理NPU设备
   - 不设置 `ASCEND_VISIBLE_DEVICES`/`ASCEND_RT_VISIBLE_DEVICES`
   - 始终使用 `NPU_DEVICE_ID=0`

### 模型路径约定
- **代码视角**: 统一使用容器内路径 `/app/models`
- **部署配置**: 通过volume挂载宿主机路径到容器内
- **环境变量**: 特定子目录（如HF cache）需要显式配置

### 环境配置优先级
1. `.env.0829` - 定义环境专用配置
2. `deploy_0829.sh` - 读取并传递给容器
3. 容器内应用 - 通过 `settings.py` 读取环境变量

---

## 相关文件

### 修改的文件
- `deploy_0829.sh` - 主部署脚本

### 相关配置
- `.env.0829` - 环境变量配置
- `backend/app/settings.py` - 应用配置
- `backend/app/indexing/visual.py` - Visual索引实现

### 参考文档
- `docs/ASCEND_SHARED_SERVER_RUNBOOK.md` - NPU服务器运维手册
- `docs/MODELS.md` - 模型管理规范
- `docs/DEPLOYMENT.md` - 部署指南

---

## 后续建议

1. **文档更新**: 将这次问题和解决方案补充到运维手册
2. **测试覆盖**: 添加NPU设备检测的自动化测试
3. **配置验证**: 在部署脚本中添加预检查步骤
4. **监控告警**: 添加NPU资源占用监控

---

**修复人员**: Claude Code  
**验证时间**: 2026-07-27 09:56  
**部署环境**: momentseek-0829-platform  
**容器镜像**: momentseek-0829-platform:e6cdc43
