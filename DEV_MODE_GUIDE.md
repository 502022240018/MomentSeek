# 开发模式使用指南

## 概述

开发模式（DEV_MODE）通过挂载代码目录和启用热重载功能，使代码修改自动生效，无需重新构建镜像，大幅提高开发效率。

## 配置说明

### 环境变量 (.env.0829)

```bash
# Development Mode (启用后代码修改自动重载，无需重构镜像)
DEV_MODE=true              # 是否启用开发模式
DEV_SKIP_BUILD=false       # true=完全跳过构建，false=仅在镜像不存在时构建
```

### 配置选项

| 变量 | 说明 | 推荐值 |
|------|------|--------|
| `DEV_MODE=true` | 启用开发模式 | 开发环境设为 `true` |
| `DEV_MODE=false` | 生产模式（每次重新构建） | 生产环境设为 `false` |
| `DEV_SKIP_BUILD=true` | 完全跳过构建，使用已有镜像 | 代码频繁修改时使用 |
| `DEV_SKIP_BUILD=false` | 首次构建，后续跳过 | 首次部署或依赖变更时使用 |

## 工作原理

### 生产模式 (DEV_MODE=false)
```
代码修改 → 重新构建镜像 → 重启容器 → 生效
耗时：~3-5分钟（完整构建）
```

### 开发模式 (DEV_MODE=true)
```
代码修改 → 自动检测 → 热重载 → 生效
耗时：~2-5秒（无需构建）
```

### 实现机制

1. **代码挂载**：将宿主机代码目录挂载到容器内
   ```bash
   -v /path/to/backend/app:/app/backend/app
   -v /path/to/frontend/dist:/app/backend/app/static:ro
   ```

2. **热重载**：使用 uvicorn 的 `--reload` 参数
   ```bash
   uvicorn app.main:app --reload --workers 1
   ```

3. **文件监控**：启用强制轮询模式
   ```bash
   -e WATCHFILES_FORCE_POLLING="true"
   ```

## 使用方法

### 1. 启用开发模式

编辑 `.env.0829`：
```bash
DEV_MODE=true
DEV_SKIP_BUILD=false  # 首次部署或依赖变更时
```

### 2. 部署服务

```bash
./deploy_0829.sh
```

输出示例：
```
[2026-07-27 11:43:38] DEV_MODE: Building only if image doesn't exist
[2026-07-27 11:43:39] DEV_MODE: Enabling code mount and hot reload

===========================================
Container: momentseek-0829-platform
Mode: true
Hot Reload: ENABLED ✓
Code Mount: /home/.../backend/app -> /app/backend/app
===========================================
```

### 3. 修改代码

直接编辑 `backend/app` 目录下的任何 Python 文件，保存后自动生效。

### 4. 监控热重载

```bash
# 实时查看日志
docker logs -f momentseek-0829-platform

# 看到这些信息表示热重载成功：
# WARNING:  StatReload detected changes in 'app/xxx.py'. Reloading...
# INFO:     Shutting down
# INFO:     Application shutdown complete.
# INFO:     Started server process [XX]
# INFO:     Application startup complete.
```

### 5. 验证服务

```bash
# 健康检查
curl http://127.0.0.1:8100/api/health

# API 文档
open http://127.0.0.1:8100/docs
```

## 测试脚本

### 运行自动化测试

```bash
./test_dev_mode.sh
```

测试内容：
- ✓ 检查容器状态
- ✓ 验证代码挂载
- ✓ 验证热重载配置
- ✓ 创建测试文件并检测变更
- ✓ 修改文件并验证重载
- ✓ 清理测试文件

## 常见场景

### 场景 1: 日常开发（代码频繁修改）

```bash
# .env.0829
DEV_MODE=true
DEV_SKIP_BUILD=true  # 完全跳过构建，使用已有镜像
```

**优点**：
- 部署速度极快（~10秒）
- 代码修改即时生效

**注意**：
- 依赖变更（requirements.txt）时需要重新构建

### 场景 2: 依赖更新后

```bash
# .env.0829
DEV_MODE=true
DEV_SKIP_BUILD=false  # 重新构建一次

# 部署
./deploy_0829.sh

# 构建完成后，再次设置
DEV_SKIP_BUILD=true  # 后续跳过构建
```

### 场景 3: 生产部署

```bash
# .env.0829
DEV_MODE=false  # 关闭开发模式

# 每次都重新构建完整镜像
./deploy_0829.sh
```

## 注意事项

### ✅ 开发模式适用场景

- 修改 Python 代码（`.py` 文件）
- 修改配置文件
- 调试和测试
- 快速迭代开发

### ⚠️ 需要重新构建的情况

以下变更需要关闭 `DEV_SKIP_BUILD` 或使用生产模式：

1. **依赖变更**
   - 修改 `requirements.txt`
   - 修改 `requirements-ascend.txt`
   - 安装新的 Python 包

2. **系统层变更**
   - 修改 `Dockerfile.ascend`
   - 修改系统依赖
   - 修改编译的二进制文件

3. **静态资源变更**
   - 前端代码修改（需要重新构建 `frontend/dist`）
   - 静态文件变更

### 🔧 热重载限制

- **不支持**：多进程模式（开发模式强制使用 `--workers 1`）
- **可能延迟**：大文件修改可能需要 5-10 秒才能检测到
- **NPU 资源**：重载时会重新初始化模型，可能占用 NPU

### 📊 性能影响

| 模式 | 启动时间 | 重载时间 | 内存占用 |
|------|----------|----------|----------|
| 生产模式 | 3-5 分钟 | N/A | 正常 |
| 开发模式（首次） | 10-30 秒 | 2-5 秒 | +5-10% |
| 开发模式（跳过构建） | 10 秒 | 2-5 秒 | +5-10% |

## 故障排查

### 问题 1: 代码修改不生效

**检查**：
```bash
# 1. 确认代码目录已挂载
docker inspect momentseek-0829-platform --format '{{range .Mounts}}{{.Source}}->{{.Destination}}{{"\n"}}{{end}}' | grep backend

# 2. 确认热重载已启用
docker inspect momentseek-0829-platform --format '{{.Config.Cmd}}'
# 应该看到 --reload 参数

# 3. 查看日志
docker logs -f momentseek-0829-platform | grep -i reload
```

**解决**：
```bash
# 确认 .env.0829 中 DEV_MODE=true
grep DEV_MODE .env.0829

# 重新部署
docker stop momentseek-0829-platform
docker rm momentseek-0829-platform
./deploy_0829.sh
```

### 问题 2: 服务重载后异常

**检查**：
```bash
# 查看完整日志
docker logs --tail 100 momentseek-0829-platform

# 检查健康状态
curl http://127.0.0.1:8100/api/health
```

**可能原因**：
- 代码语法错误
- 导入错误
- NPU 资源冲突

**解决**：
```bash
# 修复代码错误后，服务会自动重新加载
# 或手动重启容器
docker restart momentseek-0829-platform
```

### 问题 3: 依赖未安装

**现象**：
```python
ModuleNotFoundError: No module named 'xxx'
```

**解决**：
```bash
# 1. 更新 .env.0829
DEV_SKIP_BUILD=false

# 2. 重新部署（会重新构建镜像）
docker stop momentseek-0829-platform
docker rm momentseek-0829-platform
./deploy_0829.sh

# 3. 构建完成后，恢复快速模式
# 编辑 .env.0829
DEV_SKIP_BUILD=true
```

## 最佳实践

### 1. 开发工作流

```bash
# 首次启动
DEV_MODE=true DEV_SKIP_BUILD=false ./deploy_0829.sh

# 后续开发
# .env.0829: DEV_SKIP_BUILD=true
编辑代码 → 保存 → 自动重载（2-5秒） → 测试

# 依赖变更时
DEV_SKIP_BUILD=false ./deploy_0829.sh
# 完成后再设置 DEV_SKIP_BUILD=true
```

### 2. Git 工作流

```bash
# 开发分支
git checkout -b feature/xxx
# .env.0829: DEV_MODE=true

# 提交前测试（生产模式）
# .env.0829: DEV_MODE=false
./deploy_0829.sh
# 运行完整测试

# 合并到主分支
git checkout main
git merge feature/xxx
```

### 3. 团队协作

```bash
# .env.0829 不要提交到 git（已在 .gitignore）
# 每个开发者根据自己的需求配置

# 共享配置示例文档
cp .env.0829 .env.0829.example
# 提交 .env.0829.example 作为参考
```

## 配置模板

### 开发环境配置
```bash
# .env.0829 - 开发环境
DEV_MODE=true
DEV_SKIP_BUILD=true
APP_PORT=8100
SEARCH_PREWARM_ENABLED=false  # 加快启动
```

### 测试环境配置
```bash
# .env.0829 - 测试环境
DEV_MODE=false  # 使用生产构建
APP_PORT=8200
SEARCH_PREWARM_ENABLED=true
```

### 生产环境配置
```bash
# .env.0829 - 生产环境
DEV_MODE=false
APP_PORT=8000
SEARCH_PREWARM_ENABLED=true
```

## 监控和调试

### 实时日志
```bash
# 查看所有日志
docker logs -f momentseek-0829-platform

# 只看重载相关
docker logs -f momentseek-0829-platform 2>&1 | grep -i reload

# 只看错误
docker logs -f momentseek-0829-platform 2>&1 | grep -i error
```

### 性能监控
```bash
# 容器资源使用
docker stats momentseek-0829-platform

# NPU 使用情况
npu-smi info -i 4
```

### 进入容器调试
```bash
# 进入容器
docker exec -it momentseek-0829-platform bash

# 查看挂载的代码
ls -la /app/backend/app

# 手动测试
cd /app/backend
python3 -c "import app.main"
```

## 总结

开发模式通过代码挂载和热重载，将代码修改的生效时间从 **3-5分钟** 缩短到 **2-5秒**，极大提升了开发效率。

**核心优势**：
- ⚡ 快速迭代：代码修改即时生效
- 🚀 节省时间：无需等待镜像构建
- 🔧 易于调试：实时查看日志和错误
- 💾 节省资源：减少镜像构建次数

**推荐使用场景**：
- ✅ 日常开发和调试
- ✅ 快速原型验证
- ✅ Bug 修复和测试

**不推荐使用场景**：
- ❌ 生产部署
- ❌ 依赖频繁变更
- ❌ 性能压测环境
