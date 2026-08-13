# Planner Lab 快速参考（0829 环境）

> 基于 `codex/snapmind-planner-lab` 分支的 Overlay 部署方案

## 🎯 核心配置总结

### 环境发现
- ✅ 同事的 vLLM 服务已运行：`127.0.0.1:18082`（NPU 3）
- ✅ Qwen3.5-4B 模型：`/home/momentseek-29154/vlm-exp/models/Qwen3.5-4B`
- ✅ 0829 基础镜像存在：`momentseek-0829-platform:current`
- ✅ 使用 NPU 1（与 vLLM 的 NPU 3 不冲突）
- ✅ 两容器均 `--network host`，可直接访问 localhost

### 部署方式
采用 **Overlay 覆盖式**部署：
- 基于已有镜像只替换少量文件（快速构建 2-3 分钟）
- Planner Lab 独立容器（端口 8101）
- 不影响主容器（端口 8100）

---

## 🚀 快速部署

### 1. 启动 Planner Lab

```bash
cd /home/momentseek_0829_develop/workplace/MomentSeek_planner
./deploy_planner_lab_0829.sh
```

**部署步骤：**
1. 检查基础镜像和 vLLM 连通性
2. 构建 overlay 镜像（约 2-3 分钟）
3. 备份旧容器（如果存在）
4. 启动新容器（端口 8101）
5. 健康检查和能力验证
6. 询问是否删除备份

### 2. 验证部署

```bash
# 检查容器状态
docker ps | grep planner-lab

# 检查能力接口
curl http://127.0.0.1:8101/api/planner-lab/capabilities | python3 -m json.tool

# 期望输出包含：
# {
#   "planner_lab": {
#     "enabled": true,
#     "orchestration": {
#       "enabled": true,  // LLM 模式已启用
#       "profiles": [...]
#     }
#   }
# }

# 访问 Web 界面（左侧应有 "Planner Lab" 入口）
http://127.0.0.1:8101
```

### 3. 测试 LLM 规划

```bash
# 生成计划（应标记 "Qwen3.5 生成"）
curl -X POST http://127.0.0.1:8101/api/planner-lab/plans \
  -F 'query_text=找到演讲者展示产品后观众鼓掌的片段' \
  -F 'mode=assist' | python3 -m json.tool
```

### 4. 停止 Planner Lab

```bash
./stop_planner_lab_0829.sh

# 或手动停止
docker stop momentseek-0829-planner-lab
```

---

## 📂 文件说明

### 新增/修改的文件

```
MomentSeek_planner/
├── .env.0829                           # ✏️ 已更新（添加 Planner Lab 配置）
├── deploy_planner_lab_0829.sh          # ✨ 新建（Overlay 部署脚本）
├── stop_planner_lab_0829.sh            # ✨ 新建（停止脚本）
└── PLANNER_LAB_0829_SETUP.md           # ✨ 新建（本文档）
```

### .env.0829 新增配置段

```bash
# ════════════════════════════════════════════════════════════════════
# Planner Lab Configuration (Experimental - codex/snapmind-planner-lab)
# ════════════════════════════════════════════════════════════════════
PLANNER_LAB_ENABLED=true
ORCHESTRATION_ENABLED=true
QWEN35_VLLM_BASE_URL=http://127.0.0.1:18082/v1  # 注意：18082 不是 18081
QWEN35_PLANNER_MODEL=qwen3.5-4b
QWEN35_RERANKER_MODEL=qwen3.5-4b
ORCHESTRATION_CONFIG_PATH=deploy/orchestration/qwen35-vllm.json
ORCHESTRATION_PROFILE=qwen35-unified
ORCHESTRATION_FAIL_OPEN=true
ORCHESTRATION_TRACE_ENABLED=true
ORCHESTRATION_TRACE_PATH=runtime/orchestration-traces.jsonl
PLANNER_LAB_PROMPT_PATH=deploy/orchestration/prompts/snapmind-planner-v2-role-aware.txt
```

---

## 🏗️ Overlay 部署原理

### 工作流程

```
1. 基础镜像（已存在）
   momentseek-0829-platform:current (22GB)
   ├── Python 依赖
   ├── NPU 驱动
   ├── 编译好的模型
   └── 完整运行环境

2. Overlay 构建
   docker build --build-arg BASE_IMAGE=momentseek-0829-platform:current
   ├── FROM 基础镜像
   ├── 替换 4 个 Python 文件
   │   ├── backend/app/main.py
   │   ├── backend/app/api/planner_lab_routes.py
   │   ├── backend/app/orchestration/snapmind_lab.py
   │   └── deploy/orchestration/prompts/*.txt
   └── 替换前端静态文件
       └── frontend/dist/ -> app/static/
   
3. 新镜像
   momentseek-0829-planner-lab:20260812-143022 (极小增量)

4. 独立容器
   momentseek-0829-planner-lab (端口 8101)
```

### 优势

- ✅ **构建快速**：2-3 分钟 vs 完整构建 20-30 分钟
- ✅ **独立运行**：不影响主容器（8100 端口）
- ✅ **快速切换**：可保留备份容器随时回滚
- ✅ **资源共享**：共用 Milvus、模型、vLLM 服务

---

## 🔧 故障排查

### 问题 1: vLLM 连接失败

```bash
# 检查 vLLM 服务状态
docker ps | grep vllm
curl http://127.0.0.1:18082/v1/models

# 如果不可达，检查 29154 的 vLLM 容器
docker logs momentseek-29154-qwen35-compiled-test --tail 50

# 临时回退启发式模式（修改 .env.0829）
ORCHESTRATION_ENABLED=false
```

### 问题 2: 端口 8101 冲突

```bash
# 查看端口占用
ss -lntp | grep :8101

# 修改端口（在 .env.0829 添加）
PLANNER_LAB_PORT=8102

# 重新部署
./deploy_planner_lab_0829.sh
```

### 问题 3: 前端没有 Planner Lab 入口

```bash
# 检查容器环境变量
docker exec momentseek-0829-planner-lab env | grep PLANNER_LAB

# 如果为空，重新构建
docker rm -f momentseek-0829-planner-lab
./deploy_planner_lab_0829.sh
```

### 问题 4: 能力接口返回 orchestration.enabled: false

```bash
# 检查编排配置
docker exec momentseek-0829-planner-lab env | grep ORCHESTRATION

# 查看审计日志
tail -20 runtime/orchestration-traces.jsonl

# 检查 vLLM 从容器内可达性
docker exec momentseek-0829-planner-lab curl -v http://127.0.0.1:18082/v1/models
```

### 问题 5: 构建失败

```bash
# 检查基础镜像是否存在
docker images | grep momentseek-0829-platform

# 如果不存在，先部署主容器
./deploy_0829.sh

# 重试 Planner Lab 部署
./deploy_planner_lab_0829.sh
```

---

## 📊 容器对比

| 项目 | 主容器 | Planner Lab 容器 |
|------|--------|----------------|
| 名称 | momentseek-0829-platform | momentseek-0829-planner-lab |
| 端口 | 8100 | 8101 |
| NPU | 1 | 1（共享） |
| 功能 | 完整平台 | Planner Lab 实验 |
| 镜像 | 完整构建 | Overlay 增量 |
| Runtime | 共享 | 共享 |
| Milvus | 共享 | 共享 |
| vLLM | 共享（29154:18082） | 共享（29154:18082） |

**两容器可以同时运行**，互不影响。

---

## 🎨 功能特性

### Planner Lab 提供

- **三种计划策略**
  - Fast: 快速单路径检索
  - Balanced: 多模态融合
  - Deep: 包含 vlm.rerank 多帧重排

- **三种交互模式**
  - Guide: 完全手动控制每一步
  - Assist: 建议计划，手动执行
  - Auto: 全自动执行并应用质量门控

- **完整执行链路**
  - visual.search, face.search, asr.search, ocr.search
  - vlm.rerank（Qwen3.5 多帧重排）
  - 多种融合算法（RRF, CombSUM, CombMNZ）
  - 步骤质量门控和稳定性早停

- **审计追踪**
  - 所有执行写入 `runtime/orchestration-traces.jsonl`
  - 包含完整决策链路、质量指标、来源贡献

### 与现有系统的关系

- ✅ 完全独立路由（`/api/planner-lab/*`）
- ✅ 不影响现有 `/api/search`
- ✅ 共享 Milvus 索引（无需重新索引）
- ✅ 共享模型文件
- ✅ 共享 vLLM 服务（29154 的 Qwen3.5）

---

## 🔄 日常操作

### 查看日志

```bash
# 实时日志
docker logs -f momentseek-0829-planner-lab

# 最近 100 行
docker logs --tail 100 momentseek-0829-planner-lab

# 审计追踪
tail -f runtime/orchestration-traces.jsonl
```

### 重启容器

```bash
docker restart momentseek-0829-planner-lab
```

### 查看资源占用

```bash
docker stats momentseek-0829-planner-lab
```

### 回滚到备份

```bash
# 如果部署时保留了备份
docker stop momentseek-0829-planner-lab
docker start momentseek-0829-planner-lab-backup-YYYYMMDD-HHMMSS
```

### 清理

```bash
# 停止并删除容器
docker stop momentseek-0829-planner-lab
docker rm momentseek-0829-planner-lab

# 删除 overlay 镜像（保留基础镜像）
docker images | grep planner-lab | awk '{print $1":"$2}' | xargs docker rmi
```

---

## 📝 配置调优

### 调整 vLLM 地址

如果 vLLM 服务地址变化，修改 `.env.0829`：

```bash
QWEN35_VLLM_BASE_URL=http://新地址:端口/v1
```

重新部署：
```bash
./deploy_planner_lab_0829.sh
```

### 切换编排 Profile

`.env.0829` 中修改：

```bash
# 可选：qwen35-unified 或 qwen35-temporal-efficient
ORCHESTRATION_PROFILE=qwen35-temporal-efficient
```

### 关闭 LLM 模式（回退启发式）

```bash
# .env.0829 中设置
ORCHESTRATION_ENABLED=false

# 重新部署
./deploy_planner_lab_0829.sh
```

---

## 🆘 获取帮助

- **详细文档**: `docs/PLANNER_LAB.md`
- **配置注册表**: `deploy/orchestration/qwen35-vllm.json`
- **审计日志**: `runtime/orchestration-traces.jsonl`
- **容器日志**: `docker logs momentseek-0829-planner-lab`
- **API 文档**: `http://127.0.0.1:8101/docs`

---

## ✅ 部署检查清单

- [x] `.env.0829` 已更新（添加 Planner Lab 配置段）
- [x] `deploy_planner_lab_0829.sh` 已创建并可执行
- [x] `stop_planner_lab_0829.sh` 已创建并可执行
- [x] 确认基础镜像存在：`docker images | grep momentseek-0829-platform`
- [x] 确认 vLLM 服务可达：`curl http://127.0.0.1:18082/v1/models`
- [x] 确认端口 8101 未占用：`ss -lntp | grep :8101`
- [ ] 执行部署：`./deploy_planner_lab_0829.sh`
- [ ] 验证功能：`curl http://127.0.0.1:8101/api/planner-lab/capabilities`
- [ ] 访问界面：`http://127.0.0.1:8101`

---

**最后更新**: 2026-08-12
