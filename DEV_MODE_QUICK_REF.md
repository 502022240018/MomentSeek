# 开发模式快速参考卡

## 🚀 快速启用

```bash
# 1. 编辑配置
vim .env.0829
# 设置: DEV_MODE=true

# 2. 部署
./deploy_0829.sh

# 3. 监控日志
docker logs -f momentseek-0829-platform
```

## 📋 核心命令

| 操作 | 命令 |
|------|------|
| 启用开发模式 | `DEV_MODE=true` in `.env.0829` |
| 跳过构建 | `DEV_SKIP_BUILD=true` in `.env.0829` |
| 部署/重启 | `./deploy_0829.sh` |
| 查看日志 | `docker logs -f momentseek-0829-platform` |
| 测试热重载 | `./test_dev_mode.sh` |
| 健康检查 | `curl http://127.0.0.1:8100/api/health` |
| 停止服务 | `docker stop momentseek-0829-platform` |
| 进入容器 | `docker exec -it momentseek-0829-platform bash` |

## ⚡ 效率对比

| 模式 | 代码修改生效时间 |
|------|------------------|
| 生产模式 | 3-5 分钟（重新构建） |
| **开发模式** | **2-5 秒（热重载）** |
| 性能提升 | **~60倍** |

## ✅ 检查清单

```bash
# 1. 确认开发模式已启用
grep "DEV_MODE=true" .env.0829

# 2. 确认代码已挂载
docker inspect momentseek-0829-platform --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' | grep backend

# 3. 确认热重载已启用
docker inspect momentseek-0829-platform --format '{{.Config.Cmd}}' | grep reload

# 4. 测试热重载
echo "# test" >> backend/app/main.py
sleep 5
docker logs --tail 10 momentseek-0829-platform | grep -i reload
git checkout backend/app/main.py  # 恢复
```

## 🔧 常见问题

### Q: 代码修改不生效？
```bash
# 检查配置
grep DEV_MODE .env.0829

# 重新部署
docker stop momentseek-0829-platform && docker rm momentseek-0829-platform
./deploy_0829.sh
```

### Q: 依赖未安装？
```bash
# 临时关闭跳过构建
sed -i 's/DEV_SKIP_BUILD=true/DEV_SKIP_BUILD=false/' .env.0829
./deploy_0829.sh
# 完成后恢复
sed -i 's/DEV_SKIP_BUILD=false/DEV_SKIP_BUILD=true/' .env.0829
```

### Q: 查看重载日志？
```bash
docker logs -f momentseek-0829-platform 2>&1 | grep -i "reload\|detect\|restart"
```

## 📊 工作流推荐

### 日常开发
```bash
# .env.0829
DEV_MODE=true
DEV_SKIP_BUILD=true

# 工作流
编辑代码 → 保存 → 等待2-5秒 → 测试
```

### 依赖更新
```bash
# .env.0829
DEV_MODE=true
DEV_SKIP_BUILD=false  # 暂时关闭

# 执行
./deploy_0829.sh

# 完成后恢复
DEV_SKIP_BUILD=true
```

### 提交前验证
```bash
# .env.0829
DEV_MODE=false  # 使用生产构建

# 完整测试
./deploy_0829.sh
curl http://127.0.0.1:8100/api/health
```

## 🎯 最佳实践

1. **首次部署**: `DEV_SKIP_BUILD=false` → 构建镜像
2. **日常开发**: `DEV_SKIP_BUILD=true` → 快速启动
3. **依赖变更**: `DEV_SKIP_BUILD=false` → 重新构建
4. **提交代码前**: `DEV_MODE=false` → 生产模式测试
5. **监控日志**: 保持 `docker logs -f` 运行，观察重载

## 📖 完整文档

详细说明请查看: [DEV_MODE_GUIDE.md](./DEV_MODE_GUIDE.md)

## ✨ 测试结果

```
✅ 2026-07-27 测试通过
✓ 容器运行正常
✓ 代码目录已挂载
✓ 热重载已启用
✓ 文件变更检测正常
✓ 服务自动重载成功
```

---
**节省时间 = 提升效率 = 快乐开发** 🎉
