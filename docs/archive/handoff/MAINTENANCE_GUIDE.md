> Archived reference. Current documentation starts at `docs/README.md`.

# 交接目录维护规则

更新时间：2026-07-03

## 1. 文件分工

| 文件 | 用途 | 更新频率 |
|---|---|---|
| `README.md` | 交接目录入口和阅读顺序 | 目录结构变化时 |
| `SESSION_BOOTSTRAP.md` | 新 Codex 会话启动 prompt | 关键上下文变化时 |
| `CURRENT_STATUS.md` | 当前系统状态快照 | 每次重要变更后 |
| `PROGRESS_LOG.md` | 按时间记录进展和待办 | 每次阶段性工作后 |
| `MAINTENANCE_GUIDE.md` | 本目录维护规范 | 规则变化时 |

## 2. 什么时候必须更新

以下情况必须更新本目录：

- 新增/删除/重命名核心通道：visual、face、asr、ocr。
- 改变索引格式，例如 `.npz` 新增字段、向量维度变化。
- 改变召回粒度或排序逻辑。
- 服务器部署方式变化。
- 公网入口方式变化。
- 新增重要模型或替换默认模型。
- 做了影响同事接手的架构决策。
- 新装/移除 Codex 插件或 skill。

## 3. 推荐更新方式

完成一项工作后：

1. 更新 `CURRENT_STATUS.md`：记录当前状态。
2. 更新 `PROGRESS_LOG.md`：记录发生了什么、为什么重要、后续待办。
3. 如涉及新会话上下文，更新 `SESSION_BOOTSTRAP.md`。
4. 如涉及专题文档，更新对应专题文档。

## 4. 不要把什么放进来

不要在交接文档里写：

- API key
- SSH 私钥
- 密码
- 未脱敏 token
- 其他人的敏感进程信息
- 大段无关日志

如果必须记录排错信息，只保留：

- 错误摘要
- 关键命令
- 关键输出片段
- 处理结论

## 5. 服务器操作记录规范

服务器相关记录要写清：

```text
时间：
目的：
执行前检查：
执行命令：
影响范围：
验证结果：
是否释放显存/资源：
```

禁止只写：

```text
重启了服务
修好了
没问题
```

必须带证据，例如：

```text
curl /api/health 返回 200
docker ps 显示 healthy
npu-smi 显示 NPU 2 无 MomentSeek 进程
```

## 6. 完成声明规则

如果使用 Superpowers，应遵循 `verification-before-completion`：

```text
没有 fresh verification，就不要说完成。
```

也就是：

- 说“测试通过”前，必须跑测试。
- 说“后端恢复”前，必须跑 health check。
- 说“索引完成”前，必须查 job status 和索引文件。
- 说“已推送”前，必须看 git / push 输出。
