# MomentSeek 文档入口

本目录是 MomentSeek / `video_retrieval_mvp` 项目的权威文档入口。

## 阅读顺序

新会话或新同学接手时，优先按这个顺序读：

1. `docs/CURRENT.md`：当前项目、服务器、模型和部署状态。
2. `docs/ISSUES_AND_ROADMAP.md`：当前问题池和后续优化路线。
3. `docs/RETRIEVAL_CHANNELS.md`：visual / face / ASR / OCR 的索引与召回协议。
4. `docs/ARCHITECTURE.md`：系统架构、API Surface、模块边界。
5. `docs/OPERATIONS.md`：共享服务器、公网入口、安全操作规范。
6. `docs/VALIDATION.md`：验证命令和完成声明规则。
7. `docs/LESSONS_LEARNED.md`：历史踩坑、事故教训、工具注意事项。

实验结论看 `docs/experiments/README.md`。
可复现实验资产、manifest、query、schema 看 `eval/README.md` 和对应 `eval/<area>/README.md`。

## 更新规则

每类信息只维护在一个地方：

```text
系统当前状态变化 -> docs/CURRENT.md
架构或模块边界变化 -> docs/ARCHITECTURE.md
API 新增/删除/语义变化 -> docs/ARCHITECTURE.md 的 API Surface 章节
检索通道协议或索引格式变化 -> docs/RETRIEVAL_CHANNELS.md
发现问题或后续优化点 -> docs/ISSUES_AND_ROADMAP.md
服务器/公网/部署操作变化 -> docs/OPERATIONS.md
验证命令或验收规则变化 -> docs/VALIDATION.md
重复踩坑或操作经验 -> docs/LESSONS_LEARNED.md
实验结论 -> docs/experiments/<area>/<date>-<topic>.md
评测数据格式或运行方法变化 -> eval/<area>/README.md 或相邻 eval 文件
新 Codex 会话启动 prompt 变化 -> docs/handoff/SESSION_BOOTSTRAP.md
```

除 `docs/ISSUES_AND_ROADMAP.md` 外，其他活跃文档不要维护独立的问题列表或未来计划列表；只链接到问题池。

## 文件职责

| 文件 | 职责 |
|---|---|
| `CURRENT.md` | 最新事实状态快照 |
| `ARCHITECTURE.md` | 系统拓扑、数据流、模块边界、API Surface |
| `RETRIEVAL_CHANNELS.md` | visual / face / ASR / OCR 的索引 schema 和召回行为 |
| `ISSUES_AND_ROADMAP.md` | 唯一问题池和后续路线图 |
| `OPERATIONS.md` | 共享服务器、公网入口、安全 SOP |
| `VALIDATION.md` | 验证命令和完成声明规则 |
| `LESSONS_LEARNED.md` | 操作经验、事故复盘、重复踩坑 |
| `experiments/` | 人类可读的实验总结和结论 |
| `handoff/` | 新会话启动 prompt |
| `archive/` | 历史参考，不再作为权威文档维护 |

## 归档规则

旧交接文档、旧报告、历史专题文档保存在 `docs/archive/`。它们只用于追溯背景，当前工作应更新本页列出的固定文档。
