# platform/ — 平台运行时层

## context.py

平台运行时上下文，承载进程级单例与路由共享辅助函数：

- 单例：`settings` / `catalog` / `search_engine` / `search_orchestrator`
- 媒体与任务入口别名：`probe_video`、`extract_video_frame`、`launch_job` 等
- indexer daemon 监督：`start_indexer_daemon_if_configured()` /
  `stop_indexer_daemon()` / `_restart_indexer_daemon()`
- 路由共享辅助：`_safe_suffix`、`_save_upload`、`_clip_cache_path` 等

依赖规则：**本模块不 import 任何路由**，因此路由文件在顶部
`from app.platform import context` 不会形成循环；`main.py` 只负责组装
FastAPI app、挂路由和生命周期。

依赖方向：`main → routes → context ← main`（树，无环）。

测试约定：需要替换运行时状态时 monkeypatch 本模块属性，例如
`monkeypatch.setattr(context, "catalog", fake_catalog)`。

历史：2026-07-31 之前这些对象住在 `app.main`，路由靠 38 处函数内
`from app import main as runtime` 惰性导入规避循环依赖；context 化后已全部消除。
