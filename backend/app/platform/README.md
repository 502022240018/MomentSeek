# platform/ — 平台化预留层

当前为空，暂无代码。这里预留给"平台级运行时上下文"。

## 既定方向：context 化（待办）

现状：全局单例（`settings` / `catalog` / `search_engine` / `search_orchestrator`）
定义在 `app/main.py`，各路由模块通过函数内 `from app import main as runtime`
惰性导入来规避循环依赖（全仓约 38 处）。

计划：新建 `app/platform/context.py` 承载这些单例与共享辅助函数，
`main.py` 只负责组装 FastAPI app 与生命周期管理；路由模块顶部直接
`from app.platform import context`，消除惰性互引。

依赖方向由 `main → routes →（惰性）main` 变为 `main → routes → context`。

执行该改造时需同步更新对 `app.main` 属性做 monkeypatch 的测试
（如 `test_main_integration.py`、`test_retrieval_orchestration.py`）。
