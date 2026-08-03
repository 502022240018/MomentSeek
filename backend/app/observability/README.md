# observability/ — 可观测性预留层

当前为空，暂无代码。

预留给结构化日志、指标采集（索引耗时、检索延迟、通道命中率）、
链路追踪等能力。接入时约定：

- 采集点统一从本层暴露的接口打点，业务模块不直接依赖具体后端（如 Prometheus）；
- 检索侧现有的 `retrieval/retrieval_metrics.py`（RetrievalProfiler）在本层落地后迁入。
