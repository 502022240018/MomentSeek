# 当前验证记录

日期：2026-06-24

- 后端单元测试：4 passed。
- React/TypeScript production build：通过。
- 公网 `/`、`/api/health`、`/openapi.json`：HTTP 200。
- 容器隔离：`privileged=false`、`devices=null`、24 CPU、48 GiB RAM 上限。
- Visual：9 秒视频生成 2 个片段索引；英文文本查询命中 `0–9s`。
- ASR sidecar：中文“广告牌”命中 `4.0–8.5s`。
- Whisper tiny：20 秒音频约 23 秒完成转写；“journalism”命中 `11.24–14.24s`。
- Face：28 秒视频生成 tracklet；参考图检索返回 3 个片段，最高 cosine 0.855。
- 人物库：登记 `Walter` 后，纯文字查询可调用同一 face embedding。
- 所有索引子进程在任务后退出；部署前后没有新增 NPU 进程。

性能结论：CPU CLIP 和 Whisper 可作为可运行 baseline；CPU Face 约 6 分钟/28 秒视频，必须作为下一轮首要优化对象。

