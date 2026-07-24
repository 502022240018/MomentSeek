# Ascend 常驻索引进程上下文隔离回归

日期：2026-07-21  
设备：物理 NPU6（正式服务 NPU5 未参与）  
输入：64.12 秒、1280×720 视频  
顺序：`visual -> face -> ocr -> visual`

## 背景

旧的 per-stage 子进程每次退出都会销毁全部 NPU context，因此没有暴露跨运行时污染，但每个任务都重复加载模型。把 Visual、Face 和 OCR 全部常驻到同一 daemon 进程后，Torch-NPU、ONNX Runtime CANN 和原生 ACL 依次切换 context，后续 Torch 卷积可能报 `107003 stream is not in current context`。

## 方案

daemon 只负责 SQLite 队列和串行调度；visual、face、ASR、OCR 各使用一个懒启动、可常驻的 spawn 子进程。通道进程内部可以复用自己的 ModelPool，但不接触其他通道的运行时。加速器异常后丢弃整个子进程，上下文类错误最多在新进程重试一次。

## 结果

| 阶段 | 首次耗时 | 热启动耗时 | worker PID | 结果 |
|---|---:|---:|---:|---|
| visual | 37.324 s | 2.867 s | 31（两次相同） | completed |
| face | 65.000 s | - | 755 | completed |
| OCR | 26.226 s | - | 7488 | completed，32/32 帧成功 |

总计 128.561 秒，所有阶段一次成功，无 `107003`，容器退出后 NPU6 无残留进程。采样到的资源峰值为 19.7 GiB、810 PID；多数 PID/线程来自 CANN EP、GE 编译辅助进程与 ffmpeg，而不是业务 worker 数量。

## 生产结论

- 正式环境启用 `NPU_WORKER_MODE=isolated`，保留全局串行调度。
- `THREAD_LIMIT=8` 保留，用于 BLAS/OpenMP/CPU ORT；它不能覆盖 CANN EP 和 ffmpeg。
- 容器默认 24 CPU、2048 PID 上限；暂不设内存硬上限，待 ASR 也常驻后的组合峰值测完再定。
- 搜索 API 是独立 Uvicorn 进程；Face 参考图查询当前走 CPU ORT，并显式限制为 8 个 intra-op、1 个 inter-op 线程。
