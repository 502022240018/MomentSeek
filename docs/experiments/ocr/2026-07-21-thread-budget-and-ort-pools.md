# OCR CPU 线程预算与 ONNX Runtime 会话实验

## 目的

定位长视频 OCR 卡死和 OpenBLAS 报错是否由 OCR ACL 本身引起，并为共享服务器选择安全的 CPU 线程默认值。

## 环境与方法

- 服务器：Kunpeng 920，192 核，8 个 NUMA 节点，每节点 24 核。
- 正式平台：单 Uvicorn worker，索引 daemon 串行执行任务；正式 NPU 5 未改动。
- 隔离测试：正式镜像 `momentseek-29154-platform:be7d46b8422d`，NPU 6/7。
- 纯语义：同一模型、同一批 2000 条文本，每个线程值交叉运行两次。
- 组合链路：同一视频前 96 帧，ACL OCR 后接语义编码，每个线程值交叉运行两次。
- 线程值同时应用于 OpenBLAS、OpenMP、MKL、NumExpr 和 BLIS；容器 CPU 上限为 24。

## 根因一：未限制的 BLAS/OpenMP 线程池

未设置上限时，纯 CPU 语义测试达到 341 个线程，出现 `OpenBLAS warning` 和 `Bad memory unallocation`，15 分钟超时。设置上限后所有测试均成功且无上述错误。

纯语义结果：

| 上限 | 中位模型加载 | 中位编码 | 中位总耗时 | 峰值线程 | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|
| 4 | 13.093 s | 19.348 s | 32.441 s | 17 | 1946 MB |
| 8 | 13.252 s | 11.726 s | 24.979 s | 29 | 1955 MB |
| 16 | 13.250 s | 8.331 s | 21.581 s | 53 | 1992 MB |
| 24 | 13.195 s | 7.300 s | 20.495 s | 77 | 2018 MB |

16 到 24 的总耗时仅再降低约 5%，但峰值线程增加约 45%，已经越过明显的边际收益拐点。

## 根因二：RapidOCR 临时 ONNX Runtime 会话

`RapidOCRAclBackend` 会先创建 Det/Cls/Rec 三个 CPU ONNX Runtime 会话，以复用 RapidOCR 的前后处理对象，然后再把推理会话替换为 ACL OM。ORT 默认按宿主机 192 核创建线程池，三个会话使容器达到 512 PID 上限并卡在 `_create_inference_session`，日志出现 `fork: Resource temporarily unavailable`。

修复为这些仅提供元数据的临时会话显式设置 `intra_op_num_threads=1`、`inter_op_num_threads=1`。修复后 96 帧组合测试全部成功：

| BLAS/OpenMP 上限 | 中位总耗时 | 峰值线程 | 峰值 RSS |
|---:|---:|---:|---:|
| 8 | 39.400 s | 156 | 2899 MB |
| 16 | 38.433 s | 164 | 2926 MB |
| 24 | 37.072 s | 172 | 2967 MB |

组合链路中，16 相对 8 只快约 2.5%，24 相对 8 只快约 5.9%。ACL/CANN 运行时本身保留约 148 个线程，因此 BLAS 上限不是整个进程的总线程数。

## 长视频完整 soak 与识别宽度边界

输入为服务器现有长视频 `dd75f7ce5aa04f57b9a28a08d91f37ac.mkv`：875,256,892 bytes，约 1998.4 秒，原始分辨率 3840×1608、25 fps。测试按 1 fps 抽帧并把画面高度缩放为 720；由于它比标准 4K 16:9 更宽，解码后宽度约 1720，是一个偏严苛的超宽样本。

识别模型输入宽度由文字框宽高比决定，近似为 `48 × 文字框宽高比`，不等于原视频宽度。因此细长字幕框可能显著超过常见动态宽度档位。先前 1024 档出现 15 个失败帧，扩展到 1600 后仅剩 1 个失败帧，其输入宽度为 1892。最终方案采用两个 Rec OM：主模型覆盖到 1600，宽模型覆盖 1920/2048；超过 2048 的极端输入等比缩放到 2048，并统计缩放次数，避免为无界宽高比持续编译模型。

最终完整 soak 结果：

| 指标 | 结果 |
|---|---:|
| 抽取帧数 | 1998 |
| OCR 命中帧 / chunks | 743 / 743 |
| OCR 失败帧 | 0 |
| 实际最大 Rec 输入宽度 | 1892 |
| 超过 2048 的缩放次数 | 0 |
| 后端加载 | 6.282 s |
| 帧循环 | 856.074 s |
| OCR 推理 | 847.291 s |
| 解码与后处理 | 8.783 s |
| 语义向量 | 21.226 s |
| 索引保存 | 0.299 s |
| 函数总耗时 | 877.599 s |
| 含后端加载的 soak 总耗时 | 883.881 s |

本样本支持 2048 作为工程上限：它覆盖了观测最大值并保留约 8% 余量；兜底缩放使更极端文本框也不会导致整帧失败。若后续生产指标显示缩放比例明显上升，再基于真实分布重新评估，而不是继续预编译更大的无界档位。

## 结论与配置建议

- 共享在线平台默认使用 8：资源占用较低，组合链路与 24 的差距不足 6%，并显著远离 PID 上限。
- 专用离线批处理可覆盖为 16；只有独占 CPU、完成 NUMA/cpuset 绑定并确认在线延迟不受影响时才考虑 24。
- 线程环境变量必须在导入 NumPy、PyTorch、ONNX Runtime 前设置。
- 容器 CPU/cpuset、任务并发数、BLAS/OpenMP、PyTorch intra/inter-op、ONNX Runtime intra/inter-op 应分别治理，不能用一个变量假定覆盖所有运行时。

## 原始结果

- 修复前基线：`/home/momentseek-29154/logs/ocr-root-cause-suite-codex-20260721`
- 纯语义 sweep：`/home/momentseek-29154/logs/ocr-thread-sweep-v3-20260721`
- 修复后 OCR+语义 sweep：`/home/momentseek-29154/logs/ocr-thread-sweep-ocr-semantic-v2-20260721`
- 1024 档完整 soak：`/home/momentseek-29154/logs/ocr-full-soak-long-v2-20260721`
- 1600 档完整 soak：`/home/momentseek-29154/logs/ocr-full-soak-long-v3-rec1600-20260721`
- 1600/2048 双模型最终 soak：`/home/momentseek-29154/logs/ocr-full-soak-long-v4-rec2048-20260721`

这些目录属于服务器实验日志，不是仓库内测试夹具。
