# PP-OCRv6 Small 昇腾后端实验

## 目标

保持现有 PP-OCRv6 Small 检测、方向分类和识别模型，保持 OCR v3 索引与检索协议，仅替换推理运行时。不要升级 Medium，也不要引入 PaddleOCR-VL。

## 已确认事实

- 当前 RapidOCR CPU 路径可正确识别；ONNX Runtime CANN 路径虽然可初始化和预热，但真实输出错误，因此不能作为产品后端。
- PaddleX 高性能推理支持 ONNX 和 Ascend OM，但官方 NPU 插件当前列出的 Python 版本是 3.10；MomentSeek MindIE 主容器使用 Python 3.11。
- 第一阶段必须在独立实验容器完成，不能修改正式容器依赖。通过后以独立 OCR sidecar 或经过验证的兼容运行时接入 `OCRBackend`。
- 服务器 NPU 6 为 Ascend 910B4，驱动 25.5.1，CANN/ATC 9.0.0-beta.2；OM 目标 `soc_version` 为 `Ascend910B4`。该 beta 工具链生成的 OM 需要与正式运行环境做版本兼容验证。
- 三个现有 ONNX 文件已通过 checker 和 SHA-256 校验。检测输入为动态 NCHW，识别输入为动态 batch/width、固定高度 48，方向分类空间尺寸固定为 80x160。

## 实施顺序

1. 运行 `scripts/ocr_ascend_preflight.py`，固定三个 ONNX 文件校验值、输入输出名和动态维度。
2. 运行 `scripts/ocr_shape_profile.py`，通过 RapidOCR CPU 会话记录真实视频帧的检测输入尺寸和识别 crop 宽度分布，确定少量通用 shape/bucket，不为局部测试视频设计阈值。
3. 使用 ATC/PaddleX HPI 将相同 ONNX 权重转换为 OM；检测与识别均不得静默回退 CPU。
4. 在同一批真实帧上对比 RapidOCR CPU 与 OM NPU 的 box、文字、置信度和耗时。
5. 通过后实现 `PpOcrSmallOmBackend`，接入现有常驻 daemon；保持 `ocr.npz` schema 不变。

## 验收门槛

- 合成文字与真实视频帧均不得出现全空输出。
- CPU 与 NPU 的有效文本结果应基本一致；差异必须能由数值精度或固定 shape 前处理解释。
- 报告冷启动、预热后 p50/p95、吞吐、HBM 和连续运行失败率。
- 正式 NPU 配置失败即终止，不自动回退 CPU。
