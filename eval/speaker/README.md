# Speaker Diarization 与声纹模型评测

本目录用于在不接入正式索引 pipeline 的前提下，对 Speaker Diarization 和 voice embedding 模型做统一质量、速度和资源评测。

## 第一轮候选

Diarization：

1. `pyannote/speaker-diarization-community-1`
2. `nvidia/diar_sortformer_4spk-v1`
3. `modelscope/3D-Speaker` diarization pipeline

Voice embedding：

1. `Wespeaker/wespeaker-voxceleb-resnet34-LM`
2. `iic/speech_campplus_sv_zh-cn_16k-common`
3. `iic/speech_eres2netv2_sv_zh-cn_16k-common`

候选的固定标识、角色和风险见 `candidates.json`。

## 数据文件

`cases.example.json` 是可提交的格式示例。实际本地素材使用 `cases.local.json`，不提交 Git。

每个 case 包含：

```text
id
media_path
start_seconds
end_seconds
language
scenario
truth_rttm（optional）
```

先生成统一的 16kHz mono WAV：

```powershell
python scripts/speaker_eval.py prepare `
  --cases eval/speaker/cases.local.json `
  --output-dir eval/speaker/audio
```

只校验配置：

```powershell
python scripts/speaker_eval.py validate --cases eval/speaker/cases.local.json
```

## 输出协议

每个 diarization adapter 最终输出统一 JSON：

```json
{
  "case_id": "dialogue_zh",
  "model": "pyannote-community-1",
  "audio_seconds": 120.0,
  "elapsed_seconds": 8.2,
  "rtf": 0.0683,
  "peak_device_memory_mb": 1234,
  "turns": [
    {"start_ms": 100, "end_ms": 1500, "track_id": 0}
  ]
}
```

声纹 adapter 输出每个人工确认 speaker 样本的归一化 embedding，再由统一报告计算同人/异人 cosine、EER、margin、速度和资源。

## 评测阶段

1. Smoke：每个模型先跑 1 个 60-120 秒样本，确认依赖、输出和资源。
2. Local quality：全部本地 case，人工补 speaker turn/identity truth。
3. Long-form：至少一个 30-60 分钟视频，检查 label 漂移和内存增长。
4. Ascend：相同 WAV 和输出协议迁到 910B，硬门槛为 1 小时音频不超过 5 分钟。

模型下载必须通过显式准备步骤完成。实验运行时只读取本地路径，不隐式联网。

## 原始输出网页预览

启动仅绑定本机的评测服务：

```powershell
python scripts/speaker_review_server.py
```

浏览器会打开 `http://127.0.0.1:8765/eval/speaker/viewer/`。页面直接展示模型原始 turn，
不会过滤短段、合并相邻段或执行 ASR 对齐。

### 中文综艺 60 秒原始输出（2026-07-13）

| 模型 | 自动输出 speaker 数 | 原始 turn 数 | 当前观察 |
|---|---:|---:|---|
| Community-1 | 3 | 46 | 有不同男性被合并，以及 17ms 级标签抖动 |
| 3D-Speaker | 6 | 20 | 更倾向拆分说话人，需人工确认是否过度聚类 |
| Sortformer 4spk v1 | 3 | 19 | 大量片段集中到同一标签；模型最多支持 4 人 |

这只是无真值的人工试听基线，不能作为 DER 结论。下一步应在网页中标注不同男性的代表时间点，
分别统计漏分（不同人同标签）和过分（同一人不同标签）。

### 扩展场景 60 秒原始输出（2026-07-14）

所有模型使用完全相同的 16kHz mono WAV、自动人数模式和默认后处理。

| 场景 | Community-1 | 3D-Speaker | Sortformer 4spk |
|---|---:|---:|---:|
| 西语广告前 60 秒 | 3 人 / 30 turn | 5 人 / 16 turn | 2 人 / 18 turn |
| 中文电视剧前 60 秒 | 1 人 / 14 turn | 2 人 / 7 turn | 1 人 / 13 turn |
| 书籍纪录片 180–240 秒 | 2 人 / 16 turn | 3 人 / 4 turn | 2 人 / 16 turn |

人数和 turn 数只描述模型的原始聚类倾向。没有人工真值前，不能据此判断准确率。
