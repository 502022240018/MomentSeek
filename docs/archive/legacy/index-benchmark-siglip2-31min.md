> Archived reference. Current documentation starts at `docs/README.md`.

# Visual indexing benchmark: SigLIP2 31min 1080p

- Date: 2026-07-02
- Server: `110.126.0.52`
- Container: `momentseek-current-app`
- Device: container `npu:0`, physical NPU 2
- Video: `五哈团美食速度挑战纯享_31min_1080p.mp4`
- Duration: 1873.44 s
- Resolution: 1920x1080
- Visual model: `siglip2-so400m-384`
- Sampling: 5.0 fps
- Segment length: 5.0 s
- Batch size: 32
- Decode height: 256
- Output path on server: `/app/runtime/bench/visual_siglip2_31min_20260702-084826`

| Metric | Value |
|---|---:|
| Model load | 13.018 s |
| Indexing excluding model load | 298.394 s |
| Total including model load | 311.413 s |
| Frames indexed | 9367 |
| Segments indexed | 375 |

Notes:

- The benchmark wrote to a separate bench directory and did not overwrite production indexes or database rows.
- NPU memory returned to baseline after the subprocess exited.
- Hugging Face loading resolves cached models to local snapshot paths, avoiding repeated online metadata checks for cached SigLIP2/ChineseCLIP.
