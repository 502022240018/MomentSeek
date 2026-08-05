# Milvus-only 正式索引切换手册

本文用于把已有 MomentSeek 运行目录中的视频切换到当前正式模式：
**Milvus 是唯一在线检索库**。SQLite 仍保存视频和任务元数据；`indexes/<video-id>`
下的 NPZ 仅是离线诊断/重建副本，任何 API 和检索服务都不会读取它们。

适用对象是已有视频和旧索引的运行环境。全新部署且尚无视频时，不需要执行本手册。

## 1. 切换原则

- 不删除 collection，不手工修改 Milvus 行；通过正常索引阶段重建。
- 每个通道先写入新 `asset_version`，核对持久化行数，再将 manifest 指针发布到新版本；发布完成后才清理该视频该通道的旧版本。
- 单个通道失败时，旧的已发布版本仍保持可查询；失败的视频需要修复后单独重试。
- 这是一次索引数据升级，不是应用镜像回滚。发布新版本后，旧索引版本会被清理；如需再次生成索引，应从原始视频重新执行本命令。

## 2. 维护窗口前准备

1. 将本次应用镜像和配置部署完成，但不要同时执行普通索引任务。
2. 备份 `HOST_RUNTIME_DIR` 中的 `catalog.sqlite3`、`uploads/` 和 `indexes/`；Milvus 的备份按运维方既有方案执行。
3. 确认所有原视频仍位于 `uploads/`，模型目录完整，且 Milvus 可连接。
4. 为本次任务预留 NPU、磁盘和维护窗口。全量重建会重新运行视觉、脸、ASR、speaker 与 OCR。
5. 首次先选 1--3 个有代表性的视频做灰度验证，再执行全量。

`MILVUS_ENABLED=true` 与 `MILVUS_WRITE_ENABLED=true` 是必需条件。不要在切换中停止 Milvus，也不要同时运行 Web UI 发起的索引任务或 indexer daemon。

## 3. 预检查与灰度

以下命令在应用容器中运行；把 Compose 文件组合替换为本环境实际使用的组合。若本环境自带 Milvus，命令追加 `-f compose/compose.milvus.yml`。

```bash
# 只列出视频、源文件可用性和计划执行的阶段；不连接 Milvus、不写数据。
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml \
  exec app python -m app.maintenance.reindex_milvus_only --dry-run

# 灰度：替换为一个真实 video_id，执行 visual、face、asr(+speaker)、ocr。
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml \
  exec app python -m app.maintenance.reindex_milvus_only \
  --video-id VIDEO_ID --execute

# 核验灰度视频：manifest 中的发布版本必须与 Milvus 对应版本的行数一致。
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml \
  exec app python -m app.maintenance.reindex_milvus_only \
  --video-id VIDEO_ID --verify-only
```

`asr` 阶段默认同时生成并发布 `speaker`，不会再额外运行一次重复的 speaker 阶段。如果只需要某个通道，可用 `--modalities visual`、`--modalities ocr` 等；单独运行 `speaker` 时要求该视频已经有已发布的 ASR 版本。
对部分重建执行 `--verify-only` 时，必须传入与重建时相同的 `--modalities`；不传时默认核验全部正式通道，并会把未重建的通道报告为缺少发布指针。

## 4. 全量执行与核验

确认灰度视频的索引和检索正常后，保持维护窗口并执行：

```bash
docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml \
  exec app python -m app.maintenance.reindex_milvus_only --execute

docker compose --env-file .env -f compose/compose.yml -f compose/compose.ascend.yml \
  exec app python -m app.maintenance.reindex_milvus_only --verify-only
```

命令会为每个视频输出一行 JSON。只有进程退出码为 `0` 且每个通道均为 `status=ok`，才可结束迁移。退出码 `2` 表示至少一个视频失败或源视频缺失；已成功发布的其他视频不受影响。修复原因后使用 `--video-id` 精确重试失败视频，避免无谓地重建全量数据。

建议额外选择至少一段有画面、人脸、语音和文字的视频，在 UI 或 API 上验收：

- 文字视觉检索；
- 参考脸检索；
- ASR 文本检索；
- 声纹检索；
- OCR 文字检索。

## 5. 故障处理边界

- **Milvus 不可用或写入失败**：任务会失败关闭，不会把 NPZ 当线上索引继续服务。恢复 Milvus 后，对失败视频重新运行同一命令。
- **源视频缺失**：命令不会覆盖其既有已发布版本。恢复原始视频后重试；没有原始视频时，只能由经授权的离线恢复流程处理保留的 NPZ，不能把它接入线上回退。
- **新版本已发布但业务结果异常**：停止后续批量迁移，保留证据和日志；修复配置或模型后，从原视频重新索引该视频。不要手工把 manifest 改回旧版本。

## 6. NPZ 保留策略

当前版本保留 NPZ 作为离线诊断/灾后重建输入，且不会由任何运行时读路径访问。它们不等于线上双写、影子读或回退机制。若未来确认不需要离线副本，应另开一次有数据保留评审和空间评估的变更，统一移除 NPZ 写入及离线恢复工具；不要与本次 Milvus-only 切换混合执行。
