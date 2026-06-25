# 共享服务器运行规范

当前部署目录：

```text
/mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp
```

公网入口：`http://110.126.0.52:8300`

## 日常检查

```bash
cd /mnt/mog2/wyl/comfyui-wxy/video-retrieval-mvp
./scripts/check_resource.sh
docker-compose -f compose.yml -f compose.server.yml ps
docker logs --tail 100 momentseek-mvp-app
```

## 无卡模式

默认运行方式不映射 NPU：

```bash
docker-compose -f compose.yml -f compose.server.yml up -d
```

可以完成前后端、CPU CLIP、CPU InsightFace、CPU Whisper 和向量检索。

## NPU 模式

只有在卡号由使用者明确释放后才能执行：

```bash
NPU_DEVICE_ID=<approved-id> docker-compose \
  -f compose.yml -f compose.server.yml -f compose.ascend.yml up -d
```

禁止使用 `privileged: true`，禁止映射未批准的 `/dev/davinci*`。任务结束后用 `npu-smi info` 确认没有 MomentSeek 模型进程残留。

## 停止

```bash
docker-compose -f compose.yml -f compose.server.yml -f compose.ascend.yml down
```

`runtime/` 与 `models/` 是 bind mount，停止或重建容器不会删除资产和索引。

