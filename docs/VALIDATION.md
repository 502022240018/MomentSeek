# 验证与完成声明规则

本文档记录验证命令和验收标准。没有新鲜验证输出，不要声明“完成 / 修复 / 通过”。

## 文档检查

检查活跃文档是否又出现了重复的问题池：

```powershell
rg -n "<duplicate issue-list heading pattern>" docs -g "*.md" -g "!docs/archive/**" -g "!docs/superpowers/**"
```

期望：

- 活跃问题池只在 `docs/ISSUES_AND_ROADMAP.md`。
- `docs/archive/` 下出现历史问题列表是可以接受的。

检查旧 handoff 是否只作为归档引用出现：

```powershell
rg -n "<legacy handoff filename pattern>" docs -g "*.md" -g "!docs/archive/**" -g "!docs/superpowers/**"
```

## 本地后端测试

在 `video_retrieval_mvp/backend` 下运行：

```powershell
pytest
```

搜索相关改动可以先跑：

```powershell
pytest tests/test_search.py -v
```

## 本地基础 API smoke check

后端启动后，在仓库根目录运行：

```powershell
python scripts/smoke_check.py --base-url http://127.0.0.1:8000
```

该脚本只检查 `/api/health` 和 `/api/jobs`。release manifest 中应把它记录为 `verification.api_smoke`，不要把它等同于真实检索验证。

## 模型清单校验

开发模型清单校验：

```powershell
python scripts/verify_models.py --manifest deploy/models/dev-full.models.json --lock runtime/dev-models.lock.json
```

## 前端检查

在 `video_retrieval_mvp/frontend` 下运行：

```powershell
npm run build
```

开发服务器：

```powershell
npm run dev
```

## 服务器健康检查

只读：

```bash
ssh root@110.126.0.52 "docker ps --filter name=momentseek-current-app"
ssh root@110.126.0.52 "curl -s http://127.0.0.1:18300/api/health"
ssh root@110.126.0.52 "npu-smi info"
```

期望 health 至少包含：

```json
{
  "status": "ok",
  "npu_enabled": true
}
```

## 搜索 smoke tests

Visual：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/search `
  -F "query_text=football player" `
  -F "modalities=visual" `
  -F "limit=1"
```

ASR：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/search `
  -F "query_text=How many times" `
  -F "modalities=asr" `
  -F "limit=1"
```

OCR：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/search `
  -F "query_text=FIFA" `
  -F "modalities=ocr" `
  -F "limit=1"
```

## 完成声明规则

说“完成 / 修复 / 通过”前必须：

1. 明确哪个命令能证明这个结论。
2. 重新运行命令。
3. 读取输出和退出码。
4. 汇报实际证据。
