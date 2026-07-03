# 经验与踩坑记录

本文档记录未来新会话必须继承的操作经验、工具坑和事故教训。

## 记录格式

```text
日期：
类别：
现象：
根因：
影响：
经验：
以后规则：
相关文档：
```

## 2026-07-03 PowerShell UTF-8 输出

```text
类别：PowerShell
现象：
  中文 Markdown 用默认 PowerShell 输出时出现乱码。
根因：
  控制台输出编码没有设为 UTF-8。
影响：
  如果直接相信乱码输出，可能误读交接文档。
经验：
  读取中文文档时必须显式使用 UTF-8。
以后规则：
  使用：
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Get-Content -Raw -Encoding UTF8 <path>
相关文档：
  docs/README.md
```

## 2026-07-03 PowerShell 通配符和 `rg`

```text
类别：PowerShell
现象：
  类似 `rg ... docs/*.md` 的命令在 Windows 下报路径语法错误。
根因：
  PowerShell/Windows 路径和 Unix 风格 glob 行为不同。
影响：
  搜索可能失败，或者没有覆盖目标文件。
经验：
  优先使用 `rg -g "*.md" <root>` 或 `rg --files`。
以后规则：
  使用：
    rg -n "pattern" docs -g "*.md"
    rg --files docs
相关文档：
  docs/VALIDATION.md
```

## 2026-07-03 Git diff 看不到未跟踪文件

```text
类别：Git
现象：
  编辑 handoff 文档后，`git diff -- docs/handoff/...` 没有输出。
根因：
  handoff 目录当时是未跟踪目录，普通 `git diff` 不展示未跟踪文件内容。
影响：
  容易误以为没有改动，或者漏看未跟踪文档。
经验：
  `git diff` 要配合 `git status --short` 和直接内容搜索。
以后规则：
  使用：
    git status --short
    rg -n "expected text" <untracked-file-or-directory>
相关文档：
  docs/VALIDATION.md
```

## 共享服务器误杀进程事故

```text
日期：2026-07-02
类别：Server
现象：
  一次服务器操作误杀了其他人的进程。
根因：
  终止进程前没有把归属缩小到足够明确；在共享服务器上用宽泛进程匹配非常危险。
影响：
  可能中断其他人的 ComfyUI、VLLM 或其他 python/NPU 任务。
经验：
  服务器是共享环境。NPU 占用或进程名叫 python，都不能证明它属于 MomentSeek。
以后规则：
  禁止 broad kill。任何 kill/restart 前：
    1. 精确确认 MomentSeek 容器/进程。
    2. 检查 active jobs。
    3. 检查 docker ps 和 npu-smi。
    4. 确认目标归属 MomentSeek。
    5. 优先只在明确批准后操作 `momentseek-current-app`。
相关文档：
  docs/OPERATIONS.md
```

## Windows 文件名大小写

```text
日期：2026-07-03
类别：Git / Windows
现象：
  想创建 `ARCHITECTURE.md` 时，仓库里已有 `architecture.md`，Windows 下两者不能安全共存。
根因：
  当前文件系统对路径大小写不敏感。
影响：
  如果直接创建大写新文件，Git 状态可能混乱。
经验：
  先归档或改名旧文件，再创建大小写不同的新文件。
以后规则：
  仅改变文件名大小写时，使用两步 rename：
    git mv old.md temp.md
    git mv temp.md NEW.md
相关文档：
  docs/README.md
```
