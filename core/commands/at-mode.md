---
description: 切换任务模式（半自动/全自动）
agent: build
---

# /at-mode - 切换任务模式

## 用途

快速切换任务模式（半自动/全自动），修改根目录下 auto-task/memory-index.md 的任务模式字段。

## 参数

- half：半自动模式（每个阶段执行后需要用户批准）
- all：全自动模式（连续执行所有阶段，无需用户确认）

## 执行流程

1. 读取参数：确认参数为 half 或 all
2. 读取 auto-task/memory-index.md
3. 修改"任务模式"字段：
   - half → 任务模式：半自动
   - all → 任务模式：全自动
4. 写回 auto-task/memory-index.md
5. 输出确认信息

## 示例

- /at-mode half → 设置为半自动模式
- /at-mode all → 设置为全自动模式

## 注意事项

- 参数必须为 half 或 all
- 不提供参数时，输出当前模式
