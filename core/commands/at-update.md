---
description: 更新模板文件到项目根目录
agent: build
---

# /at-update - 更新模板文件

## 用途

将位于项目根目录下的 .opencode/commands/auto-task/ 最新版本模板更新到当前根目录下的 auto-task/。

## 参数

无参数

## 执行流程

1. 更新 memory-index.md 模板：
   - 源：.opencode/commands/auto-task/memory-index.md
   - 目标：auto-task/memory-index.md
   - 规则："记忆规则"必更新，其他内容直接填充新的模板，更新过程中先展示变更点，询问用户意见，用户同意后执行更新

2. 更新 templates/ 目录：
   - 源：.opencode/commands/auto-task/templates/
   - 目标：auto-task/templates/
   - 规则：全量更新（覆盖或新增）

3. 输出更新结果

## 具体更新内容

### memory-index.md 更新规则

- 保留：状态、下一步、关键文件、工作历史
- 更新：记忆规则部分（文件开头的规则说明）

### templates/ 全量更新

- task-analysis-overall-plan.md
- task-plan-phase.md
- 其他模板文件

## 示例

- /at-update → 更新模板文件

## 注意事项

- 需要先执行安装脚本，确保 .opencode/ 存在
- memory-index.md 的工作历史不会被覆盖
