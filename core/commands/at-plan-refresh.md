---
description: 优化并刷新下一阶段计划（评估gap并优化）
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

刷新当前任务的下一阶段计划：

执行流程：

1. 执行 /at-mem-read

2. 执行 /at-plan-read-next（读取下一阶段计划）

3. 如果没有下一阶段：
   - 提示"当前任务已到达最后阶段"
   - 退出

4. 执行 Gap 分析：
   - 调用 `/at-think-gap` 自动分析 gap 并保存中间件文件

5. 执行计划优化：
   - 调用 `/at-plan-opt auto` 自动应用变更到下一阶段计划文件

6. 如果计划有变化，更新 plan-phase-{next}.md 文件（已在 step 5 中完成）

7. 执行 /at-mem-fresh

关键原则：

- 无论任何情况（失败、用户指令、环境变化），都必须评估并优化计划
- 计划调整要保守，避免过度修改
- 记录每次刷新的原因和内容
