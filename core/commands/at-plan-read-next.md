---
description: 读取下一阶段计划
agent: plan
---

模板文件位置：`.opencode/commands/auto-task/`

读取当前任务的下一阶段计划：

执行流程：

1. 读取 auto-task/memory-index.md，获取：
   - 当前任务：task-{id}
   - 当前计划：phase-{current}

2. 如果当前任务为"无"：
   - 提示"当前没有执行中的任务"
   - 退出

3. 确定下一阶段：
   - 读取 tasks/task-{id}/ 目录
   - 列出所有 task-{id}-plan-phase-*.md 文件
   - 提取阶段编号，排序
   - 下一阶段 = phase-{current+1}

4. 检查下一阶段是否存在：
   - 如果存在 plan-phase-{next}.md：
     - 继续读取
   - 如果不存在：
     - 提示"当前任务已到达最后阶段，执行 /at-task-finish 完成任务"
     - 退出

5. 读取 auto-task/tasks/task-{id}/task-{id}-plan-phase-{next}.md

6. 输出阶段计划摘要：

   ```markdown
   # 下一阶段计划 - Phase {next}

   ## 阶段目标

   {本阶段的核心目标}

   ## 具体步骤

   1. {步骤1}
   2. {步骤2}
      ...

   ## 产出物

   - {产出文件路径}
     ...

   ## 验证标准

   - [ ] {标准1}
         ...

   ## 退出条件

   - [ ] {条件1}
         ...
   ```

验证标准：

- [ ] 正确识别下一阶段
- [ ] 输出完整的阶段计划内容
