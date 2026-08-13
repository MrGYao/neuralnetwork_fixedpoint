---
description: 查看当前任务状态或取出下一个待执行任务
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

检查当前任务状态：

执行流程：

1. 读取 auto-task/task-index.md

2. 在"任务记录"部分查找状态为"执行中"的任务：

   如果存在：
   - 输出提示：
     ```
     当前任务仍在执行中：
     - 任务ID：{task-id}
     - 任务内容：{内容}
     - 状态：执行中
     - 详情：tasks/task-{id}/task-{id}-analysis-overall-plan.md

     请先完成当前任务（执行 /at-plan-run），再开始新任务。
     ```
   - 退出

3. 查找状态为"未开始"的任务：

   a) 提取所有状态为"未开始"的任务记录
   b) 按创建时间排序（最早的优先）
   c) 取第一个任务（task-{id}）

4. 如果不存在未开始任务：
   - 提示"当前没有待执行的任务"
   - 提示"请执行 /at-task-new 创建新任务"
   - 退出

5. 更新 task-index.md：

   找到 task-{id} 的记录行，修改状态：
   - 状态：未开始 → 执行中

6. 执行 /at-mem-fresh

7. 输出当前任务信息：
   ```
   取出待执行任务：
   - 任务ID：{task-id}
   - 任务内容：{内容}
   - 详情索引：tasks/task-{id}/task-{id}-analysis-overall-plan.md

   下一步：
   - 执行 /at-plan-refresh 刷新计划
   - 执行 /at-plan-run 开始执行
   ```

验证标准：

- [ ] 正确识别执行中任务
- [ ] 正确取出下一个待执行任务
- [ ] 更新任务状态
