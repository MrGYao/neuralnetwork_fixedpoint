---
description: 根据当前任务执行情况刷新 memory-index
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

刷新 auto-task/memory-index.md：

执行流程：

1. 获取当前时间：
   执行 python -c "import datetime;print(datetime.datetime.now())"

2. 读取 task-index.md，查找状态为"执行中"的任务：
   - 如果存在（记录为 task-{id}）：
     - 当前任务 = task-{id}
     - 读取 tasks/task-{id}/ 目录，确定当前阶段
   - 如果不存在：
     - 当前任务 = 无
     - 当前计划 = 无

3. 确定下一步行动：
   - 如果当前任务存在：
     - 读取当前阶段计划文件
     - 检查是否存在对应 summary 文件
     - 如果存在 → 当前阶段已完成，下一步 = 下一阶段或任务完成
     - 如果不存在 → 当前阶段进行中，下一步 = 继续当前阶段
   - 如果当前任务不存在：
     - 下一步 = 执行 /at-task-pop 开始新任务

4. 更新关键文件列表：
   - 读取当前任务目录
   - 列出任务分析文档、阶段计划文档、已完成 summary 文档
   - 限制在10行以内

5. 更新关键命令列表：
   - 根据当前状态推荐命令
   - 例如：执行中 → /at-plan-run /at-mem-read
   - 例如：无任务 → /at-task-pop /at-task-new

6. 生成工作历史记录，禁止删除任何之前的工作历史：
   - 时间：{当前时间}
   - 工作内容：根据上下文生成摘要（不超过10行）
   - 追加到 memory-index.md 的"工作历史"部分

7. 检查工作历史条数：
   - 读取"工作历史"部分
   - 如果超过500条：
     - 提取所有历史记录
     - 创建归档文件：auto-task/[YYYYMMDD_HHMMSS]memory-index-history.md
     - 在 memory-index.md 中引用归档文件
     - 清空"工作历史"部分，仅保留"已归档到 {文件名}"

8. 更新 memory-index.md 文件

验证标准：

- [ ] 当前任务状态正确
- [ ] 下一步行动明确
- [ ] 关键文件列表更新
- [ ] 工作历史新增记录
