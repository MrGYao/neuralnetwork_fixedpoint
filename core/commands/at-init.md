---
description: 初始化任务系统（创建目录结构和模板文件）
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

初始化 auto-task 任务系统, 将模板文件复制到 项目根目录/auto-task/：

1. 检查并创建目录结构：
   - auto-task/
   - auto-task/tasks/
   - auto-task/templates/

2. 检查并创建初始文件：
   - auto-task/memory-index.md（记忆索引模板）
   - auto-task/task-index.md（任务索引模板）
   - auto-task/templates/task-analysis-overall-plan.md
   - auto-task/templates/task-plan-phase.md
   - auto-task/templates/task-plan-phase-summary.md

3. 如果文件已存在则跳过创建步骤，直接验证

4. 创建完成后，输出成功信息

验证标准：

- [ ] auto-task 目录存在
- [ ] auto-task/tasks 目录存在
- [ ] auto-task/templates 目录存在
- [ ] memory-index.md 存在
- [ ] task-index.md 存在
- [ ] 所有模板文件存在
