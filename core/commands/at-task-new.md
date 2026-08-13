---
description: 创建新任务（分析意图、记录任务、拆解计划）
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

创建新任务，参数：$ARGUMENTS

执行流程：

1. 分析用户意图，提取任务核心内容（一句话描述）

2. 充分思考任务：
   - 调用 `/at-think {一句话描述}`
   - 获取思考结果
   - 将思考结果作为后续步骤的输入

3. 生成任务ID：
   - 读取 task-index.md，计算下一个任务ID（task-0, task-1...）

4. 更新 task-index.md：
   - 在"任务记录"部分新增一行：
     - 记录时间：执行 python -c "import datetime;print(datetime.datetime.now())"
     - 任务ID：task-{id}
     - 任务内容：{一句话描述}
     - 状态：未开始
     - 详情索引：tasks/task-{id}/task-{id}-analysis-overall-plan.md

5. 创建任务目录：
   - auto-task/tasks/task-{id}/
   - auto-task/tasks/task-{id}/summary/

6. 创建任务分析文档 task-{id}-analysis-overall-plan.md：
   基于 /at-think 的思考结果和模板，填充：
   - 任务概述（不超过500字，直至核心）
   - 总体目标
   - 任务阶段索引
   - 关键产出
   - 退出条件

7. 拆解计划阶段：生成 task-{id}-plan-phase-X.md 文件
   基于 /at-think 的思考结果和模板，为每个阶段填充：
   - 阶段目标
   - 具体步骤（详细、可执行、涉及代码变更必须给出变更标题、目的、完整的代码）
   - 产出物
   - 验证步骤
   - 验证标准
   - 退出条件

8. 更新 memory-index.md：
   - 当前任务：task-{id}
   - 当前计划：phase-1
   - 下一步：执行 phase-1 的第一个步骤
   - 关键文件：添加任务相关文件
   - 工作历史：新增创建任务的记录

任务模式处理：

- 检查 $ARGUMENTS 中是否包含 --auto 或 --half
- --auto：设置任务模式为"全自动"
- --half 或默认：设置任务模式为"半自动"
- 更新 memory-index.md 中的"任务模式"

关键原则：

- 任务概述不超过500字，直至核心，不过于发散
- 阶段拆解要充分，每个阶段可独立执行
- 验证标准要明确、可检查
