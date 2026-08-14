---
description: 创建新任务（委派给规划专家思考与拆解）
agent: build
subtask: false
---

模板文件位置：`.opencode/commands/auto-task/`

创建新任务，参数：$ARGUMENTS

## 执行流程

### 1. 参数提取（主代理执行）

分析 $ARGUMENTS：

- 提取任务描述（一句话）
- 提取任务模式（--auto / --half，默认 half）

### 2. 任务 ID 生成（主代理执行）

读取 `auto-task/task-index.md`，计算下一个任务 ID：

- 当前最大 ID：task-{max}
- 新任务 ID：task-{max+1}

### 3. 目录初始化（主代理执行）

创建任务目录结构：

- auto-task/tasks/task-{id}/
- auto-task/tasks/task-{id}/summary/

### 4. 委派规划专家（Task 工具）

调用 Task 工具，委派给 `at-planner` subagent：

```typescript
Task({
  subagent_type: 'at-planner',
  prompt: `
    任务：${任务描述}
    任务 ID：task-{id}
    输出目录：auto-task/tasks/task-{id}/

    上下文信息：
    - 项目技术栈：{读取项目配置获取}
    - 相关代码库：{可选，通过 at-explorer 探索}

    请生成：
    1. task-{id}-analysis-overall-plan.md（总体计划）
       - 任务概述（不超过 200 字，直至核心）
       - 总体目标
       - 任务阶段索引
       - 关键产出
       - 退出条件

    2. task-{id}-plan-phase-1.md（第一阶段计划）
       - 阶段目标
       - 具体步骤（详细、可执行，涉及代码变更必须给出完整代码）
       - 产出物
       - 验证步骤
       - 验证标准
       - 退出条件

    3. task-{id}-plan-phase-2.md（第二阶段计划，如有）
    ...
  `,
  description: '规划任务 ${任务描述}',
})
```

### 5. 任务索引更新（主代理执行）

更新 `auto-task/task-index.md`：

- 在"任务记录"部分新增一行：
  - 记录时间：执行 `python -c "import datetime;print(datetime.datetime.now())"`
  - 任务ID：task-{id}
  - 任务内容：{一句话描述}
  - 状态：未开始
  - 详情索引：tasks/task-{id}/task-{id}-analysis-overall-plan.md

### 6. 记忆索引更新（主代理执行）

更新 `auto-task/memory-index.md`：

- 当前任务：task-{id}
- 当前计划：phase-1
- 任务模式：{auto/half}
- 下一步：执行 phase-1 的第一个步骤
- 关键文件：添加任务相关文件
- 工作历史：新增创建任务记录

### 7. 验证与输出（主代理执行）

验证标准：

- [ ] task-index.md 已更新
- [ ] memory-index.md 已更新
- [ ] overall-plan.md 文件存在且非空
- [ ] 至少生成一个 phase-plan.md 文件

输出：

```
========================================
任务创建完成
========================================
任务 ID：task-{id}
任务描述：{一句话描述}
计划阶段：{N} 个阶段
任务模式：{半自动/全自动}

下一步：执行 /at-task-run
========================================
```

## 关键原则

- 主代理只做参数提取、索引更新等轻量操作
- 思考和规划委派给 `at-planner`
- 通过 Task 工具实现委派
- 保持主上下文清晰
- 任务概述不超过 200 字，直至核心，不过于发散
- 阶段拆解要充分，每个阶段可独立执行
- 验证标准要明确、可检查
