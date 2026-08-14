---
description: Gap 分析与优化建议（委派给规划专家）
agent: build
subtask: true
---

模板文件位置：`.opencode/commands/auto-task/`

## 用途

分析当前阶段与下一阶段的 gap，输出优化建议，自动保存

## 参数

- $1: task-id（可选，默认从 memory-index 读取）
- $2: phase-number（可选，默认当前阶段 +1）

示例：
/at-think-gap
/at-think-gap task-10
/at-think-gap task-10 2

## 流程

### 1. 前置读取（主代理执行）

a) 读取当前状态：

- 执行：读取 auto-task/memory-index.md
- 提取：task_id、current_phase、target_phase

b) 读取计划文件：

- 下一阶段计划：auto-task/tasks/task-{id}/task-{id}-plan-phase-{target}.md
- 如不存在，读取 overall-plan 提取对应阶段

c) 读取上一阶段总结（如有）：

- 如果 current_phase > 0：
  - 读取：summary/task-{id}-plan-phase-{current}-summary.md
  - 提取：产出物、验证结果、执行摘要

### 2. 委派 Gap 分析（Task 工具）

调用 Task 工具，委派给 `at-planner` subagent：

```typescript
Task({
  subagent_type: 'at-planner',
  prompt: `
    任务：Gap 分析
    任务ID：task-{id}
    当前阶段：phase-{current}
    目标阶段：phase-{target}

    输入数据：
    - 下一阶段计划：{plan content}
    - 上一阶段总结：{summary content，如有}
    - 当前工作状态：{从 memory-index 提取}

    请执行完整的多维度 Gap 分析：

    ## 分析维度（5个维度）

    1. **范围维度**
       - 对比：下一阶段范围 vs 当前工作状态
       - 识别：范围蔓延、范围缺失、范围重叠

    2. **依赖维度**
       - 前置依赖识别：读取下一阶段计划的"具体步骤"
       - 检查依赖是否满足：
         * 检查上一阶段 summary 的产出物
         * 检查 memory-index 的工作历史
         * 检查代码仓库中的实际文件
       - 识别：依赖缺失、依赖冲突、依赖冗余

    3. **质量维度**
       - 对比：下一阶段验收标准 vs 当前产出质量
       - 识别：质量不足、标准不明确、过度质量

    4. **风险维度**
       - 识别：潜在风险点、遗漏点、冲突点、阻塞点

    5. **技术维度**（8个子维度）
       - 工具维度：检查工具是否具备
       - 环境维度：检查开发环境是否满足
       - 技术栈维度：检查技术栈是否匹配
       - 架构维度：检查架构兼容性
       - 规范维度：检查是否符合编码规范
       - 风格维度：检查代码风格是否一致
       - 功能性维度：检查功能实现的正确性
       - 实现维度：检查实现方案的可行性（代码方案合适性检查）

    ## Gap 分类

    将识别出的 gap 分为五类：
    1. 未满足的依赖：下一阶段需要但当前未满足的依赖项
    2. 已达成的部分：下一阶段的部分目标已经完成
    3. 方向偏差的部分：当前执行结果与下一阶段计划有偏差
    4. 代码实现偏差：代码实现与计划中的代码方案不一致
    5. 环境变化：开发环境、依赖版本等发生变化

    ## 输出要求

    生成标准化输出（YAML frontmatter + Markdown body）：

    ---
    task_id: task-{id}
    current_phase: {n}
    target_phase: {n+1}
    analysis_time: {时间}
    gap_count: {count}
    ---

    包含：
    1. 基本信息
    2. 维度分析概览（表格形式）
    3. 变更点列表（标准化格式，每个变更点包含：类型、目标、原因、详细检查清单、变更内容diff、风险提示、影响范围）
    4. 变更统计
    5. 建议

    并生成临时文件数据（JSON格式）。
  `,
  description: 'Gap 分析 phase-{target}',
})
```

### 3. 输出结果（主代理执行）

a) 输出完整的 Gap 分析报告到控制台

b) 输出临时文件路径：

```
临时文件：.temp/at-think-gap-{task-id}-{phase}.json
```

### 4. 自动调用保存命令（主代理执行）

执行：`/at-think-gap-save $1 $2`

## 注意事项

- **完整输出**: 不截断，完整展示所有变更点
- **自动保存**: 执行完成后自动调用 at-think-gap-save
- **无副作用**: 不修改任务文件，只生成临时文件和 gap 中间件
- **多 session 安全**: 临时文件名包含 task-id 和 phase，避免冲突
- **委派清晰**: Gap 分析委派给 `at-planner`（思考职责）
- **上下文清晰**: 主代理只做前置读取和结果输出
