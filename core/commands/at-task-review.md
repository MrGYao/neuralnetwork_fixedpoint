---
description: 任务完成后全面检查（委派给评审专家）
agent: build
subtask: false
---

模板文件位置：`.opencode/commands/auto-task/`

## 目的

任务最后阶段完成时，全面检查：

1. 各阶段summary文件是否存在
2. 各阶段产出与相应plan是否一致
3. 任务整体效果与预期是否一致
4. overall-plan退出条件是否全部达成

不一致则改进完成或标记遗留问题。

## 执行流程

### 1. 读取任务信息（主代理执行）

a) 读取 memory-index.md，获取当前任务：

- 当前任务：task-{id}

b) 读取任务目录：

- auto-task/tasks/task-{id}/

c) 统计阶段信息：

- 阶段计划文件列表：task-{id}-plan-phase-*.md
- 阶段总结文件列表：summary/task-{id}-plan-phase-*-summary.md
- 阶段数量 = 计划文件数量

d) 读取 overall-plan 文件：

- task-{id}-analysis-overall-plan.md
- 提取退出条件列表

### 2. 委派评审专家（Task 工具）

调用 Task 工具，委派给 `at-reviewer` subagent：

```typescript
Task({
  subagent_type: 'at-reviewer',
  prompt: `
    评审任务：task-{id}
    评审类型：任务整体评审

    overall-plan：auto-task/tasks/task-{id}/task-{id}-analysis-overall-plan.md
    阶段数量：{N}

    请执行：
    1. 各阶段 summary 文件检查
       - 检查每个阶段的 summary 文件是否存在
       - 记录缺失的阶段

    2. 各阶段产出与 plan 一致性检查
       - 对比每个阶段的计划产出 vs 实际产出
       - 对比每个阶段的验证标准 vs 实际验证结果
       - 记录不一致项

    3. 任务整体效果评估
       - 读取 overall-plan 的预期目标
       - 对比实际产出效果
       - 评估目标达成度

    4. overall-plan 退出条件检查
       - 对比 overall-plan 的退出条件
       - 逐个检查是否达成

    如有不一致，列出改进建议。
  `,
  description: '任务完成整体评审',
})
```

### 3. 处理评审结果（主代理执行）

读取评审专家输出的评审报告：

a) 如果存在 ❌ 或 ⚠️：

- 输出不一致项列表
- 询问用户处理方式：
  - "改进" → 执行改进逻辑
  - "补充" → 执行补充逻辑
  - "跳过" → 记录遗留问题

b) 改进逻辑：

- summary缺失 → 调用 `/at-plan-run` 重新生成
- 产出不一致 → 对指定阶段重新执行或补充
- 效果不符预期 → 执行改进措施

c) 补充逻辑：

- 缺失产出 → 执行补充操作
- 未达成条件 → 完善实现或标记风险

### 4. 更新 overall-plan 和完成任务

a) 更新 overall-plan 文件：

- 更新退出条件状态
- 记录检查结果
- 标记遗留问题（如有）

b) 更新任务完成总结：

- 添加检查报告内容
- 记录改进/补充操作

c) 提交变更：

- git add auto-task/tasks/task-{id}/
- git commit -m "[task-{id}] 任务完成检查：改进/补充产出"

### 5. 输出最终结果

```
========================================
任务完成检查完成
========================================
任务：{task-id}
状态：{全部一致 / 已改进补充 / 存在遗留问题}

处理结果：
- 补充summary：{数量}
- 改进产出：{数量}
- 遗留问题：{数量}

下一步：执行 /at-task-finish 完成任务
========================================
```

## 关键原则

1. **评审委派给 `at-reviewer`**：质量评审和一致性检查
2. **主代理只处理评审结果和用户决策**：保持主上下文清晰
3. **改进操作可选**：记录遗留问题
4. **客观评审**：给出明确判断和建议
