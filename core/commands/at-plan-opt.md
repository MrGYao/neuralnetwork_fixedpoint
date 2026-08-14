---
description: 执行计划优化（委派给执行专家）
agent: build
subtask: false
---

模板文件位置：`.opencode/commands/auto-task/`

## 用途

根据 gap 分析结果，执行计划文件变更

## 参数

- $1: mode（可选，auto | ask | dry-run，默认 ask）
- $2: task-id（可选，默认从 memory-index 读取）
- $3: phase-number（可选，默认当前阶段 +1）

示例：
/at-plan-opt
/at-plan-opt ask
/at-plan-opt auto task-10 2
/at-plan-opt dry-run

## 流程

### 1. 参数解析（主代理执行）

解析：

- mode: $1 或默认 'ask'
- task-id: $2 或从 memory-index 读取
- phase-number: $3 或当前阶段 +1

### 2. 查找 gap 文件（主代理执行）

查找顺序：

```
1. auto-task/tasks/task-{id}/gap-task-{id}-plan-phase-{n}.md
2. auto-task/tasks/task-{id}/gap-task-{id}-analysis-overall-plan.md
```

如果未找到：

- 提示：未找到 Gap 分析文件，请先执行：/at-think-gap {task-id} {phase-number}
- 退出

### 3. 读取 gap 文件（主代理执行）

读取 gap 文件，解析：

- gap_count
- 变更点列表
- 维度分析结果

### 4. 根据模式执行（主代理决策）

#### mode = dry-run

输出变更预览，**不实际写入**：

```
========================================
变更预览（dry-run 模式）
========================================

变更点 1: {标题}
目标: {文件}
类型: add
变更内容:
{展示 diff}

---

变更点 2: {标题}
目标: {文件}
类型: modify
变更内容:
{展示 diff}

---

总计 {count} 个变更，未实际写入

提示：使用 /at-plan-opt ask 或 /at-plan-opt auto 执行变更
========================================
```

退出，不执行后续步骤。

#### mode = auto

直接委派 executor 执行所有变更：

跳转到步骤 5.

#### mode = ask

逐个确认每个变更点：

```
┌─────────────────────────────────────────────┐
│ 变更点 1: {标题}                            │
│                                              │
│ 类型: {type}                                 │
│ 目标: {file}                                 │
│                                              │
│ 变更内容:                                    │
│ {展示 diff 摘要}                             │
│                                              │
│ 风险提示: {risk}                             │
│                                              │
│ 是否执行？                                   │
└─────────────────────────────────────────────┘

选择: [批准] [跳过] [查看详情] [取消全部]
```

用户操作：

- 批准 → 标记为待执行
- 跳过 → 标记为跳过
- 查看详情 → 展示完整变更，重新询问
- 取消全部 → 退出

收集所有批准的变更点，跳转到步骤 5.

### 5. 委派执行变更（Task 工具）

调用 Task 工具，委派给 `at-executor` subagent：

```typescript
Task({
  subagent_type: 'at-executor',
  prompt: `
    执行计划优化变更：task-{id} / phase-{n}

    模式：{mode}

    待执行变更点：
    {变更点列表（JSON 或标准格式）}

    执行要求：
    - 对每个变更点：
      * 读取目标文件
      * 应用变更（add/modify/delete/reorder）
      * 写回文件
      * 记录执行结果
    - 在目标文件末尾添加"计划刷新记录"章节：
      * 时间：{当前时间}
      * 命令：/at-plan-opt {mode}
      * 变更数：{count}
      * 变更点：{标题列表}
    - 更新 memory-index：
      * 追加操作记录

    自动提交：
    - git add auto-task/tasks/task-{id}/
    - git commit -m "[task-{id}] 优化 phase-{n} 计划"

    输出：执行结果报告
  `,
  description: '执行计划优化变更',
})
```

### 6. 输出执行摘要（主代理执行）

读取 executor 输出，展示执行摘要：

```markdown
========================================
计划优化完成
========================================

执行信息：

- 模式：{mode}
- 任务：task-{id}
- 阶段：phase-{n}
- 执行时间：{时间}

变更统计：

- 总计：{total} 个
- 批准：{approved} 个
- 跳过：{skipped} 个
- 失败：{failed} 个

执行详情：

已执行 ({approved})：

- ✅ 变更点 1: {标题}
  - 文件：{file}
  - 类型：{type}

已跳过 ({skipped})：

- ⏭ 变更点 3: {标题}
  - 原因：用户跳过

影响文件：

- {file1}
- {file2}

下一步：
继续执行当前任务：/at-task-run
========================================
```

## 注意事项

- **先执行 at-think-gap**: 必须存在 gap 文件才能执行
- **幂等性**: 同一变更多次执行不会重复
- **建议 git commit**: 执行前建议提交当前状态
- **无自动回滚**: 不提供回滚功能，需手动恢复
- **委派清晰**: dry-run 由主代理预览，auto/ask 委派 executor 执行
- **权限分离**: 主代理决策，executor 执行
