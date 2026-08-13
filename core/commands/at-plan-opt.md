---
description: 执行计划优化
agent: build
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

### Step 1: 参数解析

解析：

- mode: $1 或默认 'ask'
- task-id: $2 或从 memory-index 读取
- phase-number: $3 或当前阶段 +1

### Step 2: 查找 gap 文件

查找顺序：

```
1. auto-task/tasks/task-{id}/gap-task-{id}-plan-phase-{n}.md
2. auto-task/tasks/task-{id}/gap-task-{id}-analysis-overall-plan.md
```

如果未找到：

- 提示：未找到 Gap 分析文件，请先执行：/at-think-gap {task-id} {phase-number}
- 退出

### Step 3: 读取 gap 文件

执行：读取 gap 文件
解析：YAML frontmatter + Markdown 内容
提取：

- gap_count
- 变更点列表
- 维度分析结果

### Step 4: 根据模式执行

#### mode = dry-run

输出变更预览，**不实际写入**：

```
=== 变更预览 ===

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
```

#### mode = auto

自动执行所有变更：

```
执行变更：

1. 应用变更点 1 -> ✅ 成功
2. 应用变更点 2 -> ✅ 成功
...

执行完成，{count} 个变更已应用
```

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

用户操作 -> 执行对应处理

重复直到所有变更点处理完毕
```

### Step 5: 执行变更

对每个批准的变更点：

```typescript
function applyChange(point): Result {
  // 读取目标文件
  const targetPath = 'auto-task/tasks/' + point.task_id + '/' + point.目标.文件
  let content = readFileSync(targetPath)

  // 应用变更
  switch (point.类型) {
    case 'add':
      content = insertAtPosition(content, point.目标.位置, point.变更内容)
      break
    case 'modify':
      content = applyDiff(content, point.变更内容)
      break
    case 'delete':
      content = removeContent(content, point.变更内容)
      break
    case 'reorder':
      content = reorderSections(content, point.变更内容)
      break
  }

  // 写回文件
  writeFileSync(targetPath, content)

  return { success: true, file: targetPath }
}
```

### Step 5.5: 添加刷新记录

在目标计划文件末尾追加刷新记录章节：

如果文件末尾已有 `## 计划刷新记录` 章节：

- 在该章节下追加新记录：

```
- **时间**: {当前时间}
  - 命令: /at-plan-opt {mode}
  - 变更数: {count}
  - 变更点: {变更点标题列表}
```

如果文件末尾没有 `## 计划刷新记录` 章节：

- 在文件末尾新增章节：

```markdown
---

## 计划刷新记录

- **时间**: {当前时间}
  - 命令: /at-plan-opt {mode}
  - 变更数: {count}
  - 变更点: {变更点标题列表}
```

### Step 6: 更新 memory-index

追加操作记录：

```markdown
### {时间}

- 执行: /at-plan-opt {mode}
- 变更数: {count}
- 影响文件: {files}
```

### Step 7: 输出执行摘要

```markdown
# 计划优化执行摘要

## 执行信息

- 模式: {mode}
- 任务: task-{id}
- 阶段: phase-{n}
- 执行时间: {时间}

## 变更统计

- 总计: {total} 个
- 批准: {approved} 个
- 跳过: {skipped} 个
- 失败: {failed} 个

## 执行详情

### 已执行 ({approved})

- ✅ 变更点 1: {标题}
  - 文件: {file}
  - 类型: {type}

### 已跳过 ({skipped})

- ⏭ 变更点 3: {标题}
  - 原因: 用户跳过

## 影响文件

- {file1}
- {file2}

## 下一步

继续执行当前任务：/at-task-run-single
```

## 注意事项

- **先执行 at-think-gap**: 必须存在 gap 文件才能执行
- **幂等性**: 同一变更多次执行不会重复
- **建议 git commit**: 执行前建议提交当前状态
- **无自动回滚**: 不提供回滚功能，需手动恢复
