---
description: 执行当前阶段计划（委派给执行专家和评审专家）
agent: build
subtask: false
---

模板文件位置：`.opencode/commands/auto-task/`

执行当前阶段计划：

## 执行流程

### 1. 前置准备（主代理执行）

a) 提交当前暂存代码：

```bash
git add -A
git commit -m "WIP: before phase execution" || true
```

b) 读取记忆索引：

- 执行 `/at-mem-read`
- 获取当前任务和当前阶段

c) 定位阶段计划文件：

- auto-task/tasks/task-{id}/task-{id}-plan-phase-{current}.md

### 2. 委派执行专家（Task 工具）

```typescript
Task({
  subagent_type: 'at-executor',
  prompt: `
    执行阶段：task-{id} / phase-{current}
    计划文件：auto-task/tasks/task-{id}/task-{id}-plan-phase-{current}.md

    执行要求：
    - 严格按计划步骤执行
    - 失败最多重试 3 次
    - 所有产出物必须生成
    - 自动提交代码

    执行完成后生成总结文件：
    - auto-task/tasks/task-{id}/summary/task-{id}-plan-phase-{current}-summary.md
  `,
  description: '执行 Phase {current}',
})
```

### 3. 后置验证（主代理执行）

检查执行结果：

- 读取阶段总结文件
- 检查产出物是否生成
- 检查验证标准是否满足

### 4. 委派评审专家（Task 工具）

```typescript
Task({
  subagent_type: 'at-reviewer',
  prompt: `
    评审阶段：task-{id} / phase-{current}
    评审类型：阶段评审
    计划文件：auto-task/tasks/task-{id}/task-{id}-plan-phase-{current}.md
    总结文件：auto-task/tasks/task-{id}/summary/task-{id}-plan-phase-{current}-summary.md
  `,
  description: '评审 Phase {current} 执行结果',
})
```

### 5. 更新记忆索引（主代理执行）

执行 `/at-mem-fresh`：

- 更新当前状态
- 更新工作历史
- 确定下一步行动

### 6. 提交记忆索引

```bash
git add auto-task/memory-index.md
git commit -m "[task-{id}/phase-{current}] 更新记忆索引"
```

### 7. 输出执行结果

```
========================================
Phase {current} 执行完成
========================================
任务：task-{id}
执行状态：{成功/部分成功/失败}

产出物：
- {文件路径} ✅
- {文件路径} ✅

验证标准：
- [✅] 标准 1
- [✅] 标准 2
- [❌] 标准 3（需改进）

下一步：
- 如果还有下一阶段：准备执行 phase-{next}
- 如果已是最后阶段：准备任务整体评审
========================================
```

## 失败处理

如果执行专家报告失败：

1. 读取失败详情
2. 输出失败报告给用户
3. 等待用户决策：
   - "重试" → 重新委派执行专家
   - "跳过" → 标记步骤跳过，继续
   - "终止" → 退出命令

## 关键原则

- **主代理作为编排者**：不执行具体步骤
- **执行委派给 `at-executor`**：严格按计划执行
- **评审委派给 `at-reviewer`**：验证产出一致性
- **保持主上下文清晰**：避免膨胀
- 严格按计划执行，禁止跳过步骤
- 最多尝试3次解决失败，3次后等待用户决策
- 阶段完成才提交代码
- 生成完整的阶段总结
