---
description: 保存 Gap 分析结果
agent: build
---

模板文件位置：`.opencode/commands/auto-task/`

## 用途

将 at-think-gap 的分析结果保存为中间件文件

## 参数

- $1: task-id（必需）
- $2: phase-number（必需）

示例：
/at-think-gap-save task-10 2

## 触发方式

由 /at-think-gap 自动调用，用户通常不直接执行

## 流程

### Step 1: 读取临时文件

执行：读取 `.temp/at-think-gap-$1-$2.json`

如果文件不存在：

- 提示：未找到 Gap 分析数据，请先执行 /at-think-gap $1 $2
- 退出

### Step 2: 确定目标文件名

生成中间件文件名：

规则：

```
原文件: task-{id}-plan-phase-{n}.md
gap文件: gap-task-{id}-plan-phase-{n}.md

原文件: task-{id}-analysis-overall-plan.md
gap文件: gap-task-{id}-analysis-overall-plan.md
```

路径：`auto-task/tasks/task-{id}/gap-{original-name}.md`

### Step 3: 生成中间件文件

执行：写入中间件文件

内容：从临时文件的 JSON 数据转换为 YAML + Markdown 格式

示例：

```yaml
---
task_id: task-10
current_phase: 1
target_phase: 2
analysis_time: 2026-08-13T14:30:00
gap_count: 3
target_file: task-10-plan-phase2.md
generated_by: at-think-gap
---

# Gap 分析报告

## 基本信息
{基本信息}

## 维度分析概览
{维度分析}

## 变更点列表

### 变更点 1: {标题}
{完整变更点}

### 变更点 2: {标题}
{完整变更点}

...

---

## 变更统计
{统计信息}

## 建议
{建议列表}
```

### Step 4: 清理临时文件

执行：删除 `.temp/at-think-gap-$1-$2.json`

### Step 5: 输出确认

输出：

```markdown
已保存 Gap 分析结果：

文件: auto-task/tasks/task-10/gap-task-10-plan-phase2.md
变更点: 3 个
生成时间: 2026-08-13 14:30:00

下一步：

- 执行 `/at-plan-opt dry-run` 预览变更
- 执行 `/at-plan-opt ask` 交互确认
- 执行 `/at-plan-opt auto` 自动执行
```

## 注意事项

- **幂等性**: 同一 gap 文件多次保存会覆盖
- **与原文件对应**: gap 文件名与原计划文件对应
- **完整内容**: 必须包含所有变更点的完整展示
- **清理临时文件**: 保存后立即删除临时文件
