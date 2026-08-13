# 记忆索引

## 引导

### 记忆规则 (禁止变更)

- 每次执行任务前加载本文档，完成任务后生成相应的工作历史(时间使用 python -c "import datetime;print(datetime.datetime.now())" 获取, 内容描述不超过10行(git提交附上commit id))，同时更新当前阶段、下一步、关键文件与关键命令的内容。
- 执行任务中，严格按计划执行，禁止跳过任何步骤；解决问题最多尝试3次，3次后由用户决策。
- 必须回忆流程：根据任务模式读取相应的命令文件 /at-task-run。

### 状态

- 任务队列: 无
- 当前任务：无
- 当前计划：无
- 任务模式：半自动

### 下一步

执行 /at-init 初始化任务系统

### 关键文件

- auto-task/memory-index.md
- auto-task/task-index.md

### 关键命令

- /at-init: 初始化任务系统
- /at-task-new: 创建新任务

## 工作历史

（暂无）
