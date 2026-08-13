# Auto-Task 任务自动化系统

**版本：0.0.1**

自动化执行开发任务的命令系统，减少或消除人工干预，保证任务高效可靠完成。

---

## 快速开始

```bash
# 初始化任务系统
/at-init

# 创建新任务
/at-task-new 实现用户登录功能

# 半自动执行（阶段间需确认）
/at-task-run-half

# 全自动执行（阶段间自动继续）
/at-task-run-all
```

---

## 核心设计

### 三模块分离

| 模块     | 目录                      | 职责               |
| -------- | ------------------------- | ------------------ |
| 任务模块 | auto-task/tasks/          | 任务记录、计划拆解 |
| 计划模块 | tasks/task-X/             | 阶段计划、执行验证 |
| 记忆模块 | auto-task/memory-index.md | 跨会话上下文保持   |

---

## 目录结构

```
auto-task/
├── memory-index.md                       # 记忆索引（当前状态+工作历史）
├── [YYYYMMDD_HHMMSS]memory-index-history.md  # 归档记忆（超过500条）
├── task-index.md                         # 任务索引（快速指引+任务记录）
├── templates/                            # 模板文件
│   ├── task-analysis-overall-plan.md     # 任务分析模板
│   ├── task-plan-phase.md                # 阶段计划模板
│   └── task-plan-phase-summary.md        # 阶段总结模板
└── tasks/                                # 任务文件夹
    └── task-0/
        ├── task-0-analysis-overall-plan.md  # 任务分析
        ├── task-0-plan-phase1.md            # 阶段1计划
        ├── task-0-plan-phase2.md            # 阶段2计划
        └── summary/                         # 阶段总结
            ├── task-0-plan-phase1-summary.md
            └── task-0-plan-phase2-summary.md
```

---

## 命令列表

| 命令               | 用途         | 说明                       |
| ------------------ | ------------ | -------------------------- |
| /at-init           | 初始化       | 创建目录结构和模板文件     |
| /at-task-new       | 创建任务     | 分析意图→记录任务→拆解计划 |
| /at-mem-read       | 读取记忆     | 加载当前状态和上下文       |
| /at-mem-fresh      | 刷新记忆     | 根据执行情况更新记忆索引   |
| /at-plan-refresh   | 刷新计划     | 评估gap并优化下一阶段计划  |
| /at-plan-read-next | 读取下一阶段 | 查看下一阶段计划内容       |
| /at-plan-run       | 执行阶段     | 严格按计划执行当前阶段     |
| /at-task-pop       | 弹出任务     | 取出下一个待执行任务       |
| /at-task-run-half  | 半自动执行   | 阶段间需用户确认           |
| /at-task-run-all   | 全自动执行   | 阶段间自动继续             |
| /at-task-finish    | 完成任务     | 标记任务完成并更新记录     |

---

## 执行模式

### 半自动模式（默认）

```
/at-task-run-half
→ 执行 Phase 1
→ 暂停，等待用户确认
→ 用户输入"继续"
→ 执行 Phase 2
→ 暂停，等待用户确认
→ ...
→ 任务完成
```

### 全自动模式

```
/at-task-run-all
→ 执行 Phase 1
→ 自动进入 Phase 2
→ 自动进入 Phase 3
→ ...
→ 任务完成，等待用户验证
```

---

## 核心原则

### 1. 计划谨慎性

- 执行前必须刷新计划
- 评估当前成果与下一阶段的 gap
- 根据实际情况优化计划步骤

### 2. 执行严格性

- 严格按计划执行，禁止跳过步骤
- 失败最多尝试 3 次
- 3 次失败后等待用户决策

### 3. 自动提交

- 每个阶段完成自动提交代码
- Commit 格式：`[task-X/phase-Y] {阶段目标}`
- 失败不提交

### 4. 记忆持续性

- 每次执行前加载 memory-index
- 完成后生成工作历史
- 工作历史超过 500 条自动归档

---

## 任务索引结构

```markdown
# 任务索引

## 快速指引

1. 执行 /at-task-new 创建新任务
2. 执行 /at-task-run-half 半自动执行
3. 执行 /at-task-run-all 全自动执行

## 任务记录

| 时间             | 任务ID | 内容         | 状态   | 详情索引                        |
| ---------------- | ------ | ------------ | ------ | ------------------------------- |
| 2026-08-11 20:00 | task-0 | 实现用户登录 | 已完成 | tasks/task-0/task-0-analysis... |

## 历史任务

（超过500条后归档）
```

---

## 记忆索引结构

```markdown
# 记忆索引

## 引导

### 状态

- 当前任务：task-0
- 当前计划：phase-1
- 任务模式：半自动

### 下一步

执行 phase-1 的第一个步骤

### 关键文件

- auto-task/tasks/task-0/task-0-analysis-overall-plan.md
- auto-task/tasks/task-0/task-0-plan-phase1.md

### 关键命令

- /at-plan-run: 执行当前阶段
- /at-mem-read: 查看当前状态

## 工作历史

- 2026-08-11 20:00: 创建任务 task-0
- 2026-08-11 20:30: 完成 phase-1，产出 README.md
```

---

## 迁移到其他项目

本系统完全独立，可迁移到任何 opencode 项目。

### 迁移步骤

**方法1：整体复制**（推荐）

```bash
# 复制整个 .opencode/commands/ 目录
cp -r .opencode/commands/ /path/to/other-project/.opencode/

# 在目标项目执行初始化
cd /path/to/other-project
/at-init
```

**方法2：选择性复制**

```bash
# 只复制命令文件
cp .opencode/commands/at-*.md /path/to/other-project/.opencode/commands/

# 复制模板文件（供参考）
cp -r .opencode/commands/templates/ /path/to/other-project/.opencode/commands/

# 在目标项目执行初始化
/at-init
```

### 文件清单

迁移需要复制：

```
.opencode/commands/
├── at-init.md                   # 初始化命令
├── at-task-new.md               # 创建任务命令
├── at-mem-read.md               # 读取记忆命令
├── at-mem-fresh.md              # 刷新记忆命令
├── at-plan-refresh.md           # 刷新计划命令
├── at-plan-read-next.md         # 读取下一阶段命令
├── at-plan-run.md               # 执行阶段命令
├── at-task-pop.md               # 弹出任务命令
├── at-task-run-half.md          # 半自动执行命令
├── at-task-run-all.md           # 全自动执行命令
├── at-task-finish.md            # 完成任务命令
└── templates/auto-task/         # 初始模板（供参考）
    ├── memory-index.md          # 初始记忆索引
    ├── task-index.md            # 初始任务索引
    ├── README.md                # 使用说明
    └── templates/               # 文档模板
        ├── task-analysis-overall-plan.md
        ├── task-plan-phase.md
        └── task-plan-phase-summary.md
```

/at-init 会自动读取模板文件并创建 auto-task/ 目录结构。

---

## 注意事项

1. **任务概述不超过 200 字**：直至核心，不过于发散
2. **阶段拆解充分**：每个阶段可独立执行验证
3. **验证标准明确**：可检查、可量化
4. **失败不跳过**：必须等待用户决策
