# Core - 核心能力包

## 概述

core/ 目录存放项目的核心能力定义文件，是版本控制的源头。通过安装脚本分发到其他项目或全局路径。

## 目录结构

```
core/
├── agent/                   # Agent 定义（18 个）
│   ├── analyst.md
│   ├── architect.md
│   └── ...
├── agents/                  # Auto-Task Subagent 定义（4 个）
│   ├── at-planner.md        # 任务规划专家
│   ├── at-executor.md       # 执行专家
│   ├── at-reviewer.md       # 质量评审专家
│   ├── at-explorer.md       # 探索专家
│   └── README.md            # 使用说明
├── commands/                # 命令定义（18 个）
│   ├── at-init.md
│   ├── at-mode.md
│   ├── at-update.md
│   └── auto-task/           # auto-task 系统
├── skills/                  # Skills 技能（11 个）
│   ├── competitor-analysis/
│   ├── dashboard-design/
│   └── ...
├── install/                 # 安装脚本
│   ├── package.json
│   └── src/
│       ├── index.js         # CLI 入口
│       ├── installer.js     # 安装逻辑
│       └── utils.js         # 工具函数
├── install-history.json     # 安装历史记录
└── README.md                # 本文档
```

## 核心能力

### Auto-Task Subagent 系统（4 个）

四角色协作体系，支持任务的规划、执行、评审和探索：

**Subagent 列表**：

| 名称        | 文件           | 职责                                       | 权限                     |
| ----------- | -------------- | ------------------------------------------ | ------------------------ |
| at-planner  | at-planner.md  | 任务规划专家（思考、计划拆解、Gap 分析）   | write/edit（仅计划文件） |
| at-executor | at-executor.md | 执行专家（严格按计划执行、代码变更、测试） | write/edit/bash          |
| at-reviewer | at-reviewer.md | 质量评审专家（产出验证、一致性检查）       | 只读 + git               |
| at-explorer | at-explorer.md | 探索专家（只读探索、问题查询）             | 只读                     |

**架构设计**：

```
主代理
  ├─> at-planner：思考 → 规划
  ├─> at-executor：执行 → 产出
  └─> at-reviewer：验证 → 评审
```

**关键特性**：

- 职责分离：思考、执行、评审分开，质量可控
- 上下文隔离：每个 subagent 独立上下文，主上下文保持清晰
- 并行能力：独立任务可并行执行，效率提升
- 权限最小化：按 subagent 职责精简工具权限
- 失败处理：重试 3 次后等待用户决策，不丢失进度

**使用方式**：

通过 Task 工具委派，主代理作为编排者。

**配置引用**（opencode.json）：

```json
{
  "agent": {
    "at-planner": "{file:../core/agents/at-planner.md}",
    "at-executor": "{file:../core/agents/at-executor.md}",
    "at-reviewer": "{file:../core/agents/at-reviewer.md}",
    "at-explorer": "{file:../core/agents/at-explorer.md}"
  }
}
```

**命令委派**：

```typescript
Task({
  subagent_type: 'at-planner',
  prompt: '任务描述和上下文',
  description: '简短描述',
})
```

**迁移到其他项目**：

复制 `core/agents/` 目录到目标项目，并在 `opencode.json` 中引用。

### Agent 定义（18 个）

- analyst：数据驱动决策中心
- architect：架构设计
- dealer：端到端交付
- deployer：部署发布
- developer-backend：后端开发
- developer-frontend：前端开发
- developer-test：测试开发
- evaluator：价值评估
- ideation：创意挖掘
- opportunist：中间产物变现
- product-planner：产品规划
- reflector：知识沉淀
- requirement-analyst：需求分析
- strategist：商业策略
- toolsmith：可视化工具
- ui-designer：UI 设计
- 子今：全局 Agent

### 命令系统（18 个）

**已改造为 Subagent 委派模式**（8 个核心命令）：

| 命令             | 用途     | 委派对象            |
| ---------------- | -------- | ------------------- |
| /at-task-new     | 创建任务 | at-planner          |
| /at-think        | 思考任务 | at-planner          |
| /at-think-gap    | Gap 分析 | at-planner          |
| /at-plan-opt     | 执行优化 | at-executor         |
| /at-plan-refresh | 刷新计划 | planner + executor  |
| /at-plan-run     | 执行阶段 | executor + reviewer |
| /at-task-review  | 任务评审 | at-reviewer         |
| /at-task-run     | 执行任务 | 编排所有            |

**无需改造**（10 个轻量/辅助命令）：

| 命令               | 用途         | 说明         |
| ------------------ | ------------ | ------------ |
| /at-init           | 初始化       | 创建目录结构 |
| /at-mode           | 模式切换     | 轻量操作     |
| /at-update         | 更新模板     | 维护命令     |
| /at-mem-read       | 读取记忆     | 只读操作     |
| /at-mem-fresh      | 刷新记忆     | 索引更新     |
| /at-task-pop       | 弹出任务     | 队列操作     |
| /at-task-finish    | 完成任务     | 状态更新     |
| /at-plan-read-next | 读取下一阶段 | 只读操作     |
| /at-think-gap-save | 保存 gap     | 辅助命令     |
| /at-plan-review    | 计划评审     | 评审命令     |

### Skills 技能（11 个）

- competitor-analysis：竞品分析框架
- dashboard-design：可视化仪表盘设计
- idea-mining：创意挖掘方法论
- market-research：市场调研方法论
- monetization：盈利模式参考库
- packaging：应用打包指南
- post-mortem：复盘批判框架
- project-spec：项目规格定义
- python-codecheck：Python 代码检查
- submodule-manager：Git submodule 管理
- user-persona：用户创业画像

## 使用方式

### 本项目安装

将 core/ 安装到本项目的 .opencode/ 目录：

```bash
node core/install/src/index.js
```

### 全局安装

将 core/ 安装到 ~/.config/opencode/：

```bash
node core/install/src/index.js --global
```

### 部分组件安装

仅安装特定组件：

```bash
node core/install/src/index.js --components agent,commands
```

### 强制覆盖

覆盖已存在的文件：

```bash
node core/install/src/index.js --force
```

## 工作流

修改核心能力的完整流程：

1. 修改 core/ 下相应文件
2. git add core/ && git commit
3. 执行安装脚本更新 .opencode/
4. 其他项目可通过安装脚本获得更新

## 注意事项

- core/ 是版本控制源头，所有修改应在此提交
- .opencode/ 是环境临时文件，不应提交到 Git
- 安装脚本会记录安装历史到 core/install-history.json

## 改造说明

### Auto-Task 命令系统改造（v1.0.0）

**改造时间**：2026-08-13

**改造内容**：

- 新增 4 个专业 subagent 定义
- 改造 8 个核心命令为委派模式
- 遵循 OpenCode 最佳实践（Task 工具委派）

**改造收益**：

1. 职责分离：思考→planner，执行→executor，评审→reviewer
2. 上下文隔离：每个 subagent 独立上下文
3. 并行能力：独立任务可并行委派
4. 权限最小化：按 subagent 职责精简工具权限
5. 安全性提升：减少误操作风险
