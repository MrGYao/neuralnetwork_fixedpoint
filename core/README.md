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
├── commands/                # 命令定义（14 个）
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

### 命令系统（14 个）

- /at-init：初始化任务系统
- /at-mode：切换任务模式
- /at-update：更新模板文件
- /at-mem-fresh：刷新记忆索引
- /at-mem-read：读取记忆索引
- /at-plan-*：计划操作命令
- /at-task-*：任务操作命令

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
