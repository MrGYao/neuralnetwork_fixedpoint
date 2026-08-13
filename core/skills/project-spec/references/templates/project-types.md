# 项目类型总览

> 人类快速查阅视图，详细模板参见 [TEMPLATE_INDEX.md](./TEMPLATE_INDEX.md)

---

## 类型枚举

### 桌面应用 (5种)

| 类型 | 描述 | 典型技术栈 | 特有文档数 |
|------|------|-----------|-----------|
| **desktop-electron-tool** | 纯 Electron | Electron + Vue/React | 2 |
| **desktop-electron-python** | Electron 托管 Python 后端 | Electron + FastAPI/Flask | 7 |
| **desktop-electron-node** | Electron 托管 Node 后端 | Electron + Express/Fastify | 7 |
| **desktop-python-gui** | Python GUI 应用 | PyQt/PySide/Tkinter | 6 |
| **desktop-python-tool** | Python 打包工具 | PyInstaller/Nuitka | 3 |

### Web 应用 (2种)

| 类型 | 描述 | 典型技术栈 | 特有文档数 |
|------|------|-----------|-----------|
| **web-monolith** | 单体 Web | FastAPI/Flask/Express | 6 |
| **web-fullstack** | 前后端分离 | React + FastAPI | 10 |

### 其他类型

| 类型 | 描述 | 典型技术栈 | 特有文档数 |
|------|------|-----------|-----------|
| **cli-tool** | 命令行工具 | argparse/click/commander | 6 |
| **library-sdk** | 库/SDK | Python/Node 库 | 5 |
| **microservice** | 微服务架构 | Docker + 服务发现 | 目录级 |
| **generic** | 通用项目 | 不确定类型 | 0 |

---

## 文档差异速查

### 桌面应用特有文档

#### desktop-electron-python

| 层级 | 文档 | 用途 |
|------|------|------|
| solution | PROCESS_MANAGEMENT.md | Python 进程管理策略 |
| solution | DEPLOYMENT_DESKTOP.md | Electron 打包方案 |
| architecture | COMMUNICATION.md | HTTP 通信机制 |
| architecture | PROCESS.md | 主进程与子进程架构 |
| implementation | BACKEND.md | Python 后端实现 |
| implementation | FRONTEND.md | Electron 渲染进程实现 |
| implementation | PROCESS_MANAGER.md | 进程启动/监控实现 |

#### desktop-python-gui

| 层级 | 文档 | 用途 |
|------|------|------|
| solution | DEPLOYMENT_PACKAGE.md | 打包分发方案 |
| architecture | UI_FRAMEWORK.md | GUI 架构设计 |
| architecture | BUILD.md | 构建打包流程 |
| implementation | WIDGETS.md | 组件实现 |
| implementation | EVENT_LOOP.md | 事件循环处理 |
| implementation | ENTRY_POINT.md | 入口点实现 |

### Web 应用特有文档

#### web-monolith

| 层级 | 文档 | 用途 |
|------|------|------|
| solution | API.md | API 设计规范 |
| solution | DEPLOYMENT_WEB.md | Web 部署方案 |
| solution | SECURITY.md | 安全策略 |
| architecture | DATA.md | 数据架构 |
| architecture | AUTHENTICATION.md | 认证架构 |
| implementation | ROUTES.md | 路由实现 |

---

## 置信度阈值

| 置信度 | 判断 | 行为 |
|--------|------|------|
| ≥ 0.9 | 明确识别 | 直接选择类型 |
| 0.7 - 0.9 | 倾向识别 | 选择类型 + 记录证据 |
| < 0.7 | 模糊识别 | 询问用户确认 |

---

## 模板文件位置

| 类型 | 路径 |
|------|------|
| 基础模板 | `references/templates/base/` |
| 增量模板 | `references/templates/incremental/<类型>/` |
| 详细索引 | `references/TEMPLATE_INDEX.md` |

---

## 混合类型处理

当项目匹配多个类型特征时：

1. **优先级判断**：按照识别优先级表选择
2. **置信度不足**：`confidence < 0.7` 时询问用户
3. **罕见组合**：如 Electron + PyQt，询问主次关系

---

## 与识别流程的关系

- **本文档**：类型枚举 + 文档速查（人类查阅）
- **type-identification.md**：详细检测规则 + 识别流程（机器执行）
- **TEMPLATE_INDEX.md**：模板索引 + 加载策略（机器加载）

配合使用：识别流程输出类型 → 从本文档速查特有文档 → 加载对应模板
