# 项目模板索引

> 用于 skill 自动加载对应模板，人类快速查阅模板组成

---

## 基础模板

所有项目共用文档（~15个文件）

### spec 根文档

| 文件            | 层次   | 用途             | 预估行数 |
| --------------- | ------ | ---------------- | -------- |
| coding-index.md | 根级别 | 代码开发必读索引 | ~350     |

### product/ 层（必选）

| 文件            | 层次    | 用途           | 预估行数 |
| --------------- | ------- | -------------- | -------- |
| README.md       | Level 1 | 产品层概览     | ~30      |
| OVERVIEW.md     | Level 2 | 项目定位与背景 | ~80      |
| VALUE.md        | Level 2 | 核心价值主张   | ~60      |
| REQUIREMENTS.md | Level 2 | 功能需求       | ~100     |

### solution/ 层（必选）

| 文件            | 层次    | 用途                           | 预估行数 |
| --------------- | ------- | ------------------------------ | -------- |
| README.md       | Level 1 | 方案层概览                     | ~30      |
| OVERVIEW.md     | Level 2 | 解决方案概述                   | ~60      |
| ARCHITECTURE.md | Level 2 | 架构方向（引用 architecture/） | ~80      |
| TECH_STACK.md   | Level 2 | 技术栈选型                     | ~60      |
| CONFIG.md       | Level 2 | 配置策略                       | ~50      |

### architecture/ 层（必选）

| 文件                | 层次    | 用途               | 预估行数 |
| ------------------- | ------- | ------------------ | -------- |
| README.md           | Level 1 | 架构层概览         | ~30      |
| CORE.md             | Level 2 | 核心架构           | ~100     |
| DIRECTORY.md        | Level 2 | 目录结构说明       | ~80      |
| CODING_STANDARDS.md | Level 2 | 代码规范与开发模式 | ~400     |

### implementation/ 层（必选）

| 文件          | 层次    | 用途       | 预估行数 |
| ------------- | ------- | ---------- | -------- |
| README.md     | Level 1 | 实现层概览 | ~30      |
| INTERFACES.md | Level 2 | 接口契约   | ~60      |

### issues/ 层（必选）

| 文件           | 层次    | 用途         | 预估行数 |
| -------------- | ------- | ------------ | -------- |
| README.md      | Level 1 | 问题追踪概览 | ~30      |
| TODOs.md       | Level 2 | 待办事项     | ~50      |
| LIMITATIONS.md | Level 2 | 已知限制     | ~40      |

---

## 增量模板索引

按项目类型枚举，每个增量包含：

- 文档清单（相对于 base 新增/覆盖的文件）
- 文件模板路径（引用 `templates/incremental/<类型>/`）

---

### 桌面应用变体

#### desktop-electron-tool (纯 Electron)

**适用场景**：Electron 桌面应用，无独立后端进程托管

**技术特征**：

- package.json 有 "electron" 依赖
- main.js 包含 BrowserWindow
- 无独立后端进程启动（spawn/exec python/node）
- 无后端框架依赖

**增量文档**（2个）：

| 文件                           | 层次    | 用途              | 路径                                         |
| ------------------------------ | ------- | ----------------- | -------------------------------------------- |
| solution/DEPLOYMENT_DESKTOP.md | Level 2 | Electron 打包方案 | templates/incremental/desktop-electron-tool/ |
| architecture/COMMUNICATION.md  | Level 2 | IPC 通信机制      | templates/incremental/desktop-electron-tool/ |

---

#### desktop-electron-python (Electron + Python 后端)

**适用场景**：Electron 托管 FastAPI/Flask/Django 等后端

**技术特征**：

- package.json 有 "electron"
- main.js 包含 spawn/exec 启动 Python 进程
- requirements.txt/pyproject.toml 有 FastAPI|Flask|Django

**增量文档**（7个）：

| 文件                              | 层次    | 用途                       | 路径                                           |
| --------------------------------- | ------- | -------------------------- | ---------------------------------------------- |
| solution/PROCESS_MANAGEMENT.md    | Level 2 | Python 进程管理策略        | templates/incremental/desktop-electron-python/ |
| solution/DEPLOYMENT_DESKTOP.md    | Level 2 | Electron 打包方案          | templates/incremental/desktop-electron-python/ |
| architecture/COMMUNICATION.md     | Level 2 | HTTP 通信机制（覆盖 base） | templates/incremental/desktop-electron-python/ |
| architecture/PROCESS.md           | Level 2 | 主进程与子进程架构         | templates/incremental/desktop-electron-python/ |
| implementation/BACKEND.md         | Level 2 | Python 后端实现            | templates/incremental/desktop-electron-python/ |
| implementation/FRONTEND.md        | Level 2 | Electron 渲染进程实现      | templates/incremental/desktop-electron-python/ |
| implementation/PROCESS_MANAGER.md | Level 2 | 进程启动/监控实现          | templates/incremental/desktop-electron-python/ |

---

#### desktop-electron-node (Electron + Node 后端)

**适用场景**：Electron 托管独立 Node 服务（非主进程内联）

**技术特征**：

- package.json 有 "electron"
- 存在独立 Node 服务（server/ backend/ api/）
- 有 Express/Fastify/NestJS/Koa 依赖

**增量文档**（7个）：

| 文件                              | 层次    | 用途                  | 路径                                         |
| --------------------------------- | ------- | --------------------- | -------------------------------------------- |
| solution/PROCESS_MANAGEMENT.md    | Level 2 | Node 服务管理策略     | templates/incremental/desktop-electron-node/ |
| solution/DEPLOYMENT_DESKTOP.md    | Level 2 | Electron 打包方案     | templates/incremental/desktop-electron-node/ |
| architecture/COMMUNICATION.md     | Level 2 | HTTP 通信机制         | templates/incremental/desktop-electron-node/ |
| architecture/PROCESS.md           | Level 2 | 主进程与子进程架构    | templates/incremental/desktop-electron-node/ |
| implementation/BACKEND.md         | Level 2 | Node 服务实现         | templates/incremental/desktop-electron-node/ |
| implementation/FRONTEND.md        | Level 2 | Electron 渲染进程实现 | templates/incremental/desktop-electron-node/ |
| implementation/PROCESS_MANAGER.md | Level 2 | 进程启动/监控实现     | templates/incremental/desktop-electron-node/ |

---

#### desktop-python-gui (Python GUI)

**适用场景**：PyQt/PySide/Tkinter 等 GUI 框架

**技术特征**：

- requirements.txt 有 PyQt|PySide|tkinter|wxPython|DearPyGui
- 独立 Python 进程（桌面应用）

**GUI 框架映射**：

| 依赖            | 框架       | 架构特点   |
| --------------- | ---------- | ---------- |
| PyQt5/PyQt6     | Qt         | 信号槽机制 |
| PySide2/PySide6 | Qt         | 信号槽机制 |
| tkinter         | Tk         | 事件循环   |
| wxPython        | wxWidgets  | 事件绑定   |
| DearPyGui       | Dear ImGui | 即时模式   |

**增量文档**（6个）：

| 文件                           | 层次    | 用途         | 路径                                      |
| ------------------------------ | ------- | ------------ | ----------------------------------------- |
| solution/DEPLOYMENT_PACKAGE.md | Level 2 | 打包分发方案 | templates/incremental/desktop-python-gui/ |
| architecture/UI_FRAMEWORK.md   | Level 2 | GUI 架构设计 | templates/incremental/desktop-python-gui/ |
| architecture/BUILD.md          | Level 2 | 构建打包流程 | templates/incremental/desktop-python-gui/ |
| implementation/WIDGETS.md      | Level 2 | 组件实现     | templates/incremental/desktop-python-gui/ |
| implementation/EVENT_LOOP.md   | Level 2 | 事件循环处理 | templates/incremental/desktop-python-gui/ |
| implementation/ENTRY_POINT.md  | Level 2 | 入口点实现   | templates/incremental/desktop-python-gui/ |

---

#### desktop-python-tool (Python 打包工具)

**适用场景**：PyInstaller/Nuitka 等打包工具（非 GUI）

**技术特征**：

- requirements.txt 有 pyinstaller|nuitka|pyoxidizer|cx_Freeze
- 无 GUI 库依赖（否则为 desktop-python-gui）

**打包工具映射**：

| 工具        | 特点          | 配置文件       |
| ----------- | ------------- | -------------- |
| pyinstaller | .spec 文件    | app.spec       |
| nuitka      | 编译优化      | nuitka.config  |
| pyoxidizer  | 嵌入式        | pyoxidizer.bzl |
| cx_Freeze   | setup.py 集成 | setup.py       |

**增量文档**（3个）：

| 文件                           | 层次    | 用途         | 路径                                       |
| ------------------------------ | ------- | ------------ | ------------------------------------------ |
| solution/DEPLOYMENT_PACKAGE.md | Level 2 | 打包分发方案 | templates/incremental/desktop-python-tool/ |
| architecture/BUILD.md          | Level 2 | 构建打包流程 | templates/incremental/desktop-python-tool/ |
| implementation/ENTRY_POINT.md  | Level 2 | 入口点实现   | templates/incremental/desktop-python-tool/ |

---

### Web 应用变体

#### web-monolith (单体 Web)

**适用场景**：FastAPI/Flask/Django/Express/NestJS 单体应用

**增量文档**（6个）：

| 文件                           | 层次    | 用途         |
| ------------------------------ | ------- | ------------ |
| solution/API.md                | Level 2 | API 设计规范 |
| solution/DEPLOYMENT_WEB.md     | Level 2 | Web 部署方案 |
| solution/SECURITY.md           | Level 2 | 安全策略     |
| architecture/DATA.md           | Level 2 | 数据架构     |
| architecture/AUTHENTICATION.md | Level 2 | 认证架构     |
| implementation/ROUTES.md       | Level 2 | 路由实现     |

---

#### web-fullstack (全栈 Web)

**适用场景**：前后端分离（React/Vue + FastAPI/Express）

**增量文档**（10个）：

| 文件                          | 层次    | 用途             |
| ----------------------------- | ------- | ---------------- |
| solution/FRONTEND.md          | Level 2 | 前端方案         |
| solution/BACKEND.md           | Level 2 | 后端方案         |
| solution/API.md               | Level 2 | API 设计         |
| solution/DEPLOYMENT_WEB.md    | Level 2 | 部署方案         |
| solution/SECURITY.md          | Level 2 | 安全策略         |
| architecture/FRONTEND_ARCH.md | Level 2 | 前端架构         |
| architecture/BACKEND_ARCH.md  | Level 2 | 后端架构         |
| architecture/COMMUNICATION.md | Level 2 | 前后端通信       |
| implementation/FRONTEND/      | Level 3 | 前端实现（目录） |
| implementation/BACKEND/       | Level 3 | 后端实现（目录） |

---

### CLI 工具

**适用场景**：命令行工具（argparse/click/commander/yargs）

**增量文档**（6个）：

| 文件                            | 层次    | 用途         |
| ------------------------------- | ------- | ------------ |
| solution/USAGE.md               | Level 2 | 使用说明     |
| solution/DEPLOYMENT_PACKAGE.md  | Level 2 | 打包分发     |
| architecture/CLI_FLOW.md        | Level 2 | 命令执行流程 |
| architecture/COMMANDS.md        | Level 2 | 命令定义     |
| implementation/CLI_PARSER.md    | Level 2 | 参数解析实现 |
| implementation/COMMANDS_IMPL.md | Level 2 | 命令实现     |

---

### 库/SDK

**适用场景**：Python/Node 库或 SDK

**增量文档**（5个）：

| 文件                         | 层次    | 用途         |
| ---------------------------- | ------- | ------------ |
| solution/INTEGRATION.md      | Level 2 | 集成指南     |
| architecture/API.md          | Level 2 | 公开 API     |
| architecture/EXTENSION.md    | Level 2 | 扩展机制     |
| implementation/PUBLIC_API.md | Level 2 | 公开接口实现 |
| implementation/EXAMPLES.md   | Level 2 | 示例代码     |

---

### 微服务

**适用场景**：多服务架构

**增量文档**（目录级别）：

| 文件/目录                           | 层次    | 用途                 |
| ----------------------------------- | ------- | -------------------- |
| solution/SERVICES.md                | Level 2 | 服务划分             |
| solution/COMMUNICATION.md           | Level 2 | 服务通信             |
| solution/DEPLOYMENT_MICROSERVICE.md | Level 2 | 微服务部署           |
| architecture/SERVICES/              | Level 3 | 每个服务文档（目录） |
| architecture/API_GATEWAY.md         | Level 2 | API 网关             |
| implementation/SERVICES/            | Level 3 | 服务实现（目录）     |

---

## 模板组合规则

**公式**：`最终模板 = base + incremental(项目类型)`

### 组合示例

| 项目类型                | 组合方式             | 总文档数 |
| ----------------------- | -------------------- | -------- |
| desktop-electron-python | base (15) + 增量 (7) | 22       |
| desktop-python-gui      | base (15) + 增量 (6) | 21       |
| web-monolith            | base (15) + 增量 (6) | 21       |
| cli-tool                | base (15) + 增量 (6) | 21       |

---

## 加载策略

### Step 1: 加载基础模板

```
加载顺序：
  1. product/README.md
  2. solution/README.md
  3. architecture/README.md
  4. implementation/README.md
  5. issues/README.md

用途：快速概览项目结构（~150 tokens）
```

### Step 2: 根据项目类型加载增量模板

```
输入：识别到的项目类型（如 desktop-electron-python）
处理：
  1. 从 TEMPLATE_INDEX.md 查询增量文档清单
  2. 加载 templates/incremental/<类型>/ 下的文件模板
  3. 合并到基础文档结构
```

### Step 3: 合并文档清单

```
合并规则：
  - 新增文件：直接添加到对应层级
  - 覆盖文件：替换 base 中的同名文件（如 COMMUNICATION.md）
  - 记录合并结果到 spec/config.json
```

---

## 文件模板路径

### 增量模板目录结构

```
templates/
└── incremental/
    ├── desktop-electron-tool/
    │   ├── solution-DEPLOYMENT_DESKTOP.md
    │   └── architecture-COMMUNICATION.md
    ├── desktop-electron-python/
    │   ├── solution-PROCESS_MANAGEMENT.md
    │   ├── solution-DEPLOYMENT_DESKTOP.md
    │   ├── architecture-COMMUNICATION.md
    │   ├── architecture-PROCESS.md
    │   ├── implementation-BACKEND.md
    │   ├── implementation-FRONTEND.md
    │   └── implementation-PROCESS_MANAGER.md
    ├── desktop-python-gui/
    │   └── ...
    ├── web-monolith/
    │   └── ...
    └── ...
```

---

## 与 spec-README-template.md 的关系

- **本文档（TEMPLATE_INDEX.md）**：定义文档结构和加载逻辑
- **spec-README-template.md**：定义 spec/README.md 的内容模板
- **配合使用**：先根据本文档确定文档清单，再生成各文档内容，最后生成 spec/README.md 导航
