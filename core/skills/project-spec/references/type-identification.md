# 项目类型识别规则

> 详细识别流程，skill 在步骤 2 需要精确识别时加载

---

## 总体识别流程

```
输入：项目根目录
输出：{
  primary_type: "desktop-electron-python",
  tech_stack: { frontend: "Electron", backend: "FastAPI", gui: null },
  confidence: 0.95
}

执行顺序：
  1. 桌面应用检测 → 若命中则进入桌面细分流程
  2. 微服务检测
  3. Web 应用检测
  4. CLI 工具检测
  5. 库/SDK 检测
  6. 未匹配 → 通用模板
```

---

## 桌面应用识别（优先级最高）

### Step 1: 检测桌面框架

**检测规则**：

| 框架        | 检测特征                                                 | 可能类型            |
| ----------- | -------------------------------------------------------- | ------------------- |
| Electron    | package.json 有 "electron" 依赖                          | E-* 系列            |
| Python GUI  | requirements.txt/pyproject.toml 有 PyQt\|PySide\|tkinter | desktop-python-gui  |
| Python 打包 | requirements.txt/pyproject.toml 有 pyinstaller\|nuitka   | desktop-python-tool |

**检测脚本**：

```bash
# E-* 检测
grep -r "electron" package.json && echo "ELECTRON_DETECTED"

# Py-GUI 检测
grep -E "PyQt|PySide|tkinter|wxPython|DearPyGui" requirements.txt pyproject.toml && echo "PY_GUI_DETECTED"

# Py-Tool 检测（需排除 GUI）
grep -E "pyinstaller|nuitka|pyoxidizer|cx_Freeze" requirements.txt pyproject.toml && echo "PY_TOOL_DETECTED"
```

**命中处理**：

- 检测到 Electron → 进入 **Step 2: Electron 细分**
- 检测到 Python GUI → 确认为 **desktop-python-gui**
- 检测到 Python 打包（无 GUI）→ 确认为 **desktop-python-tool**

---

### Step 2: Electron 细分

#### desktop-electron-tool (纯 Electron)

**检测特征**：

```
必须满足：
  ✓ package.json 有 "electron" 依赖
  ✓ main.js/main.ts 包含 BrowserWindow
  ✗ 不存在独立后端进程启动代码（spawn/exec python/node）
  ✗ 不存在后端框架依赖（FastAPI/Flask/Express 等）
```

**检测脚本**：

```bash
# 检测 Electron 主进程
grep -E "BrowserWindow|app\.on\('ready'" main.js main.ts

# 检测是否有后端进程启动
grep -E "spawn.*python|exec.*python|spawn.*node|execFile" main.js main.ts

# 检测是否有后端框架
grep -E "fastapi|flask|django|express|fastify|nest" package.json
```

**置信度计算**：

- 基础分：50（Electron 检测）
- 有 BrowserWindow：+20
- 无后端启动代码：+20
- 无后端框架依赖：+10
- **总分 100 → confidence = 1.0**

**识别结果**：

```json
{
  "primary_type": "desktop-electron-tool",
  "tech_stack": {
    "frontend": "Electron",
    "backend": null,
    "gui": null
  },
  "confidence": 1.0,
  "incremental_docs": ["solution/DEPLOYMENT_DESKTOP.md", "architecture/COMMUNICATION.md"]
}
```

---

#### desktop-electron-python (Electron + Python 后端)

**检测特征**：

```
必须满足：
  ✓ package.json 有 "electron"
  ✓ 检测到 Python 后端启动（spawn/exec python）
  ✓ requirements.txt/pyproject.toml 有 FastAPI|Flask|Django|Falcon
```

**检测脚本**：

```bash
# 检测 main.js 中是否启动 Python 进程
grep -E "spawn.*python|exec.*python|execFile.*python" main.js main.ts

# 检测 Python 后端框架
grep -E "FastAPI|Flask|Django|Falcon|Fastify|uvicorn|gunicorn" requirements.txt pyproject.toml

# 识别具体框架
grep "FastAPI" requirements.txt && echo "FRAMEWORK: FastAPI"
grep "Flask" requirements.txt && echo "FRAMEWORK: Flask"
grep "Django" requirements.txt && echo "FRAMEWORK: Django"
```

**置信度计算**：

- 基础分：50（Electron）
- 有 Python 进程启动：+30
- 有 FastAPI/Flask/Django：+20
- **总分 100 → confidence = 1.0**

**识别结果**：

```json
{
  "primary_type": "desktop-electron-python",
  "tech_stack": {
    "frontend": "Electron",
    "backend": "FastAPI",
    "gui": null,
    "process_manager": "spawn/exec"
  },
  "confidence": 1.0,
  "incremental_docs": [
    "solution/PROCESS_MANAGEMENT.md",
    "solution/DEPLOYMENT_DESKTOP.md",
    "architecture/COMMUNICATION.md",
    "architecture/PROCESS.md",
    "implementation/BACKEND.md",
    "implementation/FRONTEND.md",
    "implementation/PROCESS_MANAGER.md"
  ]
}
```

---

#### desktop-electron-node (Electron + Node 后端)

**检测特征**：

```
必须满足：
  ✓ package.json 有 "electron"
  ✓ 存在独立 Node 服务（server/ backend/ api/）
  ✗ 不存在 Python 后端启动（排除 E-Python）
```

**检测脚本**：

```bash
# 检测独立 Node 服务目录
ls -la server/ backend/ api/ 2>/dev/null

# 检测 Node 后端框架
grep -E "express|fastify|nest|koa|hapi" package.json

# 排除 Python 后端
! grep -E "spawn.*python|exec.*python" main.js
```

**置信度计算**：

- 基础分：50（Electron）
- 有独立 Node 服务目录：+30
- 有 Express/Fastify/Nest：+20
- **总分 100 → confidence = 1.0**

**识别结果**：

```json
{
  "primary_type": "desktop-electron-node",
  "tech_stack": {
    "frontend": "Electron",
    "backend": "Express",
    "gui": null
  },
  "confidence": 1.0,
  "incremental_docs": [
    "solution/PROCESS_MANAGEMENT.md",
    "solution/DEPLOYMENT_DESKTOP.md",
    "architecture/COMMUNICATION.md",
    "architecture/PROCESS.md",
    "implementation/BACKEND.md",
    "implementation/FRONTEND.md",
    "implementation/PROCESS_MANAGER.md"
  ]
}
```

---

### Step 3: Python 桌面细分

#### desktop-python-gui (Python GUI)

**检测特征**：

```
必须满足：
  ✓ requirements.txt/pyproject.toml 有 PyQt|PySide|tkinter|wxPython|DearPyGui
```

**GUI 框架映射**：

| 依赖            | 框架       | 架构特点   | 配置文件 |
| --------------- | ---------- | ---------- | -------- |
| PyQt5/PyQt6     | Qt         | 信号槽机制 | -        |
| PySide2/PySide6 | Qt         | 信号槽机制 | -        |
| tkinter         | Tk         | 事件循环   | -        |
| wxPython        | wxWidgets  | 事件绑定   | -        |
| DearPyGui       | Dear ImGui | 即时模式   | -        |

**检测脚本**：

```bash
# 检测 GUI 框架
grep -E "PyQt|PySide" requirements.txt && echo "GUI: Qt"
grep "tkinter" requirements.txt && echo "GUI: Tkinter"
grep "wxPython" requirements.txt && echo "GUI: wxPython"
grep "DearPyGui" requirements.txt && echo "GUI: DearPyGui"
```

**识别结果**：

```json
{
  "primary_type": "desktop-python-gui",
  "tech_stack": {
    "frontend": null,
    "backend": null,
    "gui": "PyQt6"
  },
  "confidence": 1.0,
  "incremental_docs": [
    "solution/DEPLOYMENT_PACKAGE.md",
    "architecture/UI_FRAMEWORK.md",
    "architecture/BUILD.md",
    "implementation/WIDGETS.md",
    "implementation/EVENT_LOOP.md",
    "implementation/ENTRY_POINT.md"
  ]
}
```

---

#### desktop-python-tool (Python 打包工具)

**检测特征**：

```
必须满足：
  ✓ requirements.txt/pyproject.toml 有 pyinstaller|nuitka|pyoxidizer|cx_Freeze
  ✗ 无 GUI 库依赖（否则是 desktop-python-gui）
```

**打包工具映射**：

| 工具        | 特点          | 配置文件       | 输出格式    |
| ----------- | ------------- | -------------- | ----------- |
| pyinstaller | .spec 文件    | app.spec       | exe/app/dmg |
| nuitka      | 编译优化      | nuitka.config  | exe/bin     |
| pyoxidizer  | 嵌入式        | pyoxidizer.bzl | exe         |
| cx_Freeze   | setup.py 集成 | setup.py       | exe         |

**检测脚本**：

```bash
# 检测打包工具
grep "pyinstaller" requirements.txt && echo "PACK: PyInstaller"
grep "nuitka" requirements.txt && echo "PACK: Nuitka"
grep "pyoxidizer" requirements.txt && echo "PACK: PyOxidizer"
grep "cx_Freeze" requirements.txt && echo "PACK: cx_Freeze"

# 排除 GUI 框架
! grep -E "PyQt|PySide|tkinter|wxPython|DearPyGui" requirements.txt
```

**识别结果**：

```json
{
  "primary_type": "desktop-python-tool",
  "tech_stack": {
    "frontend": null,
    "backend": null,
    "gui": null,
    "packaging": "PyInstaller"
  },
  "confidence": 1.0,
  "incremental_docs": [
    "solution/DEPLOYMENT_PACKAGE.md",
    "architecture/BUILD.md",
    "implementation/ENTRY_POINT.md"
  ]
}
```

---

## Web 应用识别

### web-monolith (单体 Web)

**检测特征**：

```
必须满足：
  ✓ FastAPI/Flask/Django/Express/NestJS/Spring Boot 依赖
  ✓ 存在路由文件：routes/ / api/ / controllers/
  ✗ 无桌面应用特征
  ✗ 无微服务特征
```

**检测脚本**：

```bash
# 检测后端框架
grep -E "FastAPI|Flask|Django|Falcon" requirements.txt pyproject.toml
grep -E "express|fastify|nest|koa" package.json

# 检测路由目录
ls -la routes/ api/ controllers/ views/ 2>/dev/null
```

**识别结果**：

```json
{
  "primary_type": "web-monolith",
  "tech_stack": {
    "frontend": null,
    "backend": "FastAPI"
  },
  "confidence": 0.95,
  "incremental_docs": [
    "solution/API.md",
    "solution/DEPLOYMENT_WEB.md",
    "solution/SECURITY.md",
    "architecture/DATA.md",
    "architecture/AUTHENTICATION.md",
    "implementation/ROUTES.md"
  ]
}
```

---

### web-fullstack (全栈 Web)

**检测特征**：

```
必须满足：
  ✓ 同时存在前端和后端目录结构
  ✓ 前端：React/Vue/Angular + TypeScript
  ✓ 后端：FastAPI/Django/Express/NestJS
  ✓ 可能存在 monorepo 结构（packages/frontend, packages/backend）
```

**检测脚本**：

```bash
# 检测 monorepo
ls -la packages/frontend packages/backend 2>/dev/null

# 检测前后端分离目录
ls -la frontend/ backend/ client/ server/ 2>/dev/null

# 检测前端框架
grep -E "react|vue|angular|svelte" package.json

# 检测后端框架
grep -E "FastAPI|Flask|Django|express|fastify" requirements.txt package.json
```

---

## CLI 工具识别

**检测特征**：

```
必须满足：
  ✓ argparse/click/typer (Python)
  ✓ commander/yargs/oclif (Node.js)
  ✓ clap (Rust)
  ✓ cobra (Go)
  ✓ 主程序读取 process.argv / sys.argv
  ✗ 非桌面应用、非 Web 应用
```

**检测脚本**：

```bash
# Python CLI
grep -E "argparse|click|typer" requirements.txt pyproject.toml

# Node CLI
grep -E "commander|yargs|oclif|inquirer" package.json

# Rust CLI
grep "clap" Cargo.toml

# Go CLI
grep "cobra" go.mod

# 检测命令行入口
grep "process.argv" main.js
grep "sys.argv" main.py
```

---

## 库/SDK 识别

**检测特征**：

```
必须满足：
  ✓ package.json 的 main / exports 字段存在
  ✓ pyproject.toml 的 packages 配置
  ✓ 无主程序入口（main.js / main.py / __main__.py）
  ✓ 文档中提及 "library" / "SDK" / "package"
```

**检测脚本**：

```bash
# 检测库配置
jq '.main // .exports' package.json
grep "packages" pyproject.toml

# 检测无入口
! ls main.js main.py __main__.py 2>/dev/null
```

---

## 微服务识别

**检测特征**：

```
必须满足：
  ✓ 存在 docker-compose.yml 且定义多个服务
  ✓ 存在服务发现配置（Consul / Eureka / etcd）
  ✓ 多个独立的服务目录：services/ / apps/
  ✓ API 网关配置
```

**检测脚本**：

```bash
# 检测多服务
grep -c "service:" docker-compose.yml
ls -la services/ apps/ 2>/dev/null

# 检测服务发现
grep -E "consul|eureka|etcd|nacos" docker-compose.yml
```

---

## 识别优先级

当项目同时匹配多个类型特征时，按此优先级判断：

| 优先级  | 类型                    | 说明                                 |
| ------- | ----------------------- | ------------------------------------ |
| **1.1** | desktop-electron-python | Electron + Python 后端（特征最明显） |
| **1.2** | desktop-electron-node   | Electron + Node 后端                 |
| **1.3** | desktop-electron-tool   | 纯 Electron                          |
| **1.4** | desktop-python-gui      | Python GUI（特征明显）               |
| **1.5** | desktop-python-tool     | Python 打包                          |
| **2**   | microservice            | 多服务架构                           |
| **3**   | web-fullstack           | 前后端分离                           |
| **4**   | web-monolith            | 单体 Web                             |
| **5**   | cli-tool                | 命令行工具                           |
| **6**   | library-sdk             | 库/SDK                               |
| **7**   | generic                 | 未匹配                               |

---

## 混合类型处理

### 模糊识别场景

当 `confidence < 0.8` 或匹配多个类型时，询问用户：

```
检测到项目兼具多种特征：
  [1] Electron + Python 后端（桌面应用托管后端）
  [2] Web 应用（带桌面客户端）
  [3] 全栈项目（前后端独立）

请选择项目核心定位：
```

### 多框架共存处理

示例：Electron + PyQt（罕见但可能）

**处理策略**：

1. 记录为混合类型：`mixed: electron+pyqt`
2. 生成两套增量文档（desktop-electron-tool + desktop-python-gui）
3. 询问用户主次关系：
   ```
   检测到两套 GUI 框架：
     - Electron（桌面封装）
     - PyQt（Python GUI）

   请确认主次关系：
     [1] Electron 为主，PyQt 为辅助
     [2] PyQt 为主，Electron 为辅助
     [3] 并行（两者独立）
   ```

---

## 识别结果输出

### 标准输出格式

```json
{
  "primary_type": "desktop-electron-python",
  "tech_stack": {
    "frontend": "Electron",
    "backend": "FastAPI",
    "gui": null,
    "database": "SQLite",
    "process_manager": "spawn"
  },
  "confidence": 1.0,
  "detection_evidence": [
    "package.json contains 'electron' dependency",
    "main.js contains 'spawn.*python'",
    "requirements.txt contains 'FastAPI'"
  ],
  "incremental_docs": [
    "solution/PROCESS_MANAGEMENT.md",
    "solution/DEPLOYMENT_DESKTOP.md",
    "architecture/COMMUNICATION.md",
    "architecture/PROCESS.md",
    "implementation/BACKEND.md",
    "implementation/FRONTEND.md",
    "implementation/PROCESS_MANAGER.md"
  ],
  "template_path": "templates/incremental/desktop-electron-python/"
}
```

---

## 使用时机

**步骤 2 信息采集** 时，当需要精确识别项目类型，加载此文档执行上述检测流程。
