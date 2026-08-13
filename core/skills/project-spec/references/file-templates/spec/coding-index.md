# 代码开发必读索引

> **用途**：每次开发前必读，包含项目中最关键的编码规则与方法  
> **阅读时间**：~5 分钟  
> **更新频率**：项目规范变更时同步更新  
> **位置**：`spec/coding-index.md`

---

## 快速索引

| 章节                          | 关键点                   | 阅读时间 | 详细文档                                                           |
| ----------------------------- | ------------------------ | -------- | ------------------------------------------------------------------ |
| [命名约定](#命名约定)         | 文件、变量、函数命名规则 | 1分钟    | [CODING_STANDARDS.md](./architecture/CODING_STANDARDS.md#命名约定) |
| [错误处理模式](#错误处理模式) | 异常捕获、错误传播       | 1分钟    | [CODING_STANDARDS.md](./architecture/CODING_STANDARDS.md#错误处理) |
| [TDD 流程](#tdd-流程)         | 红-绿-重构循环           | 1分钟    | [CODING_STANDARDS.md](./architecture/CODING_STANDARDS.md#tdd)      |
| [代码组织](#代码组织)         | 目录结构、模块划分       | 30秒     | [DIRECTORY.md](./architecture/DIRECTORY.md)                        |
| [依赖管理](#依赖管理)         | 包管理、依赖更新         | 30秒     | [TECH_STACK.md](./solution/TECH_STACK.md)                          |

---

## 核心规范摘要

### 命名约定

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高  
> **配置来源**：代码约定 / `.editorconfig` / 无

#### 文件命名

**默认推荐**：

- **Python 模块**：`snake_case.py`（PEP 8 标准）
- **Python 包**：`snake_case/`
- **TypeScript 文件**：`camelCase.ts`（普通文件）/ `PascalCase.tsx`（React 组件）
- **配置文件**：`kebab-case.config.js` 或 `snake_case.config.py`

**常见选项**：

| 语言/类型   | 选项 A                | 选项 B                   | 选项 C                  |
| ----------- | --------------------- | ------------------------ | ----------------------- |
| Python 模块 | snake_case.py（标准） | camelCase.py（框架约定） | PascalCase.py（组件化） |
| TypeScript  | camelCase.ts（普通）  | PascalCase.ts（组件）    | snake_case.ts（服务端） |
| React 组件  | PascalCase.tsx        | camelCase.tsx            | -                       |

**本项目设定**：

- **Python 模块**：[待填充 / 已设定为 snake_case]
- **TypeScript 文件**：[待填充 / 已设定为 camelCase/PascalCase]
- **配置来源**：[.editorconfig / 代码约定 / 待设定]

> **自动检测策略**：
>
> - Level 1：检测 `.editorconfig` 的 `[*.{py,ts}]` 命名约定
> - Level 2：分析现有 `.py` / `.ts` 文件名，统计 snake_case/camelCase/PascalCase 比例
> - 未检测到 → 使用默认推荐（snake_case / camelCase）+ 加入遗留问题

---

#### 变量命名

**默认推荐**：

- **常量**：`UPPER_SNAKE_CASE`（如 `MAX_CONNECTIONS`）
- **变量**：
  - Python：`snake_case`（如 `user_count`）
  - TypeScript：`camelCase`（如 `userCount`）
- **私有变量**：`_leading_underscore`（如 `_internal_cache`）

**本项目设定**：

- **常量**：[已设定为 UPPER_SNAKE_CASE]
- **变量（Python）**：[已设定为 snake_case]
- **变量（TypeScript）**：[已设定为 camelCase]

> **差异程度**：中（大部分项目一致，可简化处理）

---

#### 函数命名

**默认推荐**：

- **函数/方法**：
  - Python：`verb_noun`（如 `get_user_by_id`）
  - TypeScript：`verbNoun`（如 `getUserById`）
- **类**：`PascalCase`（如 `UserService`）
- **测试函数**：`test_<scenario>_<expected>`（如 `test_login_invalid_password_fails`）

**常见选项**：

| 类型        | 选项 A                           | 选项 B                       |
| ----------- | -------------------------------- | ---------------------------- |
| Python 函数 | verb_noun（标准）                | noun_verb（某些框架）        |
| 测试函数    | test_scenario_expected（描述性） | testScenarioExpected（驼峰） |

**本项目设定**：

- **函数（Python）**：[已设定为 verb_noun]
- **函数（TypeScript）**：[已设定为 verbNoun]
- **测试函数**：[待填充 / 已设定为 test_scenario_expected]

---

#### 关键禁止

以下规则通常是固定的，不建议修改：

- 禁用单字母变量（循环计数器 `i, j, k` 除外）
- 禁用含义不明的缩写（如 `cnt` → `count`，`tmp` → `temporary`）
- 禁用保留字作为变量名

---

### 错误处理模式

> **配置状态**：[基本固定]  
> **差异程度**：低

#### 异常捕获原则

**固定规则**（不建议修改）：

```python
# ❌ 禁止裸 except
try:
    ...
except:  # 永远不要这样做
    pass

# ✅ 必须指定异常类型
try:
    ...
except ValueError as e:
    logger.error(f"Invalid value: {e}")
```

---

#### 错误传播层次

**默认推荐**：

| 层次           | 行为                   | 示例                                        |
| -------------- | ---------------------- | ------------------------------------------- |
| **业务逻辑层** | 抛出自定义异常         | `raise UserNotFoundError(user_id)`          |
| **API 层**     | 捕获并转换为 HTTP 响应 | `return JSONResponse(status_code=404, ...)` |
| **用户界面**   | 显示友好错误消息       | `toast.error("用户不存在")`                 |

**本项目设定**：[已设定为默认推荐 / 自定义：XXX]

---

#### 日志记录

**固定规则**：

- 异常必须记录上下文（相关参数、用户信息等）
- 区分 `logger.info` / `logger.warning` / `logger.error`
- 避免敏感信息进入日志（密码、token、PII 数据）

---

### TDD 流程

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高  
> **配置来源**：`pyproject.toml` / team agreement / 无

#### 标准循环

**固定流程**（不建议修改）：

```
1. 写失败测试
   ↓
2. 写最简实现
   ↓
3. 重构优化
   ↓
重复
```

---

#### 覆盖率要求

> **差异程度**：高（不同项目要求差异大）

**默认推荐**：

| 代码类型     | 默认推荐 | 理由                   |
| ------------ | -------- | ---------------------- |
| 核心业务逻辑 | ≥ 90%    | 关键逻辑必须充分测试   |
| API 端点     | ≥ 80%    | API 层易出错，需要覆盖 |
| 工具函数     | ≥ 70%    | 辅助功能，可适当降低   |
| 配置/模型    | ≥ 60%    | 主要是数据定义         |

**常见范围**：

| 代码类型     | 宽松 | 标准    | 严格 |
| ------------ | ---- | ------- | ---- |
| 核心业务逻辑 | 80%  | **90%** | 95%  |
| API 端点     | 70%  | **80%** | 90%  |
| 工具函数     | 60%  | **70%** | 80%  |
| 配置/模型    | 50%  | **60%** | 70%  |

**本项目设定**：

| 代码类型     | 本项目要求              | 配置来源                          |
| ------------ | ----------------------- | --------------------------------- |
| 核心业务逻辑 | [待填充 / 已设定为 XX%] | [pyproject.toml / team agreement] |
| API 端点     | [待填充 / 已设定为 XX%] | -                                 |
| 工具函数     | [待填充 / 已设定为 XX%] | -                                 |
| 配置/模型    | [待填充 / 已设定为 XX%] | -                                 |

> **自动检测策略**：
>
> - Level 1：检测 `pyproject.toml` 中 `[tool.pytest.ini_options]` 的 `addopts = "--cov-fail-under=XX"`
> - Level 2：分析 `pytest-cov` 报告，读取当前覆盖率
> - 未检测到 → 使用默认推荐（90%/80%/70%/60%）+ 加入遗留问题

---

#### 测试命名规范

**默认推荐**：

- **单元测试**：`test_<单元名>_<场景>_<预期结果>.py`
- **集成测试**：`test_<功能名>_integration.py`
- **端到端测试**：`test_<用户场景>_e2e.py`

**常见选项**：

| 风格               | 示例                                                                     | 适用            |
| ------------------ | ------------------------------------------------------------------------ | --------------- |
| 下划线分隔（推荐） | `test_user_login_invalid_password_fails.py`                              | Python/pytest   |
| 驼峰命名           | `testUserLoginInvalidPasswordFails.ts`                                   | TypeScript/Jest |
| describe-it 风格   | `describe('User', () => { it('should fail on invalid password', ...) })` | Jest/Mocha      |

**本项目设定**：

- **风格**：[待填充 / 已设定为 下划线分隔]
- **配置来源**：[代码约定 / 待设定]

---

#### 测试命令快速参考

> **注意**：以下命令根据项目技术栈动态生成

**Python (pytest)**：

```bash
pytest tests/unit/                       # 单元测试
pytest tests/integration/                # 集成测试
pytest -x                                # 首次失败即停止
pytest --cov=src --cov-report=html       # 覆盖率报告（HTML）
pytest --cov=src --cov-fail-under=80     # 覆盖率阈值检查
```

**TypeScript (Jest/Vitest)**：

```bash
npm run test:unit                        # 单元测试
npm run test:integration                 # 集成测试
npm run test:coverage                    # 覆盖率报告
npm run test:watch                       # 监听模式
```

---

### 代码组织

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高  
> **配置来源**：目录约定 / 无

#### 目录原则

**默认推荐**：

- 按**功能模块**划分，而非按文件类型
- 每个模块独立包含：`models` + `services` + `routes` + `tests`

**常见选项**：

| 组织方式               | 描述                                 | 优点             | 缺点     |
| ---------------------- | ------------------------------------ | ---------------- | -------- |
| **按功能模块（推荐）** | `user/models.py`, `user/services.py` | 模块独立，易维护 | 文件分散 |
| 按文件类型             | `models/user.py`, `services/user.py` | 类型集中         | 模块耦合 |
| 混合式                 | 核心按模块，工具按类型               | 灵活             | 约定复杂 |

**本项目设定**：

- **组织方式**：[待填充 / 已设定为 按功能模块]
- **示例结构**：
  ```
  [待填充 / 根据实际项目生成]
  ```

> **自动检测策略**：
>
> - Level 2：分析目录结构
>   - 检测 `models/`, `services/` 是否存在 → 按文件类型
>   - 检测 `user/`, `order/` 是否存在 → 按功能模块
> - 未检测到 → 使用默认推荐（按功能模块）+ 加入遗留问题

---

#### 模块职责

**默认推荐**：

| 文件                     | 职责         | 说明                    |
| ------------------------ | ------------ | ----------------------- |
| `models.py`              | 数据模型定义 | ORM 模型、Pydantic 模型 |
| `schemas.py`             | 数据传输对象 | API 请求/响应模型       |
| `services.py`            | 业务逻辑     | 核心业务处理            |
| `routes.py` / `views.py` | API 端点     | HTTP 请求处理           |
| `tests/`                 | 测试代码     | 单元测试、集成测试      |

**本项目设定**：

- **模块划分**：[待填充 / 已设定为 models/schemas/services/routes/tests]

---

#### 导入顺序（Python）

**默认推荐**：

```python
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
import requests
from fastapi import FastAPI

# 3. 本地模块
from myapp.models import User
from myapp.services import get_user
```

**工具支持**：

- `ruff` 自动处理导入顺序（`--select I`）
- `isort` 专门处理导入排序

**本项目设定**：

- **是否强制**：[待填充 / 已设定为 是（由 ruff 自动处理）]
- **配置来源**：[pyproject.toml 的 [tool.ruff] / 待设定]

---

### 依赖管理

> **配置状态**：[根据包管理器自动确定]  
> **差异程度**：中

#### 添加依赖

> **注意**：以下命令根据项目检测到的包管理器动态生成

**Python (uv)**（推荐）：

```bash
uv add requests                  # 添加生产依赖
uv add --dev pytest              # 添加开发依赖
uv add --group docs mkdocs       # 添加分组依赖
```

**Python (pip/poetry)**：

```bash
pip install requests             # 添加依赖（需手动更新 requirements.txt）
poetry add requests              # Poetry 方式
```

**Node (npm/yarn/pnpm)**：

```bash
npm install express              # 添加生产依赖
npm install -D jest              # 添加开发依赖
yarn add express                 # Yarn 方式
pnpm add express                 # pnpm 方式
```

> **自动检测策略**：
>
> - Level 1：检测 `pyproject.toml` / `uv.lock` → uv
> - Level 1：检测 `poetry.lock` → poetry
> - Level 1：检测 `package-lock.json` → npm
> - Level 1：检测 `yarn.lock` → yarn
> - Level 1：检测 `pnpm-lock.yaml` → pnpm

---

#### 更新策略

**固定建议**：

- **开发依赖**：可随时更新（评估兼容性）
- **生产依赖**：仔细评估兼容性 + 版本锁定
- **安全更新**：优先处理安全漏洞（`npm audit fix` / `pip-audit`）

---

#### 依赖审查

**常用命令**：

```bash
# Python
uv pip list --outdated           # 查看过期依赖
pip-audit                        # 安全检查

# Node
npm outdated                     # 查看过期依赖
npm audit                        # 安全检查
```

---

## 快速命令参考

> **注意**：以下命令根据项目技术栈动态生成

### 测试

**Python**：

```bash
pytest tests/unit/                       # 单元测试
pytest tests/integration/                # 集成测试
pytest -x                                # 首次失败即停止
pytest --cov=src --cov-report=html       # 覆盖率报告
pytest --cov=src --cov-fail-under=80     # 覆盖率阈值检查
```

**TypeScript**：

```bash
npm run test                             # 运行所有测试
npm run test:unit                        # 单元测试
npm run test:coverage                    # 覆盖率报告
npm run test:watch                       # 监听模式
```

---

### Lint

**Python (ruff)**：

```bash
ruff check .                     # 检查代码
ruff check --fix .               # 自动修复
ruff format .                    # 格式化代码
```

**TypeScript (ESLint + Prettier)**：

```bash
eslint src/                      # 检查代码
eslint --fix src/                # 自动修复
prettier --write "src/**/*.{ts,tsx}"  # 格式化代码
```

---

### 类型检查

**Python (mypy)**：

```bash
mypy src/                        # 类型检查
mypy src/ --strict               # 严格模式检查
```

**TypeScript (tsc)**：

```bash
tsc --noEmit                     # 类型检查（不生成文件）
```

---

## 未设定规范提示

> **生成时自动填充**：列出本次检测到的未设定规范项  
> **查看详情**：参见 [issues/TODOs.md](../issues/TODOs.md)

**待设定项目**（示例）：

| 规范项       | 差异程度 | 当前使用                    | 建议优先级 | 详细说明                 |
| ------------ | -------- | --------------------------- | ---------- | ------------------------ |
| 文件命名风格 | 高       | 默认推荐（snake_case）      | 中         | 查看 CODING_STANDARDS.md |
| 覆盖率要求   | 高       | 默认推荐（90%/80%/70%/60%） | 低         | 查看 CODING_STANDARDS.md |
| 目录组织方式 | 高       | 默认推荐（按功能模块）      | 低         | 查看 DIRECTORY.md        |

> ⚠️ **建议**：在项目初期明确设定核心规范，避免团队成员使用不同约定导致代码不一致。

---

## 详细规范链接

| 主题             | 详细文档                                                               |
| ---------------- | ---------------------------------------------------------------------- |
| **完整编码规范** | [architecture/CODING_STANDARDS.md](./architecture/CODING_STANDARDS.md) |
| **架构设计**     | [architecture/CORE.md](./architecture/CORE.md)                         |
| **技术栈说明**   | [solution/TECH_STACK.md](./solution/TECH_STACK.md)                     |
| **目录结构**     | [architecture/DIRECTORY.md](./architecture/DIRECTORY.md)               |
| **接口契约**     | [implementation/INTERFACES.md](./implementation/INTERFACES.md)         |

---

## 更新记录

| 日期       | 变更内容 | 更新人 |
| ---------- | -------- | ------ |
| YYYY-MM-DD | 初始版本 | [作者] |

---

> **提示**：本文档是摘要版，需要详细规范请跳转对应链接。开发前务必阅读相关章节。  
> **检测时间**：YYYY-MM-DD HH:MM  
> **检测工具**：project-spec skill v1.3.0
