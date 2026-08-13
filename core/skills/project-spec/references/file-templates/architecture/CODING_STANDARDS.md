# 代码规范与开发模式

> **说明**：此文件是模板，生成时根据实际项目填充。  
> **位置**：`spec/architecture/CODING_STANDARDS.md`  
> **版本**：v1.3.0

---

## 编码约束

### 命名约定

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高  
> **配置来源**：代码约定 / `.editorconfig` / 无

#### 文件命名

**默认推荐**：

| 语言/类型           | 默认推荐               | 理由       |
| ------------------- | ---------------------- | ---------- |
| Python 模块         | `snake_case.py`        | PEP 8 标准 |
| Python 包           | `snake_case/`          | PEP 8 标准 |
| TypeScript 普通文件 | `camelCase.ts`         | 社区约定   |
| React 组件          | `PascalCase.tsx`       | React 约定 |
| 配置文件            | `kebab-case.config.js` | 工具约定   |

**常见选项**：

| 语言/类型               | 选项 A                  | 选项 B                    | 选项 C                      |
| ----------------------- | ----------------------- | ------------------------- | --------------------------- |
| **Python 模块**         | snake_case.py（标准）   | camelCase.py（某些框架）  | PascalCase.py（组件化项目） |
| **Python 包**           | snake_case/（标准）     | CamelCase/（某些框架）    | -                           |
| **TypeScript 普通文件** | camelCase.ts（社区）    | snake_case.ts（服务端）   | PascalCase.ts               |
| **React 组件**          | PascalCase.tsx（React） | camelCase.tsx（某些团队） | -                           |
| **配置文件**            | kebab-case.config.js    | snake_case.config.py      | camelCase.config.js         |

**本项目设定**：

| 类型            | 本项目约定                     | 配置来源                            |
| --------------- | ------------------------------ | ----------------------------------- |
| Python 模块     | [待填充 / 已设定为 snake_case] | [.editorconfig / 代码约定 / 待设定] |
| Python 包       | [待填充 / 已设定为 snake_case] | -                                   |
| TypeScript 文件 | [待填充 / 已设定为 camelCase]  | -                                   |
| React 组件      | [待填充 / 已设定为 PascalCase] | -                                   |
| 配置文件        | [待填充 / 已设定为 kebab-case] | -                                   |

> **自动检测策略**：
>
> - **Level 1**：检测 `.editorconfig` 文件
>   - `[*.{py}]` → 读取命名约定
>   - `[*.{ts,tsx}]` → 读取命名约定
> - **Level 2**：分析现有文件名
>   - 统计 `.py` 文件名中 snake_case/camelCase/PascalCase 比例
>   - 取众数作为推断结果
> - **未检测到** → 使用默认推荐 + 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] 文件命名约定`
>   - 优先级：中

---

#### 变量命名

**默认推荐**：

| 类型                   | 默认推荐                     | 示例                                 |
| ---------------------- | ---------------------------- | ------------------------------------ |
| **常量**               | `UPPER_SNAKE_CASE`           | `MAX_CONNECTIONS`, `DEFAULT_TIMEOUT` |
| **变量（Python）**     | `snake_case`                 | `user_count`, `total_price`          |
| **变量（TypeScript）** | `camelCase`                  | `userCount`, `totalPrice`            |
| **私有变量**           | `_leading_underscore`        | `_internal_cache`, `_temp_value`     |
| **保护变量（Python）** | `_single_leading_underscore` | `_protected_attr`                    |

**本项目设定**：

| 类型               | 本项目约定                     | 配置来源 |
| ------------------ | ------------------------------ | -------- |
| 常量               | [已设定为 UPPER_SNAKE_CASE]    | 业界统一 |
| 变量（Python）     | [已设定为 snake_case]          | PEP 8    |
| 变量（TypeScript） | [已设定为 camelCase]           | 社区约定 |
| 私有变量           | [已设定为 _leading_underscore] | 业界统一 |

> **差异程度**：中（大部分项目一致，可简化处理）

---

#### 函数命名

**默认推荐**：

| 类型                        | 默认推荐                     | 示例                                      |
| --------------------------- | ---------------------------- | ----------------------------------------- |
| **函数/方法（Python）**     | `verb_noun`                  | `get_user_by_id`, `calculate_total_price` |
| **函数/方法（TypeScript）** | `verbNoun`                   | `getUserById`, `calculateTotalPrice`      |
| **类**                      | `PascalCase`                 | `UserService`, `OrderController`          |
| **测试函数**                | `test_<scenario>_<expected>` | `test_login_invalid_password_fails`       |

**常见选项**：

| 类型                | 选项 A（推荐）         | 选项 B                | 选项 C               |
| ------------------- | ---------------------- | --------------------- | -------------------- |
| **Python 函数**     | verb_noun（标准）      | noun_verb（某些框架） | camelCase（不推荐）  |
| **TypeScript 函数** | verbNoun（社区）       | verb_noun（部分团队） | -                    |
| **测试函数**        | test_scenario_expected | test_scenarioExpected | testScenarioExpected |

**本项目设定**：

| 类型               | 本项目约定                                 | 配置来源            |
| ------------------ | ------------------------------------------ | ------------------- |
| 函数（Python）     | [已设定为 verb_noun]                       | PEP 8               |
| 函数（TypeScript） | [已设定为 verbNoun]                        | 社区约定            |
| 类                 | [已设定为 PascalCase]                      | 业界统一            |
| 测试函数           | [待填充 / 已设定为 test_scenario_expected] | [代码约定 / 待设定] |

---

#### 命名禁用

**固定规则**（不建议修改）：

| 禁用项                | 理由                                  | 示例                                             |
| --------------------- | ------------------------------------- | ------------------------------------------------ |
| 单字母变量            | 含义不明（循环计数器 `i, j, k` 除外） | `x = 1` ❌ → `index = 1` ✅                      |
| 含义不明的缩写        | 降低可读性                            | `cnt` ❌ → `count` ✅, `tmp` ❌ → `temporary` ✅ |
| 保留字作为变量名      | 语法错误或混淆                        | `class = "foo"` ❌ → `klass = "foo"` ✅          |
| Python 混淆下划线前缀 | 可能与魔法方法冲突                    | `__dunder__` ❌                                  |

---

### 类型约束

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高（不同项目对类型提示要求不同）  
> **配置来源**：`pyproject.toml` / `tsconfig.json` / 无

#### Python 类型提示

**默认推荐**：

| 场景             | 默认推荐         | 理由                             |
| ---------------- | ---------------- | -------------------------------- |
| 核心业务逻辑函数 | **必须**类型提示 | 提升代码质量和可维护性           |
| API 端点函数     | **必须**类型提示 | FastAPI 自动生成文档依赖类型提示 |
| 工具函数         | 推荐类型提示     | 可选，但推荐                     |
| 配置/模型类      | **必须**类型提示 | Pydantic 模型需定义所有字段类型  |

**常见选项**：

| 策略                     | 描述                           | 适用项目             |
| ------------------------ | ------------------------------ | -------------------- |
| **严格类型提示（推荐）** | 所有函数参数和返回值都类型提示 | 新项目、大型项目     |
| 部分类型提示             | 仅核心函数类型提示             | 遗留项目、快速原型   |
| 无类型提示               | 不使用类型提示                 | 小型脚本、一次性脚本 |

**本项目设定**：

| 场景             | 本项目要求                       | 配置来源                          |
| ---------------- | -------------------------------- | --------------------------------- |
| 核心业务逻辑函数 | [待填充 / 已设定为 必须类型提示] | [pyproject.toml / team agreement] |
| API 端点函数     | [待填充 / 已设定为 必须类型提示] | -                                 |
| 工具函数         | [待填充 / 已设定为 推荐类型提示] | -                                 |
| 配置/模型类      | [待填充 / 已设定为 必须类型提示] | -                                 |

**语法选择**（Python 版本相关）：

| Python 版本      | 推荐语法       | 示例                                       |
| ---------------- | -------------- | ------------------------------------------ |
| **Python 3.10+** | 新语法（推荐） | `def get_user(id: int) -> User \| None:`   |
| Python 3.7-3.9   | typing 模块    | `def get_user(id: int) -> Optional[User]:` |

**本项目设定**：

- **Python 版本**：[待填充 / 已设定为 3.10+]
- **语法风格**：[待填充 / 已设定为 新语法]

> **自动检测策略**：
>
> - **Level 1**：读取 `pyproject.toml` → `[project] python = "3.10"` 或 `requires-python`
> - **Level 2**：分析现有代码
>   - 统计带有类型提示的函数比例
>   - 检测是否使用 `|` 联合语法（Python 3.10+）
> - **未检测到** → 使用默认推荐（必须类型提示，新语法）+ 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] 类型提示策略`
>   - 优先级：中

---

#### TypeScript 类型

**默认推荐**：

| 配置项           | 默认推荐                       | 理由             |
| ---------------- | ------------------------------ | ---------------- |
| **严格模式**     | `strict: true`                 | 提供最强类型检查 |
| **any 使用策略** | 避免使用 `any`，优先 `unknown` | `unknown` 更安全 |
| **数据结构定义** | 使用 `interface` 或 `type`     | 明确数据契约     |

**常见选项**：

| 配置项             | 选项 A（推荐）       | 选项 B               | 选项 C               |
| ------------------ | -------------------- | -------------------- | -------------------- |
| **严格模式**       | strict: true         | strict: false        | 部分开启             |
| **any 策略**       | 禁用 any，用 unknown | 允许 any（遗留代码） | 允许 any（快速原型） |
| **interface/type** | interface（对象）    | type（联合/交叉）    | 混合使用             |

**本项目设定**：

| 配置项       | 本项目设定                               | 配置来源        |
| ------------ | ---------------------------------------- | --------------- |
| 严格模式     | [待填充 / 已设定为 strict: true]         | [tsconfig.json] |
| any 使用策略 | [待填充 / 已设定为 禁用 any，用 unknown] | [tsconfig.json] |
| 数据结构定义 | [待填充 / 已设定为 interface]            | [代码约定]      |

> **自动检测策略**：
>
> - **Level 1**：读取 `tsconfig.json`
>   - `compilerOptions.strict` → 严格模式
>   - `compilerOptions.noImplicitAny` → any 策略
> - **未检测到** → 使用默认推荐（strict: true）+ 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] TypeScript 严格模式配置`
>   - 优先级：中

---

### 格式规范

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高（行长度、工具选择差异大）  
> **配置来源**：`pyproject.toml` / `.prettierrc` / `.editorconfig` / 无

#### Python 格式化

**工具选择**：

| 工具             | 特点                                | 推荐程度 |
| ---------------- | ----------------------------------- | -------- |
| **ruff（推荐）** | 替代 black + isort + flake8，速度快 | ★★★      |
| black            | PEP 8 兼容，固执己见                | ★★☆      |
| yapf             | Google 开发，可配置                 | ★★☆      |
| autopep8         | PEP 8 格式化                        | ★☆☆      |

**行长度**：

| 行长度         | 来源       | 适用场景         |
| -------------- | ---------- | ---------------- |
| **88（推荐）** | black 默认 | 现代 Python 项目 |
| 79             | PEP 8 标准 | 传统项目         |
| 100            | 社区习惯   | 宽屏开发         |
| 120            | 部分团队   | 超宽屏           |

**本项目设定**：

| 配置项     | 本项目设定                    | 配置来源                                 |
| ---------- | ----------------------------- | ---------------------------------------- |
| 格式化工具 | [待填充 / 已设定为 ruff]      | [pyproject.toml]                         |
| 行长度     | [待填充 / 已设定为 88]        | [pyproject.toml [tool.ruff] line-length] |
| 导入顺序   | [已设定为 标准库→第三方→本地] | [ruff 自动处理]                          |

**配置示例**（pyproject.toml）：

```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I"]  # E: pycodestyle, F: pyflakes, I: isort

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

> **自动检测策略**：
>
> - **Level 1**：读取 `pyproject.toml`
>   - 检测 `[tool.ruff]` → ruff
>   - 检测 `[tool.black]` → black
>   - 读取 `line-length`
> - **Level 2**：分析现有代码
>   - 读取最长行作为推断
> - **未检测到** → 使用默认推荐（ruff, 88）+ 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] Python 格式化配置`
>   - 优先级：中

---

#### TypeScript 格式化

**工具选择**：

| 工具                 | 特点               | 推荐程度 |
| -------------------- | ------------------ | -------- |
| **Prettier（推荐）** | 执固执见，社区标准 | ★★★      |
| dprint               | 可配置，速度快     | ★★☆      |

**行长度**：

| 行长度          | 来源          | 适用场景 |
| --------------- | ------------- | -------- |
| **80**          | Prettier 默认 | 标准     |
| **100（推荐）** | 社区习惯      | 宽屏开发 |
| 120             | 部分团队      | 超宽屏   |

**分号使用**：

| 选项             | 描述       | 适用            |
| ---------------- | ---------- | --------------- |
| **可选（推荐）** | 不强制分号 | 现代 JavaScript |
| 必需             | 强制分号   | 传统代码风格    |

**本项目设定**：

| 配置项     | 本项目设定                   | 配置来源                 |
| ---------- | ---------------------------- | ------------------------ |
| 格式化工具 | [待填充 / 已设定为 Prettier] | [.prettierrc]            |
| 行长度     | [待填充 / 已设定为 100]      | [.prettierrc printWidth] |
| 分号使用   | [待填充 / 已设定为 可选]     | [.prettierrc semi]       |

**配置示例**（.prettierrc）：

```json
{
  "printWidth": 100,
  "semi": false,
  "singleQuote": true,
  "trailingComma": "es5"
}
```

> **自动检测策略**：
>
> - **Level 1**：读取 `.prettierrc` 或 `package.json` 中 `prettier` 配置
>   - 读取 `printWidth`
>   - 读取 `semi`
> - **未检测到** → 使用默认推荐（Prettier, 100, 可选）+ 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] TypeScript 格式化配置`
>   - 优先级：中

---

#### Markdown 格式化

**固定规则**（差异程度低）：

| 规则           | 说明                                         |
| -------------- | -------------------------------------------- |
| 标题层级连续   | 不跳级（如 H1 → H3）                         |
| 代码块指定语言 | \`\`\`python 而不是 \`\`\`                   |
| 链接优先引用式 | `[text][ref]` 优于 `[text](url)`（长文档时） |

---

## 开发模式

### 测试驱动开发 (TDD)

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高（覆盖率要求差异大）  
> **配置来源**：`pyproject.toml` / team agreement / 无

#### 标准流程

**固定流程**（不建议修改）：

```
红：写失败测试 → 绿：写最简实现 → 重构：优化代码 → 重复
```

---

#### 测试覆盖率要求

**默认推荐**：

| 代码类型     | 默认推荐 | 理由                   |
| ------------ | -------- | ---------------------- |
| 核心业务逻辑 | ≥ 90%    | 关键逻辑必须充分测试   |
| API 端点     | ≥ 80%    | API 层易出错，需要覆盖 |
| 工具函数     | ≥ 70%    | 辅助功能，可适当降低   |
| 配置/模型    | ≥ 60%    | 主要是数据定义         |

**常见范围**：

| 代码类型     | 宽松 | 标准（推荐） | 严格 |
| ------------ | ---- | ------------ | ---- |
| 核心业务逻辑 | 80%  | **90%**      | 95%  |
| API 端点     | 70%  | **80%**      | 90%  |
| 工具函数     | 60%  | **70%**      | 80%  |
| 配置/模型    | 50%  | **60%**      | 70%  |

**本项目设定**：

| 代码类型     | 本项目要求              | 配置来源                          |
| ------------ | ----------------------- | --------------------------------- |
| 核心业务逻辑 | [待填充 / 已设定为 XX%] | [pyproject.toml / team agreement] |
| API 端点     | [待填充 / 已设定为 XX%] | -                                 |
| 工具函数     | [待填充 / 已设定为 XX%] | -                                 |
| 配置/模型    | [待填充 / 已设定为 XX%] | -                                 |

**配置示例**（pyproject.toml）：

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-fail-under=80"
testpaths = ["tests"]
```

> **自动检测策略**：
>
> - **Level 1**：读取 `pyproject.toml`
>   - `[tool.pytest.ini_options] addopts` → 提取 `--cov-fail-under=XX`
> - **Level 2**：运行 `pytest --cov` 获取当前覆盖率
> - **未检测到** → 使用默认推荐（90%/80%/70%/60%）+ 加入遗留问题
>   - 问题编号：`AR-XXX: [待明确] 测试覆盖率要求`
>   - 优先级：低

---

#### 测试命名规范

**默认推荐**：

| 测试类型       | 命名规范                             | 示例                                        |
| -------------- | ------------------------------------ | ------------------------------------------- |
| **单元测试**   | `test_<单元名>_<场景>_<预期结果>.py` | `test_user_login_invalid_password_fails.py` |
| **集成测试**   | `test_<功能名>_integration.py`       | `test_user_auth_integration.py`             |
| **端到端测试** | `test_<用户场景>_e2e.py`             | `test_checkout_flow_e2e.py`                 |

**常见选项**：

| 风格                   | 示例                                                 | 适用框架   |
| ---------------------- | ---------------------------------------------------- | ---------- |
| **下划线分隔（推荐）** | `test_user_login_invalid_password_fails.py`          | pytest     |
| 驼峰命名               | `testUserLoginInvalidPasswordFails.ts`               | Jest       |
| describe-it 风格       | `describe('User', () => { it('should fail', ...) })` | Jest/Mocha |

**本项目设定**：

| 类型     | 本项目设定                     | 配置来源            |
| -------- | ------------------------------ | ------------------- |
| 命名风格 | [待填充 / 已设定为 下划线分隔] | [代码约定 / 待设定] |

---

#### 测试组织

**默认推荐**：

```
tests/
├── unit/              # 单元测试
│   ├── test_user_service.py
│   └── test_order_service.py
├── integration/       # 集成测试
│   └── test_user_auth_integration.py
├── e2e/               # 端到端测试
│   └── test_checkout_flow_e2e.py
└── fixtures/          # 测试数据和 fixtures
    ├── user_data.json
    └── conftest.py
```

**本项目设定**：

- **组织方式**：[待填充 / 已设定为 按类型分离（unit/integration/e2e）]

---

### 规范驱动开发 (SDD)

#### 设计文档先行

**固定流程**：

| 阶段     | 产出文档                       | 说明         |
| -------- | ------------------------------ | ------------ |
| 需求分析 | `spec/product/REQUIREMENTS.md` | 明确做什么   |
| 方案设计 | `spec/solution/*.md`           | 明确怎么做   |
| 实施验证 | 设计评审 → 实施前检查          | 确保设计完整 |

#### 围栏机制（如有 AET）

**说明**：

- 明确可修改文件范围（`.aet/fences/`）
- 保护核心模块不被随意修改
- 适用：Agentic Engineering Team 框架

---

### Git 工作流

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：中（团队习惯差异大）  
> **配置来源**：team agreement / CONTRIBUTING.md / 无

#### 分支命名

**默认推荐**：

| 分支类型     | 命名格式                           | 示例                        |
| ------------ | ---------------------------------- | --------------------------- |
| **功能分支** | `feature/<issue-id>-<description>` | `feature/123-user-auth`     |
| **Bug 修复** | `fix/<issue-id>-<description>`     | `fix/456-login-crash`       |
| **重构**     | `refactor/<description>`           | `refactor/user-service`     |
| **发布**     | `release/v<version>`               | `release/v1.2.0`            |
| **热修复**   | `hotfix/<issue-id>-<description>`  | `hotfix/789-security-patch` |

**常见选项**：

| 工作流               | 分支策略                              | 适用               |
| -------------------- | ------------------------------------- | ------------------ |
| **Git Flow（推荐）** | feature/develop/master/release/hotfix | 中大型项目         |
| GitHub Flow          | feature/main（简化版）                | 小型项目、持续部署 |
| Trunk Based          | 直接在 main 分支工作                  | 快速迭代           |

**本项目设定**：

| 分支类型 | 本项目约定                              | 配置来源          |
| -------- | --------------------------------------- | ----------------- |
| 功能分支 | [已设定为 feature/issue-id-description] | [CONTRIBUTING.md] |
| 工作流   | [待填充 / 已设定为 Git Flow]            | [team agreement]  |

> **差异程度**：中（团队习惯差异大，不强制）  
> **遗留问题优先级**：低

---

#### Commit 消息规范

**默认推荐**：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**type 列表**：

| type       | 说明     | 示例                         |
| ---------- | -------- | ---------------------------- |
| `feat`     | 新功能   | feat(auth): 添加 JWT 认证    |
| `fix`      | Bug 修复 | fix(login): 修复登录崩溃     |
| `refactor` | 重构     | refactor(user): 重构用户服务 |
| `docs`     | 文档     | docs: 更新 README            |
| `test`     | 测试     | test(user): 添加用户服务测试 |
| `chore`    | 杂项     | chore: 更新依赖              |
| `perf`     | 性能优化 | perf(query): 优化查询性能    |
| `style`    | 格式     | style: 格式化代码            |

**示例**：

```
feat(auth): 添加 JWT 认证支持

- 实现 token 生成和验证
- 添加认证中间件
- 更新 API 文档

Closes #123
```

**本项目设定**：

- **Commit 规范**：[待填充 / 已设定为 Conventional Commits]
- **工具支持**：[commitlint / commitizen / 无]

---

## 构建规范

### 测试规范

> **注意**：以下命令根据项目检测到的技术栈动态生成

#### 单元测试

**Python**：

```bash
pytest tests/unit/                       # 运行单元测试
pytest tests/unit/ -v                    # 详细输出
pytest tests/unit/ -k "test_user"        # 匹配特定测试
```

**TypeScript**：

```bash
npm run test:unit                        # 运行单元测试
npm run test:unit -- --watch             # 监听模式
```

---

#### 集成测试

**Python**：

```bash
pytest tests/integration/                # 运行集成测试
```

**TypeScript**：

```bash
npm run test:integration                 # 运行集成测试
```

---

#### 覆盖率报告

**Python**：

```bash
pytest --cov=src --cov-report=html       # 生成 HTML 报告
pytest --cov=src --cov-report=term-missing  # 显示未覆盖行
pytest --cov=src --cov-fail-under=80     # 覆盖率阈值检查
```

**TypeScript**：

```bash
npm run test:coverage                    # 生成覆盖率报告
```

---

### Lint 规范

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高（工具选择差异大）

#### Python Lint

**工具选择**：

| 工具                   | 特点                                | 推荐程度 |
| ---------------------- | ----------------------------------- | -------- |
| **ruff（推荐）**       | 替代 flake8 + isort + black，速度快 | ★★★      |
| flake8 + isort + black | 传统组合                            | ★★☆      |
| pylint                 | 功能全面，严格                      | ★★☆      |

**本项目设定**：

| 配置项    | 本项目设定               | 配置来源         |
| --------- | ------------------------ | ---------------- |
| Lint 工具 | [待填充 / 已设定为 ruff] | [pyproject.toml] |

**命令参考**：

```bash
ruff check .                     # 检查代码
ruff check --fix .               # 自动修复
ruff format .                    # 格式化代码
ruff format --check .            # 检查格式
```

---

#### TypeScript Lint

**工具选择**：

| 工具               | 特点               | 推荐程度 |
| ------------------ | ------------------ | -------- |
| **ESLint（推荐）** | 社区标准，插件丰富 | ★★★      |

**本项目设定**：

| 配置项    | 本项目设定                 | 配置来源    |
| --------- | -------------------------- | ----------- |
| Lint 工具 | [待填充 / 已设定为 ESLint] | [.eslintrc] |

**命令参考**：

```bash
eslint src/                      # 检查代码
eslint --fix src/                # 自动修复
prettier --write "src/**/*.{ts,tsx}"  # 格式化代码
```

---

### CI 规范

> **配置状态**：[待项目自定义 / 已设定]  
> **差异程度**：高（CI 流程因项目而异）

#### 必须通过的检查

**默认推荐**：

- [ ] 单元测试通过
- [ ] 集成测试通过（如有）
- [ ] Lint 检查通过
- [ ] 类型检查通过
- [ ] 构建成功

**本项目设定**：

| 检查项   | 本项目要求               | 配置来源                   |
| -------- | ------------------------ | -------------------------- |
| 单元测试 | [待填充 / 已设定为 必须] | [.github/workflows/ci.yml] |
| 集成测试 | [待填充 / 已设定为 必须] | -                          |
| Lint     | [待填充 / 已设定为 必须] | -                          |
| 类型检查 | [待填充 / 已设定为 必须] | -                          |
| 构建     | [待填充 / 已设定为 必须] | -                          |

---

#### CI 流程示例

**GitHub Actions 示例**：

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy src/

      - name: Run tests
        run: pytest --cov=src --cov-fail-under=80
```

---

## 未设定规范汇总

> **生成时自动填充**：列出本次检测到的所有未设定规范项  
> **详细说明**：参见 [issues/TODOs.md](../issues/TODOs.md)

**待设定项列表**（生成时填充）：

| 规范项               | 差异程度 | 推荐值                               | 优先级 | 配置方式        |
| -------------------- | -------- | ------------------------------------ | ------ | --------------- |
| 文件命名风格         | 高       | snake_case (Python) / camelCase (TS) | 中     | .editorconfig   |
| 行长度（Python）     | 高       | 88                                   | 中     | pyproject.toml  |
| 行长度（TypeScript） | 高       | 100                                  | 中     | .prettierrc     |
| 类型提示策略         | 高       | 必须类型提示（核心逻辑）             | 中     | team agreement  |
| 覆盖率要求           | 高       | 90%/80%/70%/60%                      | 低     | pyproject.toml  |
| 测试命名风格         | 中       | test_scenario_expected               | 低     | 代码约定        |
| 目录组织方式         | 高       | 按功能模块                           | 低     | team agreement  |
| Git 工作流           | 中       | Git Flow                             | 低     | CONTRIBUTING.md |

---

## 配置文件示例

### Python 项目配置示例（pyproject.toml）

```toml
[project]
name = "my-project"
requires-python = ">=3.10"

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
addopts = "--cov=src --cov-fail-under=80"
testpaths = ["tests"]

[tool.mypy]
strict = true
```

---

### TypeScript 项目配置示例

**tsconfig.json**：

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "esModuleInterop": true
  }
}
```

**.prettierrc**：

```json
{
  "printWidth": 100,
  "semi": false,
  "singleQuote": true
}
```

---

### 通用配置示例（.editorconfig）

```ini
# EditorConfig - 跨编辑器统一配置
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.py]
indent_style = space
indent_size = 4

[*.{ts,tsx,js,jsx}]
indent_style = space
indent_size = 2

[*.md]
trim_trailing_whitespace = false
```

---

## 注意事项

- **一致性**：整个项目严格遵循统一规范
- **可维护性**：规范变更需更新本文档并通知全员
- **工具支持**：优先使用自动化工具（ruff, prettier, mypy 等）
- **例外处理**：特殊情况需注释说明原因
- **渐进式采用**：遗留项目可逐步引入规范，不必一次性全面改造

---

## 相关文档

- [coding-index.md](../coding-index.md) - 开发必读索引（摘要版）
- [TECH_STACK.md](../solution/TECH_STACK.md) - 技术栈说明
- [DIRECTORY.md](./DIRECTORY.md) - 目录结构

---

## 更新记录

| 日期       | 变更内容           | 更新人 |
| ---------- | ------------------ | ------ |
| YYYY-MM-DD | 初始版本（v1.3.0） | [作者] |

---

> **提示**：本文档包含完整编码规范，摘要版参见 [coding-index.md](../coding-index.md)。  
> **检测时间**：YYYY-MM-DD HH:MM  
> **检测工具**：project-spec skill v1.3.0
