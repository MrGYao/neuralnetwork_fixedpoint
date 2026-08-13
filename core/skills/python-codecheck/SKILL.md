---
name: python-codecheck
description: Python代码质量检查工具（mypy类型检查 + ruff格式化/lint）。支持四种检查模式：模块检查（指定路径）、增量检查（未提交修改）、Commit对比（最近N次提交）、全量检查。当用户提及"代码质量"、"类型检查"、"mypy"、"ruff"、"代码检查"、"lint"、"格式化Python代码"、"检查代码质量"、或指定Python模块/目录需要检查时，使用此skill。特别适用于：检查特定模块质量、验证提交前代码、检查最近提交的问题、开发中的增量验证。跨平台支持（Windows/macOS/Linux）。
---

# Python代码质量检查

> **版本**: v2.0.0  
> **更新时间**: 2026-01-26  
> **状态**: 优化完成 - 支持模块/增量/commit-diff/全量四种模式

---

## 触发场景

### 场景1: 用户指定模块检查

**典型调用**: "检查 `src/agent_sdk` 的代码质量"

| 条件                              | 行为                |
| --------------------------------- | ------------------- |
| 指定路径（CLI位置参数或环境变量） | 仅检查指定模块/目录 |

**示例**:

```bash
# CLI位置参数
python -m python_codecheck.main src/agent_sdk apps/backend

# 环境变量（agent调用友好）
CHECK_PATHS="src/agent_sdk,apps/backend" python -m python_codecheck.main
```

### 场景2: Git仓库，已全量提交

**条件**: 工作区干净，无未提交修改

| 默认行为                       | 可选行为                          |
| ------------------------------ | --------------------------------- |
| **全量检查**（所有Python文件） | `MODE=commit-diff` 对比最近提交   |
|                                | `MODE=incremental` 检查最新commit |
|                                | `CHECK_PATHS` 仅检查特定模块      |

**示例**:

```bash
# 默认：全量检查
python -m python_codecheck.main

# 对比最近2次提交
MODE=commit-diff COMPARE_N=2 python -m python_codecheck.main

# 仅检查特定模块
CHECK_PATHS=src/agent_sdk python -m python_codecheck.main
```

### 场景3: Git仓库，有未提交修改

**条件**: 开发中，有pending changes

| 默认行为                                       | 可选行为                     |
| ---------------------------------------------- | ---------------------------- |
| **Working Directory Diff**（检查所有修改文件） | `MODE=full` 全量检查         |
|                                                | `CHECK_PATHS` 仅检查特定模块 |

**示例**:

```bash
# 默认：检查未提交修改
python -m python_codecheck.main

# 全量检查
MODE=full python -m python_codecheck.main

# 仅检查特定模块
CHECK_PATHS=apps/backend python -m python_codecheck.main
```

### 场景4: 非Git项目

**条件**: 不在Git仓库中

| 默认行为                       |
| ------------------------------ |
| **全量检查**（所有Python文件） |

---

## 检查模式详解

| 模式          | 说明       | 检查范围               | 触发条件                    |
| ------------- | ---------- | ---------------------- | --------------------------- |
| `module`      | 模块检查   | 用户指定路径           | `CHECK_PATHS` 或CLI位置参数 |
| `incremental` | 增量检查   | Working Directory Diff | Git仓库+有未提交修改        |
| `commit-diff` | Commit对比 | HEAD~N vs HEAD         | `MODE=commit-diff`          |
| `full`        | 全量检查   | 所有Python文件         | Git仓库+无修改，或非Git     |

---

## 参数说明

### 优先级

```
环境变量 > CLI参数 > 默认值
```

### 参数表

| 参数     | 环境变量         | CLI参数             | 默认值   | 说明                                |
| -------- | ---------------- | ------------------- | -------- | ----------------------------------- |
| 检查模式 | `MODE`           | `--mode`            | 自动推断 | module/incremental/full/commit-diff |
| 检查路径 | `CHECK_PATHS`    | 位置参数            | 无       | 逗号分隔的路径列表                  |
| 对比范围 | `COMPARE_N`      | `--compare-commits` | 1        | 对比最近N次提交                     |
| 最大迭代 | `MAX_ITERATIONS` | `--max-iterations`  | 6        | 修复循环次数                        |
| 自动提交 | `AUTO_COMMIT`    | `--no-auto-commit`  | true     | 是否自动提交修复                    |

### 使用示例

```bash
# 方式1: CLI参数
python -m python_codecheck.main src/agent_sdk --mode=module
python -m python_codecheck.main --mode=commit-diff --compare-commits=2
python -m python_codecheck.main --mode=full --max-iterations=10

# 方式2: 环境变量（agent调用友好）
MODE=module CHECK_PATHS="src/agent_sdk,apps/backend" python -m python_codecheck.main
MODE=commit-diff COMPARE_N=2 python -m python_codecheck.main
MODE=full MAX_ITERATIONS=10 python -m python_codecheck.main

# 方式3: 便捷脚本
./run.sh src/agent_sdk --mode=module
run.bat --mode=full

# 方式4: Windows PowerShell
$env:MODE = "commit-diff"
$env:COMPARE_N = "2"
python -m python_codecheck.main
```

---

## 工作流程

```
1. 参数解析 → 环境变量 > CLI > 默认值
2. 模式推断 → Git状态 + 用户参数
3. 工具检测 → mypy/ruff，缺失则安装
4. 配置注入 → ruff配置（如有需要）
5. 质量循环 → mypy → ruff format → ruff check（最多N次）
   ├─ mypy失败 → 退出提示用户
   ├─ ruff失败 → 自动修复 → 提交 → 重新检查
   └─ 全部通过 → 完成
6. 审视结果 → 提示用户审视修复内容
```

---

## 配置注入策略

| 场景                            | 操作                     |
| ------------------------------- | ------------------------ |
| 有pyproject.toml，无[tool.ruff] | 追加配置                 |
| 有pyproject.toml，有[tool.ruff] | 使用现有配置，跳过注入   |
| 无pyproject.toml                | 创建.ruff.toml独立配置   |
| 包名检测                        | 从[project.name]自动读取 |

---

## 默认配置

### ruff配置

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM"]
ignore = ["E501", "B008", "W191"]

[tool.ruff.lint.isort]
known-first-party = ["your_package_name"]  # 自动检测

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### mypy配置（可选）

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
ignore_missing_imports = true
```

---

## 跨平台支持

✅ **Windows** (PowerShell/CMD)  
✅ **macOS** (bash/zsh)  
✅ **Linux** (bash/fish)

**实现**: 纯Python实现，无bash依赖

---

## 依赖

### 必需

- **Python 3.11+** (使用标准库tomllib)
- **uv** (Python包管理器) - https://docs.astral.sh/uv/
- **git** (版本控制，用于增量/commit-diff模式)

### 自动安装

- **mypy** (类型检查)
- **ruff** (格式化 + Linter)

### Skill依赖

- **tomli-w** (TOML写入，自动安装)

---

## 安装

### Skill位置

```
C:\Users\catch\.agents\skills\python-codecheck\
├── SKILL.md                 # 本文档
├── README.md                # 快速开始
├── run.py / .bat / .sh      # 便捷启动脚本
├── pyproject.toml           # 依赖定义
└── src/python_codecheck/
    ├── main.py              # 主入口
    ├── args.py              # 参数解析
    ├── checker.py           # 公共检查器
    ├── check_module.py      # 模块检查
    ├── check_incremental.py # 增量检查
    ├── check_commit_diff.py # Commit对比
    ├── check_full.py        # 全量检查
    ├── check_tools.py       # 工具检测安装
    ├── inject_config.py     # 配置注入
    ├── find_project_root.py # 项目根检测
    ├── commit_fix.py        # Git提交
    └── utils.py             # 工具函数
```

### 验证安装

```bash
cd ~/.agents/skills/python-codecheck
python -m python_codecheck.main --help
```

---

## monorepo支持

自动识别uv workspace定义：

```toml
[tool.uv.workspace]
members = [
    "apps/web/backend",
    "packages/agent_sdk",
]
```

全量模式会对每个workspace member执行独立检查。

---

## 故障排查

### 问题1: uv未安装

```
❌ uv未安装，请先安装uv
```

**解决**: 安装uv - https://docs.astral.sh/uv/getting-started/installation/

### 问题2: Python版本过低

```
ModuleNotFoundError: No module named 'tomllib'
```

**解决**: 升级到Python 3.11+

### 问题3: Git历史不足

```
⚠️  提交历史不足2次，当前1次提交
```

**解决**: 正常行为，自动调整为可用历史

### 问题4: 路径不存在

```
⚠️  路径不存在: src/nonexistent
```

**解决**: 检查路径拼写，确保路径存在

---

## 示例输出

### 模块检查模式

```
╔════════════════════════════════════╗
║    Python代码质量检查   ║
╚════════════════════════════════════╝

Step 1/4: 工具检测与安装
✅ mypy 已安装: v2.3.0
✅ ruff 已安装: v0.16.0

Step 2/4: 配置注入
✅ 已有ruff配置，跳过注入

Step 3/4: 检查模式确认
  模式: module
  Git仓库: 是
  未提交修改: 是
  指定路径: src/agent_sdk

Step 4/4: 质量检查循环 (最多6次)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迭代 1/6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=== 模块检查模式 ===
检查文件: 24 个
  - src/agent_sdk/core/base_tool.py
  - src/agent_sdk/core/tool_registry.py
  ...

项目: agent_sdk
  [1/3] mypy类型检查...
    ✅ mypy检查通过
  [2/3] ruff格式化...
    ✅ 已格式化
  [3/3] ruff检查...
    ✅ ruff检查通过

╔════════════════════════════════════╗
║      ✅ 质量检查通过               ║
╚════════════════════════════════════╝
```

### Commit对比模式

```
Step 3/4: 检查模式确认
  模式: commit-diff
  Git仓库: 是
  未提交修改: 否
  对比范围: 最近2次提交

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迭代 1/6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=== Commit对比检查模式 ===
对比范围: HEAD~2..HEAD

最近提交:
  4364a66 feat: 重构参数解析
  8532006 feat: 新增模块检查

检查文件: 5 个
  - src/python_codecheck/args.py
  - src/python_codecheck/main.py
  ...
```

---

## 相关文档

- [ruff官方文档](https://docs.astral.sh/ruff/)
- [mypy官方文档](https://mypy.readthedocs.io/)
- [uv官方文档](https://docs.astral.sh/uv/)

---

## 更新日志

### v2.0.0 (2026-01-26)

- ✅ 重命名: python-qa → python-codecheck
- ✅ 新增模块检查模式（`check_module.py`）
- ✅ 新增Commit对比模式（`check_commit_diff.py`）
- ✅ 参数优先级: 环境变量 > CLI > 默认值
- ✅ 重构公共检查器（`checker.py`）
- ✅ 优化触发场景分类（4种场景）
- ✅ 支持路径排除和monorepo

### v1.0.0 (2026-01-26)

- ✅ 跨平台Python实现（替代bash脚本）
- ✅ 自动工具安装
- ✅ 配置自动注入
- ✅ monorepo支持
- ✅ 迭代修复机制（最多6次）
