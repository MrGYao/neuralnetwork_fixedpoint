# Python Code Check Skill

Python代码质量检查工具，支持四种检查模式：

- **模块检查**: 检查指定模块/目录
- **增量检查**: 检查未提交修改（Working Directory Diff）
- **Commit对比**: 检查最近N次提交的差异
- **全量检查**: 检查所有Python文件

## 快速开始

```bash
# Windows
run.bat src/agent_sdk              # 检查指定模块
run.bat --mode=full                # 全量检查
run.bat --mode=commit-diff -c 2    # 对比最近2次提交

# Unix (Linux/macOS)
./run.sh src/agent_sdk             # 检查指定模块
./run.sh --mode=full               # 全量检查

# 或直接Python运行
python run.py src/agent_sdk
python -m python_codecheck.main --help
```

## 检查模式

| 模式        | 说明           | 触发方式                               |
| ----------- | -------------- | -------------------------------------- |
| module      | 检查指定路径   | `CHECK_PATHS=src/agent_sdk` 或位置参数 |
| incremental | 检查未提交修改 | Git仓库+有修改（默认）                 |
| commit-diff | 对比最近提交   | `MODE=commit-diff COMPARE_N=2`         |
| full        | 检查所有文件   | `MODE=full` 或非Git仓库                |

## 参数优先级

```
环境变量 > CLI参数 > 默认值
```

| 参数     | 环境变量         | CLI参数             | 默认值   |
| -------- | ---------------- | ------------------- | -------- |
| 检查模式 | `MODE`           | `--mode`            | 自动推断 |
| 检查路径 | `CHECK_PATHS`    | 位置参数            | 无       |
| 对比范围 | `COMPARE_N`      | `--compare-commits` | 1        |
| 最大迭代 | `MAX_ITERATIONS` | `--max-iterations`  | 6        |
| 自动提交 | `AUTO_COMMIT`    | `--no-auto-commit`  | true     |

## 详细文档

见 [SKILL.md](./SKILL.md)
