# Submodule 管理详细场景文档

本文档包含前置检查详细命令、场景判断逻辑、边缘场景处理等内容。

---

## 前置检查详细命令

执行任何操作前，必须检查以下项目：

- [ ] 当前是否在 git 仓库内
- [ ] 目标目录是否存在
- [ ] 目标目录内是否有 .git（区分普通目录 vs 独立仓库）
- [ ] 主项目对目标目录的跟踪状态（决定使用哪个场景）
- [ ] 是否有远程仓库地址（可选）
- [ ] `.gitmodules` 文件是否存在

### 检查命令

```bash
# 检查是否在 git 仓库
git rev-parse --is-inside-work-tree

# 检查目录是否存在
test -d <目标目录> && echo "存在" || echo "不存在"

# 检查目录内是否有 .git（区分场景）
test -d <目标目录>/.git && echo "独立仓库" || echo "普通目录"

# 检查主项目对目录的跟踪状态
git status --short <目标目录>
# 输出格式：<状态> <路径>
# ??  = untracked（未跟踪）
# A   = added（已暂存）
# M   = modified（已修改）
# 无输出 = 可能已提交或被忽略

# 检查是否在主项目索引中（已提交或已暂存）
git ls-files -s <目标目录>
# 输出格式：<mode> <sha> <stage> <path>
# 160000 = gitlink（场景1.9的特征）
# 040000 = tree（普通目录）
# 100644/100755 = blob（文件）

# 检查 .gitmodules
test -f .gitmodules && cat .gitmodules || echo "不存在"

# 检查 .gitmodules 是否配置该路径
test -f .gitmodules && grep -q "<目标目录>" .gitmodules && echo "已配置" || echo "未配置"
```

---

## 场景判断逻辑伪代码

```python
# 伪代码：判断应该使用哪个场景

if not 目录存在:
    return "场景二：创建新 submodule"

if 目录内有.git:
    # 独立仓库场景
    跟踪状态 = git status --short <目标目录>

    if 跟踪状态包含 "??":
        return "场景 1.7：独立仓库未跟踪"
    elif 跟踪状态包含 "A":
        return "场景 1.8：独立仓库已暂存"
    else:
        # 检查是否已提交
        索引记录 = git ls-files -s <目标目录>

        if 索引记录模式 == "160000":
            return "场景 1.9：独立仓库已提交-gitlink"
        else:
            return "场景 0：无法识别，提示用户决策"

else:
    # 普通目录场景
    跟踪状态 = git status --short <目标目录>

    if 跟踪状态包含 "!!" 或 在.gitignore中:
        return "场景 1.6：被忽略目录"
    elif 跟踪状态包含 "??":
        return "场景一：未暂存代码转 submodule"
    elif 跟踪状态包含 "A":
        return "场景 1.5：已暂存代码"
    else:
        return "场景三：已提交代码转 submodule"
```

---

## 边缘场景处理

### 场景 1.5：已暂存代码 → submodule

**适用情况**：目标目录存在，文件已在 `git add` 后（staged 状态）。

**操作流程**：

```
Step 1: 取消暂存
        git restore --staged <目录>

Step 2: 按 场景一 继续执行
```

---

### 场景 1.6：被忽略的目录 → submodule

**适用情况**：目标目录在 `.gitignore` 中。

**操作流程**：

```
Step 1: 从 .gitignore 移除该目录模式
        # 编辑 .gitignore，删除或注释对应行

Step 2: 按 场景一 继续执行
```

---

### 场景 1.7：独立仓库未跟踪 → submodule

**适用情况**：目标目录内已有 `.git` 子目录（已是独立仓库），但主项目未跟踪该目录。

**前置检查**：

```bash
test -d <目标目录>/.git && echo "已是独立仓库" || echo "需要初始化"
```

**操作流程**：

```
Step 1: 跳过初始化步骤（目录已是独立仓库）

Step 2: 从场景一的 Step 4 开始执行（注册 submodule 到主项目）
        cd <主项目根目录>
        创建 .gitmodules
        git add .gitmodules <目标目录>
        git commit -m "添加 submodule: <模块名>"
```

**注意**：这种情况下的提交仍需在子目录内完成，属于"导入已存在的独立仓库"场景。

---

### 场景 1.8：独立仓库已暂存 → submodule

**适用情况**：

- 目标目录内已有 `.git` 子目录（外部复制来的独立仓库）
- 目录已在主项目暂存（`git add` 后，未 commit）
- 用户想将其转为 submodule 管理

**典型场景**：

```bash
# 用户从外部复制了一个完整仓库到项目中
cp -r /external/repo project/libs/my-lib

# 添加到暂存区（会有嵌套仓库警告）
git add libs/my-lib
# warning: adding embedded git repository

# 此时用户想转为 submodule
```

**前置检查**：

```bash
# 检查是否是独立仓库
test -d <目标目录>/.git && echo "独立仓库" || echo "普通目录"

# 检查主项目暂存状态
git status --short <目标目录>
# 输出示例：
# A  libs/my-lib  # A表示已暂存

# 提取状态字符
git status --short <目标目录> | cut -c1-2
# 输出：'A ' 或 'AM' 等（包含A表示added/staged）
```

**操作流程**：

```
Step 1: 主项目变更：取消暂存-Step1
        git restore --staged <目标目录>

Step 2: 按 场景 1.7 继续执行
        - 跳过初始化（目录已是独立仓库）
        - 更新 .gitmodules 配置（追加模式）
        - 注册 submodule 到主项目
        - 提交变更
```

**关键点**：

- 取消暂存后，目录变为 untracked 状态
- 目录内仍有 .git，可以跳过初始化步骤
- 直接注册到主项目即可
- 比场景1.9简单，因为尚未提交到主项目历史

---

### 场景 1.9：独立仓库已提交-gitlink → submodule

**适用情况**：

- 目标目录内已有 `.git` 子目录
- 目录已在主项目提交
- 主项目记录该目录为 **gitlink**（模式160000），但没有 .gitmodules 配置

**典型场景**：

```bash
# 用户从外部复制了一个完整仓库到项目中
cp -r /external/repo project/libs/my-lib

# 添加并提交（会有嵌套仓库警告，但用户忽略了）
git add libs/my-lib
git commit -m "add my-lib"

# 此时 libs/my-lib 被记录为 gitlink
git ls-files -s | grep "libs/my-lib"
# 160000 123abc... 0  libs/my-lib  # 模式160000表示gitlink

# 但是 .gitmodules 没有配置
cat .gitmodules
# cat: .gitmodules: No such file or directory

# 用户想正规化为 submodule
```

**前置检查**：

```bash
# 检查是否是独立仓库
test -d <目标目录>/.git && echo "独立仓库" || echo "普通目录"

# 检查主项目是否已提交
git log --oneline --all -- <目标目录> | head -1

# 检查是否是 gitlink（模式160000）
git ls-files -s <目标目录> | grep "^160000"
# 输出示例：
# 160000 abc123def 0 libs/my-lib

# 检查 .gitmodules 是否配置该路径
test -f .gitmodules && grep -q "<目标目录>" .gitmodules && echo "已配置" || echo "未配置"
```

**操作流程**：

```
Step 1: 主项目变更：验证状态-Step1
        git ls-files -s <目标目录>
        # 确认模式是 160000（gitlink）
        # 确认没有 .gitmodules 配置

Step 2: 主项目变更：从历史删除-Step2
        git rm --cached <目标目录>

Step 3: 主项目变更：提交删除-Step3
        git commit -m "移除 gitlink: <目录>，准备转为 submodule"

Step 4: 主项目变更：更新.gitmodules-Step4（追加模式）
        # 准备新配置内容：
        [submodule "<模块名>"]
            path = <目标目录>
            url = ./<目标目录>  # 或远程地址

        # 跨平台追加操作（见场景一 Step 5）

Step 5: 主项目变更：注册submodule-Step5
        git add .gitmodules
        git add <目标目录>

Step 6: 主项目变更：提交变更-Step6
        git commit -m "将 <目录> 正规化为 submodule"
```

**关键坑点**：

- ⚠️ **gitlink 不是真正的 submodule**，只是主项目记录了一个指针（类似符号链接）
- ⚠️ 必须从主项目历史删除后重新添加，才能正确创建 .gitmodules 配置
- ⚠️ 子仓库内的提交历史不会丢失，仍保留在子仓库的 .git 中
- ⚠️ 从历史删除时，工作目录的文件不会丢失（子仓库独立管理）
- ⚠️ 这是最复杂的场景，需要从主项目历史中清理 gitlink 记录

**恢复子仓库历史的方法**（如果需要）：

```bash
# 如果子仓库在转为 gitlink 前有提交，这些提交历史仍在子仓库的 .git 中
cd <目标目录>
git log --oneline  # 可以看到子仓库的提交历史
```

---

### 场景 0：无法识别状态 → 提示用户决策

**适用情况**：

- 不符合以上任何场景
- 状态异常或无法判断
- 需要用户决策

**典型异常场景**：

- 目录存在但有多个 .git（嵌套异常）
- .gitmodules 配置异常（格式错误、路径不存在等）
- Git 状态冲突（同时存在 staged 和 committed 等）
- 无法识别的状态组合

**前置检查**（详细诊断）：

```bash
echo "=== 目录状态 ==="
test -d <目标目录> && echo "目录存在" || echo "目录不存在"

echo "=== .git 状态 ==="
test -d <目标目录>/.git && echo "独立仓库" || echo "普通目录"

echo "=== 主项目 git status ==="
git status <目标目录>

echo "=== 主项目 ls-files（索引状态）==="
git ls-files -s <目标目录> || echo "未在索引中"

echo "=== .gitmodules 配置 ==="
test -f .gitmodules && grep <目标目录> .gitmodules || echo "未配置"

echo "=== 最近的提交记录 ==="
git log --oneline -5 -- <目标目录> || echo "无提交记录"
```

**操作流程**：

```
Step 1: 提示用户无法处理当前场景
        "❌ 无法识别目录 <目标目录> 的状态，不符合已定义的场景。"

Step 2: 展示诊断信息
        - 目录状态
        - 主项目跟踪状态
        - .gitmodules 配置
        - 可能的异常原因

Step 3: 提供用户决策选项
        "请检查以上诊断信息，选择处理方式："

        选项 A: "跳过此目录，继续处理其他目录"
        选项 B: "查看 submodule-manager skill 文档获取更多帮助"
        选项 C: "放弃自动处理，手动处理后重新执行"
        选项 D: "强制按某个场景处理（需用户明确指定场景编号）"

Step 4: 等待用户决策
        根据用户选择执行后续操作
```

**用户决策后处理**：

- **选择 A**：跳过当前目录，返回到调用方继续处理其他目录
- **选择 B**：输出 SKILL.md 路径，建议用户查阅特定章节
- **选择 C**：终止当前操作，提示用户手动处理步骤
- **选择 D**：强制按指定场景执行（需要用户明确确认风险）
