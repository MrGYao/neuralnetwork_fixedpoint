---
name: submodule-manager
description: Git submodule 管理工具，用于在主项目中独立管理子项目（模板、SDK、共享模块、独立仓库）。【必须触发】当用户说以下任何关键词：独立管理xxx目录、将xxx作为子项目、创建submodule、git submodule、模块独立仓库、子仓库、submodule、独立版本管理、拆分为独立仓库。【功能】自动识别三种场景并执行正确的 submodule 转换，解决 Git 的坑：不能直接对已存在目录执行 git submodule add
license: MIT
metadata:
  version: v1.2
  last-updated: 2026-07-29
---

# Git Submodule 管理 Skill

本 Skill 为 Agent 提供 Git submodule 的完整管理能力，自动识别场景并执行操作。

---

## 触发条件

当用户提到以下关键词时自动触发：

- "独立管理某个目录"
- "将 xxx 作为子项目"
- "创建 submodule"
- "git submodule 管理"
- "模块独立仓库"
- "子项目独立版本控制"
- "submodule"

---

## 执行纪律

1. **每步操作必须按规范提交**：执行完一步立即提交，注释格式见下文
2. **遇到失败立即停止**：最大尝试5次自行解决，5次失败后必须报告用户等待指令
3. **验证清单必须通过**：操作完成后必须逐条验证，任何一项不通过视为失败
4. **备份关键数据**：场景三必须先备份，避免数据丢失
5. **报告用户跳过原因**：任何步骤被跳过时，必须明确告知用户并请求决策

---

## 提交注释规范

```
<变更范围>变更：<操作名称>-Step<N>：<具体操作>

变更范围：
  - 局部变更：在 submodule 目录内的操作
  - 主项目变更：在主项目根目录的操作

操作名称：
  - 初始化子仓库
  - 关联远程
  - 注册submodule
  - 创建submodule
  - 备份代码
  - 移除历史

示例：
局部变更：初始化子仓库-Step1：进入 templates/python-module/uv-package
局部变更：初始化子仓库-Step2：git init && git add . && git commit
局部变更：关联远程-Step3：git remote add origin git@github.com:xxx.git
主项目变更：注册submodule-Step4：回到主项目根目录
主项目变更：注册submodule-Step5：创建 .gitmodules 配置
主项目变更：注册submodule-Step6：添加到主项目索引
主项目变更：注册submodule-Step7：提交主项目变更
```

---

## 前置检查清单

执行任何操作前，必须检查以下项目：

- [ ] 当前是否在 git 仓库内
- [ ] 目标目录是否存在
- [ ] 目标目录内是否有 .git（区分普通目录 vs 独立仓库）
- [ ] 主项目对目标目录的跟踪状态（决定使用哪个场景）
- [ ] 是否有远程仓库地址（可选）
- [ ] `.gitmodules` 文件是否存在

🔍 **详细检查命令和场景判断逻辑** → `references/scenarios-detailed.md`

---

## 场景识别流程

```
目标目录存在？
  ├── 否 → 场景二：创建新 submodule
  │
  └── 是 → 检查目录内是否有 .git
            │
            ├── 无 .git → 检查主项目 git 状态
            │             ├── ignored（被忽略）→ 场景 1.6：移出 .gitignore
            │             ├── untracked（未跟踪）→ 场景一：未暂存代码转 submodule
            │             ├── staged（已暂存）→ 场景 1.5：取消暂存后继续
            │             └── committed（已提交）→ 场景三：已提交代码转 submodule
            │
            └── 有 .git → 检查主项目对该目录的跟踪状态
                          │
                          ├── untracked（未跟踪）→ 场景 1.7：独立仓库未跟踪
                          ├── staged（已暂存）→ 场景 1.8：独立仓库已暂存
                          ├── committed（已提交）→ 场景 1.9：独立仓库已提交-gitlink状态
                          └── 无法识别 → 场景 0：兜底场景，提示用户决策
```

**场景编号说明**：

- **场景一**：未暂存代码 → submodule（核心）
- **场景二**：创建新 submodule（核心）
- **场景三**：已提交代码 → submodule（核心）
- **场景 1.5**：已暂存代码 → submodule（边缘）
- **场景 1.6**：被忽略目录 → submodule（边缘）
- **场景 1.7**：独立仓库未跟踪 → submodule（边缘）
- **场景 1.8**：独立仓库已暂存 → submodule（边缘）
- **场景 1.9**：独立仓库已提交-gitlink → submodule（边缘）
- **场景 0**：无法识别状态 → 提示用户决策（兜底）

📄 **边缘场景详细处理流程** → `references/scenarios-detailed.md`

---

## 场景一：未暂存代码 → submodule

**适用情况**：目标目录存在，文件是 `untracked` 状态。

### 操作流程

```
Step 1: 局部变更：初始化子仓库-Step1：进入目标目录
        cd <目标目录>

Step 2: 局部变更：初始化子仓库-Step2：初始化 git 仓库并提交
        git init
        git add .
        git commit -m "初始化 submodule: <模块名>"

Step 3: 局部变更：关联远程-Step3：如用户提供远程地址
        git remote add origin <远程地址>
        git branch -M main
        git push -u origin main

        如无远程地址，跳过此步（本地管理模式）

Step 4: 主项目变更：注册submodule-Step4：回到主项目根目录
        cd <主项目根目录>

Step 5: 主项目变更：注册submodule-Step5：更新 .gitmodules 配置（追加模式）

        【重要】保留原有 .gitmodules 内容，追加新配置

        准备新配置内容：
        [submodule "<模块名>"]
            path = <相对路径>
            url = <远程地址 或 ./相对路径>

        跨平台追加操作：

        【Windows PowerShell】
        if (Test-Path .gitmodules) {
            # 追加到现有文件
            Add-Content -Path .gitmodules -Value "`n[submodule `"<模块名>`"]`n`tpath = <相对路径>`n`turl = <URL>"
        } else {
            # 创建新文件
            Set-Content -Path .gitmodules -Value "[submodule `"<模块名>`"]`n`tpath = <相对路径>`n`turl = <URL>"
        }

        【Linux/Mac bash】
        if [ -f .gitmodules ]; then
            # 追加（>> 不是 >，避免覆盖）
            echo -e "\n[submodule \"<模块名>\"]\n\tpath = <相对路径>\n\turl = <URL>" >> .gitmodules
        else
            # 创建新文件
            echo -e "[submodule \"<模块名>\"]\n\tpath = <相对路径>\n\turl = <URL>" > .gitmodules
        fi

Step 6: 主项目变更：注册submodule-Step6：添加到主项目索引
        git add .gitmodules
        git add <目标目录>

Step 7: 主项目变更：注册submodule-Step7：提交主项目变更
        git commit -m "添加 submodule: <模块名>"
```

### 关键坑点

- ❌ **不能直接 `git submodule add` 已存在的目录**，会报错 "already exists in the working tree"
- ✅ 必须先在目录内初始化仓库并提交，再手动配置 `.gitmodules`

---

## 场景二：创建新 submodule

**适用情况**：目标目录不存在或为空。

### 操作流程

```
Step 1: 询问用户以下信息
        - submodule 名称
        - 目标路径
        - 是否需要远程仓库地址
        - 是否需要初始化模板文件（README.md 等）

Step 2: 主项目变更：创建submodule-Step2：添加 submodule

        【如有远程地址】
        git submodule add <远程地址> <目标目录>

        【如仅本地管理】
        mkdir -p <目标目录>
        cd <目标目录>
        git init
        # 如用户要求初始化模板
        echo "# <模块名>" > README.md
        git add .
        git commit -m "初始化 submodule: <模块名>"
        cd <主项目根目录>
        # 更新 .gitmodules（追加模式，保留原有内容）

        【Windows PowerShell】
        if (Test-Path .gitmodules) {
            Add-Content -Path .gitmodules -Value "`n[submodule `"<模块名>`"]`n`tpath = <目标目录>`n`turl = ./<目标目录>"
        } else {
            Set-Content -Path .gitmodules -Value "[submodule `"<模块名>`"]`n`tpath = <目标目录>`n`turl = ./<目标目录>"
        }

        【Linux/Mac bash】
        if [ -f .gitmodules ]; then
            echo -e "\n[submodule \"<模块名>\"]\n\tpath = <目标目录>\n\turl = ./<目标目录>" >> .gitmodules
        else
            echo -e "[submodule \"<模块名>\"]\n\tpath = <目标目录>\n\turl = ./<目标目录>" > .gitmodules
        fi

Step 3: 主项目变更：创建submodule-Step3：提交主项目变更
        git add .gitmodules <目标目录>
        git commit -m "添加 submodule: <模块名>"
```

---

## 场景三：已提交代码 → submodule

**适用情况**：目标目录已存在，文件已在主项目 git 历史中（committed 状态）。

### 操作流程

```
Step 1: 局部变更：备份代码-Step1：备份目标目录到临时位置
        # Windows PowerShell
        Copy-Item -Recurse <目标目录> D:\temp\local\opencode\<备份目录名>

        # Linux/Mac
        cp -r <目标目录> /tmp/<备份目录名>

Step 2: 主项目变更：移除历史-Step2：从主项目索引和历史中删除
        git rm -r --cached <目标目录>

Step 3: 主项目变更：移除历史-Step3：物理删除目录
        # Windows PowerShell
        Remove-Item -Recurse -Force <目标目录>

        # Linux/Mac
        rm -rf <目标目录>

Step 4: 主项目变更：移除历史-Step4：提交删除变更
        git commit -m "移除 <目录>，准备转为 submodule"

Step 5: 局部变更：初始化子仓库-Step5：将备份初始化为独立仓库
        cd <备份目录路径>
        git init
        git add .
        git commit -m "从主项目迁移：<目录>"

Step 6: 局部变更：关联远程-Step6：如用户提供远程地址
        git remote add origin <远程地址>
        git branch -M main
        git push -u origin main

        如无远程地址，跳过此步（本地管理模式）

Step 7: 主项目变更：注册submodule-Step7：作为 submodule 重新添加
        cd <主项目根目录>

        【如有远程地址】
        git submodule add <远程地址> <目标目录>

        【如本地管理】
        Move-Item <备份目录路径> <目标目录>
        # 更新 .gitmodules（追加模式，保留原有内容）

        【Windows PowerShell】
        if (Test-Path .gitmodules) {
            Add-Content -Path .gitmodules -Value "`n[submodule `"<模块名>`"]`n`tpath = <目标目录>`n`turl = ./<目标目录>"
        } else {
            Set-Content -Path .gitmodules -Value "[submodule `"<模块名>`"]`n`tpath = <目标目录>`n`turl = ./<目标目录>"
        }

        【Linux/Mac bash】
        if [ -f .gitmodules ]; then
            echo -e "\n[submodule \"<模块名>\"]\n\tpath = <目标目录>\n\turl = ./<目标目录>" >> .gitmodules
        else
            echo -e "[submodule \"<模块名>\"]\n\tpath = <目标目录>\n\turl = ./<目标目录>" > .gitmodules
        fi
        git add .gitmodules <目标目录>

Step 8: 主项目变更：注册submodule-Step8：提交主项目变更
        git commit -m "将 <目录> 转换为 submodule"
```

### 关键坑点

- ⚠️ **必须先从主项目历史删除**，否则 submodule 引用会冲突
- ⚠️ **备份很重要**，避免数据丢失
- ⚠️ 备份路径使用 `D:\temp\local\opencode\`（Windows）或 `/tmp/`（Linux/Mac）

---

## 验证清单

操作完成后**必须**逐条验证：

### 1. 验证 submodule 独立性

```bash
cd <目标目录>
git log              # 应有独立提交记录
git remote -v        # 应显示 submodule 的远程（如有）
```

### 2. 验证主项目识别

```bash
cd <主项目根目录>
git submodule status # 应显示该路径
cat .gitmodules      # 应包含配置
```

### 3. 验证 git status

```bash
git status           # 目录应显示为 submodule 引用，而非文件改动
                     # 正确格式：modified: <路径> (new commits)
                     # 错误格式：modified: <路径>/file1.py（说明未识别为submodule）
```

### 4. 验证 .git 类型

```bash
cd <目标目录>
Get-ChildItem -Force .git  # Windows
ls -la .git                 # Linux/Mac

# 可能是文件或目录：
# - 文件：内容指向 .git/modules/<名>/（标准 submodule）
# - 目录：独立仓库模式（本地管理）
```

**验证结果说明**：

| .git 类型 | 说明                                            | 是否正常 |
| --------- | ----------------------------------------------- | -------- |
| 文件      | 标准 submodule，.git 指向主项目的 .git/modules/ | ✅ 正常  |
| 目录      | 本地管理模式，独立的 .git 仓库                  | ✅ 正常  |
| 不存在    | 未正确初始化                                    | ❌ 失败  |

---

## 常见问题

遇到问题请查阅 `references/faq.md`，包含：

- **Q1**: 为什么不能直接 `git submodule add` 已存在的目录？
- **Q2**: submodule 目录内的 .git 是文件还是目录？
- **Q3**: 主项目 clone 下来后 submodule 是空的？
- **Q4-Q16**: 本地管理、多人协作、删除 submodule、gitlink 状态等常见问题

---

### 🔥 最常见问题速答

| 问题                 | 原因                                           | 解决方法                                     |
| -------------------- | ---------------------------------------------- | -------------------------------------------- |
| **已存在目录报错**   | Git 不允许对已存在目录执行 `git submodule add` | 按场景一手动初始化 + 配置 .gitmodules        |
| **clone 后子目录空** | submodule 不会随主项目自动拉取                 | `git submodule init && git submodule update` |
| **版本不一致**       | 多人协作时 submodule 引用的 commit 不同        | `git submodule update` 恢复到锁定版本        |

详细解答见 `references/faq.md`。

---

## 总结：三种场景对比

| 场景   | 前置条件            | 关键步骤                        | 风险等级 | 是否需要备份 |
| ------ | ------------------- | ------------------------------- | -------- | ------------ |
| 场景一 | 目录存在，untracked | 初始化子仓库 + 配置 .gitmodules | 低       | 否           |
| 场景二 | 目录不存在/空       | git submodule add 或手动创建    | 低       | 否           |
| 场景三 | 目录存在，已提交    | 备份 + 删除历史 + 重新添加      | 中       | **是**       |

---

## 快速参考卡

```
# 检查目录状态
git status <目录>
test -d <目录>/.git && echo "独立仓库" || echo "普通目录"
git ls-files -s <目录>  # 查看索引记录模式（160000=gitlink）

# 场景一：untracked → submodule
cd <目录> && git init && git add . && git commit -m "init"
cd <主项目> && 追加 .gitmodules 配置 && git add .gitmodules <目录> && git commit

# 场景二：新建 submodule
git submodule add <远程地址> <路径>

# 场景三：committed → submodule
备份 → git rm -r --cached <目录> && git commit
→ 初始化备份为仓库 → 移回原位 → 追加 .gitmodules 配置 && git commit

# 场景 1.7：独立仓库未跟踪（有.git + untracked）
跳过初始化 → 直接追加 .gitmodules 配置 && git add .gitmodules <目录> && git commit

# 场景 1.8：独立仓库已暂存（有.git + staged）
git restore --staged <目录> → 按场景 1.7 处理

# 场景 1.9：独立仓库已提交-gitlink（有.git + committed + 模式160000）
git rm --cached <目录> && git commit -m "移除 gitlink"
→ 追加 .gitmodules 配置 && git add .gitmodules <目录> && git commit

# 场景 0：无法识别
展示诊断信息 → 提示用户选择：跳过/查看文档/手动处理/强制场景

# .gitmodules 追加配置（重要：用 >> 追加，不用 > 覆盖）
echo -e "\n[submodule \"<名>\"]\n\tpath = <路径>\n\turl = <URL>" >> .gitmodules

# 验证
git submodule status
git status
cd <目录> && git log
```

---

## 使用指引

本 SKILL 采用**渐进式文档结构**，详细内容按需加载：

| 内容                                           | 文档位置                           | 使用时机                 |
| ---------------------------------------------- | ---------------------------------- | ------------------------ |
| **边缘场景处理**（场景 1.5/1.6/1.7/1.8/1.9/0） | `references/scenarios-detailed.md` | 遇到边缘场景时查阅       |
| **日常管理操作**（查看状态/更新/删除/clone）   | `references/daily-operations.md`   | submodule 日常维护时查阅 |
| **常见问题解答**（Q1-Q16）                     | `references/faq.md`                | 遇到问题时查阅           |

**核心场景指导**（场景一/二/三）保留在本 SKILL.md 中，无需额外加载。
