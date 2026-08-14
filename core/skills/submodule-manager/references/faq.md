# submodule-manager 常见问题 FAQ

---

## Q1: 为什么不能直接 `git submodule add` 已存在的目录？

**A**: Git 的 `submodule add` 命令要求目标目录不存在或为空。

**错误示例**：

```bash
$ git submodule add git@github.com:user/template.git templates/my-template
fatal: 'templates/my-template' already exists in the working tree
```

**解决方案**：

1. 先在目录内 `git init` 初始化仓库
2. 提交所有文件
3. 手动创建 `.gitmodules` 配置
4. 在主项目中 `git add` 该目录

这正是 `submodule-manager` skill 的场景一流程。

---

## Q2: submodule 目录内的 .git 是文件还是目录？

**A**: 有两种情况：

| 情况               | .git 类型 | 内容/说明                                  |
| ------------------ | --------- | ------------------------------------------ |
| **标准 submodule** | 文件      | 内容是指向 `.git/modules/<模块名>/` 的路径 |
| **本地管理模式**   | 目录      | 完整的独立 git 仓库                        |

**示例**：

```bash
# 标准submodule（有远程）
$ cat templates/my-template/.git
gitdir: ../../.git/modules/templates/my-template

# 本地管理模式
$ ls -la templates/my-template/.git
drwxr-xr-x  .git/    # 目录
```

两者都可以正常工作，只是 Git 内部实现不同。本地管理模式转换到远程时无需重新配置。

---

## Q3: 主项目 clone 下来后 submodule 是空的？

**A**: 拉取主项目时，submodule 默认不自动拉取内容，只保留一个空目录。

**解决方案**：

```bash
# 方式一：clone 时自动拉取
git clone --recursive <项目地址>

# 方式二：clone 后手动拉取
git clone <项目地址>
cd <项目>
git submodule init
git submodule update

# 方式三：拉取最新版（而非锁定的版本）
git submodule update --init --remote
```

**建议**：在项目 README.md 中明确说明：

```
clone 本项目后请执行：
git submodule init && git submodule update
```

---

## Q4: 本地管理模式如何同步到其他机器？

**A**: 本地 submodule 无法自动同步到其他机器（无远程仓库）。

**建议**：

- 为 submodule 创建远程仓库（如 GitHub private repo 免费）
- 改为远程模式，方便跨机器同步

**转换步骤**：

```bash
# 1. 为 submodule 添加远程
cd <submodule路径>
git remote add origin <远程地址>
git push -u origin main

# 2. 更新主项目的 .gitmodules
# 修改 url 为远程地址
[submodule "模板名"]
    path = templates/my-template
    url = git@github.com:user/template.git  # 从 ./templates/my-template 改为远程

# 3. 主项目提交变更
cd <主项目>
git add .gitmodules
git commit -m "配置 submodule 远程地址"
```

---

## Q5: 如何查看 submodule 当前的锁定版本？

**A**:

```bash
# 查看所有 submodule 的当前版本
$ git submodule status
 abc123def456 templates/python-module/uv-package (heads/main)

# abc123def456 是 submodule 的 commit hash

# 查看特定 submodule 的详细信息
cd <submodule路径>
git log -1
git show HEAD
```

**版本锁定原理**：主项目不跟踪 submodule 的分支，只跟踪具体的 commit hash。即使 submodule 有新提交，主项目仍锁定在旧版本，直到执行 `git submodule update --remote`。

---

## Q6: 多人协作时 submodule 版本不一致怎么办？

**A**: 多人协作时，每个人的 submodule 可能指向不同的 commit。

**场景**：

- A 提交了主项目，submodule 锁定在 commit X
- B 拉取主项目后，submodule 仍在 commit Y（旧版本）

**解决方法**：

```bash
# 拉取主项目
git pull

# 更新 submodule 到主项目锁定的版本
git submodule update

# 或强制更新到最新版（慎用）
git submodule update --remote
```

**团队规范建议**：

1. 修改 submodule 后，立即更新主项目的引用并提交
2. 在 PR/MR 中提醒 reviewer 注意 submodule 变更
3. 使用 CI 检查：`git submodule status` 确保版本一致

---

## Q7: 执行场景一时报错 "already exists in the working tree"？

**A**: 这说明你错误地使用了 `git submodule add` 命令。

**错误操作**：

```bash
# 目录已存在时直接执行
git submodule add git@github.com:user/template.git existing-dir/
# 报错：fatal: 'existing-dir' already exists in the working tree
```

**正确做法**：

场景一**不能**使用 `git submodule add`，必须按照流程手动初始化：

```bash
# 1. 进入目录初始化
cd existing-dir/
git init
git add .
git commit -m "初始化"

# 2. 回到主项目配置
cd ..
echo '[submodule "模板名"]
    path = existing-dir
    url = ./existing-dir' > .gitmodules

# 3. 添加并提交
git add .gitmodules existing-dir/
git commit -m "添加 submodule"
```

**这正是 submodule-manager skill 自动化处理的流程。**

---

## Q8: 如何判断一个目录是否已经是 submodule？

**A**: 有三种方法：

**方法一：查看 .gitmodules**

```bash
cat .gitmodules | grep "<目录路径>"
# 有结果 = 是 submodule
```

**方法二：查看 git submodule status**

```bash
git submodule status | grep "<目录路径>"
# 有结果 = 是 submodule
```

**方法三：进入目录检查 .git**

```bash
cd <目录>
cat .git
# 如输出类似：gitdir: ../../.git/modules/xxx
# 则是 submodule（.git 是文件）

ls -la .git
# 如 .git 是目录，则是独立仓库（可能尚未注册为 submodule）
```

**判断逻辑**：

| .gitmodules 有记录 | .git 类型 | 状态                         |
| ------------------ | --------- | ---------------------------- |
| ✅                 | 文件      | 标准 submodule ✅            |
| ✅                 | 目录      | 本地管理 submodule ✅        |
| ❌                 | 文件      | 异常（需重新配置）           |
| ❌                 | 目录      | 独立仓库，未注册为 submodule |

---

## Q9: 如何删除一个 submodule？

**A**:

```bash
# Step 1: 取消注册
git submodule deinit -f <路径>

# Step 2: 从索引移除
git rm -f <路径>

# Step 3: 删除 .git/modules 下的实际仓库
Remove-Item -Recurse -Force .git\modules\<模块名>  # Windows
rm -rf .git/modules/<模块名>                         # Linux/Mac

# Step 4: 编辑 .gitmodules，移除对应段落
# 手动删除 [submodule "xxx"] 整段

# Step 5: 如果 .gitmodules 已清空，删除文件
git rm .gitmodules

# Step 6: 提交变更
git commit -m "移除 submodule: <模块名>"
```

---

## Q10: submodule 可以嵌套吗（submodule 内还有 submodule）？

**A**: 可以，但不推荐。

**问题**：

- clone 嵌套 submodule 需要多次 `git submodule update --init --recursive`
- 版本管理复杂，容易出错
- 多人协作时更容易出现版本不一致

**替代方案**：

- 将嵌套的依赖提取到主项目作为平级 submodule
- 或使用 Git 的 `subtree` 替代

---

## Q11: 如何批量更新所有 submodule？

**A**:

```bash
# 更新所有 submodule 到主项目锁定的版本
git submodule update

# 更新所有 submodule 到各自分支的最新版本
git submodule update --remote

# 递归更新（包含 submodule 的 submodule）
git submodule update --init --recursive

# 批量更新后提交
git submodule update --remote
git add -A  # 会添加所有 submodule 的新版本引用
git commit -m "更新所有 submodule 版本"
```

---

## Q12: submodule 的分支如何管理？

**A**:

**默认行为**：submodule 跟踪具体的 commit，**不跟踪分支**。

**指定分支**：

```bash
# 添加 submodule 时指定分支
git submodule add -b main <远程地址> <路径>

# 更新时拉取特定分支的最新提交
git submodule update --remote <路径>
# 会拉取指定分支的最新 commit
```

**查看 submodule 跟踪的分支**：

```bash
# 在主项目查看配置
git config --file .gitmodules submodule.<路径>.branch

# 或手动查看 .gitmodules
cat .gitmodules
# 有 branch = xxx 行表示指定了分支
```

**建议**：明确指定分支，避免混乱。

---

## Q13: 从外部复制了一个仓库，如何转化为 submodule？（场景 1.7/1.8/1.9）

**A**: 根据主项目对该目录的跟踪状态，选择不同场景：

### 三种情况判断

```bash
# 检查目录状态
test -d <目录>/.git && echo "独立仓库" || echo "普通目录"

# 检查主项目跟踪状态
git status --short <目录>
# ??  = 未跟踪（场景 1.7）
# A   = 已暂存（场景 1.8）

# 检查是否已提交且为 gitlink
git ls-files -s <目录> | grep "^160000"
# 有输出 = gitlink 状态（场景 1.9）
```

### 场景 1.7：独立仓库未跟踪

```bash
# 最简单：直接注册 submodule
cd <主项目>
# 追加 .gitmodules 配置
echo -e "\n[submodule \"<名>\"]\n\tpath = <路径>\n\turl = ./<路径>" >> .gitmodules
git add .gitmodules <目录>
git commit -m "添加 submodule: <名>"
```

### 场景 1.8：独立仓库已暂存

```bash
# 先取消暂存，再按场景 1.7 处理
git restore --staged <目录>
# 然后按场景 1.7 执行
```

### 场景 1.9：独立仓库已提交-gitlink

```bash
# 最复杂：需要从历史删除后重新添加
# 验证 gitlink 状态
git ls-files -s <目录>
# 输出：160000 ...（模式160000表示 gitlink）

# 从主项目历史删除
git rm --cached <目录>
git commit -m "移除 gitlink: <目录>"

# 重新添加为 submodule
echo -e "\n[submodule \"<名>\"]\n\tpath = <路径>\n\turl = ./<路径>" >> .gitmodules
git add .gitmodules <目录>
git commit -m "将 <目录> 正规化为 submodule"
```

**关键点**：

- 场景 1.9 最复杂，因为需要清理主项目历史中的 gitlink 记录
- 子仓库的提交历史不会丢失，仍保留在子仓库的 .git 中

---

## Q14: 什么是 gitlink？为什么会陷入 gitlink 状态？

**A**: **gitlink** 是 Git 对嵌套仓库的一种特殊记录方式（模式160000）。

### 产生原因

```bash
# 用户从外部复制了一个完整仓库
cp -r /external/repo project/libs/my-lib

# 添加并提交（会有警告）
git add libs/my-lib
# warning: adding embedded git repository
git commit -m "add my-lib"

# 此时 Git 记录该目录为 gitlink
git ls-files -s libs/my-lib
# 160000 abc123 0 libs/my-lib  # 模式160000
```

### gitlink 的问题

| 特征             | gitlink     | 真正的 submodule |
| ---------------- | ----------- | ---------------- |
| 主项目记录       | commit SHA  | commit SHA ✅    |
| .gitmodules 配置 | ❌ 无       | ✅ 有            |
| clone 后状态     | ❌ 空目录   | ✅ 正常初始化    |
| submodule 命令   | ❌ 无法识别 | ✅ 正常管理      |

**问题**：其他人 clone 主项目后，该目录是空的，无法获取子仓库内容。

### 解决方法

使用**场景 1.9**的处理流程，将 gitlink 正规化为 submodule。

---

## Q15: 场景判断失败怎么办？（场景 0）

**A**: 如果不符合任何已知场景，使用**场景 0** 的诊断流程：

```bash
# 详细诊断
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

### 用户决策选项

根据诊断信息，选择：

1. **跳过此目录**：继续处理其他目录
2. **查看文档**：查阅 SKILL.md 获取更多帮助
3. **手动处理**：放弃自动处理，手动修复后重新执行
4. **强制场景**：明确指定场景编号强制执行（需确认风险）

### 常见异常原因

- 目录存在多个 .git（嵌套异常）
- .gitmodules 配置格式错误
- Git 状态冲突
- 文件权限异常

---

## Q16: 如何验证 submodule 转换成功？

**A**: 执行以下检查：

```bash
# 1. 检查 .gitmodules 配置
cat .gitmodules
# 应该包含该路径的配置

# 2. 检查 submodule 状态
git submodule status
# 输出格式：<状态> <commit> <路径>
# 空格/+/− 前 + 表示初始化，- 表示未初始化，空格表示正常

# 3. 检查主项目索引
git ls-files -s <路径>
# 模式应该是 160000（gitlink 模式）

# 4. 检查子仓库历史
cd <路径>
git log --oneline
# 应该能看到提交历史

# 5. 检查 clone 后是否正常（可选）
cd /tmp
git clone <主项目地址> test-clone
cd test-clone
git submodule update --init --recursive
ls <路径>
# 应该能看到子仓库内容
```

### 验证清单

- [ ] .gitmodules 包含该路径配置
- [ ] `git submodule status` 能识别该 submodule
- [ ] 主项目索引记录模式为 160000
- [ ] 子仓库内有提交历史
- [ ] clone + submodule update 后子仓库内容完整
