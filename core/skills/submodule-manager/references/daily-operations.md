# Submodule 日常管理操作

本文档包含 submodule 的日常管理操作，如查看状态、更新版本、删除等。

---

## 追加 submodule 配置到 .gitmodules

【重要】**永远使用追加模式，保留原有配置，不要覆盖写入**

**Windows PowerShell**：

```powershell
# 检查 .gitmodules 是否存在
if (Test-Path .gitmodules) {
    # 追加到现有文件
    Add-Content -Path .gitmodules -Value "`n[submodule `"<模块名>`"]`n`tpath = <路径>`n`turl = <URL>"
} else {
    # 创建新文件
    Set-Content -Path .gitmodules -Value "[submodule `"<模块名>`"]`n`tpath = <路径>`n`turl = <URL>"
}
```

**Linux/Mac bash**：

```bash
# 检查 .gitmodules 是否存在
if [ -f .gitmodules ]; then
    # 追加（使用 >>，不要用 >）
    echo -e "\n[submodule \"<模块名>\"]\n\tpath = <路径>\n\turl = <URL>" >> .gitmodules
else
    # 创建新文件
    echo -e "[submodule \"<模块名>\"]\n\tpath = <路径>\n\turl = <URL>" > .gitmodules
fi
```

**验证追加成功**：

```bash
# 检查新配置是否写入
grep -q "<模块名>" .gitmodules && echo "✅ 追加成功" || echo "❌ 追加失败"

# 检查原有配置是否保留（假设原有模块名为 old-module）
grep -q "old-module" .gitmodules && echo "✅ 原有配置已保留" || echo "⚠️ 原有配置丢失"
```

**常见错误**：

- ❌ `echo ... > .gitmodules` - 覆盖写入，丢失原有配置
- ✅ `echo ... >> .gitmodules` - 追加写入，保留原有配置

---

## 查看所有 submodule 状态

```bash
git submodule status
```

输出格式：

```
<commit-hash> <路径> (<分支>)
```

示例：

```
 abc123def templates/python-module/uv-package (heads/main)
```

---

## 更新 submodule 到最新版本

```bash
# 更新特定的 submodule
git submodule update --remote <路径>

# 更新所有 submodule
git submodule update --remote

# 更新后锁定新版本
git add <路径>
git commit -m "更新 submodule: <模块名> 到最新版本"
```

---

## 进入 submodule 修改

```bash
cd <submodule路径>

# 在 submodule 内独立操作
git add .
git commit -m "修改内容"
git push  # 如有远程

# 回到主项目后更新引用
cd <主项目>
git add <submodule路径>
git commit -m "更新 submodule 引用"
```

---

## 删除 submodule

```bash
# Step 1: 取消注册
git submodule deinit -f <路径>

# Step 2: 从索引移除
git rm -f <路径>

# Step 3: 删除 .git/modules 下的实际仓库
Remove-Item -Recurse -Force .git\modules\<模块名>

# Step 4: 删除 .gitmodules 中的条目
# 手动编辑 .gitmodules，移除对应段落

# Step 5: 如果 .gitmodules 为空，删除文件
git rm .gitmodules  # 如已清空

# Step 6: 提交变更
git commit -m "移除 submodule: <模块名>"
```

---

## clone 含 submodule 的项目

```bash
# 方式一：clone 时自动拉取
git clone --recursive <项目地址>

# 方式二：clone 后手动拉取
git clone <项目地址>
cd <项目>
git submodule init
git submodule update

# 方式三：clone 后更新到最新版
git clone <项目地址>
cd <项目>
git submodule update --init --remote
```
