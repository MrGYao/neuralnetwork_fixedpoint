# 错误处理与降级

本文档定义各类错误场景的处理方案。当执行过程中遇到错误时加载此文档。

---

## progressive-doc 调用失败

```
错误：progressive-doc skill 不可用

降级方案：
  1. 直接生成文档（不调用 progressive-doc）
  2. 仍遵循渐进式结构（Level 1/2/3 分层）
  3. 在文档头部标注：未使用 progressive-doc（内容可能不够渐进）

不影响：
  - 五层结构（product/solution/architecture/implementation/issues）
  - 核心内容生成
  - 类型特定文档
```

---

## 项目类型识别不确定

```
错误：匹配多个类型（桌面应用 + Web 应用）

处理流程：
  1. 列出可能类型及匹配度：
     - 桌面应用：85%（存在 Electron）
     - Web 应用：70%（存在 FastAPI）

  2. 询问用户确认：
     "检测到项目兼具桌面和 Web 特征，更倾向于："
     → 桌面应用（托管后端）
     → Web 应用（带桌面客户端）

  3. 根据用户选择加载对应模板

兜底方案：
  - 用户不确认 → 选择匹配度最高的类型
  - 生成时在 issues/TODOs.md 标注：建议确认项目类型
```

---

## 代码分析失败

```
错误：无法读取源码（权限/编码问题）

降级方案：
  1. 仅分析 package.json / README.md 等元信息
  2. 询问用户关键问题（转为描述模式）
  3. 在 issues/TODOs.md 记录：未能深度分析代码

可提取信息：
  - package.json → dependencies, scripts, name, version
  - README.md → 项目描述、功能列表
  - 目录结构 → 模块组织
  - 配置文件 → 技术栈线索

无法提取：
  - 实际代码逻辑
  - 函数签名和调用关系
  - 具体实现细节
```

---

## 文档生成失败

```
错误：某层文档生成失败（如 architecture/）

处理：
  1. 跳过该层，继续生成其他层
  2. 在 issues/TODOs.md 记录：
     "ARCHITECTURE 层生成失败，原因：[error message]"
  3. 提供手动补充指引

部分文档失败：
  - 仅 architecture/CORE.md 失败
  → 生成 architecture/README.md + 其他文档
  → 在 issues/TODOs.md 记录缺失文档
```

---

## 用户取消操作

```
场景：用户在对话中取消

处理：
  1. 已生成的内容保留（不删除）
  2. 在 spec/ 目录创建 .partial 标记文件
  3. 记录取消阶段：
     "生成中断于：solution 层"

恢复：
  - 用户可重新执行 → 检测 .partial 文件 → 从中断点继续
```

---

## 外部工具不可用

```
错误：git / tree / fd 等工具不存在

降级方案：
  1. 使用内置 glob/read 替代
  2. 功能受限但可完成任务
  3. 在 issues/TODOs.md 标注：建议安装 [tool] 以获得更完整的分析

示例：
  - git log 不存在 → 无法分析提交历史 → 仅分析当前代码
  - tree 不存在 → 使用 glob 构建目录树
```

---

## 文件写入权限问题

```
错误：无法写入 spec/ 目录

处理：
  1. 尝试在项目根目录创建 spec/
  2. 如仍失败，询问用户：
     "无法在项目目录写入，是否指定其他位置？"
  3. 用户指定路径 → 在指定位置生成

临时方案：
  - 写入临时目录（如 /tmp/project-spec/）
  - 告知用户："文档生成在临时目录，请手动移动到项目目录"
```

---

## 降级流程总结

```
遇到错误 → 判断影响范围 ┐
                       ├→ 全局影响 → 询问用户是否继续
                       └→ 局部影响 → 降级处理

降级处理原则：
  1. 不中断流程（除非全局错误）
  2. 记录缺失内容到 issues/TODOs.md
  3. 提供恢复或补充指引
  4. 保证基本功能可用

错误报告：
  - 每个错误记录到 issues/TODOs.md
  - 格式："ERROR-[timestamp]: [description] → [降级方案]"
```
