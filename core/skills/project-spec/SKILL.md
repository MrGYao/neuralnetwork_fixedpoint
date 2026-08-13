---
name: project-spec
version: 1.3.0
description: 自动生成项目规范文档集，采用渐进式组织（product/solution/architecture/implementation/issues五层）。支持代码分析（现有项目）或描述创建（新项目）。触发：创建/优化/整改/补充/生成项目规范、写spec、项目文档、spec目录。优先用于项目规范化、文档化需求。
---

# 项目规范生成 Skill

自动生成分层项目规范文档集，采用渐进式组织原则，支持两种创建模式。

---

## 核心工作流

### 步骤 1：识别项目状态

**检测是否为现有项目**：

```
使用 glob 扫描当前目录或指定路径：
  - 存在 package.json / pyproject.toml / requirements.txt / Cargo.toml / go.mod → 现有项目
  - 存在 src/ / lib/ / app/ 等源码目录 → 现有项目
  - 存在 README.md / AGENTS.md → 现有项目（有价值文档）

结果：
  - 现有项目 → 进入【分析模式】
  - 新项目/描述 → 进入【描述模式】
  - 部分现成 spec → 进入【补充模式】
```

---

### 步骤 2：信息采集

#### **分析模式（现有项目）- 深度代码分析**

**维度 1：结构分析**

```
glob **/*
  → 统计文件类型分布
  → 识别 monorepo/packages 结构
  → 提取目录树摘要
```

**维度 2：依赖分析**

```
read package.json → dependencies, devDependencies, scripts
read pyproject.toml → dependencies, scripts
read requirements.txt → Python 依赖
read Cargo.toml → Rust 依赖
read go.mod → Go 依赖
```

**维度 3：逻辑分析**

```
grep "^(class|def|function|export|public)"
  → 提取核心模块
  → 识别主要类/函数
  → 理解职责划分

grep "^(import|from|require)"
  → 理解模块依赖关系
```

**维度 4：数据模型分析**

```
grep "(Model|Table|Schema|Entity|class.*Model)"
  → 识别数据实体
  → 理解数据架构
```

**维度 5：现有文档提取**

```
read README.md → 项目介绍、安装说明
read AGENTS.md → 开发指引、架构说明
read docs/** → 补充信息
```

**产出**：项目信息结构体

```javascript
{
  name: "项目名称",
  type: "桌面应用" | "Web应用" | "CLI工具" | "库" | "微服务" | "通用",
  tech_stack: ["Electron", "Vue", "FastAPI", ...],
  structure: { packages: [...], directories: [...] },
  dependencies: { frontend: {...}, backend: {...} },
  core_modules: [...],
  data_models: [...],
  existing_docs: { readme: "...", agents: "...", ... },
  key_features: ["进程管理", "认证机制", ...]
}
```

---

#### **描述模式（新项目）- 引导式对话**

**关键问题列表（3-5 个）**：

1. **项目定位**

   ```
   这个项目的核心目标是什么？解决什么问题？
   → 收集：项目价值、用户群体、成功指标
   ```

2. **技术偏好**

   ```
   计划使用什么技术栈？有特殊偏好或约束吗？
   → 收集：前端技术、后端技术、数据库、部署平台
   ```

3. **核心需求**

   ```
   项目的主要功能模块有哪些？核心场景是什么？
   → 收集：功能列表、优先级、关键流程
   ```

4. **架构约束（可选）**

   ```
   有架构上的偏好或约束吗？（如 Monorepo、微服务、前后端分离）
   → 收集：架构风格、组织方式
   ```

5. **演进规划（可选）**
   ```
   项目的短期目标和长期愿景是什么？
   → 收集：迭代计划、扩展方向
   ```

**产出**：项目信息结构体（结构相同，来源为用户描述）

详细问题模板 → [references/dialog-templates.md](./references/dialog-templates.md)

---

#### **补充模式（部分现有 spec）**

```
检测 spec/ 目录已有内容
  → 识别缺失或不完整的部分
  → 询问用户补充方向
  → 局部重新生成
```

---

### 步骤 3：识别项目类型

**自动识别规则**（优先级从高到低）：

| 规则 | 项目类型 | 判断条件                                                   |
| ---- | -------- | ---------------------------------------------------------- |
| 1    | 桌面应用 | 存在 Electron 主进程代码 或 main.js 指向 Electron          |
| 2    | 移动应用 | 存在 android/ios 目录 或 React Native/Flutter 依赖         |
| 3    | Web 应用 | 存在 FastAPI/Flask/Express/Spring Boot 且无桌面特征        |
| 4    | CLI 工具 | 存在 argparse/click/commander/yargs 且主程序读取命令行参数 |
| 5    | 微服务   | 存在多个独立服务入口 + Docker Compose + 服务发现配置       |
| 6    | 库/SDK   | 无主程序入口，主要是模块导出/类定义，无直接运行入口        |
| 7    | 通用     | 不符合以上任何特征                                         |

**用户确认流程**：

```
识别结果：【桌面应用】

确认或调整？
  → 确认：使用桌面应用模板
  → 调整：列出所有类型供选择
```

详细识别规则 → [references/type-identification.md](./references/type-identification.md)

---

### 步骤 4：生成规范文档

**调用 progressive-doc skill** 为每一层生成文档：

```
按顺序生成 5 层规范：
  1. product/ → 产品定义层
  2. solution/ → 解决方案层
  3. architecture/ → 架构设计层
  4. implementation/ → 实施细节层
  5. issues/ → 问题追踪层

每层使用 progressive-doc 的：
  - Metadata 描述
  - README.md 概要（Level 1）
  - 分类详情文档（Level 2-3）
```

根据项目类型选择对应模板（见下方"项目类型模板"）。

---

### 步骤 5：生成总览 README

在 `spec/README.md` 创建导航总览。

**必须包含的章节**：

1. 元信息块（版本、更新时间、状态、文档总数）
2. 项目定位
3. 文档集合索引
4. 文档关系（抽象层次关系图）
5. 按角色推荐阅读路径
6. AI 加载指引
7. 文档使用建议
8. 文档维护说明
9. 文档导航（末尾）

**标准模板 → [references/spec-README-template.md](./references/spec-README-template.md)**

⚠️ **重要**：必须严格遵循模板，不得自定义或省略章节。

---

### 步骤 6：预填充 issues 引导问题

在 `issues/TODOs.md` 预填充引导问题：

```markdown
# 待办事项与思考问题

## 需要进一步明确的问题

### AR-001: [待明确] 核心价值主张

- 问题：项目的核心价值主张是什么？与其他同类项目的差异点？
- 影响：影响 product/VALUE.md 的完善
- 优先级：高

### AR-002: [待明确] 扩展性设计

- 问题：项目未来可能的扩展方向？当前架构是否支持？
- 影响：影响 architecture/ 的扩展性文档
- 优先级：中

### AR-003: [待明确] 性能指标

- 问题：项目有哪些性能指标要求？如何衡量？
- 影响：影响 implementation/ 的性能优化文档
- 优先级：中

### AR-004: [待明确] 错误处理策略

- 问题：项目的错误处理和异常恢复策略是什么？
- 影响：影响 architecture/CORE.md 的鲁棒性设计
- 优先级：高

## 已记录的待办事项

（根据代码分析或用户描述补充）

## 已完成事项

（记录已完成的规范编写任务）
```

---

### 步骤 7：生成 coding-index.md（增强版 - 自动检测与填充）

**核心机制**：自动检测项目规范配置 → 智能填充模板 → 生成遗留问题提示

---

#### 自动检测与填充流程

```
1. 加载模板：references/file-templates/spec/coding-index.md

2. 对于每个可配置规范项：
   a. 执行 Level 1 检测（配置文件）
   b. 如果未检测到，执行 Level 2 检测（代码约定推断）
   c. 根据检测结果填充本项目设定
   d. 标注配置状态（已设定/待确认/待设定）

3. 收集未设定规范项：
   - 过滤：差异程度 = 高/中
   - 生成本项目特有的未设定提示
   - 添加到 issues/TODOs.md

4. 输出文件：spec/coding-index.md
```

---

#### 检测点清单

| 检测项                              | 检测方法                            | 影响的规范项                      | 优先级 |
| ----------------------------------- | ----------------------------------- | --------------------------------- | ------ |
| `pyproject.toml` 配置               | 读取 `[tool.ruff]`, `[tool.pytest]` | Python 行长度、格式化工具、覆盖率 | 中     |
| `tsconfig.json` 配置                | 读取 `compilerOptions`              | TS 严格模式、any 策略             | 中     |
| `.prettierrc` 配置                  | 读取配置                            | TS 行长度、分号                   | 中     |
| `.editorconfig` 配置                | 读取命名约定                        | 文件命名风格                      | 中     |
| 现有 `.py` 文件名                   | glob + naming 分析                  | Python 文件命名风格               | 中     |
| 现有 `.ts` 文件名                   | glob + naming 分析                  | TypeScript 文件命名风格           | 中     |
| 现有代码类型提示                    | grep 类型提示语法                   | 类型提示使用率                    | 中     |
| 目录结构                            | glob 分析                           | 目录组织原则                      | 低     |
| `requirements.txt` / `package.json` | 读取依赖                            | 包管理器检测                      | -      |

---

#### 遗留问题生成策略

**生成条件**：规范项差异程度 = 高/中 且 配置状态 = 待设定

**问题模板**：

````
### AR-XXX: [待明确] [规范项名称]

**问题**：项目未明确设定 [规范项]，可能导致 [影响描述]

**影响范围**：
- 影响新成员：可能使用不同风格，导致代码不一致
- 影响工具：lint/format 工具可能配置不一致

**默认使用**：[默认推荐值]

**建议配置方式**：
1. 在 [配置文件] 添加：
   ```toml/json
   [配置项]
````

2. 或创建 `.editorconfig` 统一约定

**优先级**：[根据规范项类别确定]

**相关文档**：[链接到 CODING_STANDARDS.md 对应章节]

```

**优先级分层规则**：

| 规范项类别 | 优先级 | 示例 |
|-----------|--------|------|
| 核心规范（文件命名、格式化工具、行长度）| **中** | 文件命名风格、Python 行长度 |
| 类型约束（类型提示、严格模式）| **中** | Python 类型提示策略、TS 严格模式 |
| 测试规范（覆盖率、测试命名）| **低** | 覆盖率要求、测试命名风格 |
| 组织规范（目录组织、模块划分）| **低** | 目录组织方式 |
| Git 规范（分支命名、Commit 规范）| **低** | Git 工作流 |

---

#### 填充示例

**场景 1：检测到 pyproject.toml 配置**

```

检测：读取 pyproject.toml → [tool.ruff] line-length = 100
填充：

- 配置状态：已设定
- 本项目设定：已设定为 100
- 配置来源：pyproject.toml [tool.ruff] line-length
- 不加入遗留问题

```

**场景 2：未检测到配置，使用 Level 2 推断**

```

检测：

- Level 1：未找到 pyproject.toml 或无 [tool.ruff]
- Level 2：分析现有 .py 文件最长行 = 98
  填充：
- 配置状态：待确认
- 本项目设定：推断为 100（代码最长行 98 → 推断行长度 100）
- 配置来源：代码约定推断
- 不加入遗留问题（已推断）

```

**场景 3：未检测到任何配置**

```

检测：

- Level 1：未找到配置
- Level 2：无法推断（代码量太少或混合风格）
  填充：
- 配置状态：待设定
- 本项目设定：默认推荐 88
- 配置来源：待设定
- 加入遗留问题：AR-XXX（优先级：中）

````

---

### 步骤 8：生成 architecture/CODING_STANDARDS.md（增强版 - 自动检测与填充）

**机制同步骤 7**，额外增加：

---

#### 额外检测点

| 检测项 | 检测方法 | 影响的规范项 |
|--------|---------|-------------|
| `.github/workflows/` | glob 检测 CI 配置 | CI 流程、检查项 |
| `CONTRIBUTING.md` | 读取贡献指引 | Git 工作流、分支命名 |
| `commitlint` 配置 | 读取配置文件 | Commit 消息规范 |

---

#### 额外生成内容

1. **配置文件示例章节**：
   - 根据检测到的工具生成对应配置示例
   - 未检测到 → 生成默认推荐工具的配置示例

2. **未设定规范汇总章节**：
   - 按优先级分组列出所有未设定项
   - 提供配置方式建议

---

#### 输出示例

```markdown
## 未设定规范汇总

**待设定项列表**：

| 规范项 | 差异程度 | 推荐值 | 优先级 | 配置方式 |
|--------|---------|--------|--------|---------|
| 文件命名风格 | 高 | snake_case (Python) / camelCase (TS) | 中 | .editorconfig |
| 行长度（Python）| 高 | 88 | 中 | pyproject.toml |
| 覆盖率要求 | 高 | 90%/80%/70%/60% | 低 | pyproject.toml |

> ⚠️ 建议在项目初期明确设定核心规范（优先级=中），避免团队成员使用不同约定。
````

---

### 步骤 9：处理项目根 README.md

**检测已存在的 README.md**：

```
检查项目根目录是否存在 README.md：
  if (存在):
    → 检测是否包含 "spec/" 或 "coding-index" 引用
    → if (无引用):
        → 追加到 README.md 末尾：

          ## 详细规范

          完整项目规范参见 [spec/README.md](./spec/README.md)

          代码开发指引参见 [coding-index.md](./spec/coding-index.md)
      → if (已有引用):
        → 验证链接有效，不重复添加
  else:
    → 调用 create-readme skill 生成 README.md
    → 确保包含 spec/ 引用
```

**内容分工原则**：

| 文档               | 包含内容                                               | 不包含内容                        |
| ------------------ | ------------------------------------------------------ | --------------------------------- |
| **根 README.md**   | 项目概要、快速上手、安装说明、核心特性列表、spec/ 链接 | 完整规范、AI 加载指引、文档关系图 |
| **spec/README.md** | 完整 5 层规范总览、AI 加载指引、代码开发指引           | 快速上手、安装说明                |

**注意**：

- 如果 create-readme skill 不存在，手动生成 README.md 模板
- 追加内容时使用 `\n\n## 详细规范\n\n...` 确保格式正确

---

### 步骤 10：展示结果并迭代优化

**自动展示生成结果**：

```
生成完成！创建的文档结构：

spec/
├── README.md              ✅ 项目规范总览
├── product/
│   ├── README.md          ✅ 产品概要
│   ├── OVERVIEW.md        ✅ 项目定位
│   └── VALUE.md           ✅ 核心价值
├── solution/
│   ├── README.md          ✅ 解决方案概要
│   ├── ARCHITECTURE.md    ✅ 架构选型
│   └── TECH_STACK.md      ✅ 技术栈
...（省略）

查看文档？
  → 用户选择查看特定文档
  → 用户要求修改/补充
```

**迭代优化机制**：

```
用户反馈："solution/ARCHITECTURE.md 不够详细，需要补充组件交互图"

处理：
  1. 读取用户要求
  2. 重新生成该文档（更详细）
  3. 自动更新相关 README（保持导航一致）
  4. 展示更新结果
```

---

## 模板与结构定义（按需加载）

以下内容已拆分到独立文件，按需加载：

### 项目类型模板

→ [references/templates/project-types.md](./references/templates/project-types.md)

**内容**：桌面应用、Web 应用、CLI 工具、库/SDK、微服务、通用模板定义

**加载时机**：步骤 4 生成规范文档时

---

### 渐进式文档组织原则

→ [references/templates/progressive-structure.md](./references/templates/progressive-structure.md)

**内容**：Level 1/2/3 三层分级体系、文档结构模板、渐进式加载策略

**加载时机**：步骤 4 生成各层文档时

---

### 质量保证流程

→ [references/templates/quality-assurance.md](./references/templates/quality-assurance.md)

**内容**：生成后检查清单、自动迭代优化、错误恢复策略

**加载时机**：步骤 5 验证与优化时

---

## 附录（按需加载）

### A. 项目类型识别细节

→ [references/type-identification.md](./references/type-identification.md)

**加载时机**：步骤 2/3 需要详细识别逻辑时

---

### B. 模板索引总览

→ [references/TEMPLATE_INDEX.md](./references/TEMPLATE_INDEX.md)

**内容**：基础模板 + 增量模板索引、模板组合规则、加载策略

**加载时机**：步骤 4 根据项目类型选择增量模板时

---

### C. 引导式对话模板

→ [references/dialog-templates.md](./references/dialog-templates.md)

**加载时机**：进入"描述模式"时

---

### D. 错误处理与降级

→ [references/error-handling.md](./references/error-handling.md)

**加载时机**：执行过程中遇到错误时

---

### E. 性能优化策略

→ [references/performance.md](./references/performance.md)

**加载时机**：检测到项目规模较大时（文件数 > 1000）

---

### F. 示例输出

→ [references/examples.md](./references/examples.md)

**加载时机**：生成文档时参考结构

---

### G. spec/README.md 标准模板

→ [references/spec-README-template.md](./references/spec-README-template.md)

**加载时机**：**步骤 5 生成总览 README 时（必须加载，严格遵循）**

⚠️ **重要**：每个项目的 spec/README.md 必须严格遵循此模板，不得自定义或省略章节。

---

## 总结

本 skill 通过自动化代码分析或引导式对话，生成结构化的项目规范文档集。核心价值：

1. **自动化**：减少手动编写文档的工作量
2. **结构化**：五层分级，清晰组织，易于维护
3. **渐进式**：适合人类阅读，也适合 AI 高效加载
4. **可迭代**：支持局部重新生成、补充、优化
5. **类型感知**：根据项目类型生成特有文档

触发时，按工作流顺序执行，最终输出完整的 spec/ 目录。
