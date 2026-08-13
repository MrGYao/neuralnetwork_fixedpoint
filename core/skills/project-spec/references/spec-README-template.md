# spec/README.md 标准模板

本文档定义 **spec/README.md 的标准格式**。每一步步骤 5 生成总览 README 时**必须严格遵循**此模板，不得自定义或省略任何章节。

---

## 模板内容

```markdown
# [项目名称] 文档总览

> **版本**: v1.0  
> **更新时间**: YYYY-MM-DD  
> **文档总数**: [N] 个文档（[M] 个文档集合）

---

## 项目定位

**[项目名称]** - [一句话描述]

**核心组成**:

1. **[组件1]** - [描述]
2. **[组件2]** - [描述]
3. **[组件3]** - [描述]

---

## 文档集合索引

| 集合                | 抽象层次 | 文档数 | 详细文档                                |
| ------------------- | -------- | ------ | --------------------------------------- |
| **product/**        | 业务层   | [N]    | [README.md](./product/README.md)        |
| **solution/**       | 方案层   | [N]    | [README.md](./solution/README.md)       |
| **architecture/**   | 设计层   | [N]    | [README.md](./architecture/README.md)   |
| **implementation/** | 实现层   | [N]    | [README.md](./implementation/README.md) |
| **issues/**         | 维护层   | [N]    | [README.md](./issues/README.md)         |

---

## 文档关系

### 抽象层次关系

┌──────────────────────────────────────────────────┐
│ product/ 业务层 - 产品定义 │
│ 职责：为什么做、为谁做、提供什么价值 │
│ 职责范围：[根据项目填写] │
└──────────────────────────────────────────────────┘
↓ 驱动
┌──────────────────────────────────────────────────┐
│ solution/ 方案层 - 整体解决方案 │
│ 职责：理清楚端到端设计，输入、输出是什么、通过何种技术选型实现 │
│ 职责范围：[根据项目填写] │
└──────────────────────────────────────────────────┘
↓ 包含/细化
┌──────────────────────────────────────────────────┐
│ architecture/ 设计层 - 架构与代码组织 │
│ 职责：展开技术选型实现，采用何种架构，代码如何组织、组件如何关联 │
│ 职责范围：[根据项目填写] │
└──────────────────────────────────────────────────┘
↓ 指导
┌──────────────────────────────────────────────────┐
│ implementation/ 实现层 - 实施细节 │
│ 职责：目录中每个文件简述，关键代码实现，编写规范 │
│ 职责范围：[根据项目填写] │
└──────────────────────────────────────────────────┘
↑↓ 贯穿全程
┌──────────────────────────────────────────────────┐
│ issues/ 维护层 - 问题追踪 │
│ 职责：遗留问题、待办事项、演进计划 │
│ 职责范围：[根据项目填写] │
└──────────────────────────────────────────────────┘

---

## 按角色推荐阅读路径

| 角色/场景           | 推荐阅读顺序                                             | 说明                         |
| ------------------- | -------------------------------------------------------- | ---------------------------- |
| **产品经理/决策者** | product/ → solution/OVERVIEW.md → solution/TECH_STACK.md | 理解产品价值和技术方向       |
| **架构师**          | solution/ → architecture/ → implementation/INTERFACES.md | 从方案到设计到接口定义       |
| **[后端开发者]**    | [根据项目填写]                                           | [说明]                       |
| **[前端开发者]**    | [根据项目填写]                                           | [说明]                       |
| **新成员入职**      | product/ → solution/ → architecture/README.md → issues/  | 从产品到方案到架构，了解待办 |
| **运维/部署**       | solution/DEPLOYMENT_*.md → implementation/CONFIG.md      | 部署方案和配置实现           |

---

## 代码开发快速指引

> 每次开发必读，包含项目中最关键的编码规则与方法（~5分钟阅读）

→ **[coding-index.md](./coding-index.md)** - 开发必读索引

快速跳转：

- [命名约定](./coding-index.md#命名约定)
- [错误处理模式](./coding-index.md#错误处理模式)
- [TDD 流程](./coding-index.md#tdd-流程)
- [代码组织](./coding-index.md#代码组织)
- [依赖管理](./coding-index.md#依赖管理)

---

## AI 加载指引

### 按场景的加载策略

| 场景             | 推荐加载文档                                                         | 加载层次  |
| ---------------- | -------------------------------------------------------------------- | --------- |
| **快速概览项目** | spec/README.md + product/README.md                                   | Level 1   |
| **理解产品价值** | product/README.md + product/VALUE.md                                 | Level 1-2 |
| **理解整体方案** | solution/README.md + solution/OVERVIEW.md + solution/ARCHITECTURE.md | Level 1-2 |
| **理解架构设计** | architecture/README.md + architecture/CORE.md                        | Level 2   |
| **开发实施**     | implementation/INTERFACES.md → 对应实现文档                          | Level 3   |
| **部署应用**     | solution/DEPLOYMENT_*.md + implementation/CONFIG.md                  | Level 2-3 |
| **处理问题**     | issues/README.md → 对应分类文档                                      | Level 1-2 |
| **完整理解项目** | 自顶向下：product → solution → architecture → implementation         | 全层次    |

### 层次化加载建议

**Level 1 - 快速概览**（~500 tokens）

- 仅加载各集合的 README.md
- 用途：快速了解项目全貌

**Level 2 - 标准加载**（~2000 tokens）

- 加载 solution/README.md + architecture/README.md + product/README.md
- 用途：理解产品方向、整体方案、架构概览

**Level 3 - 详细加载**（~5000+ tokens）

- 根据需要加载 implementation/ 的具体文档
- 用途：开发实施、问题调试

### 场景化快速查找

需要了解产品定位 → product/OVERVIEW.md
需要理解核心价值 → product/VALUE.md
需要技术选型信息 → solution/TECH_STACK.md
需要部署方案 → solution/DEPLOYMENT_*.md
需要架构细节 → architecture/CORE.md
需要目录结构 → architecture/DIRECTORY.md
需要接口定义 → implementation/INTERFACES.md
需要查看遗留问题 → issues/README.md

---

## 文档使用建议

### 对于人类读者

1. **新项目启动**：从 product/ 开始，理解产品定位和价值
2. **技术方案评审**：重点阅读 solution/ 和 architecture/
3. **开发实施**：参考 architecture/ 设计，查阅 implementation/ 实现
4. **问题追踪**：定期关注 issues/ 中的遗留问题和演进计划

### 对于 AI Agent

1. **优先加载 Level 1 文档**（各 README.md）了解全貌
2. **根据问题类型**加载对应 Level 2-3 文档
3. **避免一次性加载全部 implementation/**，按需加载
4. **issues/** 单独加载，用于追踪待办和演进

---

## 文档维护说明

### 更新规则

- **product/** 更新：产品方向、需求变更时更新
- **solution/** 更新：端到端输入、输出变化时、技术选型、部署方案变更时更新
- **architecture/** 更新：架构重构、组件调整时更新
- **implementation/** 更新：代码规范、实现变更时同步更新
- **issues/** 更新：问题状态变更、新增问题时更新

### 版本同步

- 各文档集合版本独立管理
- major 版本变更需同步更新相关引用
- 保持文档间链接有效性

---

> **文档导航**: [product/](./product/README.md) | [solution/](./solution/README.md) | [architecture/](./architecture/README.md) | [implementation/](./implementation/README.md) | [issues/](./issues/README.md)
```

---

## 模板使用说明

### 必须遵守的规则

1. **方括号 `[xxx]` 标记**：需根据实际项目填写
2. **元信息块**：必须放在文档头部，使用引用格式 `>`
3. **文档总数**：动态计算，格式为 `N 个文档（M 个文档集合）`
4. **文档关系图**：必须包含，使用 ASCII 图表
5. **不得省略**：任何章节都必须存在，即使内容简略
6. **不得自定义**：不得添加模板中未定义的章节
7. **避免重复**：如内容在其他文档中存在，使用引用：`参见 [文档名](./path) 第X章节`

### 根据项目类型调整

**桌面应用**：

```
核心组成：
1. **Electron 主进程** - 窗口管理、后端托管
2. **后端服务** - FastAPI/Flask 业务逻辑
3. **前端界面** - Vue/React 渲染进程
```

**CLI 工具**：

```
核心组成：
1. **命令解析** - argparse/click/typer
2. **核心逻辑** - 业务处理函数
3. **输出模块** - 格式化输出结果
```

**Web 应用**：

```
核心组成：
1. **API 层** - 路由和请求处理
2. **业务层** - 核心业务逻辑
3. **数据层** - 数据库和模型
```

### 示例：Electron-FastAPI 项目

```markdown
# Electron-FastAPI Demo 文档总览

> **版本**: v1.0  
> **更新时间**: 2025-01-15  
> **状态**: 渐进式文档已生成  
> **文档总数**: 28 个文档（5 个文档集合）

---

## 项目定位

**Electron-FastAPI Demo** - 演示 Electron 托管 FastAPI 后端的桌面应用架构

**核心组成**:

1. **Electron 主进程** - 窗口管理、后端进程托管、IPC 通信
2. **FastAPI 后端** - RESTful API、业务逻辑、数据处理
3. **Vue 前端** - 用户界面、状态管理、API 调用

---

## 文档集合索引

| 集合                | 抽象层次 | 职责定位                             | 文档数 | 详细文档                                |
| ------------------- | -------- | ------------------------------------ | ------ | --------------------------------------- |
| **product/**        | 业务层   | 定义产品方向（为什么做、为谁做）     | 5      | [README.md](./product/README.md)        |
| **solution/**       | 方案层   | 整体解决方案（如何解决、技术选型）   | 8      | [README.md](./solution/README.md)       |
| **architecture/**   | 设计层   | 架构与代码组织（如何设计、如何组织） | 6      | [README.md](./architecture/README.md)   |
| **implementation/** | 实现层   | 实施细节代码（如何实现、具体代码）   | 8      | [README.md](./implementation/README.md) |
| **issues/**         | 维护层   | 问题追踪管理（遗留问题、演进计划）   | 1      | [README.md](./issues/README.md)         |

...（后续章节按模板填充）
```

---

## 生成时机

**步骤 5：生成总览 README** 时，必须：

1. **读取此模板文件**：`references/spec-README-template.md`
2. **填充项目实际数据**：
   - 项目名称、版本、日期
   - 核心组成（根据项目类型）
   - 文档数量统计
   - 各层职责范围
3. **验证完整性**：
   - 所有 `[xxx]` 标记已替换
   - 所有章节都存在
   - 文档数量与实际一致
4. **写入文件**：`spec/README.md`

**禁止**：

- 跳过任何章节
- 自定义章节名称
- 省略文档关系图
- 忽略 AI 加载指引
