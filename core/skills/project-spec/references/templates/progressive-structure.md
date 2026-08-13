# 渐进式文档组织原则

本文档定义规范文档的渐进式组织结构，遵循 `progressive-doc` skill 的三层分级体系。

---

## 三层分级体系

### Level 1：概要层（README.md）

**用途**：快速理解本层级核心内容，无需深入其他文档。

**结构模板**：

```markdown
# [层级名称]

## 核心概要
（一句话描述本层级核心内容）

## 关键要素
- 要素 1：简要说明
- 要素 2：简要说明

## 详细文档导航
| 文档 | 用途 | 适用场景 |
|------|------|---------|
| [DOC.md](./DOC.md) | 详细描述 | 需要深入理解时 |

## AI 加载指引
- 快速理解：仅读本 README
- 深入细节：跳转到对应文档
```

**示例**（product/README.md）：

```markdown
# 产品层

## 核心概要
定义项目的产品价值、目标用户和核心特性。

## 关键要素
- 项目价值：为谁解决什么问题
- 目标用户：用户画像和使用场景
- 核心特性：MVP 功能清单

## 详细文档导航
| 文档 | 用途 | 适用场景 |
|------|------|---------|
| [OVERVIEW.md](./OVERVIEW.md) | 项目概览 | 了解项目背景 |
| [VALUE.md](./VALUE.md) | 价值主张 | 理解核心价值 |
| [REQUIREMENTS.md](./REQUIREMENTS.md) | 需求说明 | 了解功能需求 |

## AI 加载指引
- 快速理解：仅读本 README（约 30 行）
- 深入细节：根据需要跳转对应文档
```

---

### Level 2：分类详情层

**命名规范**：
- 大写 + 下划线：`OVERVIEW.md`, `TECH_STACK.md`, `PROCESS.md`
- 描述性强：看到文件名就知内容

**结构模板**：

```markdown
# [主题名称]

## 核心内容
（主体内容，结构化呈现）

## 关键要点
- 要点 1
- 要点 2
- 要点 3

## 注意事项
（如有）

## 相关文档
- [相关文档](./path) - 关联说明
```

**示例**（solution/TECH_STACK.md）：

```markdown
# 技术栈

## 核心内容

### 前端技术
- **框架**：Electron + React
- **语言**：TypeScript
- **构建**：Vite

### 后端技术
- **框架**：FastAPI
- **语言**：Python 3.10+
- **数据库**：SQLite（开发）/ PostgreSQL（生产）

## 关键要点
- 前后端分离：Electron 渲染进程与后端进程独立
- 类型安全：TypeScript + Pydantic
- 开发优先：优先保证开发体验

## 注意事项
- Python 版本要求：≥ 3.10（使用了新语法）
- Node 版本要求：≥ 18（Electron 要求）

## 相关文档
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 架构设计
- [PROCESS_MANAGEMENT.md](./PROCESS_MANAGEMENT.md) - 后端进程管理
```

---

### Level 3：子目录详细层（可选）

**使用场景**：
- 文档内容过多（>1000 行）
- 需要按模块/服务/功能分别组织

**示例结构**：
```
architecture/
├── README.md
├── CORE.md
└── SERVICES/              # Level 3 子目录
    ├── README.md
    ├── user-service.md
    ├── auth-service.md
    └── notification-service.md
```

**子目录 README.md 模板**：

```markdown
# 服务架构

## 核心概要
微服务架构的服务划分和通信机制。

## 服务列表
| 服务 | 职责 | 文档 |
|------|------|------|
| user-service | 用户管理 | [详细](./user-service.md) |
| auth-service | 认证授权 | [详细](./auth-service.md) |

## AI 加载指引
- 快速理解：读本 README（约 20 行）
- 深入服务：跳转对应服务文档
```

---

## 文档层级应用

### coding-index.md（spec 根文档）

**层级定位**：特殊文档，介于 Level 1 和 Level 2 之间

**特点**：
- 包含摘要（Level 1）+ 快速命令参考（Level 1.5）+ 详细链接（Level 2）
- 阅读时间：~5 分钟
- 每次开发必读

**适用场景**：
- 开发前快速回顾关键规则
- 需要 lint/test/tsc 命令速查

---

### CODING_STANDARDS.md（architecture 层）

**层级定位**：Level 2 详情文档

**与前者的关系**：
- coding-index.md 是摘要版（~200行）
- CODING_STANDARDS.md 是完整版（~150行详细规范）
- 前者链接到后者

---

### spec/README.md（项目根概要）

```markdown
# 项目规范文档

## 核心概要
本项目采用 5 层规范结构：product/solution/architecture/implementation/issues。

## 文档结构
| 层级 | 用途 | 适用角色 |
|------|------|---------|
| [product/](./product/) | 产品定义 | 产品经理、业务 |
| [solution/](./solution/) | 方案设计 | 架构师、技术负责人 |
| [architecture/](./architecture/) | 技术架构 | 开发人员 |
| [implementation/](./implementation/) | 实现细节 | 开发人员 |
| [issues/](./issues/) | 问题追踪 | 全员参与 |

## AI 加载指引
- 快速理解：读本 README + 各层级 README（约 5×30=150 行）
- 方案级理解：深入到 solution/*.md（约 5×100=500 行）
- 完全理解：深入所有文档（视项目规模而定）
```

### 各层级 README 应用

**原则**：
- 每个 `product/`, `solution/`, `architecture/`, `implementation/`, `issues/` 都有 README.md
- README.md 遵循 Level 1 结构（核心概要 + 关键要素 + 导航）
- Level 2 文档提供详细内容
- Level 3 子目录仅在必要时创建

---

## 渐进式加载策略

### AI Agent 加载顺序

**场景 1：快速理解项目**
```
加载量：约 150 行
  1. spec/README.md（约 30 行）
  2. 各层级 README.md（5×30≈150 行）
```

**场景 2：理解方案设计**
```
加载量：约 500 行
  1. 快速理解项目（150 行）
  2. product/*.md（约 3×100=300 行）
  3. solution/*.md（约 5×100=500 行）
```

**场景 3：深入实现细节**
```
加载量：约 1500 行
  1. 理解方案设计（500 行）
  2. architecture/*.md（约 5×150=750 行）
  3. implementation/*.md（约 5×150=750 行）
```

**场景 4：完全理解**
```
加载量：约 2000-5000 行（视项目规模）
  1. 深入实现细节（1500 行）
  2. issues/*.md（约 2×100=200 行）
  3. Level 3 子目录（如有）
```

### 人类读者加载顺序

**建议**：
1. 先读 `spec/README.md` 了解整体结构（1 分钟）
2. 根据角色跳转到对应层级 README：
   - 产品经理 → `product/README.md`
   - 架构师 → `solution/README.md`
   - 开发人员 → `architecture/README.md` + `implementation/README.md`
3. 需要时深入 Level 2 文档

---

## 与 progressive-doc skill 的集成

### 集成策略

**调用时机**：

| 场景 | 调用方式 | 输入 |
|------|---------|------|
| **生成 Level 1 README** | 调用 progressive-doc 模式 2（从零创建） | 层级名称 + 关键要素 |
| **生成 Level 2 详情文档** | 调用 progressive-doc 模式 2 | 文档主题 + 内容大纲 |
| **优化现有文档** | 调用 progressive-doc 模式 1 | 现有文档内容 |

**映射关系**：

```
progressive-doc Level 1 (Metadata) 
  → project-spec README.md (概要)

progressive-doc Level 2 (Summary)
  → project-spec Level 2 详情文档（TECH_STACK.md, CORE.md, ...）

progressive-doc Level 3 (Details)
  → project-spec Level 3 子目录/implementation/
```

### 降级策略

如果 progressive-doc skill 不可用：
1. 直接使用 file-templates/ 下的模板
2. 手动填充内容
3. 记录体验，后续可改进模板

### 特殊文档处理

**coding-index.md**：
- 不适合直接调用 progressive-doc
- 手动生成摘要 + 链接结构
- 确保包含"快速命令参考"章节

**CODING_STANDARDS.md**：
- 可调用 progressive-doc 模式 2
- 输入：编码约束大纲 + 开发模式大纲
- 输出：完整规范文档

---

## 验证清单

生成后验证：

- [ ] 每个层级都有 `README.md`
- [ ] `README.md` 包含"核心概要"、"关键要素"、"详细文档导航"、"AI 加载指引"
- [ ] Level 2 文档命名规范（大写+下划线）
- [ ] Level 2 文档包含"核心内容"、"关键要点"、"相关文档"
- [ ] Level 3 子目录（如有）也遵循渐进式组织
- [ ] 文档间交叉引用正确无误
- [ ] coding-index.md 存在且包含"快速索引"和"核心规范摘要"
- [ ] CODING_STANDARDS.md 存在且包含完整编码规范
- [ ] coding-index.md 正确链接到 CODING_STANDARDS.md
