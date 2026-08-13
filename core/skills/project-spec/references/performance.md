# 性能优化策略

本文档定义大项目分析的优化策略。当检测到项目规模较大时加载此文档。

---

## 大项目检测条件

```
触发条件（满足任一）：
  - 文件数 > 1000
  - 目录深度 > 5
  - glob 扫描时间 > 5s
  - 依赖数 > 100（package.json dependencies）
```

---

## 优化策略

### 1. 限制扫描范围

```
关键目录优先：
  - src/
  - lib/
  - app/
  - packages/（monorepo）
  - backend/ / frontend/（全栈）

跳过目录：
  - node_modules/
  - venv/ / .venv/
  - build/ / dist/
  - .git/
  - __pycache__/
  - .next/ / .nuxt/
```

### 2. 限制扫描深度

```
默认：depth <= 5
大项目：depth <= 3

实现：
  glob **/* depth=3
  → 如未覆盖关键信息，递归增加深度
```

### 3. 并行执行分析任务

```
任务并行化：
  - 结构分析（目录树）
  - 依赖分析（package.json / requirements.txt）
  - 代码分析（源码扫描）

使用 Task 工具：
  → spawn 多个子任务并行处理
```

### 4. 缓存分析结果

```
临时缓存（内存）：
  - 文件类型统计
  - 依赖列表
  - 关键文件内容

不持久化原因：
  - 避免过期数据
  - 每次执行获取最新信息
```

### 5. 降级标注

```
在 issues/TODOs.md 标注：
  "项目规模较大（[文件数] 文件），进行了以下优化：
   - 仅扫描关键目录：<list>
   - 扫描深度限制：3
   - 建议手动补充以下内容：<缺失部分>"
```

---

## 按项目类型优化

### Monorepo 项目

```
结构：
  packages/
    ├── package-a/
    ├── package-b/
    └── package-c/

优化策略：
  1. 识别 packages/* 列表
  2. 每个 package 单独分析（并行）
  3. 汇总生成总体 spec
  4. 每个 package 可有独立的 implementation/ 文档
```

### 微服务项目

```
结构：
  services/
    ├── auth-service/
    ├── api-gateway/
    └── user-service/

优化策略：
  1. 识别 services/* 列表
  2. 分析通用部分（网关、通信）
  3. 每个服务单独生成 spec/architecture/SERVICES/[service-name]/
  4. 总览 spec/solution/SERVICES.md 汇总
```

### 大型 Web 应用

```
识别：
  - src/controllers/ 文件数 > 50
  - src/models/ 文件数 > 30

优化策略：
  1. 仅分析代表性文件（前 20 个）
  2. 根据命名规则推断其他文件内容
  3. 在 issues/TODOs.md 标注：
     "已分析代表性文件，建议补充：
      - controllers/ 下未分析的文件
      - models/ 中的关联关系"
```

---

## 性能监控

```
自动计时：
  - glob 扫描时间
  - 代码分析时间
  - 文档生成时间

阈值警告：
  - 如总时间 > 60s → 提示用户："项目较大，分析耗时较长"
  - 如某步骤 > 30s → 考虑降级："步骤 X 耗时过长，已简化处理"
```

---

## 用户选择优化级别

```
可选询问：
  "检测到项目较大（X 文件），分析范围："
  → 完整分析（可能较慢）
  → 快速分析（仅关键目录，推荐）
  → 自定义（指定目录）

默认：快速分析（平衡速度和完整性）
```
