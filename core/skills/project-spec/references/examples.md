# 示例输出

本文档提供典型项目生成的文档示例，供参考。

---

## 示例 1：Electron-FastAPI 项目 - solution/ARCHITECTURE.md

```markdown
# 架构选型

## 总体架构

本项目采用 **Monorepo + 前后端分离** 架构：

- **Monorepo 组织**：使用 pnpm workspaces 管理多个包
- **前端托管后端**：Electron 主进程负责启动/管理 FastAPI 子进程
- **动态端口通信**：Python 后端动态分配端口，通过 stdout 输出，前端捕获并建立连接
- **认证机制**：生成 AUTH_TOKEN 并通过环境变量传递，确保前后端通信安全

## 为什么这样设计？

### 选择 Monorepo 的原因

1. **统一版本管理**：前后端版本号统一，避免版本不匹配问题
2. **依赖共享**：protocol 包在前后端间共享，类型定义一致
3. **构建编排**：pnpm 协调前后端构建顺序，自动处理依赖关系

### 前端托管后端的原因

1. **用户体验**：用户无需手动启动后端，点击桌面图标即可使用
2. **生命周期管理**：前端启动时启动后端，退出时自动清理
3. **跨平台一致**：无论 Windows/macOS/Linux，体验一致

### 动态端口的原因

1. **避免端口冲突**：硬编码端口可能与其他应用冲突
2. **多实例支持**：每个实例可独立运行在不同端口
3. **灵活性**：开发/测试/生产环境无需修改端口配置

## 架构图
```

┌─────────────────────────────────────┐
│ Electron 主进程 (index.ts) │
│ - 启动 FastAPI 子进程 │
│ - 管理 BrowserWindow │
│ - 提供 IPC API │
│ - 监控后端健康状态 │
└─────────────┬───────────────────────┘
│ spawn + stdout
↓
┌─────────────────────────────────────┐
│ FastAPI 后端 (main.py) │
│ - 提供 RESTful API │
│ - 业务逻辑处理 │
│ - 输出 PORT:xxxx 到 stdout │
└─────────────────────────────────────┘
↑
│ HTTP + AUTH_TOKEN
│
┌─────────────────────────────────────┐
│ Vue 渲染进程 (renderer-ui) │
│ - 用户界面 │
│ - 通过 IPC 或 fetch 调用后端 │
└─────────────────────────────────────┘

```

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Monorepo 工具 | pnpm workspaces | 依赖管理高效，支持包间引用 |
| 进程通信 | HTTP (同端口动态分配) | 简单直接，跨语言通用 |
| 认证方式 | AUTH_TOKEN + HTTP Header | 轻量级，无需复杂认证框架 |
| 打包方案 | PyInstaller + Electron Builder | 成熟稳定，跨平台支持好 |
| 日志方案 | Python logging → stdout → Electron logger | 统一输出，集中管理 |

## 注意事项

- **端口探测**：后端启动后必须在 stdout 输出 `PORT:xxxx`，否则前端会超时
- **进程清理**：关闭窗口时必须清理 Python 进程（包括孤儿进程）
- **虚拟环境路径**：打包后的 Python 执行器路径需在打包脚本中正确配置
- **认证安全**：AUTH_TOKEN 每次启动随机生成，不持久化

## 相关文档

- [solution/TECH_STACK.md](./TECH_STACK.md) - 详细技术栈说明
- [solution/PROCESS_MANAGEMENT.md](./PROCESS_MANAGEMENT.md) - 进程管理方案
- [architecture/PROCESS.md](../architecture/PROCESS.md) - 进程架构详情
- [architecture/COMMUNICATION.md](../architecture/COMMUNICATION.md) - 通信机制详情
```

---

## 示例 2：CLI 工具 - solution/USAGE.md

````markdown
# 使用说明

## 安装

```bash
pip install complexity-analyzer
```
````

## 基本用法

```bash
# 分析单个文件
complexity-analyzer analyze src/main.py

# 分析目录
complexity-analyzer analyze ./src --recursive

# 输出 JSON 格式
complexity-analyzer analyze ./src --format json --output result.json

# 按复杂度排序
complexity-analyzer analyze ./src --sort-by complexity --descending
```

## 命令参考

### analyze

分析代码复杂度。

**参数**：

- `<path>` - 文件或目录路径
- `--recursive, -r` - 递归分析目录
- `--format <fmt>` - 输出格式（table/json/yaml），默认 table
- `--output <file>` - 输出到文件
- `--sort-by <field>` - 排序字段（complexity/name）
- `--threshold <n>` - 仅显示复杂度 ≥ n 的函数

**示例**：

```bash
# 仅显示复杂度 ≥ 10 的函数
complexity-analyzer analyze ./src --threshold 10

# 生成报告并导出
complexity-analyzer analyze ./src --format json --output report.json
```

### report

生成详细报告（HTML）。

```bash
complexity-analyzer report ./src --output report.html
```

## 配置文件

可在项目根目录创建 `.complexityrc.yaml`：

```yaml
# 默认配置
threshold: 10
exclude:
  - '*/tests/*'
  - '*/migrations/*'
format: table
```

## 退出码

- `0` - 成功
- `1` - 分析失败（文件不存在/解析错误）
- `2` - 配置错误

````

---

## 示例 3：Web 应用 - architecture/API.md

```markdown
# API 设计

## API 风格

本项目采用 **RESTful API** 风格，使用 FastAPI 框架实现。

## 端点设计

### 用户相关

| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| POST | /api/auth/register | 用户注册 | 无 |
| POST | /api/auth/login | 用户登录 | 无 |
| GET | /api/users/me | 当前用户信息 | JWT |
| PUT | /api/users/me | 更新用户信息 | JWT |

### 项目相关

| 方法 | 路径 | 描述 | 认证 |
|------|------|------|------|
| GET | /api/projects | 获取项目列表 | JWT |
| POST | /api/projects | 创建项目 | JWT |
| GET | /api/projects/{id} | 获取项目详情 | JWT |
| PUT | /api/projects/{id} | 更新项目 | JWT |
| DELETE | /api/projects/{id} | 删除项目 | JWT |

## 请求/响应格式

### 标准响应格式

```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功"
}
````

### 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "参数验证失败",
    "details": [...]
  }
}
```

## 认证机制

使用 **JWT Bearer Token** 认证：

```
Authorization: Bearer <token>
```

**Token 结构**：

```json
{
  "sub": "user_id",
  "exp": 1234567890,
  "iat": 1234560000
}
```

## 速率限制

- 默认：100 请求/分钟
- 认证端点：10 请求/分钟
- 超限返回：HTTP 429 Too Many Requests

## 相关文档

- [architecture/AUTHENTICATION.md](./AUTHENTICATION.md) - 认证实现详情
- [implementation/ROUTES.md](../implementation/ROUTES.md) - 路由实现
- [implementation/MIDDLEWARE.md](../implementation/MIDDLEWARE.md) - 中间件实现

```

---

## 使用示例的时机

生成文档时参考这些示例：

1. **相同项目类型** → 直接参考结构
2. **相似技术栈** → 参考设计和说明风格
3. **不同类型** → 仍参考文档组织方式（章节、表格、图表）

**注意**：不要直接复制示例内容，应根据实际项目调整。
```
