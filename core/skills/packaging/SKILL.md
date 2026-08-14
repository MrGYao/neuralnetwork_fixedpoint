---
name: packaging
description: 应用打包指南，帮助 developer-backend/developer-frontend/deployer 按技术栈完成打包并验证产物
license: MIT
metadata:
  version: v1.0
  last-updated: 2026-07-13
---

# 应用打包 Skill

加载本 Skill 后，Agent 根据自身角色执行对应技术栈的打包流程。

---

## 谁在何时加载

| Agent              | 加载时机                            |
| ------------------ | ----------------------------------- |
| developer-frontend | 前端所有测试通过 → 交付前的最后一步 |
| developer-backend  | 后端所有测试通过 → 交付前的最后一步 |
| deployer           | 正式部署/发布前 → 验证全部打包产物  |

---

## 技术栈: Electron + Vite + TypeScript + Vue

适用于桌面端 Electron 应用。

### 打包方案

#### 方案一: electron-builder（默认推荐）

| 项目     | 内容                                                             |
| -------- | ---------------------------------------------------------------- |
| 命令     | `pnpm build`（已在 package.json 中配置 electron-builder 构建链） |
| 配置     | `electron-builder.yml`                                           |
| 产物     | `release/*.exe` / `.msi` / `.dmg`（根据平台）                    |
| 适用场景 | 首次打包、通用场景                                               |

#### 方案二: electron-forge

| 项目     | 内容                      |
| -------- | ------------------------- |
| 命令     | `npx electron-forge make` |
| 配置     | `forge.config.ts`         |
| 产物     | `out/make/*`              |
| 适用场景 | 需要自定义安装器逻辑时    |

### 验证清单

- [ ] `pnpm build` 命令零报错退出
- [ ] 产物文件存在于预期路径
- [ ] 安装包体积在合理范围内（< 500MB）
- [ ] 无 `.env` / 密钥文件被打包进去
- [ ] 无 `node_modules` / 临时文件泄漏

---

## 技术栈: Python + FastAPI

适用于 Python 后端服务。

### 打包方案

#### 方案一: PyInstaller

| 项目       | 内容                                              |
| ---------- | ------------------------------------------------- |
| 命令       | `pyinstaller --onefile --name {应用名} server.py` |
| 产物       | `dist/{应用名}.exe`                               |
| 体积       | 较大（约 50–200MB，因打包了整个 Python 运行时）   |
| 启动速度   | 较慢（首次运行需解压）                            |
| 调试便利性 | 容易，`--onedir` 模式可逐文件排查                 |
| 构建环境   | 仅需 pip install pyinstaller                      |

#### 方案二: Nuitka

| 项目       | 内容                                                        |
| ---------- | ----------------------------------------------------------- |
| 命令       | `nuitka --standalone --onefile --output-dir=dist server.py` |
| 产物       | `dist/server.exe`                                           |
| 体积       | 较小（C 编译后体积优于 PyInstaller）                        |
| 启动速度   | 较快（原生代码，无解压步骤）                                |
| 调试便利性 | 困难，编译后堆栈信息不直观                                  |
| 构建环境   | 需要 C 编译器（MSVC/MinGW），编译时间较长                   |

### 方案选择流程

当存在多种可行打包方案时，**禁止直接选择**，必须按以下步骤执行：

```
Step 1: 读取 projects/{项目}/delivery/architecture.md 了解项目约束
Step 2: 读取 projects/{项目}/packaging/（如有）参考历史经验
Step 3: 按以下维度对比各方案
        - 目标用户环境（是否要求免安装 Python？）
        - 产物体积（用户有无传输限制？）
        - 启动速度（冷启动是否影响体验？）
        - 调试便利性（后续是否频繁更新？）
        - 构建环境复杂度（是否需要额外工具？）
Step 4: 给出推荐方案 + 理由（不超过 3 句话）
Step 5: 用 question 工具提交用户做最终决策
        "推荐使用 {方案名}，理由：... 是否采用？"
```

**例外**：如果 `projects/{项目}/packaging/` 下已有明确的历史记录表明某个方案经过验证，则直接复用，跳过用户选择。

### 验证清单

- [ ] 打包产物可独立启动（不依赖 Python 环境）
- [ ] 所有 API 端点响应正常（可用 curl 快速验证）
- [ ] 产物中无 `.py` 源码泄漏（单文件模式应只保留 `.exe`）
- [ ] 端口号可在外部配置（未硬编码在二进制的只读区）

---

## 技术栈: 静态站点 / SPA（新增预留）

（待扩展）

---

## 产品总验证清单

deployer 部署前必须逐条检查：

- [ ] 前端打包产物存在且可启动
- [ ] 后端打包产物存在且可启动
- [ ] 前后端端口无冲突
- [ ] 无 `.env` / `*.db` / 密钥文件被意外打包
- [ ] 安装包可在目标操作系统正常运行
- [ ] 核心用户流程走通（启动 → 登录 → 操作 → 退出）

---

## 项目特有覆写

每个项目可以在 `projects/{项目}/packaging/config.md` 中覆写默认值：

```markdown
# {项目名} 打包配置

## 前端

- 根目录: code/{项目}/frontend/
- 构建命令: pnpm build
- 产物目录: code/{项目}/frontend/release/

## 后端

- 入口: code/{项目}/backend/server.py
- 打包方案: PyInstaller
- 产物目录: code/{项目}/backend/dist/

## 总体验证

- 后端端口: 9821
- 启动顺序: 先后端 → 再前端
```

### AutoTag 默认值（未覆盖时生效）

```
前端根目录:   code/AutoTag/frontend/
后端入口:     code/AutoTag/backend/server.py
后端端口:     9821
后端方案:     PyInstaller（首次打包时需按选择流程确认）
```

---

## 经验回写

每次打包成功后，**必须**向 `projects/{项目}/packaging/` 回写经验：

```markdown
# {日期} - {打包方案} 打包记录

## 基本信息

- 打包方案: PyInstaller (v6.x)
- 产物路径: code/{项目}/backend/dist/{产物名}
- 产物体积: xx MB

## 遇到的坑

- 问题描述 + 解决方案

## 后续建议

- 下次打包可改进的点
```

累积的经验将成为后续项目打包时的参考依据。新项目打包时，优先查阅历史记录。
