---
name: idea-mining
description: 创意系统化挖掘方法论，帮助Agent从趋势、痛点、技术等维度主动发掘和构建创意，并可调用创意系统插件API
license: MIT
metadata:
  version: v1.1
---

# 创意系统化挖掘

本Skill为Agent提供主动挖掘创意的框架，避免等创意上门。同时提供与创意系统插件的API交互能力。

---

## 创意系统插件 API 调用

创意系统插件运行在 `packages/backend/` 后端，提供REST API。当后端运行时（http://127.0.0.1:9820），Agent可通过bash调用curl命令直接操作创意数据。

### API端点速查

```bash
# 创建创意
curl -X POST http://127.0.0.1:9820/api/ideas \
  -H "Content-Type: application/json" \
  -d '{"name":"创意名称","description":"描述","stage":"new","parent_id":null}'

# 查询创意列表
curl http://127.0.0.1:9820/api/ideas?page=1&page_size=20

# 搜索创意
curl "http://127.0.0.1:9820/api/ideas?search=关键词"

# 按状态筛选
curl "http://127.0.0.1:9820/api/ideas?stage=validating"

# 获取创意详情
curl http://127.0.0.1:9820/api/ideas/{id}

# 更新创意
curl -X PUT http://127.0.0.1:9820/api/ideas/{id} \
  -H "Content-Type: application/json" \
  -d '{"stage":"priority","score_survival":85}'

# 删除创意
curl -X DELETE http://127.0.0.1:9820/api/ideas/{id}

# 获取完整创意树
curl http://127.0.0.1:9820/api/ideas/tree/all

# 获取统计
curl http://127.0.0.1:9820/api/ideas/stats/overview
```

### Agent调用示例

```
# 1. 检查后端是否运行
curl -s http://127.0.0.1:9820/api/health

# 2. 批量录入Brainstorm创意
curl -X POST http://127.0.0.1:9820/api/ideas -H "Content-Type: application/json" -d '{...}'

# 3. 查看当前创意池
curl http://127.0.0.1:9820/api/ideas?sort_by=score_survival&sort_desc=true

# 4. 查看统计看板
curl http://127.0.0.1:9820/api/ideas/stats/overview
```

### Brainstorm后自动录入

Brainstorm session结束后，Agent应：

1. 将Top 3创意通过API录入系统
2. 如有推理关系，设置parent_id和relation_type
3. 更新现有创意的评分或状态

---

## 创意挖掘维度

### 维度1：技术趋势扫描

| 方法             | 说明                                         | 频率     |
| ---------------- | -------------------------------------------- | -------- |
| GitHub Trending  | 关注Star增速快的项目，寻找可商业化的开源方向 | 每周     |
| Product Hunt Top | 分析上榜产品的共性和趋势                     | 每周     |
| Hacker News      | 技术社区热议话题                             | 每日扫描 |
| arXiv            | 最新研究论文的实用化潜力                     | 每月     |

**挖掘方法**：发现热门技术 → 问"这个技术能做什么产品？" → 按画像权重打分

### 维度2：痛点迁移

用户的个人经历和技能可以解决哪些领域的痛点：

```
用户精通领域：
- 电力电子 → 能源管理、硬件监控、智能电网相关软件
- AI/ML → 各类智能工具、自动化、预测分析
- Agent开发 → AI Agent平台、开发者工具
- 嵌入式 → IoT设备管理、边缘计算软件
- 全栈开发 → 任意Web/桌面应用

交叉机会：电力电子 × AI × Web全栈
→ 示例创意：智能电源管理SaaS平台
```

### 维度3：自己的痒点（Dogfooding）

最有效的创意来源：

```
你在工作/生活中遇到什么问题？
- 现有的工具哪里不爽？
- 有什么重复性劳动可以自动化？
- 有什么"如果有个工具就好了"的瞬间？

Agent建议：记录在 projects/idea-pool.md 中
```

### 维度4：竞品不满提取

从竞品差评中提取创意：

```
方法：
1. 浏览App Store / Trustpilot / G2的低分评价
2. 提取"如果xxx就好了"的模式
3. 评估：这个改进能否独立成一个产品？

示例：
竞品A差评："导出格式太少"→ 创意：通用格式转换工具
竞品B差评："学习曲线太陡"→ 创意：简化版新手友好工具
```

### 维度5：平台机会扫描

新平台、新API、新市场带来的机会窗口：

- Apple Vision Pro生态 → 空间计算应用
- 新发布的LLM能力（如function calling、vision）→ 智能工具
- 政策变化 → 合规工具
- 新社交平台崛起 → 内容工具

---

## Brainstorm流程

Agent在brainstorm模式下的行为：

### Session结构

```
Round 1: 发散（15分钟）
- 目标：量 > 质，至少产出20个原始创意
- 规则：不评判、不筛选、不分析可行性
- 输出：创意列表（一句话描述）

Round 2: 收敛（15分钟）
- 目标：从20个中选出5个最有潜力的
- 规则：快速按当前阶段权重打分
- 输出：Top 5 创意卡

Round 3: 深化（15分钟）
- 目标：对Top 3进行初步验证
- 规则：快速搜索验证（竞品存在？搜索量？）
- 输出：Top 3 验证结果 + 推荐排序
```

### 创意卡片模板

```markdown
## 创意：[一句话描述]

- 解决什么问题：
- 目标用户：
- 变现方式：
- 技术可行性：[高/中/低]
- 开发周期预估：
- 与用户技能匹配度：[高/中/低]
- 温饱阶段评分：xx/100
- 自我实现评分：xx/100
```

---

## 创意管道管理

```
创意状态流转：
  新创意 → 初步验证 → 进入管道 → 高优先级 → 开发中 → 已上线
                ↓            ↓
             放弃/暂存    低优先级（择机）
```

### 创意池文件

`projects/idea-pool.md`：

```markdown
# 创意池

## 高优先级（下个月开发）

| 创意 | 评分 | 预估收入 | 开发周期 | 状态     |
| ---- | ---- | -------- | -------- | -------- |
| xxx  | 85   | $x/月    | 2周      | 准备开发 |

## 管道中（择机启动）

| 创意 | 评分 | 备注 |
| ---- | ---- | ---- |

## 暂存（需要更多信息）

| 创意 | 评分 | 待验证 |
| ---- | ---- | ------ |

## 已放弃

| 创意 | 放弃原因 |
| ---- | -------- |
```

---

## 创意挖掘频率建议

| 频率   | 动作                          | 产出         |
| ------ | ----------------------------- | ------------ |
| 每日   | 浏览HN/Product Hunt，记录灵感 | 原始创意笔记 |
| 每周   | 正式Brainstorm session        | 更新创意池   |
| 每月   | 创意池大盘点，重新排序        | 月度创意报告 |
| 每季度 | 趋势扫描 + 方向校准           | 战略方向建议 |
