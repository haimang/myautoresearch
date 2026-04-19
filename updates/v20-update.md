# v20 Update — v2x Roadmap Node 1 / Point Frontier Observation Layer

> 2026-04-19  
> 状态：**已完成并已验证**  
> 路线归属：`updates/v20-roadmap.md` 的第一个节点（v20）

---

## 1. 节点身份（一句话）

> **v20 的任务不是替我们重新发现 v15 最后的长期训练路线，而是先把 point frontier 的观察层和执行层补齐：会出图、会过滤、会持久化、会在 sweep 结束后自动回到 Pareto 分析。**

这是整个 v2x 路线的第一步。  
它解决的是：

- “现在 front 长什么样？”
- “哪些点在同 protocol 下可比？”
- “sweep 结束后能否自动回到 frontier 观察？”

它**不解决**：

- 长周期参数发现
- multi-fidelity 晋升
- checkpoint continuation 分叉
- 主动选点 / 贝叶斯优化

---

## 2. 为什么 v20 仍然有效

在重新审视 v15/v16 的长期实验之后，结论已经很清楚：

> **仅靠当前的 v20 工具，不能从零找回 v15 最终那条有效训练轨迹。**

但这并不意味着 v20 失效。相反，v20 仍然是后续所有节点的必要前置，因为：

1. 没有 point frontier 的统一观察面，就没有后续 campaign / promotion / trajectory 的共同语境
2. 没有 protocol 内比较，就无法判断哪些候选值得继续投预算
3. 没有 frontier snapshot 持久化，就无法让后续主动选点建立在稳定历史之上

因此，v20 的准确定位应改写为：

> **它是 v2x 系列的“观察层 + 执行层”版本，不是“最终参数发现器”。**

---

## 3. 本节点的正式目标

### 3.1 目标

1. 把 `--pareto` 从文本表格升级到 **可视化散点图**
2. 把 Pareto 轴从硬编码升级到 **CLI 可配置**
3. 把 frontier 分析从“一次性输出”升级到 **可持久化快照**
4. 把 sweep 收尾从“手动分析”升级到 **自动触发 pareto --plot**

### 3.2 非目标

1. 不实现 search-space schema
2. 不实现 campaign entity
3. 不实现 multi-fidelity promotion
4. 不实现 continuation / trajectory explorer
5. 不实现 next-point selection
6. 不实现 Bayesian / MOBO

---

## 4. 本节点在 roadmap 中的位置

```text
v20 (本文件)
  ↓
v20.1 search-space schema + campaign ledger
  ↓
v20.2 multi-fidelity promotion
  ↓
v20.3 continuation / trajectory explorer
  ↓
v21 surrogate-guided selection
  ↓
v21.1 Bayesian multi-objective exploration
```

v20 的职责非常克制：

> **先把 point frontier 看见、记住、在同协议下比较出来。**

---

## 5. 实际交付范围

| 工作线 | 交付内容 | 状态 |
|---|---|---|
| P1 | Pareto 散点图可视化 | ✅ |
| P2 | 任意轴配置 + protocol 过滤 + frontier snapshot | ✅ |
| P3 | sweep 结束后自动 Pareto 出图 | ✅ |
| P4 | 基于新训练 campaign 的实证 findings | ❌ 不属于本节点闭环 |

这里最重要的修正是：

> **旧版 v20 草案里把“实际 gomoku sweep 实验”和“v20-findings-v2 数据分析”也算进了 v20。现在这些内容不再作为 v20 发布门槛，而是转交给后续 roadmap 节点。**

原因很简单：  
v20 的职责是补齐 point-frontier 工具层，而不是用一次短 sweep 假装完成长期参数发现。

---

## 6. 受影响文件树状结构

```text
mag-gomoku/
├── framework/
│   ├── analyze.py                 # [改] Pareto 文本/绘图/轴配置/过滤/快照
│   ├── sweep.py                   # [改] sweep 后自动触发 Pareto 分析
│   ├── pareto_plot.py             # [新] Pareto 散点图模块
│   └── core/
│       └── db.py                  # [改] frontier_snapshots 表
├── pyproject.toml                 # [改] matplotlib 依赖
├── updates/
│   ├── v20-findings.md            # [读] 主线判断与边界分析
│   └── v20-update.md              # [改] 本文件，作为 roadmap 节点文档
└── docs/
    └── v20-roadmap.md             # [新] v2x 系列路线图（后补治理文档）
```

---

## 7. 分阶段交付内容

### 7.1 P1 — Pareto 可视化

| 编号 | 文件 | 实际交付 |
|---|---|---|
| P1-A | `pyproject.toml` | 新增 `matplotlib>=3.8.0` |
| P1-B | `framework/pareto_plot.py` | 新建 `plot_pareto()`：scatter + frontier line + annotations |
| P1-C | `framework/analyze.py` | `cmd_pareto()` 集成绘图逻辑 |
| P1-D | `framework/analyze.py` | 新增 `--plot` / `--output` |
| P1-E | `framework/analyze.py` | 跨 `eval_level` 自动择最大组并警告 |

### 7.2 P2 — 轴可配置 + protocol 过滤 + frontier 持久化

| 编号 | 文件 | 实际交付 |
|---|---|---|
| P2-A | `framework/analyze.py` | `--maximize` / `--minimize` 自定义维度 |
| P2-B | `framework/analyze.py` | `--eval-level` 过滤 |
| P2-C | `framework/analyze.py` | `--sweep-tag` 过滤 |
| P2-D | `framework/analyze.py` | 支持 `cycles` / `steps` / `throughput` / `lr` |
| P2-E | `framework/core/db.py` | 新表 `frontier_snapshots` |
| P2-F | `framework/pareto_plot.py` | `_AXIS_META` 标签和格式化映射 |
| P2-G | `framework/pareto_plot.py` | 图表轴跟随 CLI 配置 |

### 7.3 P3 — sweep → pareto 自动闭环

| 编号 | 文件 | 实际交付 |
|---|---|---|
| P3-A | `framework/sweep.py` | sweep 结束自动调用 `analyze.py --pareto --plot` |
| P3-B | `framework/sweep.py` | 输出文件名包含 tag：`output/pareto_{tag}.png` |
| P3-C | `framework/sweep.py` | 支持 `--no-auto-pareto` |

---

## 8. 关键实现事实

### 8.1 代码事实

1. `framework/pareto_plot.py` 已存在并使用 `matplotlib.use("Agg")`，可在 headless 环境下出图
2. `framework/analyze.py` 已支持：
   - `--plot`
   - `--maximize`
   - `--minimize`
   - `--eval-level`
   - `--sweep-tag`
   - `--output`
3. `frontier_snapshots` 已进入 `tracker.db` schema
4. `framework/sweep.py` 已在 sweep 收尾阶段自动执行 Pareto 分析

### 8.2 行为事实

当前系统已经能：

1. 只看 L1 的 runs 画图
2. 只看指定 `sweep_tag` 的 runs 画图
3. 把 `WR vs params` 改成 `WR vs wall_s`
4. 在无数据时稳定返回友好提示
5. 在只有单点时也能出图，不依赖 frontier 折线

---

## 9. 验证证据

### 9.1 已执行测试

共执行 **10 项验证**，全部通过。

| # | 测试 | 命令 / 场景 | 结果 |
|---|---|---|---|
| T1 | 纯文本 Pareto 输出 | `--pareto` | ✅ 自动选 L1，输出正确 |
| T2 | L0 过滤 + 绘图 | `--pareto --eval-level 0 --plot` | ✅ PNG 95KB |
| T3 | L1 过滤 + 自定义输出 | `--pareto --eval-level 1 --plot --output pareto_L1.png` | ✅ PNG 106KB |
| T4 | 自定义轴 | `--pareto --maximize wr --minimize wall_s --eval-level 1 --plot` | ✅ PNG 92KB |
| T5 | 空结果处理 | `--pareto --eval-level 99` | ✅ 友好提示 |
| T6 | 单点数据出图 | `--pareto --eval-level 2 --plot` | ✅ PNG 62KB |
| T7 | JSON 输出 | `--pareto --eval-level 0 --format json` | ✅ 结构正确 |
| T8 | 快照持久化 | 查询 `frontier_snapshots` | ✅ 6 条记录 |
| T9 | 语法检查 | 四文件 AST parse | ✅ 全通过 |
| T10 | 依赖安装 | `uv sync` | ✅ 成功 |

### 9.2 已生成产物

```text
output/
├── pareto_front.png        95 KB
├── pareto_L1.png          106 KB
├── pareto_L2.png           62 KB
└── pareto_wr_vs_time.png   92 KB
```

### 9.3 已落库证据

`frontier_snapshots` 已有 **6 条记录**，覆盖：

- 默认轴
- 自定义轴
- 不同 `eval_level`
- 不同 frontier 过滤条件

---

## 10. 节点收口标准

作为 roadmap 节点，v20 以“point-frontier 观察层是否闭环”为唯一收口标准。

| 标准 | 事实证据 | 状态 |
|---|---|---|
| `--pareto --plot` 能出图 | 4 个 PNG 产物 | ✅ |
| 比较维度可配置 | `--maximize/--minimize` 已验证 | ✅ |
| protocol 过滤可用 | `--eval-level` / `--sweep-tag` 已实现 | ✅ |
| frontier 可持久化 | `frontier_snapshots` 6 条记录 | ✅ |
| sweep 收尾自动回 Pareto | `framework/sweep.py` 已接入 | ✅ |
| 行为被文档化 | 本文件 + `updates/v20-roadmap.md` | ✅ |

**结论：v20 已达到 roadmap 节点的正式发布条件。**

---

## 11. 本节点没有解决什么

这是理解 v20 边界最关键的一节。

v20 **没有**解决：

1. search space 语义
2. campaign 一等实体
3. multi-fidelity 晋升
4. checkpoint continuation / branch
5. trajectory frontier
6. next-point / next-branch selection
7. Bayesian multi-objective exploration

所以 v20 不能回答的问题包括：

- “从零开始时，什么参数会在 2000+ cycle 后最强？”
- “什么时候该从 5e-4 衰减到 2e-4 / 1e-4 / 5e-5？”
- “什么时候该把 MCTS 从 400 提到 800？”
- “什么时候该从 L1 切到 L2？”

这些问题必须交给后续节点解决。

---

## 12. v20 对后续节点的输入

| 下一节点 | v20 提供什么前置 |
|---|---|
| **v20.1** | 已有 point frontier 观察面；已有 filter 纪律；已有快照历史 |
| **v20.2** | 已有可比较 runs 的基本视图；可开始为 stage/campaign 建模 |
| **v20.3** | 已有 frontier 记录，可用于判断哪些 checkpoint 值得分叉 |
| **v21** | 已有最基础的历史 front 数据，可作为推荐器冷启动输入 |

---

## 13. 对旧版 v20 草案的范围修正

旧版 v20 文档里，以下内容曾被写成 v20 自身的闭环：

1. 运行一轮新的 9 点 gomoku sweep
2. 基于这轮新数据写 `v20-findings-v2.md`
3. 用这轮短 sweep 来回答“Pareto frontier 在 gomoku 上是否有效”

现在这三件事统一改判为：

> **它们不是 v20 节点的发布门槛，而是后续 roadmap 节点里的研究活动。**

原因有二：

1. 一轮短 sweep 只能证明 point-frontier 工具可用，不能证明长期参数发现问题已经解决
2. 如果把它们硬塞进 v20，会继续模糊“观察层”和“决策层”的边界

---

## 14. 工作日志（整理版）

### 14.1 实际完成项

1. 增加 `matplotlib` 依赖
2. 新建 `framework/pareto_plot.py`
3. 重写 `analyze.py:cmd_pareto()`
4. 新增 6 个 Pareto CLI 参数
5. 新增 `frontier_snapshots` 表
6. 接通 `sweep.py` 自动 Pareto 收尾

### 14.2 实际验证结果

1. 共 10 项验证全部通过
2. 产出 4 张 PNG
3. 写入 6 条 `frontier_snapshots`
4. 自定义轴、跨 level 保护、单点图、空结果处理全部稳定

### 14.3 实际经验

1. CJK 字体在 matplotlib 默认字体上不稳定，因此图表标签改为英文
2. `eval_level` 自动择组很重要，否则 L0/L1/L2 混画会直接污染判断
3. `sweep.py` 自动回 Pareto 非常值钱，它让“看完 sweep 还得手动收尾”这个断点消失了

---

## 15. 一句话结论

> **v20 现在应被视为 v2x 路线上的第一个正式更新：它完成了 point frontier 的观察层与执行层，但并不声称已经解决长期参数发现。**
>
> **它的成功，不在于“找到了最终参数”，而在于“让后续节点终于有了统一、可重复、可持久化的 frontier 观察面”。**
