# v20 Roadmap — 从 Point Frontier 到 Trajectory Frontier

> 2026-04-19  
> 适用范围：gomoku × autoresearch 主线前半段  
> 配套文档：`updates/v20-findings.md`、`updates/v20-update.md`

---

## 1. 一句话目标

> **把当前“会跑 sweep、会画 Pareto 图”的点状探索能力，升级成“能从冷启动逐步发现可行参数、做长周期晋升、分叉 continuation、并越来越少地浪费实验预算”的轨迹型研究系统。**

---

## 2. 为什么需要这份 roadmap

当前代码已经证明两件事：

1. **我们能看到 point frontier。**
   - 已有 `sweep.py`
   - 已有 `analyze.py --pareto --plot`
   - 已有 `frontier_snapshots`

2. **但我们还不能从零找回 v15 最后那条成功训练路线。**
   - 找不到“何时降 LR”
   - 找不到“何时把 MCTS 提到 800”
   - 找不到“何时把 eval 从 L1 切到 L2”
   - 也不能把 resume 分支当成一等研究对象

所以这份 roadmap 的任务不是继续吹大 v20，而是把路线说清楚：

> **v20 解决观察层；v20+ 之后的节点，才开始逐步解决 search space 语义、multi-fidelity、trajectory、主动选点、贝叶斯探索。**

---

## 3. 路线总判断

如果目标只是：

> “从已有 runs 里画一个 Pareto front”

那么现在已经够用。

如果目标是：

> “从冷启动开始，最终重新逼近 v15/v16 那种长期有效参数，并且越来越接近真正 Pareto front”

那么还必须补 4 层能力：

1. **search space 语义层**
2. **multi-fidelity / promotion 层**
3. **trajectory / continuation 层**
4. **active selection 层**

---

## 4. 节点依赖关系

```text
v20
  ↓
v20.1
  ↓
v20.2
  ↓
v20.3
  ↓
v21
  ↓
v21.1
```

含义：

- **v20** 先把 frontier 看见、记住、按协议比较
- **v20.1** 再让 search space / campaign 变成一等公民
- **v20.2** 再让实验从“平面 runs”升级成“分层晋升”
- **v20.3** 再把 continuation / branch 变成 trajectory
- **v21** 再让系统学会推荐下一步
- **v21.1** 才有资格接贝叶斯多目标探索

---

## 5. 节点总表

| 节点 | 定位 | 核心产出 | 收口证据 | 解锁下一步 |
|---|---|---|---|---|
| **v20** | Point frontier observation layer | 出图、可配置比较、frontier snapshot、sweep 自动出图 | PNG 产物、frontier_snapshots、命令验证 | v20.1 |
| **v20.1** | Search-space schema + campaign ledger | search space 元数据、campaign 实体、protocol 锁 | 至少 1 个 gomoku campaign 正式入库并可汇总 | v20.2 |
| **v20.2** | Multi-fidelity promotion engine | Stage A/B/C/D 预算层、晋升规则、seed 复验纪律 | 至少 1 条 A→B→C 晋升链入库 | v20.3 |
| **v20.3** | Continuation / trajectory explorer | checkpoint 分叉、branch reason、trajectory 对比 | 至少 1 个父 checkpoint 产生 2 条以上 continuation 并完成对比 | v21 |
| **v21** | Surrogate-guided selector | next-point / next-branch 排名器 | 系统建议至少命中 1 个新 front 或近 front 候选 | v21.1 |
| **v21.1** | Bayesian multi-objective frontier search | MOBO / EHVI 级主动探索 | 同预算下 frontier 质量优于或不差于基线 sweep | 后半段路线 |

---

## 6. 节点详述

### 6.1 v20 — Point Frontier Observation Layer

**定位**

> 先把 point frontier 看见、画出来、存下来，并且确保比较发生在同 protocol 内。

**为什么它是第一步**

没有这个节点，后面所有节点都没有统一观察面。系统连“当前 front 长什么样”都说不清，就更不可能做晋升、分叉或主动选点。

**代码工作包**

1. `framework/analyze.py`
   - `--pareto --plot`
   - `--maximize` / `--minimize`
   - `--eval-level` / `--sweep-tag`
2. `framework/pareto_plot.py`
   - 2D scatter + frontier line
3. `framework/core/db.py`
   - `frontier_snapshots`
4. `framework/sweep.py`
   - sweep 结束后自动触发 Pareto 分析与出图

**收口证据**

1. `uv run python framework/analyze.py --pareto --plot` 可生成 PNG
2. `frontier_snapshots` 表成功落库多条记录
3. 同一数据库上能按 `eval_level`、`sweep_tag` 稳定过滤
4. `updates/v20-update.md` 完整记录实际交付与验证

**节点完成定义**

- 系统已经具备 point-frontier 观察能力
- 但不声称能发现长期最优训练路线

**它给 v20.1 的输入**

- 已有 point-frontier 观察面
- 已有比较维度
- 已有最基础的实验分组能力（`sweep_tag`）

---

### 6.2 v20.1 — Search-Space Schema + Campaign Ledger

**定位**

> 把“可枚举参数列表”升级成“带研究语义的 search space”，并把一轮研究变成 campaign，而不是零散 runs。

**必须解决的问题**

当前系统不知道：

- 哪个轴是离散 / 连续 / log-scale
- 哪个轴是结构变量 / 训练变量 / 慢变量
- 哪个轴适合短预算筛查
- 哪个轴只该通过 continuation 验证
- 哪些 runs 本来就不该拿来直接比较

**建议代码触点**

1. `framework/core/db.py`
   - 新增 `campaigns`
   - 新增 `campaign_runs`
   - 新增 `search_spaces` 或等价表
2. `framework/sweep.py`
   - 支持 `--campaign`
   - 支持读取 schema/profile
3. `framework/analyze.py`
   - 新增 campaign 视图 / summary
   - 强化 protocol comparability guard
4. `domains/gomoku/`
   - 首份 gomoku search-space profile

**收口证据**

1. 至少 1 个 gomoku 冷启动 campaign 正式入库
2. 至少 1 个 search-space profile 明确标注：
   - 轴类型
   - 默认范围
   - 是否 log-scale
   - 是否允许 continuation
3. `analyze.py` 能输出 campaign summary
4. 跨 protocol 比较被明确拒绝或警告

**节点完成定义**

- 系统开始知道“点为什么这么选”
- 但仍然不会自动晋升，也不会主动推荐下一点

**它给 v20.2 的输入**

- 同一 campaign 的可比 runs 集合
- search space 的语义元数据

---

### 6.3 v20.2 — Multi-Fidelity Promotion Engine

**定位**

> 把实验从“全平面的 runs”变成“短预算筛查 → 中预算复筛 → 长预算晋升 → continuation”的分层流程。

**核心原因**

v15 的经验已经证明：

> **短预算只能筛掉明显差点，不能决定长期最优路线。**

所以系统必须内建 fidelity 分层，否则永远会把短期信号误当长期结论。

**建议代码触点**

1. `framework/core/db.py`
   - 新增 `campaign_stages`
   - 新增 `promotion_decisions`
2. `framework/sweep.py`
   - 支持按 stage 预算运行
3. `framework/analyze.py`
   - 输出 stage summary
   - 输出 promotion rationale
4. 可能的 gomoku 配置文件
   - 定义 Stage A/B/C/D 预算与门槛

**建议 stage 语义**

1. **Stage A**：短预算筛查
2. **Stage B**：中预算复筛
3. **Stage C**：长预算晋升
4. **Stage D**：continuation / branch 验证

**收口证据**

1. 至少 1 条 gomoku campaign 跑通 A→B→C
2. 至少 1 个候选被晋升，至少 1 个候选被淘汰
3. 数据库中能查出 promotion reason
4. 同一 stage 内的 seed 方差被记录并参与决策

**节点完成定义**

- 系统不再把所有 run 混成一个平面
- 仍然没有把 continuation 本身做成一等对象

**它给 v20.3 的输入**

- 哪些候选值得被延长
- 哪些 checkpoint 值得被分叉

---

### 6.4 v20.3 — Continuation / Trajectory Explorer

**定位**

> 把 resume 分支从“训练技巧”升级成“研究对象”，正式开始研究 trajectory frontier。

**核心问题**

v15 最终有效的解，不是单次 run 的参数点，而是多段 continuation：

- LR 逐段衰减
- MCTS 逐段提高
- benchmark 逐段升级

如果系统不会表示 branch，就永远只能找 point frontier。

**建议代码触点**

1. `framework/core/db.py`
   - 新增 `run_branches` 或等价表
   - 明确 `branch_reason`
2. `framework/analyze.py`
   - branch tree / trajectory report
   - parent vs child 对比
3. `framework/sweep.py` 或新工具
   - 支持从 checkpoint 批量分叉 continuation

**branch reason 最少应支持**

- `lr_decay`
- `mcts_upshift`
- `eval_upgrade`
- `seed_recheck`
- `buffer_or_spc_adjust`

**收口证据**

1. 至少 1 个父 checkpoint 产生 2 条以上 continuation
2. 至少 1 份 branch tree 报告可视化展示 parent/child 关系
3. 至少 1 条 continuation 在长期结果上优于 parent 或同层候选
4. `analyze.py` 能明确说明每条 branch 的 reason 与结果

**节点完成定义**

- 系统从 point frontier 进入 trajectory frontier
- 但仍未实现自动推荐下一条 trajectory

**它给 v21 的输入**

- 可比较的 trajectory 数据
- 可学习的“什么 continuation 更值钱”

---

### 6.5 v21 — Surrogate-Guided Next-Point / Next-Branch Selection

**定位**

> 先不上最重的贝叶斯，先让系统学会“基于已有 campaign/trajectory 数据推荐下一步”。

**为什么不直接上 BO**

因为在 v20.3 之前：

- comparability 还不稳
- fidelity 还没建模
- continuation 还不是一等实体

直接 BO 很容易学到噪声。

**建议代码触点**

1. `framework/analyze.py`
   - `--recommend-next`
2. 新模块
   - frontier gap scoring
   - uncertainty scoring
   - seed variance penalty
3. 数据库
   - 持久化 recommendation 及其理由

**最小可行策略**

不要求一上来就是完整 surrogate。先做：

1. frontier 邻域稀疏区补点
2. 高方差候选优先复验
3. 慢变量候选优先 continuation
4. 已明显 dominated 的区域停止加预算

**收口证据**

1. `--recommend-next --campaign X` 能输出排序后的候选列表
2. 推荐结果包含理由和所依据的证据
3. 至少 1 个被推荐候选进入新 front 或接近新 front
4. 相比盲扫，浪费在明显 dominated 点上的预算减少

**节点完成定义**

- 系统具备轻量主动选点能力
- 但仍未进入标准意义上的 BO / MOBO

**它给 v21.1 的输入**

- 有组织的 recommendation 历史
- 可用的 acquisition 基线

---

### 6.6 v21.1 — Bayesian Multi-Objective Frontier Search

**定位**

> 在 comparability、fidelity、trajectory 都已经建模后，再引入多目标贝叶斯探索，把昂贵实验数量压下来。

**建议目标**

- maximize: `win_rate`
- minimize: `params`, `wall_time`
- 可选扩展：stability、memory、promotion_success

**建议前提**

只在下列条件都成立时启用：

1. campaign protocol 固定
2. fidelity 层次明确
3. continuation/trajectory 语义明确
4. recommendation 日志已存在，可做基线比较

**建议代码触点**

1. 新增 optimizer / acquisition 模块
2. `framework/analyze.py`
   - acquisition summary
3. 数据库
   - 记录 surrogate snapshot、acquisition score、chosen point

**收口证据**

1. 在固定预算下，与网格或启发式基线对比
2. frontier 质量达到“更好或不差”
3. 昂贵 runs 数量下降
4. 所有 BO 决策都有可追溯记录

**节点完成定义**

- 系统开始具备真正的多目标主动探索能力

---

## 7. 适用于全部节点的研究纪律

### 7.1 不跨 protocol 假比较

以下条件不一致时，不直接比较 frontier：

- `eval_level`
- `eval_opponent`
- fidelity stage
- 是否 continuation
- campaign 目标

### 7.2 不把短 sweep 当最终参数发现器

短预算 sweep 的职责只有两个：

1. 排除明显差点
2. 建立 response surface 的第一层地图

它**不能**单独证明长期最优路线。

### 7.3 每个节点都必须有事实证据收口

每次发布不能只说“代码已写完”，还必须给出至少一种真实证据：

- 数据库记录
- 可复现命令输出
- 图表产物
- campaign / branch / recommendation 报告

### 7.4 先做轻量主动选点，再做完整贝叶斯

原因不是保守，而是为了避免：

> **在错误的数据纪律上堆更复杂的自动化。**

---

## 8. 当前状态与下一步

### 8.1 当前状态

- **v20：已完成**
  - 观察层和 point-frontier 工具已落地
- **v20.1：应为下一个代码节点**
  - 先做 search-space schema 和 campaign ledger

### 8.2 为什么 v20 仍然有效

因为它虽然不能直接找回 v15 的最终路线，但它解决了一个不可绕开的前提：

> **没有 point frontier 的统一观察面，就没有后面的 campaign、promotion、trajectory、selector。**

### 8.3 最终判断

> **这条 roadmap 的核心，不是把 v20 否掉，而是承认 v20 只是第一层。**
>
> **从这里开始，v2x 系列的目标应该是：从 point frontier 稳定过渡到 trajectory frontier。**
