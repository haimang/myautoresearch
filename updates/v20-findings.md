# v20 Findings — 回到 autoresearch 主线后的基础设施盘点

> 2026-04-16 | 本文只讨论 **autoresearch 主线能力**：实时观测、参数空间探索、Pareto 前沿、实验数据库。
> 不讨论当前 S2 训练节奏，也不依赖 v15 / v16 的版本叙事。
> 备注：本文初稿写作时 `updates/v20-update.md` 尚未合入，因此这里直接基于现有代码做一次基础设施审计。

---

## 1. 一句话结论

> **我们已经有了“实验记录器 + 批量 sweep + 固定轴 Pareto 分析器”的雏形，但还没有到“只给任务、不懂 domain，也能自动探索可用参数区并持续逼近 Pareto 前沿”的完成态。**
>
> 更具体地说：
> - **实时接口：未完成。** 现在有 TUI 渲染和 Web 服务基础，但没有训练 telemetry 的 websocket / SSE / structured stream API。
> - **Pareto 探索：部分完成。** 现在能做手工定义网格的 sweep，并对已完成 runs 做固定轴 Pareto 排序；但还不能自动定义搜索空间、主动选点、逐步逼近 frontier。
> - **数据库：中等偏强，但不够审计级。** 现在足够记录 run / cycle / checkpoint / eval breakdown / recordings；但没有完整事件日志、stdout/stderr、代码版本、决策轨迹、agent 行为审计。

---

## 2. 当前已经具备的基础设施

先把“已有能力”说清楚，避免把系统说得过弱。

### 2.1 实验执行与扫描

- `framework/sweep.py`
  - 已支持 **笛卡尔积 sweep**
  - 已支持 `--resume` 跳过已完成组合
  - 已支持 `--dry-run` 预览矩阵
  - 已支持 `--train-script`，因此入口层面是 **domain-agnostic**

### 2.2 实验分析

- `framework/analyze.py`
  - 已支持 `--runs`
  - 已支持 `--matrix`
  - 已支持 `--compare` / `--compare-by-steps`
  - 已支持 `--report --format json`
  - 已支持 `--pareto`

### 2.3 实验数据库

- `framework/core/db.py`
  - `runs`：记录一次运行的硬件、核心超参、最终结果
  - `cycle_metrics`：记录周期级 loss / win_rate / eval 元数据
  - `checkpoints`：记录阶段性模型与 full eval 结果
  - `eval_breakdown`：记录更细的 opening 级别评估
  - `recordings`：记录对局录像元数据
  - `opponents`：记录注册对手及晋升链

### 2.4 训练内观测

- `domains/gomoku/train.py`
  - 训练过程中维护了大量 **内存态指标**：
    - `events`
    - `wr_history`
    - `loss_history`
    - `policy_loss_history`
    - `value_loss_history`
    - `mcts_sims_per_sec_history`
    - `mcts_entropy_history`
    - `last_mcts_stats`
  - `framework/core/tui.py` 已将其抽成纯渲染 helper

**结论：** 我们不是从零开始；基础骨架已经在了，问题在于这些能力还没有被统一成“autoresearch 控制面”。

---

## 3. 问题 1：TUI / websocket 实时接口是否具备？

### 3.1 当前状态：**只有终端 TUI，没有训练 telemetry API**

现状分三层：

1. **渲染层存在**
   - `framework/core/tui.py` 是纯文本渲染工具
   - `domains/gomoku/train.py` 中 `_draw_panel()` 会把当前训练状态拼成完整 panel

2. **状态层存在，但在内存里**
   - `train.py` 中的 `events`、`wr_history`、`last_mcts_stats`、`eval_status_str` 都是运行期内存变量
   - `_log_event()` 只是把消息 append 到 `events`，并在非 TTY 时直接 print

3. **网络层只服务于对弈 UI，不服务于训练 telemetry**
   - `domains/gomoku/web/web_app.py` 是 FastAPI
   - 但它当前只提供 **HTTP REST** 接口给浏览器下棋
   - 没有 `WebSocket` endpoint
   - 也没有训练状态订阅接口

### 3.2 这意味着什么

如果你现在说：

> “我想通过 websocket 拿到 MCTS 训练过程中的实时效果”

答案是：

> **当前基础设施不支持。**

更准确地说，不是“完全没有基础”，而是：

- **数据源有**：训练进程内部已有大量实时状态
- **展示端有**：已有 TUI、已有 FastAPI Web 服务
- **缺的就是中间层**：没有把训练状态抽象成结构化事件流并向外发布

### 3.3 距离“可用实时接口”还缺什么

至少缺 4 件事：

1. **structured telemetry schema**
   - 现在 TUI 依赖的是 Python 局部变量
   - 还没有统一的 `RunSnapshot` / `RunEvent` 数据结构

2. **publisher**
   - 训练循环需要周期性产出 snapshot
   - 例如每个 cycle、每次 probe、每次 checkpoint、每条 event

3. **transport**
   - websocket / SSE / NDJSON stdout 三选一或并存

4. **subscriber-safe persistence**
   - 如果前端断线，当前 `events` 会丢
   - 没有 replay 机制

### 3.4 判断

| 维度 | 结论 |
|---|---|
| 终端 TUI | **有** |
| FastAPI 基础 | **有** |
| 训练 telemetry API | **没有** |
| websocket 训练实时接口 | **没有** |
| 作为 v20 主线支撑是否足够 | **不够** |

**结论：** 实时观测这一块，当前代码只完成了“本地单进程终端可见”，还没有完成“autoresearch 可远程消费的控制面接口”。

---

## 4. 问题 2：Pareto 前沿探索基础设施是否准备完全？

### 4.1 当前状态：**能做“手工 sweep + 事后排序”，不能做“自动 frontier exploration”**

现有能力如下。

#### A. 能扫参数

`framework/sweep.py` 已支持：

- `num_blocks`
- `num_filters`
- `learning_rate`
- `steps_per_cycle`
- `buffer_size`
- `seed`

它能：

- 生成网格
- 顺序运行
- 用 `sweep_tag` 分组
- 用 `--resume` 跳过已完成项

#### B. 能做矩阵汇总

`framework/analyze.py --matrix <tag_prefix>` 已能汇总：

- mean WR
- WR std
- mean loss
- throughput
- params

#### C. 能做 Pareto 排序

`framework/analyze.py:cmd_pareto()` 已实现非支配排序，但它现在是：

- **对象固定：completed runs**
- **目标固定：maximize WR**
- **成本固定：minimize params + wall_time**

也就是说，它已经能算一个 **固定定义的 Pareto front**。

### 4.2 但为什么我仍然判断“未准备完全”

因为如果我们的目标是：

> “在对 gomoku 几乎一无所知时，让系统自己探索参数可用区，并逐步构建 Pareto frontier”

那当前系统还缺少下面这些关键能力。

#### 4.2.1 搜索空间不是一等公民

现在 sweep 的搜索空间是 CLI 上写死的。

系统不知道：

- 哪些参数是离散/连续/有序
- 哪些参数是条件依赖的
- 参数边界是什么
- 什么是“已探索区域”
- 什么是“未探索但高价值区域”

换句话说，**系统能执行 sweep，但还不能理解 search space**。

#### 4.2.2 没有“主动选点”逻辑

当前流程是：

1. 人类给一组参数网格
2. `sweep.py` 全部跑掉
3. `analyze.py` 事后看结果

缺的是：

1. 根据已有点自动选下一个点
2. 识别 frontier 附近哪些点最值得补样
3. 判断哪里需要更多 seed 复验
4. 判断哪里已经 dominated、可以停止投入

这意味着当前系统更像 **batch runner**，还不是 **autoresearch explorer**。

#### 4.2.3 Pareto 轴是硬编码的，不是通用探索面

当前 `cmd_pareto()` 只支持：

- maximize: `wr`
- minimize: `params`, `wall_time`

但真实 autoresearch 里，我们未来可能需要：

- `steps`
- `games`
- `gpu_hours`
- `memory`
- `eval_stability`
- `promotion_eligible`
- 领域特定 truth metric

现在这些都还不能配置。

#### 4.2.4 结果对象还是 run-centric，不是 frontier-centric

当前数据库的核心单位是 `run`。

这很好，但对 frontier exploration 来说还不够，因为缺少：

- search campaign / experiment batch 的一等实体
- frontier snapshot 的版本化
- dominated / non-dominated 状态的持久化
- exploration decision 的理由记录

#### 4.2.5 “对 domain 一无所知也能找到好参数”这一目标，还差 benchmark discipline

系统现在已经有：

- `is_benchmark`
- `eval_level`
- `eval_opponent`

这是好的。

但真正的 autoresearch 还需要系统层保证：

- 不同 run 的 truth 是否可比
- 预算是否等价
- frontier 是否只在同 benchmark protocol 内比较

当前这套 discipline 主要还靠人类自觉，不是系统硬约束。

### 4.3 判断

| 能力 | 当前状态 |
|---|---|
| 手工定义 sweep grid | **有** |
| 批量执行实验 | **有** |
| sweep 结果汇总 | **有** |
| 固定轴 Pareto 排序 | **有** |
| 自动定义搜索空间 | **没有** |
| 自动选择下一实验点 | **没有** |
| 任意目标/成本轴 Pareto | **没有** |
| frontier 版本化/持久化 | **没有** |
| domain-agnostic 的主动探索闭环 | **没有** |

### 4.4 结论

> **如果问题是“现在能不能帮我画出一个已有实验集合上的 Pareto front”——可以。**
>
> **如果问题是“现在能不能在几乎不懂 gomoku 的前提下，自己探索参数可用区并逼近最优 Pareto frontier”——还不行。**

当前系统最多算：

> **frontier analysis ready**

但还不是：

> **frontier exploration ready**

---

## 5. 问题 3：数据库是否支持完整日志、监控、行为、审计？

### 5.1 当前数据库的优点

`framework/core/db.py` 这套 tracker.db 已经比普通训练脚本强很多。

它已经记录了：

#### runs

- 硬件信息：chip / cpu / gpu / memory / mlx_version
- 核心超参：blocks / filters / lr / batch / parallel / mcts / replay / spc / budget / eval_level
- 结果汇总：cycles / games / steps / final_loss / final_win_rate / wall_time / checkpoints
- 实验分组：`sweep_tag`
- 随机种子：`seed`
- benchmark 标记：`is_benchmark`

#### cycle_metrics

- cycle 时间戳
- total loss
- policy loss
- value loss
- games / steps / buffer
- probe win_rate
- eval type / eval games / eval level
- `eval_submitted_cycle`

#### checkpoints

- tag / cycle / step
- full eval 的 wins / losses / draws / avg_length
- eval 时间
- promotion 结果

#### eval_breakdown

- 更细的 opening 粒度 breakdown

#### recordings

- 对局录像元数据

#### opponents

- 注册对手
- source run / source tag
- prev_alias 晋升链

### 5.2 但为什么仍然不是“审计级数据库”

因为当前数据库更像：

> **experiment summary DB**

而不是：

> **full observability / audit DB**

主要缺口如下。

#### 5.2.1 没有事件表

`train.py` 有 `_log_event()`，但事件只在：

- 内存 `events` 列表
- stdout

里存在。

**没有持久化到 DB。**

所以以下信息在训练结束后不可查询：

- “什么时候触发了 probe”
- “什么时候 crossing threshold”
- “什么时候 auto-stop”
- “什么时候 fallback / warning / backend 选择”

#### 5.2.2 没有 stdout / stderr / command capture

数据库里没有记录：

- 训练启动命令全文
- stdout 日志
- stderr 日志
- 异常栈
- 退出原因全文

这意味着很多问题只能靠终端滚动日志或手工保存。

#### 5.2.3 没有代码版本审计

当前 DB 没有：

- git commit SHA
- dirty worktree 状态
- train.py diff / hash
- framework 版本号

这对 autoresearch 是致命缺口之一，因为 frontier 上的每个点最终都需要回答：

> “这个点到底对应哪一版代码？”

现在这件事不能靠 DB 自己回答。

#### 5.2.4 没有 agent / human decision trail

系统没有记录：

- 这次实验是谁发起的
- 为什么选这个参数
- 它来自哪条假设
- 它是 frontier 补点、回归验证，还是新方向探索

这意味着“研究过程”没有形成结构化审计链。

#### 5.2.5 缺少 sweep/campaign 一等实体

虽然 `sweep_tag` 可以分组，但数据库里没有：

- `campaigns`
- `campaign_runs`
- `frontier_snapshots`
- `experiment_decisions`

因此：

- 不能系统回答“这轮探索一共做了什么”
- 不能回答“为什么这一批组合被选中”
- 不能回答“frontier 是如何演化的”

#### 5.2.6 周期级监控仍偏摘要，缺少行为细节

当前 cycle_metrics 够做趋势图，但还不够做深审计：

- 没有 cycle 级 MCTS stats 落库
- 没有实时 health flags 落库
- 没有 replay buffer 采样行为记录
- 没有动作分布 / entropy 的持久化时间序列

这些值很多在 TUI 里看得到，但不会进入 DB。

### 5.3 判断

| 维度 | 当前状态 |
|---|---|
| 运行摘要 | **强** |
| 周期级指标 | **中等偏强** |
| checkpoint/eval 细节 | **强** |
| 录像与复盘元数据 | **有** |
| 事件日志持久化 | **没有** |
| stdout/stderr 审计 | **没有** |
| git/code 版本审计 | **没有** |
| agent 决策审计 | **没有** |
| campaign/frontier 演化记录 | **没有** |

### 5.4 结论

> **现在的 tracker.db 足够支撑“训练结果分析”，但还不足以支撑“完整的 autoresearch 审计与治理”。**

它擅长回答：

- 这个 run 的最终结果是什么？
- 哪个 checkpoint 最好？
- 最近几次 probe 怎么样？

但不擅长回答：

- 这轮研究为什么这样探索？
- 这个 frontier 点的代码上下文是什么？
- 这个异常和回退过程在哪里发生？
- 这次实验在整个研究计划里扮演什么角色？

---

## 6. 综合判断：我们距离 autoresearch 主线还差什么

如果把目标重新定义为：

> **让系统高效探索参数可用区，持续构造 Pareto frontier，并让 agent 基于这些数据做决策**

那么当前代码库的位置可以概括为：

### 6.1 已完成的部分

- **实验执行器**：有
- **实验记录器**：有
- **批量 sweep**：有
- **基础 Pareto 排序**：有
- **结构化报告**：有

### 6.2 尚未完成的部分

- **实时控制面 / telemetry API**
- **主动探索器（next-point selection）**
- **通用 search-space 描述**
- **frontier 版本化与 campaign 化**
- **完整行为/日志/代码审计**

### 6.3 因此目前的准确定位

> **当前系统是一个“可被 agent 使用的实验平台”，但还不是一个“真正的 autoresearch operating system”。**

---

## 7. 对 v20 主线的直接建议

如果 v20 真要回到主线，我建议目标不要再写成“继续提升 gomoku 棋力”，而应改成下面三条。

### P0 — 训练 telemetry 控制面

把 `train.py` 里的运行态指标抽成结构化 snapshot / event，并提供：

- stdout JSONL
- websocket / SSE
- 断线后可 replay 的最近 N 条 event

### P0 — frontier exploration 最小闭环

在现有 `sweep.py + analyze.py + tracker.db` 之上补齐：

- search space schema
- experiment campaign entity
- next-point selection
- frontier snapshot persistence

### P0 — 审计级 experiment ledger

至少补上：

- run_event 表
- command/stdout/stderr capture
- git SHA / dirty flag / code hash
- experiment rationale / decision reason

---

## 8. 最后的判断

> **如果我们坚持把 v20 定义为“回到 autoresearch 主线”，那现在最该做的不是再优化某个 domain 的 MCTS，而是把“实验平台”升级成“可观测、可审计、可主动探索 frontier 的研究系统”。**
>
> 当前仓库已经有足够好的第一层地基；但第二层——实时接口、主动探索、审计链——还没有真正建起来。

