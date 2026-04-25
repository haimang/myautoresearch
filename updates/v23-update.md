# v23 Update — Constrained Bayesian Frontier Refinement / Visual Decision Layer 执行计划

> 2026-04-25  
> 前置：`updates/v22-update.md`、`updates/v22-findings.md`  
> 定位：把 v22 的 **multi-scenario frontier observation** 推进到下一阶段的 **run-scoped evidence organization + human-readable frontier visualization + constraint-active search space + Bayesian frontier refinement**。

---

## 1. v23 一句话目标

> **把 `fx_spot` 从“能批量生成 Pareto front 证据的多场景 mock domain”推进到“能按 run id 归档实验、输出人类可读 front、让约束真正塑形、并用贝叶斯式主动补点去逼近 frontier / knee”的研究系统。**

v23 不做：

1. 不做历史行情回测。
2. 不做趋势预测。
3. 不把 live conversion 接进自动研究 loop。
4. 不用“更大的笛卡尔积 brute-force”冒充 Bayesian refinement。
5. 不继续容忍 `output/` 根目录平铺 artifacts。
6. 不把“图导出了 PNG”误判为“图已经可供人类决策”。

v23 要做：

1. 为每次 `fx_spot` 执行建立 **run-scoped output workspace**。
2. 把 Pareto 图从“文字墙”修成“概览图 + 决策图 + companion table”。
3. 让 hard constraint 真正进入 front 竞争，而不是只存在于 schema。
4. 修正 route template / effective leg / bridge degeneration 的语义边界。
5. 扩大 mock pricing surface，使 uplift、amount tier、corridor 差异成为可观察变量。
6. 引入 constraint-aware Bayesian / surrogate-driven active sampling。
7. 用 replay benchmark 证明 Bayesian loop 比 random / heuristic / brute-force budget 更有信息效率。

---

## 2. 为什么 v23 不能停留在 v22 的状态

v22 通过 756 个点的多 scenario smoke，已经证明了 4 件事：

1. `fx_spot` 不再是单点 demo。
2. treasury scenario 会改变 front 结构。
3. 2-leg path 有结构性价值，不是偶发噪音。
4. objective-profile Pareto、knee point、quote evidence lineage 这条路径是成立的。

但 v22 findings 也同时明确暴露了 5 个断点：

1. **hard constraint 没有真正被压测**
   - `infeasible = 0`
   - 当前 front 更像“约束友好的局部再平衡 front”，不是贴边 constrained frontier。

2. **route template 与实际复杂度仍然混淆**
   - `via_usd / via_eur / via_hkd` 等模板会退化为 1-leg path。
   - 模板表达的是搜索意图，不是执行复杂度事实。

3. **mock pricing 还不足以支持“保值基础上的增值”探索**
   - preservation 目前大多接近 `1.0` 且略小于 `1.0`。
   - uplift 仍未被真正证明存在。

4. **当前还不是 Bayesian frontier approximation**
   - 只是更大的 Cartesian product + 事后 Pareto sort。
   - 没有 posterior uncertainty、没有 acquisition-driven resampling、没有 knee-neighborhood refinement。

5. **证据层和可视化层都还不合格**
   - PNG 被 annotation 压垮。
   - `output/` 根目录平铺 artifacts，实验边界不清楚。

所以，v23 的第一优先级不是“再多跑几百个点”，而是：

> **先把实验组织、可视化表达、约束塑形能力和贝叶斯迭代框架补起来，再谈更深的 frontier 逼近。**

---

## 3. v23 内部 Phase 总览

| Phase | 名称 | 核心产物 | 完成后解锁 |
|---|---|---|---|
| **Phase 0** | Run-scoped artifact workspace | `fx_run_id`、输出目录分层、manifest、artifact index | 干净的实验证据边界 |
| **Phase 1** | Human-readable frontier visualization | overview/front-only/knee plots、front tables、短标签体系 | 人类可读 trade-off 分析 |
| **Phase 2** | Constraint activation | floor-probe / split / aggressive amount candidate、infeasible evidence | 真实 constrained frontier |
| **Phase 3** | Route / pricing semantics repair | 退化桥接过滤、effective route signature、tiered pricing mock | 更有信息量的 search surface |
| **Phase 4** | Bayesian refinement loop | surrogate、uncertainty、acquisition、iterative resampling | 从 observation 走向 approximation |
| **Phase 5** | Replay benchmark & budget evaluation | random / heuristic / Bayesian 对比、front coverage 曲线 | 证明 BO 的实际价值 |
| **Phase 6** | 测试与文档收口 | regression、smoke、findings / README 回填 | v23 完成 |

---

## 4. Phase 0 — Run-scoped artifact workspace

### 4.1 当前问题

v22 findings 已明确写出：

1. `fx-*.log`
2. `fx-*_pareto.png`
3. `fx-*_pareto.json`
4. `fx-*_pareto.txt`
5. `tracker.db`

都直接堆在 `output/` 根目录。

这会导致：

1. 无法一眼看出本轮实验边界。
2. 多 campaign 的 artifacts 彼此混在一起。
3. 同名实验多次重跑后，人类很难复原哪张图对应哪批 quotes。
4. 后续新增 interactive/CSV/front tables/acquisition traces 后，根目录会完全失控。

### 4.2 v23 要引入的身份层次

v23 需要明确三层 identity：

1. **`campaign_id`**
   - 表示实验主题，例如 `fx-cn-exporter`。

2. **`runs.id`**
   - 表示数据库中单个 candidate / 单次执行记录。

3. **`fx_run_id` / `execution_run_id`**
   - 表示“一次 `fx_spot` 实验执行”的外层容器。
   - 这是文件系统层必须补上的那一层。

### 4.3 建议目录

推荐目录：

```text
output/
  fx_spot/
    <fx_run_id>/
      manifest.json
      tracker.db
      logs/
      campaigns/
        <campaign_id>/
          pareto/
            overview.png
            front_only.png
            knee_zoom.png
            front.json
            front.csv
            front.md
          traces/
            acquisition_trace.json
            frontier_snapshots.json
          quotes/
            quote_windows.json
            route_legs.json
          reports/
            summary.txt
```

可接受的最小版本：

```text
output/fx_spot/<fx_run_id>/
  tracker.db
  manifest.json
  <campaign artifacts...>
```

### 4.4 建议 schema / metadata

新增：

```sql
CREATE TABLE IF NOT EXISTS experiment_runs (
    id                    TEXT PRIMARY KEY,
    domain                TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    objective_profile_id  TEXT,
    output_root           TEXT NOT NULL,
    manifest_json         TEXT NOT NULL
);
```

以及：

```sql
ALTER TABLE campaigns ADD COLUMN experiment_run_id TEXT;
ALTER TABLE runs ADD COLUMN artifact_dir TEXT;
ALTER TABLE frontier_snapshots ADD COLUMN artifact_dir TEXT;
```

`manifest.json` 最低应包含：

```json
{
  "fx_run_id": "fx-20260425-cn-exporter-a",
  "domain": "fx_spot",
  "created_at": "2026-04-25T16:30:39+08:00",
  "campaign_ids": ["fx-cn-exporter"],
  "objective_profile_id": "fx-spot-v23",
  "search_space_hash": "...",
  "git_head": "...",
  "notes": "constraint smoke"
}
```

### 4.5 CLI / runtime 约束

v23 要求：

1. `framework/sweep.py` 支持 `--run-id`。
2. `domains/fx_spot/train.py` 支持接收并沿用该 `--run-id`。
3. 若不显式传入，则自动生成，但必须：
   - 落到 manifest
   - 出现在 log header
   - 出现在 summary/report 顶部
4. 默认输出目录不再是 `output/` 根层，而是：
   - `output/fx_spot/<fx_run_id>/...`

### 4.6 这一 phase 的完成标准

1. 任意一次 `fx_spot` 执行都有唯一 `fx_run_id`。
2. 同一次执行产生的 DB / plot / JSON / CSV / logs 都落在同一目录树下。
3. 根目录不再平铺实验正文 artifacts。
4. 人可以仅凭文件路径定位实验边界。

---

## 5. Phase 1 — Human-readable frontier visualization

### 5.1 v22 暴露出的可视化问题

当前 `pareto_plot.py` 的问题不是“小挪一下字号”，而是编码策略错误：

1. dominated 点也画 annotation。
2. 所有 front 点都画 annotation。
3. annotation 直接使用完整 `sweep_tag`。
4. 高维语义压缩到二维后，再试图用长文本补救。

结果就是：

> **图上首先被人眼看到的是大面积文本框，而不是 frontier 几何结构。**

### 5.2 v23 的 immediate fix

第一轮必须落地：

1. **停用 dominated annotation**
2. **front 默认只标注 knee + 少数关键点**
3. **标签改成短标签**
4. **完整明细拆到 front table**
5. **图与表成对输出**

短标签建议：

```text
F7 = MXN->USD | via_eur | 0.15 | europe
K  = USD->CNY | direct | 0.10 | cn_exporter
```

完整 sweep tag / candidate payload 只留在：

1. `front.json`
2. `front.csv`
3. `front.md`
4. hover payload（如果启用 HTML）

### 5.3 v23 要输出的图类型

每个 campaign 最低输出 4 类产物：

1. **overview plot**
   - front + dominated cloud
   - 不写 dominated annotation

2. **front-only decision plot**
   - 只画 non-dominated points + knee

3. **knee neighborhood plot**
   - 局部放大 knee 附近区域

4. **front companion table**
   - CSV / Markdown 双份输出

推荐再加：

5. **interactive HTML plot**
   - hover 显示 candidate payload
   - filter by route / scenario / sell currency / effective legs

### 5.4 视觉编码分工

v23 不应继续把所有信息都塞进 annotation。

推荐默认编码：

```text
x = embedded_spread_bps
y = liquidity_headroom_ratio
color = effective_route_family
marker = quote_scenario
size = rebalance_fraction or sell_amount_ratio
edge/highlight = knee
```

如果切换投影，图标题必须显式写出：

```text
x = ...
y = ...
projection note = this is one 2D slice of a higher-dimensional Pareto set
```

### 5.5 与 run-scoped output 的耦合

这一 phase 必须和 Phase 0 绑定：

1. 所有 plot 必须写到 `output/fx_spot/<fx_run_id>/campaigns/<campaign_id>/pareto/`
2. 所有 table 必须与 plot 同目录
3. summary 中必须回写 plot path / table path

否则：

> 图修好了，文件仍然乱，分析链路依旧不合格。

---

## 6. Phase 2 — Constraint activation

### 6.1 v22 的核心缺口

当前统计：

```text
feasible = 756
infeasible = 0
```

这并不说明系统已经足够稳健，而是说明：

```text
sell_amount = surplus_above_floor * rebalance_fraction
```

这个生成逻辑天生就规避了大部分 breach。

因此 v23 必须让约束真正进入竞争。

### 6.2 新增 candidate 维度

至少新增：

1. `sell_amount_mode = surplus_fraction | explicit_ratio | floor_probe`
2. `sell_amount_ratio`
3. `floor_buffer_target`
4. `split_ratio`
5. `parallel_route_mode`
6. `retry_or_requote_mode`

含义：

1. `surplus_fraction`
   - 延续 v22 的保守模式。

2. `explicit_ratio`
   - 允许更激进的卖出比例。

3. `floor_probe`
   - 故意贴近 liquidity floor，观察 frontier 如何在约束前沿变形。

4. `split_ratio`
   - 把一笔卖出量拆成两条路径或两种 route family。

5. `parallel_route_mode`
   - 支持多腿 / 多段局部并行执行的 mock。

### 6.3 约束结果必须显式分层

v23 不应只保留 `feasible/infeasible` 二元标签，还要记录：

1. `constraint_status_json`
2. `breach_reason_codes`
3. `breach_magnitude`
4. `headroom_before/after`

建议的 reason codes：

```text
liquidity_floor_breach
quote_expired
unsupported_pair
insufficient_balance
path_not_found
split_execution_conflict
```

### 6.4 分析层要求

`analyze.py --pareto` 必须明确区分：

1. infeasible points
2. feasible dominated points
3. feasible Pareto front
4. knee

并且能够回答：

1. 约束主要是被哪类 candidate 触发？
2. 哪类 infeasible 邻近 front？
3. 哪些点虽然 infeasible，但离 knee 很近？

### 6.5 这一 phase 的完成标准

1. 冒烟实验中出现非零 infeasible 点。
2. infeasible 原因可统计、可分布化、可画图。
3. constraint-first Pareto 不再只是“schema 上存在”，而是能真实改变 front 形状。

---

## 7. Phase 3 — Route / pricing semantics repair

### 7.1 退化桥接过滤

v22 findings 已指出：

```text
bridge_currency == sell_currency
bridge_currency == buy_currency
```

会产生“名义两跳、实际一跳”的候选。

v23 必须在 candidate generation 阶段过滤这些退化桥接点。

### 7.2 route identity 要从模板转向事实

v23 需要把以下概念拆开：

1. `route_template`
   - 搜索意图，例如 `via_usd`

2. `effective_leg_count`
   - 实际执行复杂度

3. `route_signature`
   - 实际路径，例如 `MXN->USD->CNY`

4. `bridge_family`
   - 可供聚类和画图的中间表示，例如 `usd_bridge / eur_bridge / direct`

分析、plot、surrogate 应优先用：

1. `effective_leg_count`
2. `route_signature`
3. `bridge_family`

而不是继续把 template 当唯一真相。

### 7.3 mock pricing 要支持更丰富的 price surface

如果 v23 还停留在“所有 route 都只是 slightly worse than par”的价格结构，那么“保值基础上的增值”仍然无从研究。

因此 mock provider 至少要新增：

1. **corridor-specific tier**
   - 不同 corridor 有不同价差结构。

2. **amount bucket tier**
   - 金额规模变化会影响 client rate。

3. **bridge advantage / penalty**
   - 某些桥接路径在特定 scenario 下优于 direct。

4. **provider regime skew**
   - 同一 scenario 下，不同 provider / source 模式可以有不同 price surface。

5. **validity / latency / settlement 联动**
   - 更优价格不必然伴随更优 validity 或更短 lag。

### 7.4 v23 应新增的 route/pricing 指标

建议至少增加：

| metric | 作用 |
|---|---|
| `effective_leg_count` | 实际复杂度事实 |
| `route_signature_hash` | 路径稳定标识 |
| `route_family` | 用于聚类 / 画图 |
| `sell_amount_ratio` | 约束压力与结果关联 |
| `breach_margin_ratio` | 距离 liquidity floor 的安全边际 |
| `tier_bucket` | 价格分层来源 |
| `provider_mode` | fixture / mock / sandbox_quote_only |

### 7.5 这一 phase 的完成标准

1. 退化桥接点被过滤。
2. 图和分析优先使用实际路径事实，而不是模板名。
3. mock pricing 能产生更丰富的 preservation / uplift / validity trade-off。

---

## 8. Phase 4 — Bayesian frontier refinement loop

### 8.1 v23 要解决的不是“更大扫表”，而是“更快逼近”

v22 的问题不是点数不够，而是：

> **即使把 Cartesian product 扩到更大，本质上仍然是在 brute-force + 事后 Pareto。**

v23 必须真正引入：

1. surrogate
2. posterior uncertainty
3. acquisition
4. iterative resampling
5. knee-neighborhood refinement

### 8.2 适合当前项目的实际落地路径

由于搜索空间同时包含：

1. categorical（currency / route / scenario）
2. ordinal（max_legs）
3. continuous-ish（rebalance_fraction / sell_amount_ratio）

v23 建议采用 **两层 constrained BO 架构**，而不是一上来追求最重的多目标 BO 学术实现：

#### Layer A：Feasibility surrogate

目标：

1. 预测 `P(feasible)`
2. 识别最可能触发哪类 breach

输入特征：

1. sell currency / buy currency
2. route family / effective legs
3. sell amount ratio / split ratio
4. quote scenario / treasury scenario
5. quote validity / latency / settlement features

输出：

1. `p_feasible`
2. `top_breach_reason`

#### Layer B：Objective surrogate

目标：

1. 预测 preservation / uplift / headroom / spread / lag
2. 给出 uncertainty estimate

候选实现优先级：

1. bootstrapped ensemble regressor（易实现、对 mixed feature 更稳）
2. TPE-style density estimator
3. one-hot categorical + GP kernel（只在维度受控时尝试）

v23 的要求不是“学术上最纯的 BO”，而是：

> **在当前 mixed FX search space 上，能稳定给出 uncertainty-guided next points。**

### 8.3 Acquisition 设计

v23 acquisition 至少应综合：

1. `p_feasible`
2. frontier-gap bonus
3. knee-neighborhood bonus
4. uncertainty bonus
5. complexity penalty
6. quote cost / call budget penalty

一个可执行的第一版形式：

```text
acquisition =
  p_feasible
  * (
      frontier_gap_bonus
    + knee_bonus
    + uncertainty_bonus
    + uplift_bonus
    - spread_penalty
    - lag_penalty
    - complexity_penalty
    )
```

### 8.4 Iterative loop

v23 目标 loop：

```text
1. seed observations
2. fit feasibility surrogate
3. fit objective surrogate
4. score unseen candidates
5. pick next batch
6. evaluate
7. update frontier snapshot
8. repeat until quote budget / iteration budget reached
```

每轮必须输出：

1. new front points
2. front coverage delta
3. knee shift
4. infeasible count delta
5. uncertainty reduction

### 8.5 Knee-neighborhood refinement

v23 不应只追 front coverage，还应单独追：

1. knee 附近采样密度
2. knee 稳定性
3. knee 附近 candidate 的解释性

建议输出：

1. `knee_trace.json`
2. `knee_zoom.png`
3. `knee_candidates.md`

### 8.6 这一 phase 的完成标准

1. 系统不再只靠大 sweep 发现 front。
2. 在固定 quote budget 下，Bayesian loop 的 front coverage 优于 random baseline。
3. knee neighborhood 能被主动加密，而不是碰运气扫到。

---

## 9. Phase 5 — Replay benchmark / budget evaluation

### 9.1 为什么 v23 必须有 benchmark

如果没有 benchmark，v23 很容易再次陷入：

> “看起来像在主动探索，但其实只是换了一种更复杂的 brute-force。”

所以 v23 必须设计固定 replay corpus，用来比较不同策略的 quote efficiency。

### 9.2 Benchmark 参与者

至少比较：

1. **random sampling**
2. **heuristic selector**
3. **large Cartesian sweep**（作为近似上界）
4. **Bayesian refinement loop**

### 9.3 Benchmark 维度

至少覆盖：

1. `cn_exporter_core`
2. `usd_importer_mix`
3. `global_diversified`
4. `asia_procurement_hub`
5. 至少 1 个高压 scenario（更贴近 liquidity floor）
6. 至少 1 个 uplift-capable scenario（mock provider 明确允许非单调 corridor pricing）

### 9.4 比较指标

推荐指标：

| 指标 | 含义 |
|---|---|
| `front_hit_rate_at_budget_b` | 给定 quote budget 的 front 命中率 |
| `knee_hit_rate_at_budget_b` | 给定 budget 的 knee 命中率 |
| `hypervolume_proxy` | 近似多目标覆盖质量 |
| `best_preservation_at_budget_b` | 在预算内的最好 preservation |
| `best_uplift_at_budget_b` | 在预算内的最好 uplift |
| `infeasible_rate` | 约束代价 |
| `quote_calls_used` | 信息成本 |
| `time_to_first_front_hit` | 首次命中 front 的效率 |

### 9.5 Benchmark 输出

每次 benchmark 至少输出：

1. `benchmark_summary.md`
2. `benchmark_curves.csv`
3. `strategy_comparison.png`
4. `frontier_coverage_trace.json`

并全部落到：

```text
output/fx_spot/<fx_run_id>/benchmarks/
```

### 9.6 这一 phase 的完成标准

1. fixed replay corpus 可重复运行。
2. random / heuristic / Bayesian 在同一 budget 下可直接比较。
3. findings 能用数据而不是口头描述来说明“Bayesian loop 值不值得”。

---

## 10. Phase 6 — 测试计划

### 10.1 Run-scoped output tests

新增：

```text
tests/test_fx_run_workspace.py
tests/test_experiment_run_manifest.py
```

覆盖：

1. `--run-id` 透传。
2. 自动生成 run id。
3. artifacts 写入正确目录。
4. 根目录不再平铺正文文件。

### 10.2 Visualization tests

新增：

```text
tests/test_pareto_plot_outputs.py
tests/test_front_table_export.py
tests/test_knee_plot.py
```

覆盖：

1. dominated annotation 默认关闭。
2. front 只标关键点。
3. 短标签生成稳定。
4. CSV / Markdown companion table 与 front 一致。
5. plot 文件写到 run workspace。

### 10.3 Constraint activation tests

新增：

```text
tests/test_fx_constraint_modes.py
tests/test_fx_infeasible_reasons.py
tests/test_constraint_first_pareto.py
```

覆盖：

1. `floor_probe` 会产生接近约束边界的点。
2. aggressive ratio 可触发 infeasible。
3. reason code 稳定写入。
4. infeasible 点不会混入 feasible Pareto。

### 10.4 Route / pricing tests

新增：

```text
tests/test_fx_route_signature.py
tests/test_fx_degenerate_bridge_filter.py
tests/test_fx_pricing_tiers.py
```

覆盖：

1. 退化桥接被过滤。
2. effective legs 与 template 解耦。
3. corridor tier / amount tier 能改变结果。
4. uplift-capable fixture 的行为可复现。

### 10.5 Bayesian loop tests

新增：

```text
tests/test_fx_surrogate_features.py
tests/test_fx_acquisition_loop.py
tests/test_fx_replay_benchmark.py
```

覆盖：

1. surrogate 输入特征稳定编码。
2. acquisition 结果 deterministic（在固定 seed 下）。
3. Bayesian loop 在小 fixture 上比 random 更快命中 front。
4. knee neighborhood refinement 可复现。

### 10.6 Regression 要求

仍必须保证：

1. Gomoku legacy path 不回退。
2. `run_metrics` / objective profiles 不回退。
3. v22 的 `fx_spot` smoke chain 不被破坏。

---

## 11. 验收标准

v23 完成必须同时满足：

1. 每次 `fx_spot` 执行都有 `fx_run_id`，并写入独立 output workspace。
2. `output/` 根目录不再堆放本轮实验正文 artifacts。
3. Pareto 输出至少包含：
   - overview plot
   - front-only plot
   - knee plot
   - front CSV
   - front Markdown
4. 新一轮 smoke 中出现非零 infeasible 点，且原因可统计。
5. 退化桥接被过滤，分析优先使用 `effective_leg_count / route_signature`。
6. mock provider 能生成更丰富的 preservation / uplift / validity trade-off。
7. Bayesian loop 在固定 quote budget 下优于 random baseline。
8. benchmark 输出可复现，并能回溯到 `fx_run_id`。
9. findings、README 与实际系统状态同步。

---

## 12. 建议实施顺序

1. **先做 run-scoped output**
   - 否则后面所有 plot / benchmark / trace 都没有干净容器。

2. **再做 Pareto 可视化止血**
   - 先去掉乱码，恢复人类可读性。

3. **再激活 constraint**
   - 让 front 真正开始被约束塑形。

4. **再修 route / pricing semantics**
   - 否则 surrogate 学到的是污染过的表征。

5. **再上 Bayesian refinement**
   - 让主动补点建立在更可信的 surface 上。

6. **最后跑 benchmark 和文档收口**
   - 用数据判断 v23 是否真正比 v22 前进。

---

## 13. v23 的最终边界

v23 的研究边界是：

```text
当前头寸
  + 当前 quote surface
  + 30 分钟内的局部执行窗口
  + 流动性硬约束
  + run-scoped evidence workspace
  + human-readable Pareto visualization
  + constraint-aware Bayesian refinement
```

它仍然不是：

```text
历史趋势预测
真实资金自动换汇
无人工审批的 live conversion
宏观 FX 交易系统
```

更准确地说，v23 要交付的是：

> **一个能组织证据、能被人读懂、能让约束真正进入 front 形状、并能在固定 quote budget 下主动逼近 frontier / knee 的 FX autoresearch 实验系统。**

---

## 14. v23 完成后应回填的内容

v23 完成后，建议在本文件底部追加独立收口章节，至少包含：

1. 工作日志（按列表列出新增 / 修改项）
2. 目录与 schema 变更清单
3. smoke / benchmark 结果
4. Bayesian loop 与 random / heuristic 的对比结论
5. plot 改造前后对比
6. `fx_run_id` 输出目录示例
7. README / findings 回填说明
