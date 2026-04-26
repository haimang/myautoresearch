# v22 Update — Spot FX Quote-Surface Autoresearch 执行计划

> 2026-04-25  
> 前置：`updates/v21.1-update.md`、`updates/v22-study-v2.md`  
> 定位：把 v21.1 的 acquisition-driven research loop 从 Gomoku 收口域迁移到第二个 domain：**当前报价图上的受约束 spot FX route / quote-surface 搜索**。

---

## 1. v22 一句话目标

> **把 myautoresearch 从“Gomoku 参数 / trajectory frontier”推进到真正的 multi-domain 研究框架：先完成数据库、分析、绘图、测试与文档叙事的 domain-generic 改造，再落地一个不依赖真实 Airwallex sandbox 密钥的 `spot_trader` mock domain，用当前头寸 + 当前报价 + 流动性硬约束去搜索 spot quote-surface 的 Pareto front 与 knee point。**

v22 不做：

1. 不预测行情。
2. 不做历史汇率回测。
3. 不自动执行 live conversion。
4. 不把 Airwallex sandbox 当作真实市场收益来源。
5. 不把 FX domain 强行塞进 Gomoku 的 `win_rate / params / wall_time` 语义。

v22 要做：

1. 把框架层的核心指标系统从 Gomoku legacy columns 泛化为 `run_metrics`。
2. 让 Pareto / plot / recommendation / acquisition 由 objective profile 驱动。
3. 让 sweep / candidate execution 支持动态 axis 与 JSON candidate。
4. 新增 `domains/spot_trader/`，用 mock provider 完成 quote graph、route enumeration、local mock execution、Pareto front、knee point。
5. 形成未来接 Airwallex live quote / sandbox conversion 的安全 adapter 边界。

---

## 2. 为什么 v22 必须先做 multi-domain 泛化

v21.1 已经完成：

```text
campaign
  → stage
  → branch / trajectory
  → recommendation
  → acquisition
  → outcome / replay
```

但当前代码仍有明显 Gomoku hardcode：

1. `runs.final_win_rate` 被当作主 truth。
2. `runs.num_params` 被当作成本轴。
3. `runs.wall_time_s` 被当作成本轴。
4. `cycle_metrics.win_rate` / `checkpoints.win_rate` 被大量分析命令读取。
5. `sweep.py` 的 axis 仍固定为 `num_blocks / num_filters / learning_rate / steps_per_cycle / buffer_size`。
6. `branch.py` 假设 branch 来自 model checkpoint / resume。
7. `acquisition.py` 的特征仍是 `predicted_wr / mean_params / mean_wall_s`。
8. `pareto_plot.py` 默认图轴和格式仍是 `wr / params / wall_s`。

这些对 Gomoku 是正确 legacy；对 FX spot quote-surface 是错误抽象。

因此 v22 的第一优先级不是直接接 Airwallex，而是：

> **把 framework 的“指标、候选、执行、分析、绘图”统一升级为 domain-generic。**

---

## 3. v22 内部 Phase 总览

| Phase | 名称 | 核心产物 | 完成后解锁 |
|---|---|---|---|
| **Phase 0** | 文档与硬编码审计 | Gomoku hardcode 清单、清理策略、v22-update 本文件 | Phase 1 |
| **Phase 1** | DB generic metric schema | `run_metrics`、`objective_profiles`、FX quote tables、迁移策略 | Phase 2/3 |
| **Phase 2** | Dynamic candidate execution | dynamic axes、candidate JSON、generic subprocess contract | FX mock domain |
| **Phase 3** | Objective-aware analyze / plot | generic Pareto、metric-driven plot、knee point | FX frontier 可视化 |
| **Phase 4** | FX spot mock domain | quote graph、mock provider、route eval、metrics | 无密钥 smoke 测试 |
| **Phase 5** | Selector / acquisition 泛化 | quote-surface candidate pool、constrained acquisition | active quote sampling |
| **Phase 6** | Airwallex adapter boundary | quote-only adapter、sandbox mode、no-live-conversion guard | 未来真实接入 |
| **Phase 7** | 测试与文档收口 | multi-domain regression、mock FX E2E、README 同步 | v22 完成 |

---

## 4. Phase 0 — 文档与 Gomoku hardcode 审计

### 4.1 当前统计

对 `updates/` 下 Markdown 做 broad scan：

```text
pattern:
  gomoku | Gomoku | 五子棋 | final_win_rate | win_rate | WR
  num_params | wall_time | wall_s | params

total_md = 44
matching_docs = 44
current_matching = 12
legacy_matching = 32
```

当前层 12 份全部命中：

| 文档 | 命中数 | 判断 |
|---|---:|---|
| `updates/pareto-frontier.md` | 138 | 设计文档里高度绑定 Gomoku / WR，需要改成“Gomoku 作为实例” |
| `updates/v20-findings.md` | 33 | 历史实验发现，可保留 |
| `updates/v20-roadmap.md` | 10 | roadmap 语义基本通用，但例子仍是 Gomoku |
| `updates/v20-update.md` | 10 | 历史交付记录，可保留 |
| `updates/v20.1-update.md` | 59 | search-space/campaign 设计含 Gomoku 例子 |
| `updates/v20.2-update.md` | 52 | stage/promotion 设计含 WR metric |
| `updates/v20.3-update.md` | 99 | branch/trajectory 设计强绑定 checkpoint |
| `updates/v21-code-review.md` | 23 | 历史审查记录，可保留 |
| `updates/v21-update.md` | 41 | selector 设计含 WR/params/wall |
| `updates/v21.1-update.md` | 36 | acquisition 设计含 Gomoku handoff 说明 |
| `updates/v22-study.md` | 46 | 第一版 FX 研究，已被 v2 修正 |
| `updates/v22-study-v2.md` | 20 | 故意对比 Gomoku / WR，保留 |

legacy 32 份全部命中，这是合理的历史记录，不应批量改写。

### 4.2 清理原则

1. **不重写历史**
   - `updates/legacy-*` 保持为历史归档。
   - v1-v16 的 Gomoku 数据是事实记录，不做“去 Gomoku 化”。

2. **当前层做语义升级**
   - v22 之后的文档必须把 Gomoku 表述为“第一个 domain 实例”，不是 framework 身份。
   - `WR / params / wall_time` 必须标注为 Gomoku legacy objective，不再作为 framework 默认真理。

3. **保留叙事，不改成执行流水账**
   - README 仍是项目叙事。
   - v22 只做事实同步和 multi-domain 叙事扩展，不做大幅风格重写。

### 4.3 文档清理计划

| 优先级 | 文档 | 操作 |
|---|---|---|
| P0 | `README.md` | 小范围同步：加入 `spot_trader` 作为第二 domain，说明 Gomoku 是第一个实证域 |
| P0 | `updates/pareto-frontier.md` | 增补 “domain-generic objective profile” 章节；保留 Gomoku 示例 |
| P0 | `updates/v22-study-v2.md` | 作为 spot FX 研究定义保留 |
| P0 | `updates/v22-update.md` | 本文件作为 v22 执行源头 |
| P1 | `updates/v20-roadmap.md` | 不改历史 roadmap，只在 v22 文档中说明其已由 v21.1 收口 |
| P1 | `updates/v21.1-update.md` | 可在末尾补充 v22 handoff note，不重写主体 |
| P2 | `updates/v20.1~v21-update.md` | 仅在需要引用时加注：“此处 WR/params 是 Gomoku 实例” |
| Archive | `updates/legacy-*` | 不清理，只保持归档 |

---

## 5. Phase 1 — Database schema multi-domain 泛化

### 5.1 基本原则

v22 不删除 legacy columns：

```text
runs.final_win_rate
runs.num_params
runs.wall_time_s
checkpoints.win_rate
cycle_metrics.win_rate
```

它们继续服务 Gomoku 和历史分析。

v22 新增 canonical generic layer：

```text
run_metrics
objective_profiles
domain_artifacts / quote-specific tables
```

新 domain 必须优先写 `run_metrics`，Gomoku 可继续写 legacy columns，同时逐步镜像到 `run_metrics`。

### 5.2 新增 `objective_profiles`

用途：定义一个 campaign 的目标、约束、显示格式、Pareto 方向、knee point 规则。

建议 schema：

```sql
CREATE TABLE IF NOT EXISTS objective_profiles (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    domain              TEXT NOT NULL,
    name                TEXT NOT NULL,
    version             TEXT NOT NULL,
    profile_json        TEXT NOT NULL,
    profile_hash        TEXT NOT NULL UNIQUE
);
```

FX spot 示例：

```json
{
  "domain": "spot_trader",
  "name": "spot-quote-frontier",
  "version": "0.1",
  "hard_constraints": [
    {"metric": "liquidity_breach_count", "op": "eq", "value": 0},
    {"metric": "unsupported_action_count", "op": "eq", "value": 0},
    {"metric": "expired_quote_count", "op": "eq", "value": 0}
  ],
  "maximize": [
    "liquidity_headroom_min",
    "preservation_ratio",
    "spot_quote_uplift_bps",
    "quote_validity_seconds_remaining"
  ],
  "minimize": [
    "embedded_spread_bps",
    "route_leg_count",
    "settlement_lag_seconds",
    "locked_funds_ratio"
  ],
  "display": {
    "preservation_ratio": {"label": "Preservation", "format": "ratio"},
    "spot_quote_uplift_bps": {"label": "Spot Uplift", "format": "bps"},
    "embedded_spread_bps": {"label": "Embedded Spread", "format": "bps"}
  },
  "knee": {
    "method": "utopia_distance",
    "constraint_first": true
  }
}
```

### 5.3 新增 `run_metrics`

用途：代替 `final_win_rate / num_params / wall_time_s` 成为跨 domain 指标事实表。

建议 schema：

```sql
CREATE TABLE IF NOT EXISTS run_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    metric_unit     TEXT,
    metric_role     TEXT NOT NULL,  -- objective / constraint / diagnostic / execution
    direction       TEXT NOT NULL,  -- maximize / minimize / eq / le / ge / none
    source          TEXT,           -- domain / framework / provider / derived
    created_at      TEXT NOT NULL,
    UNIQUE(run_id, metric_name)
);
```

FX spot 必写 metrics：

| metric | role | direction |
|---|---|---|
| `liquidity_breach_count` | constraint | eq |
| `unsupported_action_count` | constraint | eq |
| `expired_quote_count` | constraint | eq |
| `liquidity_headroom_min` | objective | maximize |
| `preservation_ratio` | objective | maximize |
| `spot_quote_uplift_bps` | objective | maximize |
| `quote_validity_seconds_remaining` | objective | maximize |
| `embedded_spread_bps` | objective | minimize |
| `route_leg_count` | objective | minimize |
| `settlement_lag_seconds` | objective | minimize |
| `locked_funds_ratio` | objective | minimize |
| `quote_latency_ms` | diagnostic | minimize |

Gomoku mirror metrics：

| legacy column | run_metric |
|---|---|
| `runs.final_win_rate` | `win_rate` |
| `runs.num_params` | `num_params` |
| `runs.wall_time_s` | `wall_time_s` |
| `runs.total_games` | `total_games` |
| `runs.total_steps` | `total_steps` |

### 5.4 将 campaign 绑定 objective profile

在 `campaigns` 上新增：

```sql
ALTER TABLE campaigns ADD COLUMN objective_profile_id TEXT;
```

约束：

1. 如果 campaign 有 `objective_profile_id`，`analyze.py --pareto` 默认读取 profile。
2. 如果没有，走 legacy Gomoku 默认：maximize `wr`，minimize `params/wall_s`。
3. 新 domain 必须提供 objective profile。

### 5.5 泛化 `frontier_snapshots`

当前 `frontier_snapshots` 已有：

```text
maximize_axes
minimize_axes
front_run_ids
dominated_count
total_runs
campaign_id
```

v22 新增：

```sql
ALTER TABLE frontier_snapshots ADD COLUMN objective_profile_id TEXT;
ALTER TABLE frontier_snapshots ADD COLUMN metric_source TEXT;       -- legacy_columns / run_metrics
ALTER TABLE frontier_snapshots ADD COLUMN constraints_json TEXT;
ALTER TABLE frontier_snapshots ADD COLUMN knee_run_id TEXT;
ALTER TABLE frontier_snapshots ADD COLUMN knee_rationale_json TEXT;
```

### 5.6 FX quote-specific tables

这些表不进入 framework generic core 的最小必需层，但 v22 FX domain 需要：

```sql
CREATE TABLE IF NOT EXISTS quote_windows (
    id                          TEXT PRIMARY KEY,
    campaign_id                 TEXT NOT NULL REFERENCES campaigns(id),
    anchor_currency             TEXT NOT NULL,
    started_at                  TEXT NOT NULL,
    expires_at                  TEXT NOT NULL,
    max_quote_age_seconds       INTEGER NOT NULL,
    portfolio_snapshot_json     TEXT NOT NULL,
    liquidity_floor_json        TEXT NOT NULL,
    provider_config_json        TEXT NOT NULL,
    status                      TEXT NOT NULL
);
```

```sql
CREATE TABLE IF NOT EXISTS fx_quotes (
    id                  TEXT PRIMARY KEY,
    quote_window_id      TEXT NOT NULL REFERENCES quote_windows(id),
    provider             TEXT NOT NULL,
    environment          TEXT NOT NULL,
    quote_source         TEXT NOT NULL, -- fixture / sandbox / live
    sell_currency        TEXT NOT NULL,
    buy_currency         TEXT NOT NULL,
    sell_amount          REAL,
    buy_amount           REAL,
    client_rate          REAL,
    mid_rate             REAL,
    awx_rate             REAL,
    quote_id             TEXT,
    valid_from_at        TEXT,
    valid_to_at          TEXT,
    conversion_date      TEXT,
    quote_latency_ms     REAL,
    raw_json             TEXT NOT NULL,
    created_at           TEXT NOT NULL
);
```

```sql
CREATE TABLE IF NOT EXISTS fx_route_legs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      TEXT NOT NULL REFERENCES runs(id),
    leg_index                   INTEGER NOT NULL,
    sell_currency               TEXT NOT NULL,
    buy_currency                TEXT NOT NULL,
    sell_amount                 REAL,
    buy_amount                  REAL,
    quote_ref                   TEXT REFERENCES fx_quotes(id),
    route_state_before_json     TEXT NOT NULL,
    route_state_after_json      TEXT NOT NULL
);
```

### 5.7 Recommendation / acquisition 泛化

现有表可继续用，但 v22 应增加通用 payload：

```sql
ALTER TABLE recommendations ADD COLUMN candidate_payload_json TEXT;
ALTER TABLE recommendations ADD COLUMN objective_metrics_json TEXT;
ALTER TABLE recommendation_outcomes ADD COLUMN constraint_status_json TEXT;
```

原因：

1. FX candidate 不一定能被 `axis_values_json` 表达完整。
2. route legs / provider / quote validity / quote source 需要保存在 candidate payload。
3. outcome 不只是 `new_front / near_front / no_gain`，还必须记录是否破红线、quote 是否过期、是否 mock。

---

## 6. Phase 2 — Dynamic candidate execution

### 6.1 `sweep.py` 改造目标

当前 `sweep.py` 的 axis 是 Gomoku 固定参数。v22 要支持：

```bash
uv run python framework/sweep.py \
  --campaign fx-spot-demo \
  --search-space domains/spot_trader/manifest/search_space.json \
  --axis route_topology=direct,two_hop \
  --axis amount_bucket_ratio=0.1,0.2,0.5 \
  --axis quote_validity=MIN_1,MIN_15,MIN_30 \
  --train-script domains/spot_trader/train.py
```

以及：

```bash
uv run python framework/sweep.py \
  --campaign fx-spot-demo \
  --candidate-json candidate.json \
  --train-script domains/spot_trader/train.py
```

### 6.2 通用 subprocess 协议

Domain `train.py` 最低必须支持：

```text
--db <path>
--sweep-tag <tag>
--candidate-json <json>
--time-budget <seconds>        # 可选，对 FX 表示 quote window / execution timeout
```

Gomoku legacy flags 继续保留：

```text
--num-blocks
--num-filters
--learning-rate
...
```

但 framework 不应再把这些当成 universal flags。

### 6.3 Candidate identity

`candidate_key` 生成规则：

```text
stable_json(candidate_payload without volatile fields)
```

volatile fields 包括：

```text
quote_id
valid_from_at
valid_to_at
created_at
latency
```

这样同一条 route strategy 可跨 quote window 比较，但实际 quote observation 仍单独落库。

---

## 7. Phase 3 — Analyze / Pareto / plot 适配新 domain

### 7.1 `analyze.py --pareto` 改造

新增参数：

```text
--objective-profile <path-or-id>
--metric-source run_metrics|legacy
--knee
--constraints
```

默认行为：

1. 如果 campaign 绑定 objective profile：从 `run_metrics` 读 metric。
2. 如果未绑定：走 legacy Gomoku 逻辑。
3. 如果用户显式传 `--maximize / --minimize`：覆盖 profile，但仍要校验 metrics 存在。

### 7.2 Generic point builder

新增内部函数：

```python
build_metric_points(conn, campaign_id, objective_profile) -> list[dict]
```

输出：

```json
{
  "run": "run-abc",
  "run_full": "...",
  "label": "USD->HKD->CNY 10k",
  "metrics": {
    "preservation_ratio": 1.0003,
    "spot_quote_uplift_bps": 3.1,
    "embedded_spread_bps": 18.4,
    "route_leg_count": 2
  },
  "constraint_status": {
    "liquidity_breach_count": 0
  }
}
```

为了兼容现有 `_pareto_front(points, maximize, minimize)`，可把 metric 展平成：

```python
point["preservation_ratio"] = ...
point["spot_quote_uplift_bps"] = ...
```

### 7.3 Constraint-first Pareto

流程：

```text
1. 读取 hard_constraints
2. 过滤 infeasible runs
3. 对 feasible runs 做 Pareto
4. infeasible runs 单独报告，不进入 dominated
5. 计算 knee point
```

输出必须显示：

```text
Feasible: N
Infeasible: M
Front: K
Dominated feasible: D
Knee: run_x
```

### 7.4 `pareto_plot.py` 改造

当前 `_AXIS_META` 是硬编码：

```text
wr -> Win Rate
params -> Params
wall_s -> Wall Time
```

v22 应新增：

```python
plot_pareto(
    front,
    dominated,
    axis_meta={...},
    constraints_meta={...},
    knee_point=...,
)
```

FX plot 应支持：

1. x = `embedded_spread_bps` 或 `route_leg_count`
2. y = `preservation_ratio` 或 `spot_quote_uplift_bps`
3. color = `route_topology`
4. size = `liquidity_headroom_min`
5. marker = provider / quote_source
6. knee point 高亮
7. infeasible points 可选灰色单独层

### 7.5 新增 plot 类型

除了 2D Pareto scatter，v22 建议新增：

1. **route graph plot**
   - currency nodes + quote edges。

2. **knee plot**
   - normalized utility vs complexity。

3. **quote validity plot**
   - uplift / preservation vs quote validity remaining。

4. **liquidity headroom plot**
   - route 执行前后每个 currency 的 headroom bar。

这些可以后续拆到 `domains/spot_trader/plot.py`，framework 只提供 generic scatter。

---

## 8. Phase 4 — FX spot mock domain

### 8.1 目录

```text
domains/spot_trader/
├── train.py
├── portfolio.py
├── quote_graph.py
├── provider_base.py
├── mock_provider.py
├── route_eval.py
├── metrics.py
├── knee.py
├── search_space.json
├── objective_profile.json
├── stage_policy.json
├── branch_policy.json
├── selector_policy.json
└── acquisition_policy.json
```

### 8.2 `mock_provider.py`

必须支持 deterministic fixture：

```json
{
  "USD/CNY": {
    "client_rate": 7.2300,
    "mid_rate": 7.2350,
    "awx_rate": 7.2330,
    "validity_seconds": 900
  },
  "USD/HKD": {
    "client_rate": 7.8100,
    "mid_rate": 7.8120
  },
  "HKD/CNY": {
    "client_rate": 0.9258,
    "mid_rate": 0.9260
  }
}
```

mock provider 要能制造：

1. direct route 最优。
2. two-hop route 最优。
3. quote 过期。
4. unsupported pair。
5. insufficient balance。
6. liquidity breach。
7. 多个 Pareto front 点。
8. 明确 knee point。

### 8.3 `portfolio.py`

职责：

1. 读 current holdings。
2. 读 liquidity floors。
3. 应用 route 前检查 sell amount。
4. 应用 route 后计算 headroom。
5. 锚定货币估值。

### 8.4 `quote_graph.py`

职责：

1. 根据 held currencies + supported pairs 生成 directed graph。
2. 生成 direct / two-hop route candidates。
3. 生成 amount buckets。
4. 限制 max path length。

### 8.5 `route_eval.py`

职责：

1. 请求 quote。
2. 本地 mock route execution。
3. 写 `fx_quotes` / `fx_route_legs`。
4. 计算 `run_metrics`。
5. 给 `runs` 写 legacy compatibility summary：
   - `final_win_rate = normalized_preservation_score`
   - `wall_time_s = quote_latency_s`
   - `num_params = route_leg_count`

compatibility summary 只为旧工具过渡，不作为 v22 正式语义。

---

## 9. Phase 5 — Selector / acquisition 泛化

### 9.1 Selector

FX selector 候选类型：

```text
new_route
route_requote
try_bridge
split_amount
extend_quote_validity
reduce_complexity
liquidity_repair
```

候选生成不依赖历史趋势，只依赖：

1. 当前 quote window。
2. 当前 portfolio。
3. 已观测 quote graph 的稀疏区。
4. 已发现 front 的邻域。
5. 当前 infeasible 原因。

### 9.2 Acquisition

FX acquisition features：

```text
predicted_preservation_ratio
predicted_uplift_bps
uncertainty_over_quote_surface
frontier_gap_bonus
liquidity_headroom_bonus
quote_validity_bonus
embedded_spread_penalty
route_complexity_penalty
settlement_lag_penalty
quote_staleness_penalty
liquidity_risk_penalty
```

硬约束：

```text
liquidity_breach_count == 0
unsupported_action_count == 0
expired_quote_count == 0
```

### 9.3 Replay benchmark

在没有真实 Airwallex 密钥前，replay benchmark 用 fixture quote windows：

1. 固定 quote matrix。
2. 固定 portfolio。
3. 固定 liquidity floor。
4. 比较：
   - random route sampling
   - heuristic selector
   - acquisition reranker
5. 指标：
   - front hit rate
   - knee hit rate
   - quote calls used
   - infeasible candidate rate
   - best preservation ratio
   - best spot uplift bps

---

## 10. Phase 6 — Airwallex adapter 边界

### 10.1 adapter capability matrix

每个 provider adapter 必须声明：

```json
{
  "provider": "airwallex",
  "environment": "sandbox",
  "supports_current_rate": true,
  "supports_guaranteed_quote": true,
  "supports_conversion_booking": true,
  "quote_reality": "sandbox_or_demo_until_verified",
  "conversion_reality": "sandbox",
  "requires_secret": true,
  "default_allow_live_conversion": false
}
```

production quote-only：

```json
{
  "provider": "airwallex",
  "environment": "production",
  "supports_current_rate": true,
  "supports_guaranteed_quote": true,
  "supports_conversion_booking": true,
  "quote_reality": "live",
  "conversion_reality": "disabled_by_default",
  "requires_secret": true,
  "default_allow_live_conversion": false
}
```

### 10.2 credential discipline

1. 不提交 API key。
2. 测试默认使用 mock provider。
3. sandbox integration test 默认 skip，除非环境变量存在：

```text
AIRWALLEX_CLIENT_ID
AIRWALLEX_API_KEY
AIRWALLEX_ENV=sandbox
```

4. production quote test 默认 skip，需要显式：

```text
FX_ALLOW_LIVE_QUOTE=1
```

5. live conversion test 不进入自动测试。

### 10.3 no-live-conversion guard

任何调用 create conversion 的路径必须检查：

```text
allow_create_conversion == true
environment != production OR FX_ALLOW_LIVE_CONVERSION == 1
approval_packet_id exists
quote_not_expired
liquidity_recheck_passed
idempotency_key exists
```

否则直接失败，并写入 negative outcome。

---

## 11. Phase 7 — 测试计划

### 11.1 Framework generic schema tests

新增：

```text
tests/test_run_metrics_db.py
tests/test_objective_profile.py
tests/test_generic_pareto.py
```

覆盖：

1. `run_metrics` upsert / round-trip。
2. objective profile hash / validation。
3. constraint-first Pareto。
4. legacy Gomoku columns 仍可用。
5. campaign 绑定 objective profile。

### 11.2 Dynamic sweep tests

新增：

```text
tests/test_dynamic_sweep.py
```

覆盖：

1. `--axis name=v1,v2` 生成笛卡尔积。
2. `--candidate-json` 透传给 fake train。
3. candidate_key 排除 volatile quote fields。
4. Gomoku legacy flags 不回退。

### 11.3 Analyze / plot tests

新增：

```text
tests/test_generic_pareto_cli.py
tests/test_generic_pareto_plot.py
tests/test_knee_point.py
```

覆盖：

1. 从 `run_metrics` 读取 arbitrary metrics。
2. hard constraint 过滤 infeasible。
3. knee point 输出稳定。
4. plot axis label / formatter 来自 objective profile。
5. WR legacy plot 仍通过。

### 11.4 FX mock domain tests

新增：

```text
tests/test_fx_portfolio.py
tests/test_fx_quote_graph.py
tests/test_fx_route_eval.py
tests/test_fx_mock_provider.py
tests/test_fx_smoke_chain.py
```

覆盖：

1. 当前头寸 + liquidity floor。
2. direct route / two-hop route。
3. unsupported pair。
4. insufficient balance。
5. quote expiry。
6. liquidity breach。
7. route metrics 写入 `run_metrics`。
8. Pareto front 和 knee point 可复现。

### 11.5 Acquisition tests

新增：

```text
tests/test_fx_selector.py
tests/test_fx_acquisition.py
tests/test_fx_replay_benchmark.py
```

覆盖：

1. candidate pool 生成。
2. infeasible candidate 不进入 top recommendations。
3. acquisition 比 random 更快命中 front fixture。
4. replay benchmark 在固定 fixture 上 deterministic。

### 11.6 Airwallex sandbox tests

默认不运行，需要环境变量：

```text
tests/test_airwallex_adapter_contract.py
tests/test_airwallex_sandbox_optional.py
```

默认 CI 只跑 contract test：

1. request payload 构造。
2. response parser。
3. error parser。
4. idempotency key。
5. no-live-conversion guard。

optional sandbox test 只验证：

1. auth。
2. current rate / quote endpoint。
3. quote fields parse。
4. 可选 sandbox conversion。

不验证市场收益。

---

## 12. 验收标准

v22 完成必须同时满足：

1. `run_metrics` 和 `objective_profiles` 已落库并有测试。
2. Gomoku legacy tests 不回退。
3. 至少一个 non-Gomoku fake domain 使用 `run_metrics` 完成 Pareto。
4. `domains/spot_trader/` 在 mock provider 下完成 smoke chain。
5. FX mock smoke 能生成：
   - quote window
   - fx quotes
   - route legs
   - run metrics
   - Pareto front
   - knee point
   - recommendation outcome
6. `analyze.py --pareto` 可由 objective profile 驱动。
7. `pareto_plot.py` 支持 generic axis metadata。
8. Airwallex adapter 默认不会执行 live conversion。
9. `updates/v22-update.md`、`updates/v22-study-v2.md` 与 README 当前阶段叙事一致。

---

## 13. 建议实施顺序

1. **先做 DB generic metrics**
   - 否则所有 FX 指标都会被迫塞进 `final_win_rate`。

2. **再做 analyze / plot generic path**
   - 先让任意 metrics 可以画 Pareto。

3. **再做 dynamic candidate execution**
   - 让 `sweep.py` 能执行 FX candidate。

4. **再做 `spot_trader` mock domain**
   - 不需要 Airwallex 密钥，先证明完整 loop。

5. **再做 selector / acquisition 泛化**
   - 在 mock quote graph 上证明 active quote sampling 有价值。

6. **最后接 Airwallex adapter**
   - quote-only first，sandbox optional，live conversion disabled。

---

## 14. v22 的最终边界

v22 的研究边界是：

```text
当前头寸
  + 当前可报价图
  + 30 分钟 quote window
  + 流动性红线
  + 本地 mock execution
  + Pareto / knee point
```

它不是：

```text
历史趋势预测
宏观 FX 策略
自动交易机器人
真实资金自动换汇系统
```

这正好承接 v21.1 的 portable acquisition handoff：

> **Gomoku 证明了 framework 能从复杂训练实验里逼近 frontier；v22 要证明同一套机制能在完全不同的 spot quote-surface domain 中，通过可控、可审计、可 mock 的局部实验，逼近一个新的 Pareto front。**

---

## 15. v22 完成回填 — multi-domain / spot FX 收口报告

### 15.1 工作日志

1. **创建并执行 v22 todo-list**
   - 建立 `v22-db-generic`、`v22-dynamic-sweep`、`v22-analyze-plot`、`v22-fx-domain`、`v22-selector-acquisition`、`v22-tests`、`v22-doc-backfill` 等任务。
   - 按依赖顺序完成：schema 泛化 → dynamic candidate → generic Pareto/plot → FX mock domain → selector/acquisition 泛化 → 测试 → 文档回填。

2. **完成 database multi-domain 泛化**
   - 新增 `objective_profiles`，用于持久化 objective profile、profile hash、domain/name/version。
   - 新增 `run_metrics`，作为跨 domain 的 canonical metric fact table。
   - 新增 `quote_windows`、`fx_quotes`、`fx_route_legs`，用于保存 spot FX quote-window、报价观测和本地 mock route leg evidence。
   - 为 `campaigns` 增加 `objective_profile_id`，使 campaign 可以绑定 domain-specific objective。
   - 为 `frontier_snapshots` 增加 `objective_profile_id`、`metric_source`、`constraints_json`、`knee_run_id`、`knee_rationale_json`。
   - 为 `recommendations` / `recommendation_outcomes` 增加 generic payload / metrics / constraint status 字段。

3. **完成 objective profile 支撑**
   - 新增 `framework/objective_profile.py`。
   - 支持 profile validation、normalization、stable hash、loader、human-readable describe。
   - 支持 objective direction、hard constraints、display formatter、knee method。

4. **完成 search-space 与 sweep 泛化**
   - `search_space.py` 新增 `strategy / execution / risk / provider` axis role。
   - `sweep.py` 新增 `--axis name=v1,v2` dynamic axis。
   - `sweep.py` 新增 `--candidate-json`，支持 inline JSON 或 JSON 文件。
   - candidate identity 改为 stable JSON，并排除 `quote_id / valid_from_at / valid_to_at / created_at / latency / quote_latency_ms` 等 volatile quote fields。
   - campaign 创建时支持 `--objective-profile`，并把 objective profile 绑定到 campaign。
   - objective-profile campaign 的 domain train 脚本会收到 `--campaign-id`，用于写 quote-window evidence。
   - generic recommendation execution 会把 `axis_values` 重新包装成 `--candidate-json`，避免 FX route candidate 退回 Gomoku legacy flags。

5. **完成 analyze / Pareto / plot 泛化**
   - `analyze.py --pareto` 新增：
     - `--objective-profile <path-or-id>`
     - `--metric-source legacy|run_metrics`
     - `--knee`
   - 如果 campaign 绑定 objective profile，Pareto 自动走 `run_metrics`。
   - 新增 constraint-first Pareto：先过滤 infeasible，再对 feasible runs 做 non-dominated sort。
   - 输出中区分 feasible / infeasible / front / dominated。
   - knee point 使用 normalized utopia-distance。
   - frontier snapshot 会保存 objective profile、metric source、constraints、knee run 和 rationale。
   - `pareto_plot.py` 支持 objective profile 提供的 axis label / formatter，并高亮 knee point。

6. **完成 selector / acquisition 泛化**
   - `selector.py` 在 campaign 绑定 objective profile 时，改用 `run_metrics` 生成 generic point candidates。
   - generic selector 支持基于 objective profile 的 primary maximize/minimize signal 做 frontier-neighbour candidate 和 seed recheck。
   - `acquisition.py` 支持 `predicted_objective / mean_cost`，保留 legacy `predicted_wr / mean_params` 兼容路径。
   - `analyze.py --recommend-next` 会把 generic candidate payload 与 objective metrics 写入 recommendation ledger。

7. **新增 `domains/spot_trader/` mock domain**
   - 新增当前头寸、liquidity floor、mock provider、quote graph、route evaluation、metric mapping、train entry。
   - 支持 direct route 与 USD intermediate route。
   - `train.py` 支持 `--candidate-json`、`--campaign-id`、`--sweep-tag`、`--time-budget`。
   - 每个 mock run 会写：
     - `runs`
     - `run_metrics`
     - `quote_windows`（有 campaign id 时）
     - `fx_quotes`（有 quote window 时）
     - `fx_route_legs`
   - 新增 `search_space.json`、`objective_profile.json`、`selector_policy.json`、`acquisition_policy.json`、`stage_policy.json`、`branch_policy.json`。

8. **建立 Airwallex adapter 安全边界**
   - 新增 `domains/spot_trader/airwallex_provider.py`。
   - 当前只定义 quote-only provider adapter，不创建 conversion，不移动资金。
   - adapter 从环境变量读取 base URL / API key / client id。
   - 真实 sandbox / production 接入不进入默认测试；默认使用 `MockQuoteProvider`。

9. **README 小范围同步**
   - 保留 README 原有叙事，不改成执行流水账。
   - 同步当前事实：项目已有 `gomoku` 与 `spot_trader` 两个 domain。
   - 增补 `objective_profile.py`、`run_metrics`、`spot_trader` 目录和 v22 roadmap 条目。
   - 将 domain 接入指南从 legacy columns-only 更新为 legacy columns + v22 `run_metrics` 双轨。

### 15.2 新增文件

1. `framework/objective_profile.py`
2. `domains/spot_trader/__init__.py`
3. `domains/spot_trader/train.py`
4. `domains/spot_trader/portfolio.py`
5. `domains/spot_trader/provider_base.py`
6. `domains/spot_trader/mock_provider.py`
7. `domains/spot_trader/airwallex_provider.py`
8. `domains/spot_trader/quote_graph.py`
9. `domains/spot_trader/route_eval.py`
10. `domains/spot_trader/metrics.py`
11. `domains/spot_trader/manifest/search_space.json`
12. `domains/spot_trader/manifest/objective_profile.json`
13. `domains/spot_trader/manifest/selector_policy.json`
14. `domains/spot_trader/manifest/acquisition_policy.json`
15. `domains/spot_trader/manifest/stage_policy.json`
16. `domains/spot_trader/manifest/branch_policy.json`
17. `tests/test_objective_profile.py`
18. `tests/test_run_metrics_pareto.py`
19. `tests/test_sweep_dynamic.py`
20. `tests/test_spot_trader_domain.py`
21. `updates/v22-study-v2.md`
22. `updates/v22-update.md`

### 15.3 修改文件

1. `framework/core/db.py`
   - 增加 objective profile、run metrics、FX quote evidence schema 与 CRUD helper。
   - 扩展 campaign、frontier snapshot、recommendation schema。

2. `framework/search_space.py`
   - 新增 FX / generic domain axis role。

3. `framework/sweep.py`
   - 新增 dynamic axis、candidate JSON、objective profile campaign binding、generic candidate execution。

4. `framework/analyze.py`
   - 新增 run_metrics-backed Pareto、constraint-first filtering、knee point、generic plot metadata、objective metrics recommendation persistence。

5. `framework/pareto_plot.py`
   - 新增 generic axis metadata formatter 与 knee marker。

6. `framework/selector.py`
   - 新增 objective-profile campaign 的 generic run_metrics candidate generation。

7. `framework/acquisition.py`
   - 新增 generic `predicted_objective / mean_cost` scoring path。

8. `README.md`
   - 仅做 v22 事实同步，不改变原叙事风格。

### 15.4 测试结果

1. **v22 targeted regression**
   - 命令：
     ```bash
     uv run python -m pytest tests/test_objective_profile.py tests/test_run_metrics_pareto.py tests/test_sweep_dynamic.py tests/test_spot_trader_domain.py tests/test_search_space.py tests/test_campaign_db.py tests/test_recommendation_execution.py -q
     ```
   - 结果：`26 passed`

2. **FX sweep smoke**
   - 命令核心：
     ```bash
     uv run python framework/sweep.py \
       --train-script domains/spot_trader/train.py \
       --campaign fx-v22-smoke \
       --search-space domains/spot_trader/manifest/search_space.json \
       --objective-profile domains/spot_trader/manifest/objective_profile.json \
       --axis sell_currency=EUR \
       --axis buy_currency=CNY \
       --axis route_template=direct \
       --axis rebalance_fraction=0.5 \
       --axis max_legs=2 \
       --axis provider=mock \
       --axis quote_scenario=base
     ```
   - 结果：
     - 1 个 config 成功。
     - 自动 `run_metrics` Pareto 成功。
     - 输出 feasible/front/knee。
     - 生成 plot 后已清理临时 artifact。

3. **FX recommendation execution smoke**
   - 流程：
     - 运行一个 `spot_trader` campaign。
     - `analyze.py --recommend-next fx-v22-rec --limit 1` 生成 generic recommendation。
     - 手动将 recommendation 标记为 `accepted`。
     - `sweep.py --execute-recommendation <rec_id>` 执行。
   - 结果：`executed 1`，即 recommendation 状态更新为 executed，且写入 1 条 recommendation outcome。

4. **全量测试**
   - 命令：
     ```bash
     uv run python -m pytest tests/ -q
     ```
   - 结果：`182 passed in 3.84s`

### 15.5 v22 收口判断

v22 的核心断点已经成立：

1. framework 不再只能用 `final_win_rate / num_params / wall_time_s` 表达研究结果。
2. 新 domain 可以通过 `objective_profile.json + run_metrics` 定义自己的 truth / cost / hard constraints。
3. `sweep.py` 可以执行 dynamic JSON candidate，不再只服务 Gomoku hyperparameters。
4. `analyze.py --pareto` 可以从 objective profile 读取任意 metrics，并输出 feasible / infeasible / knee。
5. `spot_trader` 在没有 Airwallex key 的情况下，已经能完成当前报价窗口的 local mock execution、quote evidence、route metrics、Pareto 和 recommendation execution closure。

仍然明确不做的事情：

1. 不预测 FX trend。
2. 不读取历史汇率做回测。
3. 不声称 mock quote route 等于真实可盈利交易。
4. 不自动创建 Airwallex conversion。
5. 不把 sandbox response 当作 production price guarantee。

因此，v22 已经完成从 `Gomoku-only autoresearch` 到 `multi-domain autoresearch` 的结构性切分。后续 v23 可以在 `spot_trader` 上继续做 provider contract tests、route graph visualization、live quote-only integration，或直接进入第三个 domain。

### 15.6 Airwallex sandbox base URL 与密钥填写方式

当前仓库不提交任何 Airwallex secret。后续接 sandbox / demo quote endpoint 时，通过环境变量注入：

```bash
export AIRWALLEX_ENV=sandbox
export AIRWALLEX_BASE_URL="<Airwallex dashboard / developer docs 给出的 sandbox 或 demo API base URL>"
export AIRWALLEX_API_KEY="<your sandbox API key>"
export AIRWALLEX_CLIENT_ID="<your client id, if required by the account>"
```

使用原则：

1. **默认 provider 仍是 `mock`**
   - 自动测试、CI、本地无密钥 smoke 全部走 `MockQuoteProvider`。

2. **Airwallex adapter 只做 quote-only**
   - `domains/spot_trader/airwallex_provider.py` 当前只封装 quote request boundary。
   - 不创建 conversion。
   - 不移动资金。

3. **base URL 不硬编码**
   - 不同 Airwallex account / region / sandbox program 可能给出不同 base URL。
   - 以 dashboard 或官方 developer docs 当前值为准，填入 `AIRWALLEX_BASE_URL`。

4. **密钥不写入仓库**
   - 本地 shell 使用环境变量。
   - CI 使用 secret manager。
   - 不写入 `.env` 到 git。

5. **production quote 必须显式开启**
   - 建议未来加：
     ```bash
     export FX_ALLOW_LIVE_QUOTE=1
     ```
   - 未开启时，production quote integration test 应 skip。

6. **live conversion 默认禁止**
   - 即使未来加入 conversion adapter，也必须要求：
     ```text
     allow_create_conversion == true
     environment != production OR FX_ALLOW_LIVE_CONVERSION == 1
     approval_packet_id exists
     quote_not_expired
     liquidity_recheck_passed
     idempotency_key exists
     ```
   - 否则只允许本地 mock execution，不允许真实 conversion。
