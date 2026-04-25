# v22 Study — FX Portfolio Domain / Constrained Bayesian Frontier Search

> 2026-04-25  
> 前置：`updates/v20-roadmap.md`、`updates/v21.1-update.md`、当前 `framework/` 代码  
> 目标：评估把 v21.1 已收口的 autoresearch / acquisition loop 拓展到“进出口企业外汇头寸组合管理”domain 的可行性、边界、建模方式与执行路线。

---

## 1. 一句话结论

> **可以做，而且这个 domain 很适合承接 v21.1 的“candidate-pool Bayesian acquisition + Pareto frontier”骨架；但不能把 Airwallex 等支付平台的 sandbox 误认为真实市场。正确做法是：用历史行情 / 情景生成器 / 企业现金流假设产生 truth signal，用 sandbox 验证账户、报价、换汇、结算、余额约束等执行链；再在硬性流动性约束下寻找“价值稳定性、成本、流动性缓冲、操作复杂度”的 Pareto 前沿。**

这意味着 v22 不应被设计成“自动帮企业交易外汇”的系统，而应被设计成：

1. **treasury decision-support research loop**：研究不同头寸调整策略在给定约束下的 trade-off。
2. **sandbox-backed execution rehearsal**：用支付平台 sandbox 演练报价、换汇、余额、结算状态和幂等请求。
3. **constrained multi-objective optimizer**：把最低流动性红线当作硬约束，而不是可牺牲的优化目标。
4. **human-approved recommendation system**：系统推荐候选策略，人类 treasury / finance owner 审批；初期绝不自动执行 live conversion。

---

## 2. 当前 v21.1 框架能直接复用什么

v21.1 已经把 v20 roadmap 的前半段收成了完整链路：

```text
point frontier
  → campaign / protocol
  → multi-fidelity stage
  → continuation / branch
  → recommendation ledger
  → acquisition reranker
  → outcome / replay evidence
```

这些能力对外汇头寸 domain 仍然成立，只是每个概念需要重新命名。

| 当前 Gomoku / framework 概念 | FX portfolio domain 映射 |
|---|---|
| `campaign` | 一个企业 / 业务线 / 锚定货币 / 期限 / 现金流假设 / 支付平台配置固定的一轮研究 |
| `search_space.json` | 可枚举的 hedge ratio、rebalance band、buffer multiplier、quote validity、trade size bucket、currency-pair route |
| `stage_policy.json` | 从离线合成情景，到历史回放，到 sandbox quote rehearsal，再到 paper-trading 的 fidelity 分层 |
| `branch_policy.json` | 对当前策略做局部分叉：提高某币种 liquidity buffer、缩短 rebalance interval、切换 quote validity、降低单笔换汇上限 |
| `selector.py` | 基于已跑策略生成下一批候选头寸调整策略 |
| `acquisition.py` | 在合法候选池上做不确定性 + 多目标权衡的重排序 |
| `frontier_snapshots` | 某一协议下非支配策略集合：价值稳定性 / 成本 / 流动性 / 操作复杂度 |
| `recommendation_outcomes` | 某个被执行 / 回放的策略候选最终产生的收益、风险、违约、成本、流动性结果 |

这正是 v21.1 做 candidate-pool acquisition，而不是 gomoku-only optimizer 的价值：外汇 domain 的动作空间天然混合了离散动作、交易约束、平台限制和监管边界，不适合直接做无限 raw-space BO。

---

## 3. Airwallex 等支付平台 sandbox 的定位

### 3.1 sandbox 能提供什么

以 Airwallex 文档暴露的能力为例，平台 API 通常覆盖：

1. **多币种账户 / wallet / balance**
   - 查询多币种余额。
   - 管理收款、付款和资金流。

2. **Transactional FX**
   - 查询当前 rate。
   - 创建 guaranteed quote。
   - 在 quote validity 内创建 conversion。
   - 指定 buy / sell currency、amount、conversion date。
   - 使用 request id / idempotency key 防止重复请求。

3. **sandbox / demo endpoint**
   - 使用 demo API endpoint 演练认证、报价、换汇、余额、状态流转。
   - 验证系统是否正确处理 quote expiry、余额不足、币种不支持、结算日期、失败响应等工程问题。

这些能力非常适合验证 **execution discipline**：也就是 recommendation 被接受后，系统是否能按协议完成“取报价 → 检查余额和红线 → dry-run / sandbox conversion → 写回 outcome”的闭环。

### 3.2 sandbox 不能提供什么

sandbox 不应被当成真实 market simulator，原因是：

1. **sandbox rate 未必代表真实可成交市场**
   - demo rate 主要服务 API 测试，不保证真实流动性、真实 spread、真实冲击成本。

2. **sandbox 没有企业真实现金流约束**
   - 企业真实 receivables / payables、账期、付款日、收款概率、授信额度、税务和资金监管，必须由企业侧 ledger / ERP / treasury system 提供。

3. **sandbox 不等于 forward / lock FX 全产品覆盖**
   - 不同地区、账户、币种、资金模式、结算模型支持不同，不能假设所有锁汇 / 远期 / 掉期能力都可由同一 sandbox 完整模拟。

4. **sandbox 无法证明策略在未来市场有效**
   - 真正的 “truth signal” 必须来自历史回放、情景压力测试、蒙特卡洛路径、以及后续 paper/live observation。

因此 v22 的正确架构是：

```text
market truth layer      历史行情 / 情景生成 / 企业现金流 / 费用模型
        ↓
strategy simulator     头寸演化、换汇动作、红线约束、成本、风险
        ↓
autoresearch loop      campaign / stage / frontier / acquisition
        ↓
sandbox executor       支付平台 API 执行链演练，不作为唯一 truth
```

---

## 4. 外汇头寸 domain 的核心建模

### 4.1 状态变量

设企业持有币种集合为 `C`，锚定货币为 `A`（通常 CNY 或 USD）。

每个时间点 `t` 的状态至少包括：

1. `balance[c, t]`
   - 每个币种当前可用余额。

2. `receivable[c, t+h]`
   - 未来某期限内预计收到的外币现金流。

3. `payable[c, t+h]`
   - 未来某期限内预计支付的外币现金流。

4. `rate[c -> A, t]`
   - 每个币种折算到锚定货币的报价。

5. `liquidity_floor[c, t]`
   - 业务红线：某币种必须保留的最低可用头寸。

6. `platform_constraints`
   - 支持币种、交易对、最小 / 最大金额、quote validity、settlement delay、fee、spread、funding mode、open position limit。

7. `risk_scenario`
   - 汇率路径、跳空、波动率、相关性、收付款延迟、客户违约、结算失败等情景。

### 4.2 动作变量

一个候选策略不是单个点，而是一组可执行规则：

```json
{
  "anchor_currency": "CNY",
  "rebalance_interval": "daily",
  "rebalance_band_bps": 80,
  "hedge_ratio_usd": 0.6,
  "hedge_ratio_eur": 0.5,
  "liquidity_buffer_multiplier": 1.25,
  "max_conversion_fraction_per_day": 0.2,
  "quote_validity": "HR_1",
  "conversion_route": "direct_or_anchor",
  "stress_policy": "preserve_liquidity_first"
}
```

对应到当前 framework，这些字段就是 search-space axes。它们可以通过笛卡尔积生成候选池，再由 selector / acquisition 逐步减少无效探索。

### 4.3 红线约束

流动性红线不应该放进 Pareto 轴里让优化器“权衡牺牲”。它应是硬约束：

```text
for every currency c and time t:
    projected_available_balance[c, t] >= liquidity_floor[c, t]
```

如果候选策略违反红线：

1. 不能进入 executable candidate pool；
2. 或者进入 pool 但标记为 infeasible，acquisition score 必须为不可选；
3. outcome 必须记录 `liquidity_breach=true`，且 label 不应为正例。

这是外汇 domain 与 Gomoku 最大的差别之一：Gomoku 的低 WR 只是差结果，外汇的流动性破线可能是业务事故。

---

## 5. Pareto 前沿应该怎么定义

### 5.1 不建议直接最大化“头寸价值”

如果把 `portfolio_anchor_value` 简单当成 Gomoku 的 `win_rate` 去最大化，系统会被鼓励做 FX speculation。这不符合进出口企业 treasury 的目标。

更合理的 truth 概念是：

> **在满足流动性红线和企业现金流义务的前提下，最大化组合价值稳定性 / 最小化短缺风险 / 控制换汇成本。**

因此，Gomoku 的 `WR` 在 FX domain 中不应直接映射为“赚更多钱”，而应映射为一个或多个 treasury utility / risk-adjusted metrics。

### 5.2 建议目标

| 类型 | 指标 | 方向 | 说明 |
|---|---|---|---|
| hard constraint | `liquidity_breach_count` | 必须为 0 | 任一币种低于最低流动性红线即不可接受 |
| hard constraint | `unsupported_action_count` | 必须为 0 | 平台不支持币种、金额、日期、资金模式时不可执行 |
| maximize | `utility_score` | 越高越好 | 综合价值稳定性、低风险、低成本后的归一化分数，可作为临时 `final_win_rate` 兼容字段 |
| maximize | `liquidity_headroom_min` | 越高越好 | 所有币种跨时间的最小流动性缓冲 |
| minimize | `anchor_value_drawdown_p95` | 越低越好 | 锚定货币口径下组合价值的高分位回撤 |
| minimize | `cash_shortfall_cvar95` | 越低越好 | 付款义务无法覆盖的 CVaR |
| minimize | `fx_cost_bps` | 越低越好 | spread、fee、quote slippage、conversion cost |
| minimize | `settlement_lag_hours` | 越低越好 | 换汇或资金可用延迟 |
| minimize | `trade_count` | 越低越好 | 操作复杂度、财务对账压力 |

第一版 Pareto front 可以使用：

```text
maximize: utility_score, liquidity_headroom_min
minimize: cash_shortfall_cvar95, fx_cost_bps, trade_count
```

但在当前代码完全泛化之前，可以先把 `utility_score` 写入 `runs.final_win_rate` 做兼容桥接，同时把真实 FX 指标写入 domain-specific JSON / side table。这个桥接只能作为 v22 早期过渡，不能长期把 `final_win_rate` 当成外汇语义。

---

## 6. 贝叶斯 / acquisition 应该怎样落地

### 6.1 不建议直接做 raw continuous BO

外汇头寸策略不是一个简单连续空间：

1. 币种支持是离散的；
2. 交易金额受最小金额、余额、红线、限额约束；
3. quote validity 是离散产品参数；
4. settlement date / funding mode 影响可行性；
5. 企业现金流时间桶让状态高度路径依赖；
6. 某些动作只能在特定账户 / 地区 / 时点执行。

因此 v21.1 的 candidate-pool acquisition 仍然是正确路线：

```text
domain-aware candidate generator
    先生成合法策略 / 合法交易动作
constrained acquisition reranker
    再对这些候选做不确定性 + 多目标权衡
```

### 6.2 第一版 acquisition 可以怎么做

v21.1 当前 `framework/acquisition.py` 是轻量 UCB-style score：

```text
score =
  predicted_value
  + uncertainty_bonus
  + frontier_bonus
  + candidate_type_bonus
  - cost_penalty
  - wall_time_penalty
```

FX domain 可以继承这个形态，但需要把特征替换成：

```text
score =
  predicted_utility
  + uncertainty_bonus
  + frontier_gap_bonus
  + liquidity_headroom_bonus
  - cvar_penalty
  - fx_cost_penalty
  - settlement_lag_penalty
  - trade_count_penalty
  - infeasibility_penalty
```

其中：

1. `predicted_utility`
   - 来自历史 replay / scenario simulation 的 surrogate。

2. `uncertainty_bonus`
   - 对样本少、情景覆盖不足、波动敏感的策略增加探索价值。

3. `frontier_gap_bonus`
   - 鼓励探索当前 Pareto front 稀疏区域。

4. `liquidity_headroom_bonus`
   - 只奖励红线以上的缓冲，不奖励违反红线的候选。

5. `infeasibility_penalty`
   - 对 sandbox / policy 判定不可执行的候选直接降为不可选。

### 6.3 v22 中期可以升级为 constrained EHVI

当有足够 replay/outcome 历史后，可以把 acquisition 升级为：

1. **feasibility classifier**
   - 预测候选是否会破流动性红线、是否会平台不可执行。

2. **multi-output surrogate**
   - 同时预测 `utility_score`、`cash_shortfall_cvar95`、`fx_cost_bps`、`liquidity_headroom_min`。

3. **constrained EHVI / scalarized qUCB**
   - 只在可行概率足够高的候选中最大化 expected hypervolume improvement。

4. **scenario-aware posterior**
   - 不只看一个历史路径，而是跨多组市场情景估计后验不确定性。

但这不应该是 v22.0 的第一步。第一步应先让 domain 跑通 replay / sandbox / ledger / recommendation 闭环。

---

## 7. v22 domain 的建议目录结构

建议新增一个独立 domain，而不是把 FX 逻辑塞进 Gomoku：

```text
domains/fx_portfolio/
├── train.py                         # framework subprocess 入口；实际是 strategy episode runner
├── portfolio.py                     # 多币种头寸、现金流、锚定货币估值
├── market_data.py                   # 历史汇率 / 情景路径加载
├── scenario.py                      # 压力测试 / Monte Carlo path generator
├── strategy.py                      # 策略参数 → rebalancing / conversion decisions
├── simulator.py                     # 离线 replay truth engine
├── sandbox_client.py                # Airwallex-like sandbox adapter
├── ledger.py                        # quote / conversion / balance / settlement event ledger
├── metrics.py                       # utility、CVaR、cost、liquidity headroom
├── search_space.json                # v22 search space
├── stage_policy.json                # offline → historical → sandbox → paper fidelity
├── branch_policy.json               # 策略局部分叉规则
├── selector_policy.json             # FX candidate generation/scoring policy
└── acquisition_policy.json          # constrained acquisition weights / priors
```

这里的 `train.py` 只是为了复用当前 framework 的 subprocess 协议。它不是真的训练神经网络，而是执行一次“候选策略 episode”：

```text
read candidate axes
  → load portfolio scenario
  → simulate strategy over horizon
  → optionally rehearse quote/conversion in sandbox
  → write run + metrics to tracker.db
```

---

## 8. v22 需要先修正的 framework 泛化点

当前 README 叙事已经说 framework 是 domain-agnostic，但代码仍有 Gomoku / WR 硬编码。FX domain 可以承接框架，但如果不先做泛化，会出现语义断点。

### 8.1 `sweep.py` 的扫描轴仍是 Gomoku 专属

当前 `framework/sweep.py` 只认识：

```text
--num-blocks
--num-filters
--learning-rate
--steps-per-cycle
--buffer-size
--seeds
```

并且 `run_one()` 会把它们翻译为 Gomoku train.py 参数。FX domain 需要：

```text
--axis hedge_ratio_usd=0.3,0.5,0.7
--axis rebalance_band_bps=50,100,200
--axis liquidity_buffer_multiplier=1.1,1.25,1.5
```

或：

```text
--candidate-json '{"hedge_ratio_usd":0.5,...}'
```

建议 v22.0 第一项修改就是让 `sweep.py` 从 search-space profile 动态生成 axis，而不是继续 hard-code Gomoku 超参。

### 8.2 `stage_policy.py` 的 metric 白名单仍是 WR

当前 `ALLOWED_METRICS = {"win_rate", "final_win_rate"}`。FX domain 至少需要：

```text
utility_score
liquidity_headroom_min
cash_shortfall_cvar95
fx_cost_bps
trade_count
```

建议改为：

1. stage policy 从 objective profile 读取可用 metric；
2. DB 查询只允许 profile 中声明的 metric；
3. 不再把 metric 名称硬编码为 WR。

### 8.3 `selector.py` 和 `acquisition.py` 仍以 WR / params / wall time 为中心

当前 selector 聚合的是 `AVG(r.final_win_rate)`，acquisition 的核心特征也是 `predicted_wr`、`mean_params`、`mean_wall_s`。

FX domain 需要改成 objective-aware：

```json
{
  "objectives": {
    "maximize": ["utility_score", "liquidity_headroom_min"],
    "minimize": ["cash_shortfall_cvar95", "fx_cost_bps", "trade_count"]
  },
  "hard_constraints": {
    "liquidity_breach_count": 0,
    "unsupported_action_count": 0
  }
}
```

推荐做法不是把 FX 指标塞进 WR 字段，而是新增通用 metric ledger。

### 8.4 `branch.py` 仍假设 checkpoint / resume 语义

Gomoku 的 branch 是从 checkpoint resume。FX domain 的 branch 更像：

```text
parent strategy
  → 修改某几个 policy 参数
  → 在相同 scenario window 或更高 fidelity stage 重放
```

不需要 model checkpoint。v22 应把 branch 抽象成：

1. `state_ref`：可以是 checkpoint，也可以是 strategy config / portfolio ledger snapshot；
2. `branch_reason`：仍然可复用；
3. `delta_json`：仍然可复用；
4. executor：由 domain adapter 决定是否需要 `--resume`。

### 8.5 `runs` 表缺少 domain-generic metrics

当前 framework 把比较依赖压在 `final_win_rate`、`num_params`、`wall_time_s`。这对 Gomoku 合理，对 FX 不够。

建议 v22.0 新增：

```sql
run_metrics(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL NOT NULL,
  metric_unit TEXT,
  metric_role TEXT,       -- objective / constraint / diagnostic
  direction TEXT,         -- maximize / minimize / equals / le / ge
  created_at TEXT NOT NULL,
  UNIQUE(run_id, metric_name)
)
```

并让 Pareto / selector / acquisition 优先读取 `run_metrics`。`final_win_rate` 可以保留为 legacy alias。

---

## 9. v22 推荐阶段划分

### v22.0 — Framework generic objective layer

目标：先修通 domain-agnostic 承诺。

工作项：

1. 新增 `run_metrics` 或等价 metric ledger。
2. 新增 objective profile / objective mapping。
3. `analyze.py --pareto` 支持从 objective profile 读取 maximize/minimize。
4. `sweep.py` 支持 dynamic axes / `--axis` / `--candidate-json`。
5. `stage_policy.py` metric 白名单改为 profile-driven。
6. selector / acquisition 保留 Gomoku 兼容路径，但新增 generic metric path。

验收：

1. Gomoku 现有 177-test 级别行为不回退。
2. 一个 fake non-gomoku domain 能写入 `utility_score` 并生成 Pareto front。

### v22.1 — FX offline simulator

目标：不碰真实 API，先把 truth engine 跑通。

工作项：

1. `domains/fx_portfolio/portfolio.py`
2. `market_data.py`
3. `scenario.py`
4. `strategy.py`
5. `simulator.py`
6. `metrics.py`
7. `search_space.json`

验收：

1. 给定初始多币种头寸、现金流、锚定货币和历史汇率路径，能计算：
   - anchor value path
   - liquidity breach
   - shortfall CVaR
   - FX cost
   - trade count
2. `sweep.py` 能跑一个 FX campaign。
3. `analyze.py --pareto` 能输出 FX strategy frontier。

### v22.2 — Sandbox-backed execution rehearsal

目标：用 Airwallex-like sandbox 验证执行链，而不是验证市场 alpha。

工作项：

1. `sandbox_client.py`
   - auth
   - get balances
   - retrieve current rate
   - create quote
   - create conversion
   - handle quote expiry / insufficient funds / unsupported pair
2. `ledger.py`
   - quote event
   - conversion event
   - balance snapshot
   - settlement state
3. sandbox dry-run mode
4. idempotency key discipline

验收：

1. accepted recommendation 能在 sandbox 中完成 quote/conversion rehearsal。
2. outcome 写回 `recommendation_outcomes`。
3. 不提交任何 API key；全部 secret 只能来自环境变量或本地未跟踪配置。
4. sandbox failure 被记录为明确 negative outcome，而不是 silent fallback。

### v22.3 — Constrained acquisition

目标：让 acquisition 真正理解 FX 的硬约束和多目标。

工作项：

1. `acquisition_policy.json` 增加 hard constraints。
2. candidate generator 过滤红线破坏策略。
3. acquisition score 引入：
   - predicted utility
   - uncertainty
   - liquidity headroom
   - CVaR penalty
   - cost penalty
   - infeasibility probability
4. replay benchmark 比较：
   - random grid
   - heuristic selector
   - constrained acquisition

验收：

1. 同等 candidate budget 下，constrained acquisition 的 positive outcome hit-rate 不低于 heuristic。
2. liquidity breach rate 必须显著低于无约束 baseline，理想为 0。
3. 所有 recommendation 有 frontier/surrogate/outcome lineage。

### v22.4 — Paper-trading / human approval boundary

目标：如果未来要接近真实业务，也只能先进入 human-approved paper-trading。

工作项：

1. live API 仍默认禁用。
2. 所有 conversion recommendation 输出 human-readable approval packet。
3. 强制记录：
   - rationale
   - constraints checked
   - quote used
   - expiry
   - expected impact
   - worst-case liquidity headroom
4. 人类确认后才允许执行，且执行日志不可篡改。

验收：

1. 系统不会在无人审批下发起 live conversion。
2. 所有 live-like action 都可审计。

---

## 10. 第一版 search space 草案

```json
{
  "domain": "fx_portfolio",
  "name": "import-export-treasury-core",
  "version": "0.1",
  "protocol": {
    "anchor_currency": "CNY",
    "horizon_days": 30,
    "scenario_set": "historical_2022_2025_major_pairs",
    "sandbox_provider": "airwallex_demo",
    "live_execution": false
  },
  "axes": {
    "rebalance_interval_hours": {
      "type": "int",
      "values": [6, 12, 24, 48],
      "scale": "linear",
      "role": "strategy",
      "allow_continuation": true
    },
    "rebalance_band_bps": {
      "type": "int",
      "values": [50, 100, 200, 400],
      "scale": "linear",
      "role": "strategy",
      "allow_continuation": true
    },
    "liquidity_buffer_multiplier": {
      "type": "float",
      "values": [1.05, 1.15, 1.25, 1.50],
      "scale": "linear",
      "role": "risk",
      "allow_continuation": true
    },
    "max_conversion_fraction_per_day": {
      "type": "float",
      "values": [0.10, 0.20, 0.35],
      "scale": "linear",
      "role": "risk",
      "allow_continuation": true
    },
    "quote_validity": {
      "type": "categorical",
      "values": ["MIN_15", "HR_1", "HR_4", "HR_24"],
      "scale": "linear",
      "role": "execution",
      "allow_continuation": true
    },
    "hedge_ratio_usd": {
      "type": "float",
      "values": [0.25, 0.50, 0.75, 1.00],
      "scale": "linear",
      "role": "strategy",
      "allow_continuation": true
    },
    "hedge_ratio_eur": {
      "type": "float",
      "values": [0.25, 0.50, 0.75, 1.00],
      "scale": "linear",
      "role": "strategy",
      "allow_continuation": true
    }
  }
}
```

注意：当前 `search_space.py` 只允许 role 为 `structure / training / slow`。v22 要么扩展 role 集合，要么把 FX role 暂时映射到旧 role。推荐扩展，而不是继续污染语义。

---

## 11. 第一版 stage policy 草案

```text
Stage A — synthetic smoke
  目标：验证策略不破红线、不产生非法 action
  数据：合成汇率路径 + 小组合
  成本：低

Stage B — historical replay
  目标：在真实历史行情路径上评估 utility / CVaR / cost
  数据：多币种历史 OHLC / mid / spread model
  成本：中

Stage C — scenario stress replay
  目标：加入跳空、相关性断裂、收款延迟、结算失败
  数据：Monte Carlo / stress scenarios
  成本：中高

Stage D — sandbox execution rehearsal
  目标：用支付平台 sandbox 演练 quote / conversion / balance / settlement
  数据：sandbox API + replay candidate
  成本：API 调用成本 / 工程成本

Stage E — paper-trading with human approval
  目标：真实行情观察，不自动执行 live trade
  数据：live quote read-only + human-approved paper ledger
  成本：高，必须审计
```

与 Gomoku 类似，短 budget 只能排除明显坏策略，不能证明长期最优 treasury 策略。FX domain 尤其如此，因为市场 regime 会改变。

---

## 12. 第一版 branch reasons 草案

```json
{
  "increase_liquidity_buffer": {
    "description": "Raise buffer multiplier for currencies close to liquidity floor.",
    "allowed_deltas": {
      "liquidity_buffer_multiplier": {"type": "add", "default_delta": 0.10}
    },
    "preserves_protocol": true
  },
  "tighten_rebalance_band": {
    "description": "React earlier to FX drift by reducing rebalance band.",
    "allowed_deltas": {
      "rebalance_band_bps": {"type": "multiply", "default_factor": 0.5}
    },
    "preserves_protocol": true
  },
  "reduce_daily_conversion_limit": {
    "description": "Lower execution concentration risk.",
    "allowed_deltas": {
      "max_conversion_fraction_per_day": {"type": "multiply", "default_factor": 0.75}
    },
    "preserves_protocol": true
  },
  "quote_validity_upgrade": {
    "description": "Use longer guaranteed quote validity when settlement timing is unstable.",
    "allowed_deltas": {
      "quote_validity": {"type": "set", "default_value": "HR_4"}
    },
    "preserves_protocol": true
  },
  "scenario_upgrade": {
    "description": "Replay the same strategy under stricter stress scenarios.",
    "allowed_deltas": {
      "scenario_set": {"type": "set", "default_value": "stress_major_pairs"}
    },
    "preserves_protocol": false
  }
}
```

这里的 branch 不依赖 neural checkpoint，而依赖 parent strategy config / scenario ledger。

---

## 13. 数据与合规边界

### 13.1 必须保护的数据

1. 企业真实余额。
2. 客户 / 供应商现金流。
3. API key / client secret。
4. 银行账户 / wallet / linked account id。
5. 实时交易报价和 quote id。
6. 任何可识别企业经营状况的数据。

这些都不能提交进 git，也不能写入 update 文档。测试必须使用 synthetic portfolio。

### 13.2 不能做的事

1. 不输出“买入 / 卖出某币种必然获利”的投资建议。
2. 不在无人审批下执行 live conversion。
3. 不把 sandbox 结果宣传为真实市场收益。
4. 不绕过平台限额、KYC、地区限制、资金模式限制。
5. 不把流动性红线作为可牺牲目标。

### 13.3 应做的安全默认值

1. 默认 `live_execution=false`。
2. sandbox credentials 只来自环境变量。
3. 所有外部 API 请求必须有 request id / idempotency key。
4. 所有 conversion recommendation 必须先生成 approval packet。
5. 所有失败响应必须写入 outcome，不允许 silent fallback。

---

## 14. 最小 smoke chain

v22 的第一条完整 smoke chain 应该是：

```text
1. 准备 synthetic import/export portfolio
   - anchor=CNY
   - currencies=[USD, EUR, JPY, GBP, CNY]
   - 每个币种有 initial balance / receivable / payable / liquidity floor

2. 创建 fx campaign
   - search_space=domains/fx_portfolio/search_space.json
   - stage=A
   - protocol 固定 anchor / horizon / scenario_set / sandbox_provider

3. sweep Stage A
   - 动态 axes 生成候选策略
   - simulator 写入 run_metrics

4. analyze Pareto
   - maximize utility_score, liquidity_headroom_min
   - minimize cvar95, cost_bps, trade_count

5. recommend-next
   - selector 生成候选
   - constrained acquisition rerank
   - 写入 frontier_snapshot / surrogate_snapshot / recommendations

6. accept recommendation
   - 人工把计划状态改为 accepted

7. execute recommendation in sandbox rehearsal
   - 取 quote
   - 检查余额与红线
   - sandbox conversion dry-run / demo execution
   - 写回 recommendation_outcomes

8. acquisition-summary / replay benchmark
   - 对比 heuristic vs constrained acquisition
```

验收标准不是“赚钱”，而是：

1. 红线约束被正确执行；
2. 策略指标可复现；
3. Pareto front 可解释；
4. sandbox 执行链可审计；
5. recommendation outcome 可回放。

---

## 15. 对当前问题的直接回答

### 是否可以用类似 Airwallex 的 sandbox 对进出口公司头寸进行真实演绎？

**可以用于执行链演绎，不可以单独用于真实市场演绎。**

sandbox 适合演练：

1. 多币种余额查询；
2. quote 获取；
3. conversion booking；
4. quote validity；
5. settlement date；
6. insufficient balance；
7. unsupported currency pair；
8. idempotency；
9. API failure handling。

但真实策略效果必须来自：

1. 历史汇率数据；
2. 企业现金流账本；
3. spread / fee / settlement model；
4. Monte Carlo / stress scenario；
5. 未来 paper-trading observation。

### 是否可以找到进出口企业用于权衡的 Pareto 前沿？

**可以，而且这正是 v22 domain 最有价值的目标。**

推荐的 Pareto 前沿不是“收益最大 vs 流动性最低”，而是：

```text
在 liquidity floor 永不破线的可行策略集合中，
寻找 value stability / shortfall risk / FX cost / liquidity headroom / operational complexity 的非支配边界。
```

### 是否可以把“头寸价值”当作 Gomoku 的 WR？

**不建议直接等同。**

Gomoku 的 WR 是越高越好的 benchmark truth。FX 的头寸价值如果直接最大化，会把系统推向投机。更好的 WR 类比是：

```text
utility_score =
  value preservation
  - shortfall risk
  - drawdown risk
  - FX cost
  + liquidity safety
```

但长期应使用多目标指标，而不是压成单一 WR。

### 是否可以通过 autoresearch 机制完成检索？

**可以，但前提是先完成 v22.0 的 framework 泛化。**

当前 v21.1 骨架足够好，但这些地方必须先改：

1. dynamic search axes；
2. generic run metrics；
3. objective-aware Pareto；
4. constrained acquisition；
5. branch without checkpoint；
6. sandbox executor with no live default。

---

## 16. 建议的最终 v22 定位

> **v22 不应是“外汇交易机器人”，而应是“受约束的企业 treasury 策略研究系统”。**

它的研究问题不是：

> 哪个币今天会涨？

而是：

> 在一组收付款义务、最低流动性红线、支付平台约束和汇率不确定性下，哪类头寸调整策略能以更低成本维持组合价值稳定，并给 treasury owner 一个可解释的 Pareto 选择面？

这与 myautoresearch 的原始叙事完全一致：

1. agent 读实验数据；
2. 形成假设；
3. 生成候选；
4. 跑受约束实验；
5. 记录结果；
6. 在 Pareto 边界上留给人类做最终取舍。

Gomoku 的 WR frontier 是第一个 domain 的“真理曲线”；FX portfolio 的 frontier 将是第二个 domain 的“经营约束曲线”。这正好完成从游戏训练到真实业务决策支持的跨 domain 跃迁。

