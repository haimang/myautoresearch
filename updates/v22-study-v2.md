# v22 Study v2 — Spot FX Quote-Surface Autoresearch

> 2026-04-25  
> 前置：`updates/v21.1-update.md`、`updates/v22-study.md`、当前 `framework/` 代码  
> 触发原因：上一版把 FX domain 错误地引向了历史账本 / 历史汇率 / trend replay；本版按“当前头寸 + 当前报价 + 30 分钟局部窗口”的 spot-only 设定重新分析。

---

## 1. 先纠正上一版的误解

上一版 `v22-study.md` 的主要问题是：它把外汇头寸 domain 解释成了一个“历史行情 / 情景回放 / 风险预测”问题。这不是当前要研究的方向。

本版采用完全不同的定义：

> **v22 的 FX domain 不是 trend research，不是宏观预测，不是历史回测；它是 spot quote-surface research。**

也就是说，系统只回答一个局部、当前、可控的问题：

> **在此刻已知的多币种头寸与流动性红线下，如果我可以向一个或多个支付 / FX 平台请求当前报价，那么所有合法的直接换汇、多跳换汇、锁定报价、拆单路径，会在“流动性安全、锚定货币价值、成本、路径复杂度、quote validity”之间形成怎样的 Pareto front？是否存在 knee point？**

这与 myautoresearch 的 scope 是一致的，甚至比上一版更符合：

1. **实验边界局部**
   - 不预测明天。
   - 不解释趋势。
   - 不引入宏观变量。
   - 只研究当前 quote graph。

2. **实验对象可枚举**
   - sell currency / buy currency。
   - trade amount bucket。
   - direct route / multi-hop route。
   - quote validity。
   - settlement date。
   - provider / account / funding mode。

3. **实验结果可观测**
   - quote 返回的 `client_rate`、`mid_rate`、`awx_rate`。
   - quote `valid_from_at` / `valid_to_at`。
   - conversion 可用性。
   - 余额与流动性红线是否被破坏。

4. **Pareto 目标明确**
   - 首要：流动性红线不破。
   - 次要：锚定货币价值保值。
   - 再次：在保值基础上寻找增值 / uplift。
   - 同时最小化成本、路径长度、settlement lag、quote expiry risk。

因此，本版结论是：

> **你的需求符合 autoresearch 的 scope，而且是一个比“历史趋势型外汇研究”更适合当前框架的 domain：它是一个有限窗口内的 constrained quote-surface exploration。**

---

## 2. 这个问题应该怎样被数学化

### 2.1 当前状态

给定一个时刻 `t0`：

```text
portfolio P0:
  balance[currency] = amount
  liquidity_floor[currency] = minimum required amount
  anchor_currency = CNY or USD
  quote_window = [t0, t0 + 30min]
  providers = [airwallex, ...]
```

这就是本 domain 的全部初始世界。不要引入历史趋势，不要预测未来路径。

### 2.2 报价图

把当前可获得的报价看成一个有向图：

```text
node:
  currency

edge:
  provider
  sell_currency
  buy_currency
  sell_amount or buy_amount
  client_rate
  mid_rate
  awx_rate
  quote_id
  valid_from_at
  valid_to_at
  conversion_date
  settlement_lag
  supported / unsupported
```

每条边代表一次可报价的转换动作：

```text
USD --sell 10,000 USD / buy CNY--> CNY
CNY --sell 50,000 CNY / buy EUR--> EUR
EUR --sell 8,000 EUR / buy USD--> USD
```

路径则代表一组连续动作：

```text
USD -> CNY
USD -> EUR -> CNY
USD -> HKD -> CNY
EUR -> USD -> CNY
CNY -> USD -> CNY
```

这个 graph 本身就是实验对象。

### 2.3 候选策略

一个候选不是长期策略，而是当前 30 分钟窗口内的一条可执行 / 可模拟路线：

```json
{
  "route": ["USD", "HKD", "CNY"],
  "provider": "airwallex",
  "sell_amount": 10000,
  "quote_validity": "MIN_15",
  "conversion_date": "same_day",
  "execution_mode": "quote_only",
  "max_quote_age_seconds": 60
}
```

多路线组合也可以作为候选：

```json
{
  "legs": [
    {"sell": "USD", "buy": "CNY", "sell_amount": 5000},
    {"sell": "EUR", "buy": "CNY", "sell_amount": 3000}
  ],
  "quote_validity": "MIN_30",
  "execution_mode": "mock_conversion"
}
```

因此，v22 的 search space 不是历史参数，而是当前报价动作空间。

---

## 3. “保值 / 增值”在 spot-only 设定下是什么意思

### 3.1 不能定义成“未来一定上涨”

在 spot-only 设定下，我们不能说：

> 这条线路未来 30 分钟一定增值。

因为那需要预测行情。预测行情不是本 domain 目标。

### 3.2 应定义成“以当前可执行报价重新标价后的组合改善”

更准确的定义是：

> **在同一个 quote window 内，用平台返回的可执行 / 可锁定报价，把初始组合和转换后组合都折算成锚定货币，比较它们的当前可变现价值。**

也就是：

```text
initial_anchor_value =
  mark_to_anchor(P0, current executable quotes)

terminal_anchor_value(route) =
  mark_to_anchor(apply_route(P0, quoted conversions), current executable quotes)

value_uplift =
  terminal_anchor_value - initial_anchor_value
```

这里的 “uplift” 不是趋势收益，而是当前报价图中的相对改善。

如果存在 uplift，它可能来自：

1. 不同 currency pair 的报价差异。
2. direct route 与 multi-hop route 的隐含价差。
3. quote validity / amount bucket 带来的 rate tier 差异。
4. provider 内部 pricing / spread / rounding。
5. settlement / funding mode 差异。

但必须明确：

> 支付平台通常会通过 spread、rate tier、风控和条款抑制无风险套利。系统可以搜索 quote-surface inefficiency，但不能假设一定存在稳定套利。

因此，“增值”应被写成 `spot_quote_uplift`，不是 investment alpha。

---

## 4. 目标优先级：流动性第一，保值第二，增值第三

这个 domain 必须使用 lexicographic objective，而不是普通加权求和。

### 4.1 第 0 层：硬约束

候选路线必须满足：

```text
for every currency c:
  post_route_available_balance[c] >= liquidity_floor[c]
```

同时还要满足：

```text
supported_pair = true
quote_not_expired = true
balance_sufficient = true
conversion_amount_within_limit = true
settlement_mode_allowed = true
idempotency_safe = true
```

任何破坏硬约束的候选都不能进入 Pareto front。

### 4.2 第 1 层：流动性质量

在硬约束之上，优先最大化：

```text
liquidity_headroom_min =
  min_c((post_balance[c] - liquidity_floor[c]) / max(liquidity_floor[c], epsilon))
```

这代表路线执行后最薄弱币种还剩多少缓冲。

### 4.3 第 2 层：保值

核心指标：

```text
preservation_ratio =
  terminal_anchor_value / initial_anchor_value
```

如果：

```text
preservation_ratio >= 1.0
```

说明当前报价图下至少没有损失锚定价值。

### 4.4 第 3 层：增值

增值指标：

```text
spot_quote_uplift_bps =
  (terminal_anchor_value - initial_anchor_value) / initial_anchor_value * 10000
```

这只表示当前 quote-surface uplift，不表示未来价格判断。

### 4.5 第 4 层：代价

代价包括：

```text
embedded_spread_bps
route_leg_count
quote_expiry_seconds
settlement_lag_seconds
locked_or_unavailable_funds
api_call_count
execution_failure_risk
```

因此 Pareto front 可以定义为：

```text
hard constraints:
  liquidity_breach_count = 0
  unsupported_action_count = 0
  expired_quote_count = 0

maximize:
  liquidity_headroom_min
  preservation_ratio
  spot_quote_uplift_bps
  quote_validity_seconds

minimize:
  embedded_spread_bps
  route_leg_count
  settlement_lag_seconds
  locked_funds_ratio
  api_call_count
```

---

## 5. 为什么这特别适合 Pareto frontier / knee point

外汇 spot quote routes 天然存在 trade-off：

1. **单跳路径**
   - 成本低、执行简单、失败点少。
   - 但可能 rate 不如多跳路径。

2. **多跳路径**
   - 可能找到更好的隐含交叉汇率。
   - 但每一跳都有 spread，路径越长越容易被 spread 吃掉。

3. **长 quote validity**
   - 更像“锁汇”，降低短窗口内价格变化风险。
   - 但平台可能给更保守的 client_rate。

4. **大额一次性转换**
   - 操作简单。
   - 但可能降低某币种流动性缓冲，或进入更差 rate tier。

5. **拆单**
   - 可以保护流动性、探索 amount tier。
   - 但 API 调用、执行复杂度、quote expiry 风险上升。

所以它不是单目标优化问题，而是典型 Pareto 问题：

```text
最安全的路线不一定最保值；
最保值的路线不一定最简单；
理论 uplift 最高的路线可能流动性缓冲最差；
路径最短的路线可能 embedded spread 更高。
```

knee point 的意义是：

> **在继续牺牲复杂度 / 流动性 / quote validity 之前，已经拿到大部分保值或 uplift 好处的那个点。**

这与 gomoku 中 “WR vs params vs wall_time” 的结构非常像，只是 WR 换成了 spot quote value quality。

---

## 6. 贝叶斯算法在这里应该做什么

### 6.1 它不是预测未来行情

这里的 Bayesian / acquisition 不应该预测：

```text
30 分钟后 USD/CNY 会涨还是跌
```

它应该做的是：

> **在有限 quote API 调用预算、quote expiry、路径组合爆炸的限制下，用尽可能少的报价请求逼近当前 quote-surface 的 Pareto front。**

这与 v21.1 的 candidate-pool acquisition 完全一致。

### 6.2 为什么需要 acquisition

假设有：

```text
currencies = 8
amount_buckets = 10
validities = 4
providers = 2
max_path_len = 3
```

只看 direct quote，就有：

```text
8 * 7 * 10 * 4 * 2 = 4,480
```

如果加入两跳、三跳路径，组合会迅速爆炸。虽然 quote API 可能免费，但它仍然受限于：

1. rate limit；
2. quote expiry；
3. 请求延迟；
4. 平台风控；
5. 同一窗口内报价时效一致性；
6. 人类最终阅读和审批成本。

所以 acquisition 的任务是：

```text
先用少量 quote 采样当前报价面
  → 建立局部 surrogate
  → 判断哪些 pair / amount / route 最可能靠近 Pareto front
  → 继续请求这些 quote
  → 更新 frontier
```

这不是 trend BO，而是 **active quote sampling**。

### 6.3 当前 v21.1 acquisition 如何迁移

v21.1 当前 score 结构是：

```text
predicted_wr
+ uncertainty
+ frontier_bonus
+ candidate_type_bonus
- params_penalty
- wall_penalty
```

FX spot domain 可以替换为：

```text
predicted_preservation_ratio
+ predicted_uplift_bps
+ uncertainty_over_quote_surface
+ frontier_gap_bonus
+ liquidity_headroom_bonus
+ quote_validity_bonus
- embedded_spread_penalty
- route_complexity_penalty
- settlement_lag_penalty
- quote_staleness_penalty
- liquidity_risk_penalty
```

硬约束先过滤，acquisition 只在 feasible candidate pool 上排序。

---

## 7. Airwallex pricing / quote model 的可用事实

根据 Airwallex Transactional FX 文档，至少可以确认以下能力：

1. **Retrieve current rate**
   - 可以指定 `buy_currency`、`sell_currency`。
   - 可以可选指定 `sell_amount` 或 `buy_amount`。
   - 可以可选指定 `conversion_date`。
   - 返回当前 rate，用于了解如果现在 book conversion 会得到的价格。

2. **Create Quote**
   - 可以请求 guaranteed quote。
   - 参数包含 `buy_currency`、`sell_currency`、`buy_amount` / `sell_amount`、`conversion_date`、`validity`。
   - 文档列出的 validity 包括：1 minute、15 minutes、30 minutes、1 hour、4 hour、8 hour、24 hour。
   - 返回 `quote_id`、`valid_from_at`、`valid_to_at`、`client_rate`、`mid_rate`、`awx_rate`、`rate_details`、buy/sell amount。

3. **Create Conversion**
   - 可以用 `quote_id` booking conversion。
   - 如果提供有效 `quote_id`，conversion 使用 quote 中的 `client_rate`。
   - 如果不提供 quote_id，会以当前 market rate 执行。
   - 请求包含 `request_id`，作为 idempotency key 防止重复请求。

4. **Funding / settlement**
   - pre-funding 模式下，转换或付款不能超过可用 wallet 余额。
   - post-funding 模式下，可以在 open position limit 内先 book conversion。
   - settlement timing 与 conversion date / funding mode 相关。
   - 文档说明某些结算重试可能按 30 分钟 batch 发生。

这些事实对本 domain 很重要，因为它们意味着：

```text
quote 本身可作为实验观测值；
quote validity 可作为“短时间锁价”变量；
client_rate / mid_rate / awx_rate 可用于估算隐含成本；
conversion_date / funding mode / settlement timing 可作为流动性和执行约束；
request_id 可满足 autoresearch 的可追溯执行纪律。
```

---

## 8. “mockup 交易，但报价是真实的”是否可行

### 8.1 最理想模式：live quote + local mock execution

最符合你设想的模式是：

```text
1. 用真实 API credential 请求真实 quote
2. 不调用 create conversion
3. 在本地 portfolio ledger 中模拟执行
4. 用 quote_id / valid_to_at / client_rate 作为证据
5. 计算 post-route portfolio 与 Pareto metrics
```

这叫：

> **live-pricing / local-mock execution**

优点：

1. 报价是真实当前 quote。
2. 不发生真实资金转移。
3. 不需要历史行情。
4. 可以在 30 分钟窗口内高频探索。
5. 风险远低于真实 conversion。

限制：

1. 需要真实账户权限和 API key。
2. 平台可能对 quote 请求频率有限制。
3. quote 可能因账户、地区、金额、币种、funding mode 不同而变化。
4. quote_id 有有效期，过期后不能继续作为可执行证据。
5. 如果平台条款限制 quote 滥用，需要遵守。

这是 v22 最推荐的首选模式。

### 8.2 第二模式：sandbox quote + sandbox conversion

如果只用 developer sandbox：

```text
quote endpoint -> sandbox quote
conversion endpoint -> sandbox conversion
```

它适合验证：

1. API authentication。
2. request schema。
3. quote_id lifecycle。
4. conversion booking lifecycle。
5. balance insufficient / unsupported pair。
6. idempotency。
7. database outcome backfill。

但是否“报价是真实的”，必须通过平台文档或实测确认。公开文档中的 demo endpoint 可以返回 `client_rate`、`mid_rate`、`awx_rate` 等字段，但这不自动等于真实生产可成交报价。

因此代码层必须记录：

```json
{
  "quote_source": "sandbox_demo",
  "price_reality": "unknown_or_mock",
  "execution_reality": "mock"
}
```

不能把 sandbox quote 直接标记为 live market truth。

### 8.3 第三模式：production quote + sandbox conversion 不一定可行

有些平台不会允许把 production quote_id 带到 sandbox conversion，也不会允许 sandbox conversion 使用 production pricing。即便 API 形态相同，环境通常隔离。

因此不要设计成：

```text
production quote_id -> sandbox conversion
```

除非平台明确支持。

### 8.4 推荐 capability matrix

v22 的 provider adapter 必须为每个平台声明能力：

```json
{
  "provider": "airwallex",
  "environment": "production",
  "supports_live_quote": true,
  "supports_guaranteed_quote": true,
  "supports_mock_conversion": false,
  "supports_sandbox_conversion": true,
  "quote_is_executable": true,
  "quote_reality": "live",
  "conversion_reality": "none_unless_explicitly_booked"
}
```

以及：

```json
{
  "provider": "airwallex",
  "environment": "sandbox",
  "supports_live_quote": "must_verify",
  "supports_guaranteed_quote": true,
  "supports_mock_conversion": true,
  "quote_is_executable": true,
  "quote_reality": "demo_or_mock_until_verified",
  "conversion_reality": "sandbox"
}
```

这能避免把“API 字段看起来真实”误解成“市场真实性已证明”。

---

## 9. 当前报价路径的实验设计

### 9.1 Quote window

每轮研究应创建一个 quote window：

```json
{
  "window_id": "qw-20260425-143000",
  "started_at": "2026-04-25T14:30:00+08:00",
  "expires_at": "2026-04-25T15:00:00+08:00",
  "anchor_currency": "CNY",
  "max_quote_age_seconds": 60,
  "portfolio_snapshot_hash": "...",
  "liquidity_floor_hash": "..."
}
```

所有 quote、candidate、frontier snapshot 都必须绑定这个 window。

如果 quote 超过 `max_quote_age_seconds`，则候选必须重新报价，不能继续使用旧结果。

### 9.2 可行路径生成

输入：

```text
currencies held
anchor currency
supported currency pairs
balance
liquidity floor
amount buckets
max path length
quote validity options
```

生成：

```text
direct routes:
  c -> anchor
  anchor -> c
  c1 -> c2

two-hop routes:
  c1 -> bridge -> anchor
  c1 -> bridge -> c2

portfolio routes:
  route A for USD partial balance
  route B for EUR partial balance
  route C keeps JPY untouched due liquidity floor
```

必须先做静态过滤：

1. 不卖出低于 liquidity floor 的金额。
2. 不生成 unsupported pair。
3. 不生成超过余额或限额的 amount。
4. 限制 max path length，建议第一版 `K <= 2`。
5. 限制每个币种 amount buckets，避免组合爆炸。

### 9.3 报价采样

对候选边请求 quote：

```text
get current rate / create quote
```

写入：

```text
quote_id
client_rate
mid_rate
awx_rate
sell_amount
buy_amount
valid_from_at
valid_to_at
created_at
quote_latency_ms
provider
environment
```

### 9.4 路径求值

对一条路线：

```text
P1 = apply quoted route to P0
```

然后计算：

```text
liquidity_breach_count
liquidity_headroom_min
initial_anchor_value
terminal_anchor_value
preservation_ratio
spot_quote_uplift_bps
embedded_spread_bps
route_leg_count
settlement_lag_seconds
quote_validity_seconds_remaining
locked_funds_ratio
```

### 9.5 Pareto 与 knee point

过滤不可行路线后，做 Pareto：

```text
maximize:
  liquidity_headroom_min
  preservation_ratio
  spot_quote_uplift_bps
  quote_validity_seconds_remaining

minimize:
  embedded_spread_bps
  route_leg_count
  settlement_lag_seconds
  locked_funds_ratio
```

knee point 可先用两种方法：

1. **归一化 utopia distance**
   - 各轴归一化后，选择离理想点最近且不破红线的点。

2. **marginal gain elbow**
   - 观察多增加一跳 / 多牺牲流动性 / 多增加 quote expiry risk 后，保值或 uplift 的边际收益是否明显下降。

---

## 10. 搜索空间草案：spot quote surface

```json
{
  "domain": "fx_spot_portfolio",
  "name": "spot-quote-surface-core",
  "version": "0.1",
  "protocol": {
    "anchor_currency": "CNY",
    "quote_window_seconds": 1800,
    "max_quote_age_seconds": 60,
    "live_execution": false,
    "pricing_mode": "live_quote_local_mock"
  },
  "axes": {
    "route_topology": {
      "type": "categorical",
      "values": ["direct", "two_hop", "portfolio_split"],
      "role": "strategy",
      "allow_continuation": true
    },
    "bridge_currency": {
      "type": "categorical",
      "values": ["USD", "HKD", "EUR", "SGD"],
      "role": "strategy",
      "allow_continuation": true
    },
    "amount_bucket_ratio": {
      "type": "float",
      "values": [0.1, 0.2, 0.35, 0.5, 0.75],
      "role": "execution",
      "allow_continuation": true
    },
    "quote_validity": {
      "type": "categorical",
      "values": ["MIN_1", "MIN_15", "MIN_30", "HR_1"],
      "role": "execution",
      "allow_continuation": true
    },
    "conversion_date": {
      "type": "categorical",
      "values": ["same_day"],
      "role": "execution",
      "allow_continuation": false
    },
    "liquidity_buffer_multiplier": {
      "type": "float",
      "values": [1.0, 1.05, 1.10, 1.25],
      "role": "risk",
      "allow_continuation": true
    }
  }
}
```

当前 `framework/search_space.py` 还只允许 `structure / training / slow` 三种 role。v22.0 要扩展 role：

```text
strategy
execution
risk
provider
```

否则 FX domain 的语义会被迫伪装成 Gomoku 训练语义。

---

## 11. Stage policy 草案：不是历史回测，而是 quote fidelity

```text
Stage A — static quote graph smoke
  使用 mock provider / fixture quote
  验证路径枚举、流动性过滤、Pareto 计算

Stage B — sandbox quote rehearsal
  使用 developer sandbox quote / conversion mock
  验证 API schema、quote_id、validity、conversion lifecycle

Stage C — live quote / local mock
  使用真实 quote endpoint
  不 booking conversion
  本地模拟路径执行，生成真实报价下的 Pareto front

Stage D — human-approved sandbox / paper execution
  人类接受 recommendation
  可在 sandbox booking conversion 或 paper ledger 中执行

Stage E — live execution guarded
  默认不启用
  只有明确人工审批、最小金额、强审计时才允许
```

这里的 fidelity 不是“历史数据越来越真实”，而是：

```text
mock quote
  → sandbox quote
  → live quote
  → paper execution
  → guarded live execution
```

这更符合你的 spot-only 设想。

---

## 12. Branch reasons 草案：局部探索当前路线

```json
{
  "increase_liquidity_buffer": {
    "description": "Keep more of the scarce currency after route execution.",
    "allowed_deltas": {
      "liquidity_buffer_multiplier": {"type": "add", "default_delta": 0.05}
    },
    "preserves_protocol": true
  },
  "try_two_hop_bridge": {
    "description": "Try one bridge currency to test cross-rate surface.",
    "allowed_deltas": {
      "route_topology": {"type": "set", "default_value": "two_hop"},
      "bridge_currency": {"type": "set", "default_value": "HKD"}
    },
    "preserves_protocol": true
  },
  "split_amount": {
    "description": "Split one large conversion into smaller buckets.",
    "allowed_deltas": {
      "route_topology": {"type": "set", "default_value": "portfolio_split"},
      "amount_bucket_ratio": {"type": "multiply", "default_factor": 0.5}
    },
    "preserves_protocol": true
  },
  "extend_quote_validity": {
    "description": "Trade a potentially worse rate for longer guaranteed quote time.",
    "allowed_deltas": {
      "quote_validity": {"type": "set", "default_value": "MIN_30"}
    },
    "preserves_protocol": true
  },
  "reduce_route_complexity": {
    "description": "Collapse multi-hop candidate back to direct route for lower execution risk.",
    "allowed_deltas": {
      "route_topology": {"type": "set", "default_value": "direct"}
    },
    "preserves_protocol": true
  }
}
```

这些 branch 不需要 checkpoint。它们是对当前 route candidate 的局部变体。

---

## 13. 当前 framework 是否能画曲线

### 13.1 能复用的部分

可以复用：

1. `campaigns`
   - 一轮 quote window / portfolio snapshot。

2. `frontier_snapshots`
   - 当前 quote window 的 Pareto front。

3. `recommendation_batches`
   - 一批候选路径推荐。

4. `surrogate_snapshots`
   - acquisition 对当前 quote surface 的证据。

5. `recommendation_outcomes`
   - 被接受路线的 mock / sandbox / paper outcome。

6. `acquisition.py`
   - candidate-pool rerank 思路可复用。

### 13.2 必须改的部分

当前代码还有 Gomoku 硬编码，不能直接用于 FX：

1. `sweep.py`
   - 只支持 `num_blocks / num_filters / learning_rate / steps_per_cycle / buffer_size`。
   - v22 需要 dynamic axes 或 `--candidate-json`。

2. `search_space.py`
   - role 只允许 `structure / training / slow`。
   - v22 需要 `strategy / execution / risk / provider`。

3. `selector.py`
   - 聚合 `final_win_rate`、`num_params`、`wall_time_s`。
   - v22 需要 objective-generic metric。

4. `acquisition.py`
   - 特征是 `predicted_wr / params / wall_s`。
   - v22 需要 quote-surface features。

5. `branch.py`
   - 假设 branch 来自 checkpoint / resume。
   - v22 需要 route/strategy config branch。

6. `analyze.py --pareto`
   - 已支持 maximize/minimize，但很多默认语义仍是 WR / params / wall time。
   - v22 需要 objective profile 驱动。

因此，答案是：

> **框架理念和 ledger 能支持；当前代码需要一层 v22 generic metric / dynamic candidate executor 才能优雅支持。**

---

## 14. 推荐新增 domain 结构

```text
domains/fx_spot/
├── train.py                         # subprocess 入口：执行一次 quote-route episode
├── portfolio.py                     # 当前头寸、红线、锚定币种估值
├── quote_graph.py                   # 当前报价图 / route enumeration
├── provider_base.py                 # provider adapter interface
├── airwallex_client.py              # Airwallex quote / conversion adapter
├── mock_provider.py                 # fixture quote provider
├── route_eval.py                    # 路径执行、本地 mock ledger、指标计算
├── metrics.py                       # preservation / uplift / spread / liquidity / complexity
├── knee.py                          # knee point detection
├── search_space.json
├── stage_policy.json
├── branch_policy.json
├── selector_policy.json
└── acquisition_policy.json
```

这里的 `train.py` 不训练模型，而是执行一次实验：

```text
read candidate route
  → fetch quotes or use fixture
  → locally apply route
  → compute metrics
  → write run + run_metrics
```

为了兼容当前 framework，早期可以临时写：

```text
runs.final_win_rate = normalized preservation / uplift score
runs.wall_time_s = quote_latency_s
runs.num_params = route_leg_count
```

但这只是过渡。正式 v22 应新增 `run_metrics`。

---

## 15. 推荐新增数据库表

### 15.1 `quote_windows`

```sql
CREATE TABLE quote_windows (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL,
  anchor_currency TEXT NOT NULL,
  started_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  max_quote_age_seconds INTEGER NOT NULL,
  portfolio_snapshot_json TEXT NOT NULL,
  liquidity_floor_json TEXT NOT NULL,
  provider_config_json TEXT NOT NULL,
  status TEXT NOT NULL
);
```

### 15.2 `fx_quotes`

```sql
CREATE TABLE fx_quotes (
  id TEXT PRIMARY KEY,
  quote_window_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  environment TEXT NOT NULL,
  quote_source TEXT NOT NULL,
  sell_currency TEXT NOT NULL,
  buy_currency TEXT NOT NULL,
  sell_amount REAL,
  buy_amount REAL,
  client_rate REAL,
  mid_rate REAL,
  awx_rate REAL,
  quote_id TEXT,
  valid_from_at TEXT,
  valid_to_at TEXT,
  conversion_date TEXT,
  raw_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

### 15.3 `run_metrics`

```sql
CREATE TABLE run_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value REAL NOT NULL,
  metric_unit TEXT,
  metric_role TEXT NOT NULL,
  direction TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, metric_name)
);
```

### 15.4 `fx_route_legs`

```sql
CREATE TABLE fx_route_legs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  leg_index INTEGER NOT NULL,
  sell_currency TEXT NOT NULL,
  buy_currency TEXT NOT NULL,
  sell_amount REAL,
  buy_amount REAL,
  quote_ref TEXT,
  route_state_before_json TEXT NOT NULL,
  route_state_after_json TEXT NOT NULL
);
```

这些表能把“当前报价实验”变成可追溯事实，而不是一次性脚本输出。

---

## 16. Airwallex adapter 的安全边界

### 16.1 默认只允许 quote，不允许 conversion

默认配置：

```json
{
  "live_execution": false,
  "allow_create_conversion": false,
  "allow_live_quote": true,
  "execution_mode": "local_mock"
}
```

也就是说：

1. 可以请求 quote。
2. 可以保存 quote_id。
3. 可以本地 mock portfolio route。
4. 不调用 create conversion。

### 16.2 如果允许 sandbox conversion

只能在：

```json
{
  "environment": "sandbox",
  "allow_create_conversion": true
}
```

并明确标记：

```text
conversion_reality = sandbox
```

### 16.3 如果未来允许 live conversion

必须同时满足：

1. 人类审批。
2. 最小金额。
3. 流动性红线再次检查。
4. quote 未过期。
5. idempotency key 已生成。
6. approval packet 已签名 / 记录。
7. live execution 明确开启。

否则系统不得调用 conversion booking。

---

## 17. 这是否能“找到 30 分钟内保值或增值路线”

可以，但需要精确定义：

### 可以找到的是

```text
在当前 quote window 内，
基于当前可请求 / 可锁定报价，
执行某条路线后，
按同一报价口径折算，
组合锚定价值是否不下降，甚至是否存在 spot quote uplift。
```

### 不能承诺的是

```text
30 分钟后真实市场一定还这样；
这个 uplift 可以无限重复；
多平台之间可以无风险套利；
平台不会因风控、限额、报价更新而拒绝执行；
锁定 quote 等于完整金融意义上的远期锁汇。
```

所以文档和 UI 必须使用：

```text
spot quote preservation
spot quote uplift
quote-surface frontier
```

而不是：

```text
future profit
market prediction
risk-free arbitrage
```

---

## 18. 最小可行 smoke chain

### 18.1 Stage A：fixture provider

```text
1. 给定 portfolio:
   CNY 1,000,000
   USD 100,000
   EUR 50,000
   HKD 200,000

2. 给定 liquidity floor:
   CNY >= 300,000
   USD >= 20,000
   EUR >= 10,000
   HKD >= 50,000

3. mock provider 返回固定 quote matrix

4. route enumerator 生成 direct / two-hop routes

5. route_eval 计算 preservation / uplift / cost / liquidity

6. analyze 输出 Pareto front + knee point
```

### 18.2 Stage B：Airwallex sandbox

```text
1. 使用 sandbox credential
2. 请求 rates / quotes
3. 记录 quote_id、client_rate、mid_rate、valid_to_at
4. 本地 mock route
5. 可选 sandbox conversion
6. 写入 recommendation_outcomes
```

### 18.3 Stage C：live quote / no conversion

```text
1. 使用 production quote 权限
2. 只请求 guaranteed quote
3. 不调用 create conversion
4. 本地模拟 portfolio
5. 生成当前 quote-surface Pareto front
6. 给出 human-readable route recommendation
```

### 18.4 Stage D：human-approved paper route

```text
1. 人类选择 knee point
2. 系统重新请求 fresh quote
3. 再次检查 liquidity floor
4. 生成 approval packet
5. 只进入 paper ledger，不真实换汇
```

---

## 19. v22.0 代码实施建议

### 19.1 先做 generic metric / dynamic axes

这是 blocker。否则 FX spot domain 会被迫塞进 Gomoku 字段。

最低改动：

1. `search_space.py`
   - role 扩展。

2. `sweep.py`
   - 支持 `--axis name=v1,v2,v3`。
   - 支持 `--candidate-json path_or_inline`。
   - `run_one()` 不再硬编码 Gomoku 参数。

3. `db.py`
   - 新增 `run_metrics`。

4. `analyze.py`
   - Pareto 支持从 `run_metrics` 读取任意 metric。

5. `acquisition.py`
   - 新增 generic policy path，保留 Gomoku path。

### 19.2 再做 fx_spot domain

1. `portfolio.py`
2. `quote_graph.py`
3. `mock_provider.py`
4. `route_eval.py`
5. `metrics.py`
6. `train.py`

### 19.3 最后接 Airwallex adapter

1. `airwallex_client.py`
2. quote-only mode
3. sandbox mode
4. capability matrix
5. no-live-conversion guard

---

## 20. 最终判断

你的修正非常重要。按 spot-only 重新定义后，v22 domain 的核心判断变成：

> **这是一个当前报价图上的受约束多目标搜索问题，不是趋势预测问题。**

它非常适合 myautoresearch：

1. search space 是当前 route / amount / validity / provider 的笛卡尔积。
2. truth signal 是平台当前 quote 返回值和本地 portfolio 约束计算。
3. acquisition 的价值是减少 quote API 调用并更快逼近当前 Pareto front。
4. Pareto front 可以表达流动性、保值、增值、成本、复杂度之间的真实 trade-off。
5. knee point 可以给人类 treasury owner 一个可解释的行动候选。

因此，v22 的最佳题目应从上一版的：

```text
FX Portfolio Domain / Constrained Bayesian Frontier Search
```

改为：

```text
Spot FX Quote-Surface Autoresearch
```

它的边界是：

```text
不预测行情；
不做历史趋势；
不自动 live 交易；
只在当前可报价、可验证、可审计的局部空间中搜索。
```

这正好符合 myautoresearch 的叙事：agent 不需要理解整个真实世界，它只需要在一个定义良好、可实验、可追溯的局部平面上，越来越快地逼近 Pareto frontier。

