# v22 Findings — Spot FX Quote-Surface Smoke Experiments

> 2026-04-25  
> 前置：`updates/v22-study-v2.md`、`updates/v22-update.md`  
> 目的：在一次 clean-break 之后，对扩展过的 `fx_spot` mock domain 进行多 treasury scenario 的大样本冒烟实验，用数据而不是单次样本来判断当前 v22 代码到底证明了什么、没有证明什么。

---

## 1. 本轮 findings 要回答的问题

上一轮只有一次小 sweep，信息量不足，不能支持对 v22 的强判断。

本轮要回答的是：

1. 如果把 `fx_spot` 的 mock 棋盘做大，加入更多货币、更多桥接路径、更多 treasury 资金池，前沿会不会发生结构性变化？
2. 在多种更接近外贸企业的当前头寸场景下，哪些 quote regime 更容易进入 Pareto front？
3. 两跳桥接路径到底只是噪音，还是会在真实多目标下稳定留在 front 上？
4. 当前 v22 的数据足够支持什么结论，又还缺什么？

---

## 2. 本轮实验前的代码扩展

为了让 smoke 不再只围绕 `USD/EUR/JPY/CNY` 的小盘面，本轮先扩展了 `fx_spot` mock domain：

1. **货币池扩大**
   - 新增：`HKD`、`SGD`、`GBP`、`AUD`、`MXN`
   - 当前 mock 货币集合：
     - `CNY`
     - `USD`
     - `EUR`
     - `JPY`
     - `HKD`
     - `SGD`
     - `GBP`
     - `AUD`
     - `MXN`

2. **桥接路径扩大**
   - 原先只有：
     - `direct`
     - `via_usd`
   - 本轮扩展为：
     - `direct`
     - `via_usd`
     - `via_eur`
     - `via_hkd`
     - `via_sgd`

3. **quote regime 扩大**
   - 保留旧的：
     - `base`
     - `wide_spread`
     - `short_validity`
   - 新增更接近 treasury/market-microstructure 的 regime：
     - `cn_exporter`
     - `asia_hub`
     - `usd_liquidity`
     - `europe_corridor`
     - `latam_volatility`

4. **新增 treasury scenario materialization**
   - `fx_spot/train.py` 现在支持 `treasury_scenario`
   - 它会把候选自动扩展成：
     - `anchor_currency`
     - `portfolio`
     - `liquidity_floors`

---

## 3. 本轮设想的 4 个 treasury scenario

这些 scenario 不是“真实公司数据”，而是为了贴近外贸资金池结构而做的 **可解释 mock**。

### 3.1 `cn_exporter_core`

**设想对象：** 中国出口企业，人民币为锚点，持有大量美元/欧元/日元回款。

**anchor:** `CNY`

**portfolio 结构：**

- `CNY`: 8,000,000
- `USD`: 1,200,000
- `EUR`: 350,000
- `JPY`: 120,000,000
- `HKD`: 2,500,000
- `SGD`: 600,000

**liquidity floor：**

- `CNY`: 5,000,000
- `USD`: 400,000
- `EUR`: 120,000
- `JPY`: 45,000,000
- `HKD`: 900,000
- `SGD`: 180,000

### 3.2 `usd_importer_mix`

**设想对象：** 进口企业，美元为锚点，日常需要保留 USD，但也持有 CNY/AUD/SGD/MXN 等运营头寸。

**anchor:** `USD`

### 3.3 `global_diversified`

**设想对象：** 多区域 treasury，人民币锚定，但同时持有欧美、亚太、拉美币种。

**anchor:** `CNY`

### 3.4 `asia_procurement_hub`

**设想对象：** 亚洲采购中心，美元锚定，但 HKD/SGD/JPY 是运营流动性的重要节点。

**anchor:** `USD`

---

## 4. 本轮 quote regime 的含义

### 4.1 `cn_exporter`

模拟：

1. `USD/CNY`、`HKD/CNY`、`SGD/CNY` spread 更紧
2. `MXN/CNY` 等非核心走廊 spread 更宽
3. validity 中等偏长

### 4.2 `asia_hub`

模拟：

1. `JPY/HKD`、`HKD/CNY`、`SGD/CNY` 更适合作为亚洲桥
2. validity 对 `JPY`/`MXN` 缩短
3. settlement 稍慢于基础档

### 4.3 `usd_liquidity`

模拟：

1. 含 `USD` 的走廊更紧
2. `USD` 流动性最好
3. validity 比较短，但比 `short_validity` 温和

### 4.4 `europe_corridor`

模拟：

1. `EUR/USD`、`GBP/USD`、`GBP/EUR`、`EUR/CNY` 走廊更紧
2. 欧洲收款/换汇通道较顺

### 4.5 `latam_volatility`

模拟：

1. `MXN` 相关走廊 spread 大幅扩大
2. validity 很短
3. settlement lag 更长

---

## 5. 实验矩阵

本轮 clean-break 后重新清空 `output/`，然后运行 4 个 campaign：

| campaign | treasury scenario | sell currencies | buy | route templates | fractions | quote regimes | runs |
|---|---|---|---|---|---|---|---:|
| `fx-cn-exporter` | `cn_exporter_core` | `USD, EUR, JPY, HKD, SGD` | `CNY` | `direct, via_usd, via_hkd, via_sgd` | `0.15, 0.4, 0.75` | `cn_exporter, asia_hub, usd_liquidity` | 180 |
| `fx-usd-importer` | `usd_importer_mix` | `CNY, EUR, AUD, SGD, MXN` | `USD` | `direct, via_eur, via_sgd, via_hkd` | `0.15, 0.4, 0.75` | `usd_liquidity, europe_corridor, latam_volatility` | 180 |
| `fx-global-diversified` | `global_diversified` | `USD, EUR, GBP, AUD, JPY, MXN` | `CNY` | `direct, via_usd, via_eur, via_hkd` | `0.15, 0.4, 0.75` | `cn_exporter, europe_corridor, latam_volatility` | 216 |
| `fx-asia-procurement` | `asia_procurement_hub` | `CNY, HKD, SGD, JPY, AUD` | `USD` | `direct, via_hkd, via_sgd, via_eur` | `0.15, 0.4, 0.75` | `asia_hub, usd_liquidity, short_validity` | 180 |

**总样本量：**

- `runs = 756`
- `quote_windows = 756`
- `fx_quotes = 1242`
- `fx_route_legs = 1242`

说明：

1. 平均每个 run 产生 `1.64` 条 quote / route leg。
2. 这意味着 front 已经不是一批几乎重复的单腿点，而是有大量两跳路径进入竞争。

---

## 6. 总体结果

### 6.1 campaign 级别统计

| campaign | runs | front points | knee | avg legs/run | avg quotes/run | front route mix | front quote mix |
|---|---:|---:|---|---:|---:|---|---|
| `fx-asia-procurement` | 180 | 36 | `ed74533a` | 1.65 | 1.65 | `direct:13, via_eur:3, via_hkd:11, via_sgd:9` | `asia_hub:18, usd_liquidity:18` |
| `fx-cn-exporter` | 180 | 39 | `098dc8df` | 1.60 | 1.60 | `direct:9, via_hkd:15, via_sgd:4, via_usd:11` | `asia_hub:6, cn_exporter:23, usd_liquidity:10` |
| `fx-global-diversified` | 216 | 49 | `4bd4deff` | 1.67 | 1.67 | `direct:18, via_eur:6, via_hkd:12, via_usd:13` | `cn_exporter:32, europe_corridor:17` |
| `fx-usd-importer` | 180 | 45 | `a77f61ed` | 1.65 | 1.65 | `direct:16, via_eur:16, via_hkd:10, via_sgd:3` | `europe_corridor:23, usd_liquidity:22` |

### 6.2 quote regime 命中率

| quote scenario | runs | front hits | hit rate |
|---|---:|---:|---:|
| `cn_exporter` | 132 | 55 | 0.417 |
| `europe_corridor` | 132 | 40 | 0.303 |
| `usd_liquidity` | 180 | 50 | 0.278 |
| `asia_hub` | 120 | 24 | 0.200 |
| `latam_volatility` | 132 | 0 | 0.000 |
| `short_validity` | 60 | 0 | 0.000 |

### 6.3 route / currency 命中率

| route template | runs | front hits | hit rate |
|---|---:|---:|---:|
| `direct` | 189 | 56 | 0.296 |
| `via_hkd` | 189 | 48 | 0.254 |
| `via_usd` | 99 | 24 | 0.242 |
| `via_eur` | 144 | 25 | 0.174 |
| `via_sgd` | 135 | 16 | 0.119 |

| effective legs | runs | front hits | hit rate |
|---|---:|---:|---:|
| `1` | 270 | 74 | 0.274 |
| `2` | 486 | 95 | 0.195 |

注意：

> 有些 `via_*` 模板在桥接币正好等于起点或终点时，会退化成 direct path。  
> 因此，**分析 route complexity 时应优先看 `route_leg_count`，而不是只看 route template 标签。**

---

## 7. 每个 campaign 的 knee point

| campaign | knee run | treasury scenario | sell | buy | route | quote scenario | fraction | spread | headroom | preservation | validity | legs |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `fx-asia-procurement` | `ed74533a` | `asia_procurement_hub` | `HKD` | `USD` | `via_hkd` | `asia_hub` | 0.15 | 9.00 | 1.17622 | 0.999989 | 1200 | 1 |
| `fx-cn-exporter` | `098dc8df` | `cn_exporter_core` | `USD` | `CNY` | `direct` | `cn_exporter` | 0.15 | 7.00 | 0.77268 | 0.999980 | 1500 | 1 |
| `fx-global-diversified` | `4bd4deff` | `global_diversified` | `USD` | `CNY` | `via_usd` | `cn_exporter` | 0.15 | 7.00 | 0.92785 | 0.999979 | 1500 | 1 |
| `fx-usd-importer` | `a77f61ed` | `usd_importer_mix` | `EUR` | `USD` | `via_eur` | `europe_corridor` | 0.15 | 7.00 | 1.01907 | 0.999996 | 1500 | 1 |

共同点非常明显：

1. **所有 knee 都落在低 rebalance fraction (`0.15`)**
2. **所有 knee 都来自更紧的 corridor regime**
3. **所有 knee 都是“先保头寸缓冲，再压 spread，再保 preservation”的中等复杂度点**

换句话说：

> 当前 mock 下，knee 不是大幅调仓点，而是**保守、轻量、低 friction 的局部再平衡点**。

---

## 8. 数据支持的结论

### 8.1 不是所有 quote regime 都值得继续探索

这是本轮最清晰的结论之一：

1. `latam_volatility` **0 front hit**
2. `short_validity` **0 front hit**

这意味着，在当前 mock 结构下：

- **高波动 + 很短 validity** 的 regime 只是在系统里制造劣势点；
- 它们没有带来“高 risk 高 reward”的前沿补充；
- 至少在 smoke 阶段，它们更像是应该被快速排除的劣质搜索区。

这条结论是有数据支持的，不是主观看法。

### 8.2 两跳桥接路径不是噪音

如果只看 hit rate，1-leg 更强：

- `1-leg hit rate = 27.4%`
- `2-leg hit rate = 19.5%`

但如果看 **front 点的绝对数量**：

- `1-leg front points = 74`
- `2-leg front points = 95`

而且 4 个 campaign 里，每个 campaign 的 front 都是 **2-leg 点数量 >= 1-leg 点数量**。

这说明：

> 两跳路径虽然平均命中率更低，但它们提供了大量单腿点无法提供的 trade-off，因此仍然稳定留在 front 上。

尤其是：

1. `cn_exporter_core` 的 `via_hkd` / `via_usd`
2. `asia_procurement_hub` 的 `via_hkd` / `via_sgd`
3. `usd_importer_mix` 的 `via_eur`

这些都不是偶发噪音，而是可以稳定进入非支配集的结构性选择。

### 8.3 不同 treasury profile 会偏好不同 corridor

从 front quote mix 可以看到：

1. **出口型 CNY 锚定资金池**
   - `cn_exporter` regime 命中最多
   - `via_hkd` / `via_usd` 很常见

2. **USD 锚定进口资金池**
   - `europe_corridor` 与 `usd_liquidity` 平分 front
   - `via_eur` 在 front 上很活跃

3. **亚洲采购型**
   - `asia_hub` 与 `usd_liquidity` 各占一半
   - `HKD/SGD` 作为桥接币有明显存在感

这说明：

> 当前 mock 已经足以让“企业头寸结构”改变 front 形状，而不是所有 campaign 都收敛到同一类答案。

### 8.4 当前 front 仍然是“保值型 front”，不是“增值型 front”

本轮所有 best preservation 都非常接近 1，但没有超过 1：

| campaign | best preservation |
|---|---:|
| `fx-asia-procurement` | 0.999998 |
| `fx-cn-exporter` | 0.999995 |
| `fx-global-diversified` | 0.999993 |
| `fx-usd-importer` | 0.999998 |

这说明当前 mock provider 的经济含义是：

1. 它能表达 **spread / validity / route complexity / headroom** 的权衡；
2. 但它还没有制造出可以稳定观察到的 **positive spot uplift**。

也就是说，当前 front 更像：

> **“怎样用尽可能小的 friction 去做 liquidity-safe rebalancing”**

而不是：

> **“怎样在当前 quote surface 上找到可验证的正向 uplift”**

这不是坏事，但必须如实说明。

---

## 9. 当前代码暴露出的边界

### 9.1 当前 hard constraint 其实没有被真正压测

本轮统计：

- `feasible = 756`
- `infeasible = 0`

原因不是说系统已经足够鲁棒，而是因为当前 evaluator 的卖出量逻辑是：

```text
sell_amount = surplus_above_floor * rebalance_fraction
```

这保证了：

1. sell leg 从设计上就不会打穿 floor
2. 所以 `liquidity_floor_ok` 几乎总是成立

因此：

> 当前 Pareto 是“约束友好的局部再平衡 front”，不是“接近红线时的真实 constrained frontier”。

如果要真的研究流动性第一的 frontier，下一步必须让 candidate 能提出：

1. 更激进的 sell amount
2. split route
3. 多腿并行执行
4. 靠近 floor 的主动试探

否则 hard constraint 只是在 schema 里存在，没有真正成为 front 形状的主要驱动。

### 9.2 route template 与有效路径复杂度要分开

由于：

```text
via_usd on USD->CNY
via_eur on EUR->USD
via_hkd on HKD->USD
```

这类候选会退化成 1-leg path，所以：

1. route template 适合表示“搜索意图”
2. `route_leg_count` 才是“实际复杂度”

后续 findings / plotting / BO surrogate 最好直接基于实际 legs，而不是 template label。

### 9.3 当前还不是 Bayesian frontier approximation

本轮数据比上轮多得多，但这仍然只是：

1. 更大的 Cartesian product
2. 更多 treasury scenario
3. 更多 quote regime
4. 事后 Pareto sort

它仍然 **不是**：

1. GP surrogate
2. EI / UCB / Thompson / qNEHVI
3. 迭代式主动补点
4. uncertainty-guided frontier approximation

所以，当前 findings 证明的是：

> **frontier observation 已经从单场景提升到多 treasury scenario 的稳定观察。**

但它还没有证明：

> **我们已经在 FX domain 上实现了贝叶斯式前沿逼近。**

---

## 10. 对 v22 当前状态的判断

本轮 756 点 smoke 之后，可以更稳地说：

### 10.1 已经被数据支持的部分

1. `fx_spot` 不是一个只会产出单点的玩具 demo
2. 多 treasury scenario 下，front 的组成会真实变化
3. 多跳桥接路径会稳定留在前沿，不是偶发噪音
4. 某些 regime（`short_validity`、`latam_volatility`）可以被数据性地排除
5. 当前框架已经能支撑：
   - multi-campaign
   - multi-scenario
   - objective-profile Pareto
   - knee point
   - quote evidence lineage

### 10.2 仍然没有被证明的部分

1. 正向 uplift 是否存在
2. liquidity constraint 贴边时 front 如何变形
3. split amount / multi-leg portfolio routing 是否会产生更丰富的 knee
4. 贝叶斯 acquisition 是否能比大 sweep 更快逼近 front

---

## 11. 下一步最值得做什么

如果只允许从数据里推下一步，而不是拍脑袋，优先级应是：

### P1. 让约束真正参与竞争

新增 candidate 维度：

1. `sell_amount_mode = surplus_fraction | explicit_ratio | floor_probe`
2. `floor_buffer_target`
3. `split_ratio`

目标：

- 让一部分点变成 infeasible
- 让 constraint-first Pareto 真正有分叉结构

### P2. 去掉退化桥接点

在 candidate generation / sweep 侧过滤：

```text
bridge_currency == sell_currency
bridge_currency == buy_currency
```

避免“名义两跳、实际一跳”的模板污染 route analysis。

### P3. 扩大正向 uplift 的 mock 机制

如果要研究“保值基础上的增值”，需要让 mock provider 能表达：

1. direct vs bridge 的真实 price tier 差异
2. amount bucket 改变 client rate
3. provider 内部 corridor pricing 非一致性

否则 preservation 永远只会接近 1 且略小于 1。

### P4. 进入真正的 Bayesian loop

在 P1-P3 之后，再做：

1. surrogate features
2. posterior uncertainty
3. acquisition-driven resampling
4. frontier-gap targeting
5. knee-neighborhood refinement

否则只是在更大的笛卡尔积上继续 brute-force。

---

## 12. 最终结论

这轮 findings 的核心结论可以压缩成一句话：

> **v22 的 `fx_spot` domain 经过 756 个点的多 scenario smoke 后，已经足以证明“当前头寸 + 当前报价 + route graph + objective-profile Pareto”这条路是成立的；但它证明的是多场景 frontier observation，不是贝叶斯式 frontier approximation。**

更具体地说：

1. **成立的部分**
   - 多场景 front 会变
   - corridor 选择会变
   - 2-leg path 有结构性价值
   - 劣质 quote regime 可以被 front hit rate 清晰淘汰

2. **还没成立的部分**
   - 正向 uplift discovery
   - 贴边 liquidity constraint frontier
   - 贝叶斯主动逼近

因此，v22 到这里的最佳定位不是：

> “我们已经找到 FX Pareto frontier 的贝叶斯算法。”

而是：

> **“我们已经把 FX quote-surface domain 的棋盘、指标、证据和多场景 front 观察系统搭起来了；下一阶段应该用这些数据面，去做真正的 constrained Bayesian frontier refinement。”**

---

## 13. 附加章节 — 当前 Pareto PNG 在视觉上失效的原因与缓解方案

在本轮输出的：

- `output/fx-cn-exporter_pareto.png`
- `output/fx-usd-importer_pareto.png`
- `output/fx-global-diversified_pareto.png`
- `output/fx-asia-procurement_pareto.png`

中，**图片本身已经不再是“信息密度高”，而是“对人类视觉无效”**。

从实际图像可以直接看到：

1. **文字层彻底淹没了几何层**
   - 图中本来真正需要让人看的对象只有：
     - dominated 点云
     - Pareto front 点
     - knee 点
     - frontier 连线
   - 但当前图上首先进入人眼的不是这些几何结构，而是大面积蓝色/灰色长文本框。

2. **annotation 文本过长，而且使用完整 sweep_tag**
   - 当前 label 类似：
     ```text
     fx-usd-importer_treasury_scenariousd_importer_mix_sell_currencyMXN_buy_currencyUSD_route_templatevia_sgd_rebalance_fraction0.4_max_legs2_providermock_quote_scenarioeurope_corridor_sd42
     ```
   - 这对数据库和日志是好 ID，但对静态图是灾难。

3. **大量点在二维投影中重合或近重合**
   - 当前很多点共享或近似共享：
     - `embedded_spread_bps`
     - `liquidity_headroom_ratio`
   - 结果是 annotation 被堆叠成一整条“文字墙”。

4. **高维 front 被粗暴压缩成二维**
   - 当前真正参与 Pareto 的 objective 有 8 个：
     - `liquidity_headroom_ratio`
     - `preservation_ratio`
     - `spot_uplift_bps`
     - `quote_validity_remaining_s`
     - `embedded_spread_bps`
     - `route_leg_count`
     - `settlement_lag_s`
     - `locked_funds_ratio`
   - 现在却只投影成：
     - `x = embedded_spread_bps`
     - `y = liquidity_headroom_ratio`
   - 一旦二维平面承载不了全部语义，系统就试图用长文字补救，结果把图彻底压垮。

5. **当前 static PNG 没有做到“概览”和“细节”分层**
   - 概览图应该回答：
     - front 在哪里
     - dominated cloud 在哪里
     - knee 在哪里
     - 哪类 route / regime 更常出现
   - 明细表才应该回答：
     - 每个点的完整 route / scenario / fraction / exact metrics
   - 现在这两层被硬塞在一张静态图里，所以两边都失败。

6. **这不是调字号能解决的问题，而是编码策略本身错位**
   - 当前 `framework/pareto_plot.py` 会：
     1. 给 dominated 点画 annotation
     2. 给所有 front 点画 annotation
     3. annotation 使用完整 `label_key`
     4. knee 也使用完整 label
   - 这意味着，只要点数上来，图就必然失效。

### 13.1 当前图片为什么不能支持“人类权衡”

从决策角度看，当前 PNG 至少无法让人直接回答：

1. 哪些 front 点是 **1-leg**，哪些是 **2-leg**
2. 哪些点属于 `cn_exporter / europe_corridor / usd_liquidity / asia_hub`
3. 哪些点是低 spread 但 preservation 更差
4. 哪个区域是 knee 附近，哪些只是边界极端点
5. dominated 点云有没有 cluster 或明显分层

原因不是数据太少，而是：

> **图把所有细节都直接写成字贴在图上，导致几何关系完全不可读。**

因此，当前 PNG 不能被当作可决策的可视化产物，只能被视为“静态导出成功”的 artifact。

### 13.2 立即可做的缓解方案（不改变研究逻辑）

这些措施不涉及算法升级，只是把图从“乱码”拉回“可读”：

1. **默认不标注 dominated 点**
   - dominated cloud 只保留灰点，不再逐点写字。
   - 这一步单独就能清掉最大一层视觉噪音。

2. **front 点默认只标注少数关键点**
   - 仅标注：
     - knee
     - x 轴最小点
     - y 轴最大点
     - preservation 最优点
     - validity 最优点
     - 人工指定 top N
   - 其余 front 点只画点，不写文本框。

3. **label 改成短标签，不再直接用完整 sweep_tag**
   - 静态图标签应压缩成类似：
     ```text
     USD->CNY | direct | 0.15 | cn_exporter
     EUR->USD | via_eur | 0.15 | europe
     ```
   - 完整 sweep_tag 保留在 CSV / JSON / Markdown 表里，不放在 PNG 上。

4. **把完整明细从图里拆出去**
   - 对每张 Pareto 图同时输出：
     - `*_pareto_front.csv`
     - `*_pareto_front.md`
   - 图给概览，表给细节。

5. **front 图只展示 front，不和 100+ dominated annotation 混画**
   - 可以保留 dominated 点云，但不要再画 dominated 文本框。
   - 甚至可以直接提供两个版本：
     - overview：front + dominated cloud
     - decision：front + knee only

6. **标题写清楚视觉编码**
   - 例如：
     ```text
     x = Embedded Spread (bps)
     y = Liquidity Headroom Ratio
     color = route template
     marker = quote scenario
     size = rebalance fraction
     ```
   - 不要把这些语义全部压到 annotation 里。

### 13.3 中期修复方案（v22.x 级别）

这些是下一步真正应该落到 plot 代码里的修复：

1. **引入视觉通道分工**
   - `color = route_template`
   - `marker = quote_scenario`
   - `size = rebalance_fraction` 或 `preservation_ratio`
   - `edge/highlight = knee`

2. **增加 small multiples / facet**
   - 不要把所有 regime 混在一张图里。
   - 至少可以按以下维度切面：
     - treasury scenario
     - quote scenario
     - effective legs (`1-leg` / `2-leg`)

3. **增加局部放大图**
   - 当前大量 front 点挤在很窄的 x 区间。
   - 对 knee 附近加 inset zoom，会比拉长整张图更有效。

4. **增加 front-only 决策图**
   - 一张图只画 non-dominated points + knee。
   - dominated cloud 只在 overview 图中保留。

5. **用编号替代全文字标签**
   - 图上只标：
     - `F1, F2, F3 ...`
     - `K`
   - 旁边表格解释每个编号对应的 candidate。

6. **输出 route summary label**
   - 例如：
     ```text
     F7 = MXN->USD via_eur | 0.15 | europe
     ```
   - 这样人能看懂，也能回查。

### 13.4 更合适的最终方案（如果目标是真正支持人类权衡）

如果目标不是“导出一个 PNG”，而是“支持人类做 trade-off decision”，那么最终合理的形态应该是：

1. **interactive HTML plot**
   - 鼠标悬停显示完整 candidate payload
   - 默认图上不写大段文字
   - 支持按：
     - route template
     - quote scenario
     - sell currency
     - effective legs
     - treasury scenario
     过滤

2. **linked table**
   - 图上只显示编号或 hover
   - 右侧表显示：
     - route
     - scenario
     - spread
     - headroom
     - preservation
     - validity
     - legs
     - settlement lag

3. **多图联动，而不是单张万能图**
   - overview scatter
   - front-only scatter
   - knee neighborhood plot
   - route complexity vs preservation
   - quote validity vs preservation
   - headroom before/after bar

4. **把高维 front 的二维投影显式化**
   - 让用户知道：
     - 当前看到的只是某一组 x/y 投影
     - 某些 front 点在这个投影中几乎重叠，但在其他 objective 上并不相同

### 13.5 对当前 v22 可视化状态的判断

当前 v22 的数据面已经足够让 front 发生结构变化，但 **可视化层还没有跟上数据复杂度**。

因此，这轮 findings 应该把结论写清楚：

> **当前 `output/*.png` 不能被当作可供人类直接做权衡的决策图。它们是“数据存在”的证明，不是“视觉分析可用”的证明。**

更准确地说，v22 现在已经完成了：

1. multi-scenario data generation
2. objective-profile Pareto sorting
3. knee point extraction
4. front observation

但还没有完成：

1. human-readable frontier visualization
2. decision-grade trade-off UI
3. high-density Pareto projection design

### 13.6 建议的优先执行顺序

如果只选最少动作、最快止血，优先级应该是：

1. **立刻停用 dominated annotation**
2. **front 只标注 knee + top N**
3. **引入短标签 + companion table**
4. **把 color / marker / size 用起来**
5. **再考虑 interactive HTML**

也就是说，先把图从“乱码”变成“概览图”，再把细节外移，而不是继续往静态 PNG 上堆字。

---

## 14. 附加章节 — `fx_spot` 的输出必须按 run id 归档

除了图本身不可读之外，当前 v22 还暴露出另一个非常实际的问题：

> **每次 `fx_spot` 执行产生的日志、Pareto JSON、PNG、TXT、数据库快照都被直接堆在 `output/` 根目录下。**

这会带来 4 个直接后果：

1. **实验边界不清楚**
   - 人很难一眼判断：
     - 哪些文件属于这一次 smoke
     - 哪些文件属于上一次 clean-break 前后的残留
     - 哪些图和哪一批 quote window / tracker.db 是同一轮证据

2. **多 campaign 输出彼此污染**
   - 即使 campaign 名不同，只要都扔在 `output/` 根目录，文件数量一多，查找、比较、回溯都会迅速失控。
   - 如果未来加入更多 scenario、更多 plot 类型、更多 front table，这种混放会比当前更乱，而不是更清晰。

3. **复现实验时缺少“执行容器”**
   - 当前数据库里的 `runs.id` 是 candidate / execution record 的主键，但文件系统层面缺少一个“本轮实验容器”。
   - 也就是说，**数据库有 run identity，磁盘上的 artifacts 却没有对应的 run workspace**。

4. **可视化整改之后，文件管理问题会进一步放大**
   - 一旦 v23 开始输出：
     - overview plot
     - front-only plot
     - knee zoom
     - companion CSV / Markdown table
     - acquisition trace
     - surrogate snapshot
   - 如果仍然全部平铺到 `output/` 根目录，视觉层刚修好，文件层就会立刻失效。

### 14.1 这不是命名习惯问题，而是实验组织问题

当前 `output/` 的问题，不是“文件名还可以更规范一点”，而是：

> **缺少一个以本轮 `fx_spot` 执行为单位的 run-scoped artifact boundary。**

也就是说，未来每次 `fx_spot` 跑，都应该有一个外层的 **run id / execution id**，把这一次实验产生的所有 artifacts 收拢到同一个目录里。

这里需要区分两层身份：

1. **framework / DB 内的 `runs.id`**
   - 表示单个 candidate / 单个执行记录。

2. **文件系统层的 `fx_run_id` / `execution_run_id`**
   - 表示“这一次 `fx_spot` 实验”的总容器。
   - 同一轮实验里的多个 campaign、多个 plot、多个 report，都应该归到这个目录下。

### 14.2 建议的输出目录形态

推荐从 v23 开始，把输出改成：

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
          quotes/
            quote_window.json
            route_legs.json
          reports/
            summary.txt
            acquisition_trace.json
```

最少也应该做到：

```text
output/fx_spot/<fx_run_id>/
```

作为每次执行的根目录，而不是继续把：

- `fx-*.log`
- `fx-*_pareto.json`
- `fx-*_pareto.png`
- `fx-*_pareto.txt`
- `tracker.db`

全部扔在 `output/` 根目录。

### 14.3 run id 的最低要求

每次 `fx_spot` 执行都应满足：

1. **有明确 run id**
   - 可以显式传入：
     ```bash
     --run-id fx-20260425-cn-exporter-a
     ```
   - 或者自动生成，但必须落到 manifest，并反写到日志与报告里。

2. **所有 artifacts 默认写入 run-scoped 目录**
   - 包括：
     - tracker.db
     - log
     - Pareto PNG / JSON / TXT
     - front table
     - quote evidence
     - acquisition trace

3. **根目录只保留索引，不保留散乱正文**
   - `output/` 或 `output/fx_spot/` 根层最多保留：
     - latest pointer
     - run index
     - 汇总清单
   - 不应继续承担所有明细文件的平铺存放。

4. **报告必须能从文件路径反推出实验边界**
   - 当人看到某个 plot 时，应该能立刻知道它属于：
     - 哪个 `fx_run_id`
     - 哪个 campaign
     - 哪个 objective profile
     - 哪一轮 smoke / benchmark

### 14.4 这项改造对 v23 是 P0，不是附属优化

因此，这个问题不应被视为“顺手整理一下 output”的小修饰，而应该被写入下一阶段工作优先级：

> **v23 必须把 `fx_spot` 的实验输出从“根目录平铺”升级为“run id 归档的实验工作区”。**

否则：

1. findings 会越来越难复核
2. plots 会越来越难比较
3. benchmark 会越来越难留证
4. Bayesian refinement 的迭代轨迹会越来越难回放

更直接地说：

> **如果没有 run-scoped output workspace，那么即使后续把 Pareto 图修好、把贝叶斯 loop 做出来，整个实验系统在证据组织层仍然是不合格的。**
