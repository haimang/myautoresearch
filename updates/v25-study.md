# v25 Study — Floorplan-Checker 首次真实世界 Autoresearch 探索计划

> 2026-04-26  
> 前置：`updates/v24-study.md`、`updates/v24.1-update.md`、`updates/v24.1-findings.md`  
> 定位：在 `floorplan_checker` 已完成 benchmark-clean 数据切分、`run_metrics` 接入、Pareto / recommendation 基线打通之后，正式开展 **第一次真实世界的 autoresearch 式系统探索**。

---

## 1. 一句话目标

> **用 Gomoku 阶段验证过的方法论，在真实的 floorplan CV 任务上启动第一轮分阶段、多目标、数据库留痕、可解释图表驱动的系统探索：从 `4x64` 中等容量锚点出发，先找可行工作带，再找 Pareto front，再用 knee point 做第一次“真实世界的研究决策”。**

这次 v25 不是“再做一个 smoke”，而是：

1. 真正用 `campaign + run_metrics + pareto + recommendation + continuation` 的整套 autoresearch 方法跑起来
2. 第一次在真实世界数据上，不靠单次试验，而靠**系统性探索**来找模型结构与超参的 trade-off
3. 让人类研究者拿到**有意义的 front / knee / recommendation 证据**，而不是只拿到一堆孤立跑分

---

## 2. 为什么 v25 可以开始

v25 之前，`floorplan_checker` 还不适合做真实探索，原因是：

1. 数据切分存在 `listingId` leakage
2. Pareto hard constraints 会被 benchmark 污染直接拦死
3. loader / metrics / recommendation 虽然能跑，但 benchmark 本身不可信

现在这些前提已经满足：

1. **clean split 已完成**
   - `dataset_leakage_count = 0`
2. **strict contract 已通过**
   - `missing_file_count = 0`
   - `missing_field_count = 0`
   - `summary_mismatch_count = 0`
3. **真实链路已打通**
   - strict smoke train 可跑
   - `analyze --pareto --metric-source run_metrics` 可输出前沿
   - `analyze --recommend-next` 可正常出推荐

因此，v25 可以从“修系统”转向“真实探索”。

---

## 3. 从 Gomoku 学到什么，如何翻译到 Floorplan

Gomoku 给我们的不是某个具体模型结构，而是一套研究纪律。

### 3.1 Lesson 1 — 先找中等容量锚点，不要一开始铺满全空间

Gomoku 的经验不是“越大越好”，而是先用一个**中等容量、可跑、可比较**的结构建立坐标系，再向上向下展开。

映射到 floorplan：

1. 不从 `2x32` 开始，因为太小，容易把“容量不足”和“超参错误”混在一起
2. 不从 `8x128` 开始，因为太贵，容易让吞吐和显存先成为噪声
3. **从 `4x64` 开始最合理**
   - 它位于当前搜索空间中间
   - 往下可以比较 `2x32` / `4x32`
   - 往上可以比较 `6x64` / `4x128` / `6x128`

### 3.2 Lesson 2 — 先找工作带，再找前沿

Gomoku 中如果一开始就大规模扫全空间，绝大多数预算会浪费在：

1. 不收敛点
2. 低效点
3. 早期就已明显 dominated 的点

所以 v25 也必须遵守：

1. 先用 `4x64` 找出 **learning rate / batch / resolution 的合法工作带**
2. 再固定工作带去比较结构
3. 再对 frontier 邻域做 loss balance 和 BO refine

### 3.3 Lesson 3 — 决策不能只看平均分

Gomoku 看 `WR` 还能勉强单指标决策，但 floorplan 是多头任务。

因此 v25 的解释顺序必须是：

1. 先过红线约束
2. 再看 `val_acc_min_head`
3. 再看 `val_acc_macro`
4. 再看 `wall_time_s` / `inference_latency_ms`
5. 最后才看“某个头是不是特别高”

### 3.4 Lesson 4 — knee point 不是图上最漂亮的点，而是最值得继续投资的点

对 v25 来说，knee point 的意义不是“数学上某个点”，而是：

> **在成本显著上升之前，已经拿到比较均衡多头表现的最小代价候选。**

---

## 4. v25 的总探索原则

v25 全程遵守以下原则：

1. **Benchmark-first**
   - 所有比较都在当前 clean split 上进行

2. **Stage-first**
   - 不允许直接大预算全空间长跑

3. **Metrics-first**
   - 决策优先看 `run_metrics`
   - `final_win_rate` 只作兼容代理

4. **Min-head-first**
   - 平衡性先于“某一个头特别高”

5. **Seed-aware**
   - 不接受单 seed 幻觉

6. **Frontier-first**
   - 不把“单次最好点”当结论
   - 只谈 front / knee / dominated region

7. **Human-readable artifacts**
   - 每一轮都要产出人能看懂的图和表

---

## 5. v25 的第一个真实世界研究问题

这次 v25 不是泛泛地“提高模型”，而是回答下面这个明确问题：

> **在当前 clean split、当前 MLX 资源预算下，Floorplan-Checker 的第一代多头 ResNet，哪一个结构与超参组合位于“准确率、头部平衡、训练成本、推理延迟”之间最合理的 knee region？**

我们要找的不是：

1. 单头最强
2. 纯粹最便宜
3. 纯粹最大的模型

而是：

1. **balanced knee**
2. **cheap baseline**
3. **expensive specialist**

这三类点都要找出来。

---

## 6. v25 研究对象与基线冻结

### 6.1 数据与协议冻结

v25 开始前，以下对象视为冻结：

1. 数据目录：
   - `domains/floorplan_checker/dataset/`
2. clean manifest：
   - `train = 348,615`
   - `eval = 36,582`
   - `test = 20,234`
3. 搜索空间：
   - `domains/floorplan_checker/manifest/search_space.json`
4. 目标定义：
   - `domains/floorplan_checker/manifest/objective_profile.json`
5. stage / selector / acquisition policy：
   - 当前 manifest 版本

v25 不允许中途一边改 benchmark，一边解释 Pareto。

### 6.2 初始锚点：`4x64`

本轮探索锚点：

| 轴 | 值 |
|---|---|
| `num_res_blocks` | `4` |
| `num_filters` | `64` |
| `image_resolution` | `224`（起点） |
| `learning_rate` | 待扫 |
| `batch_size` | 待扫 |
| `loss_weight_*` | 初始全 `1.0` |

选择原因：

1. 是当前搜索空间的中位容量点
2. 成本、表达能力、后续上下扩张都比较自然
3. 适合作为 v25 的第一代真实锚点

---

## 7. v25 分阶段探索路线

v25 不建议一开始就把全部搜索空间做全笛卡尔积。  
推荐分 5 个阶段。

---

### Phase 0 — Benchmark / throughput 冻结确认

**目标**：确认探索使用的是干净 benchmark，而不是继续在脏前提上跑。

执行内容：

1. 跑一次 strict dataset contract
2. 跑一次 strict smoke train
3. 记录 clean split 的 manifest 计数到 findings
4. 确认 workspace / DB / pareto artifact 路径无误

通过标准：

1. `dataset_leakage_count == 0`
2. smoke 可跑
3. `run_metrics` 可写入

---

### Phase 1 — `4x64` 锚点工作带搜索

**目标**：先找 `4x64` 的可行工作带，而不是先比结构。

固定：

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `loss_weight_bed = 1.0`
4. `loss_weight_bath = 1.0`
5. `loss_weight_park = 1.0`

探索轴：

1. `image_resolution ∈ {160, 224, 256}`
2. `batch_size ∈ {16, 32}`
3. `learning_rate ∈ {5e-5, 1e-4, 5e-4}`

总点数：

1. `3 × 2 × 3 = 18` 个点

执行建议：

1. Stage A：全部 18 点，`seed_count = 1`，短预算
2. Stage B：选出 4–6 个合法且不明显 dominated 的点，再补 seed 和更长预算

本阶段回答的问题：

1. 224 分辨率是否明显优于 160
2. 256 是否带来成本飙升但收益不足
3. 哪个 LR 组合最容易稳定收敛
4. batch 16 / 32 哪个更接近 knee

---

### Phase 2 — 结构阶梯探索

**目标**：在已知工作带内比较容量，而不是让 LR/resolution 噪声掩盖结构结论。

固定：

1. 使用 Phase 1 中表现最稳的：
   - `learning_rate`
   - `batch_size`
   - `image_resolution`

结构候选：

1. `2x32`
2. `4x32`
3. `4x64`
4. `6x64`
5. `4x128`
6. `6x128`

为什么不一开始上 `8x128`：

1. 成本过高
2. 在第一次真实探索中更像“上界探针”，不是主战点

执行建议：

1. Stage B：上述 6 个结构，`seed_count = 2`
2. Stage C：frontier 邻域保留 2–3 个结构，进入更长预算验证

本阶段输出：

1. 第一版 capacity frontier
2. 哪些结构是“显著过小”
3. 哪些结构是“明显过大但收益不成比例”

---

### Phase 3 — 多头 loss balance 探索

**目标**：解决“某个头好、某个头差”的 head imbalance。

只对 Phase 2 里最有希望的 2 个结构展开。

建议的 loss pattern 子集：

1. `(1.0, 1.0, 1.0)` neutral
2. `(2.0, 1.0, 1.0)` bed-up
3. `(1.0, 2.0, 1.0)` bath-up
4. `(1.0, 1.0, 2.0)` park-up
5. `(0.5, 1.0, 1.0)` bed-down
6. `(1.0, 0.5, 1.0)` bath-down
7. `(1.0, 1.0, 0.5)` park-down

为什么不直接扫 27 个组合：

1. 第一次真实探索更应该先看方向，而不是先把组合数炸开
2. 先用结构化 pattern 看哪一个 head 最值得补偿

本阶段关注指标：

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `wall_time_s`

输出目标：

1. 找到 balanced candidate
2. 找到 specialist candidate
3. 区分“真平衡提升”和“平均分掩盖退化”

---

### Phase 4 — Bayesian / recommendation 驱动 frontier refine

**目标**：让 autoresearch 真正接管“下一步试什么”。

输入：

1. 前 3 个阶段累积的已完成 runs
2. `run_metrics`
3. 当前 front / dominated region

执行方式：

1. 每轮使用 `analyze --recommend-next`
2. 每轮推荐 batch 保持在 `3–5` 个候选
3. 优先接受：
   - `seed_recheck`
   - `new_point`
4. 对“非常像 knee 邻域”的点，再接受少量：
   - `continue_branch`

不建议：

1. 还没看清 capacity frontier 就大量 branch
2. 还没确定 head imbalance 方向就大规模 loss 微调

本阶段的真正目标不是“让 BO 接管一切”，而是：

> **让 recommendation 开始在 frontier 邻域做更高性价比的探索，而不是继续人工盲扫。**

---

### Phase 5 — Knee confirm / final branch

**目标**：确认第一个真实世界 knee point，并给出是否继续投入的判断。

最终至少保留三类候选：

1. **cheap baseline**
   - 成本最低且不离谱
2. **balanced knee**
   - 多头平衡最好，成本尚可
3. **expensive specialist**
   - 明显更贵，但有明确性能特长

然后再决定：

1. 是否对 knee 做 `lr_decay`
2. 是否对 knee 做 `resolution_upshift`
3. 是否对 specialist 做 continuation

---

## 8. 推荐的 v25 campaign 组织方式

### 8.1 一个 experiment run，多个 stage

推荐使用：

1. `experiment_run_id = v25-floorplan-real-001`
2. `campaign = v25_floorplan_real_001`

优点：

1. 所有 frontier、recommendation、surrogate snapshot 都挂在同一研究主题下
2. 后续 findings / report 更容易追溯
3. artifact 路径整齐

### 8.2 建议的 stage 语义

沿用当前 A/B/C/D，但在 v25 中明确解释为：

| Stage | 语义 | 用途 |
|---|---|---|
| A | legal + smoke | 大面积排除崩溃点 |
| B | early signal | 看收敛趋势、头部失衡、吞吐表现 |
| C | frontier validation | 真正用于 Pareto / knee 比较 |
| D | continuation handoff | 只给 knee / specialist 使用 |

### 8.3 推荐的命名

| 对象 | 命名 |
|---|---|
| experiment run | `v25-floorplan-real-001` |
| campaign | `v25_floorplan_real_001` |
| findings | `updates/v25-findings.md` |
| main Pareto artifact dir | `output/floorplan_checker/v25-floorplan-real-001/campaigns/v25_floorplan_real_001/pareto/` |

---

## 9. 数据库如何配合这次真实探索

这次 v25 不是“文件夹里跑几个图”，而是必须让数据库成为完整研究账本。

### 9.1 必须使用的数据库对象

| 表 / 账本 | 用途 |
|---|---|
| `search_spaces` | 冻结本轮搜索空间版本 |
| `objective_profiles` | 冻结本轮 Pareto / hard constraint 定义 |
| `campaigns` | 记录 v25 这轮真实研究主题 |
| `campaign_stages` | 记录 A/B/C/D 的预算语义 |
| `campaign_runs` | 每个配置、seed、stage 的归属 |
| `runs` | 训练 run 的超参快照与最终汇总 |
| `run_metrics` | v25 的真理面：多头质量、吞吐、延迟、约束 |
| `promotion_decisions` | 从 A/B/C 晋升时的门禁证据 |
| `frontier_snapshots` | 每次 Pareto front 的快照 |
| `surrogate_snapshots` | 每次 recommendation 的代理面快照 |
| `recommendation_batches` | 每轮推荐批次 |
| `recommendations` / `recommendation_outcomes` | 推荐内容与执行效果 |

### 9.2 v25 的数据库使用原则

1. **一个重要结论必须能在 DB 中追溯**
2. **一个 Pareto front 必须有 frontier snapshot**
3. **一次 recommendation 必须有 surrogate snapshot**
4. **一个被采纳的下一步动作必须记录 outcome**

### 9.3 v25 期间建议常看的 DB 问题

1. 哪些 `4x64` 点在 A 阶段就失败？
2. 哪些结构在 B 阶段已有明显 dominated 迹象？
3. 哪些 frontier 点只是单 seed 幻觉？
4. 当前 knee 相对上一轮 frontier 是否稳定？
5. recommendation 是在补 frontier gap，还是只是在重复看起来漂亮的老点？

### 9.4 推荐的人工 SQL 视角

下面这些查询足够支撑 v25 日常巡检：

```sql
-- 1. 看某个 campaign 的 stage 覆盖情况
SELECT stage, COUNT(*) AS runs
FROM campaign_runs
WHERE campaign_id = ?
GROUP BY stage
ORDER BY stage;

-- 2. 看 4x64 锚点在不同 lr/batch/resolution 下的 run_metrics
SELECT r.id, r.learning_rate, r.batch_size,
       MAX(CASE WHEN rm.metric_name = 'val_acc_macro' THEN rm.metric_value END) AS val_acc_macro,
       MAX(CASE WHEN rm.metric_name = 'val_acc_min_head' THEN rm.metric_value END) AS val_acc_min_head,
       MAX(CASE WHEN rm.metric_name = 'wall_time_s' THEN rm.metric_value END) AS wall_time_s,
       MAX(CASE WHEN rm.metric_name = 'inference_latency_ms' THEN rm.metric_value END) AS inference_latency_ms
FROM runs r
JOIN campaign_runs cr ON cr.run_id = r.id
JOIN run_metrics rm ON rm.run_id = r.id
WHERE cr.campaign_id = ?
  AND r.num_res_blocks = 4
  AND r.num_filters = 64
GROUP BY r.id, r.learning_rate, r.batch_size;

-- 3. 最近 frontier snapshot
SELECT id, created_at, total_runs, dominated_count, knee_run_id
FROM frontier_snapshots
WHERE campaign_id = ?
ORDER BY created_at DESC
LIMIT 5;

-- 4. 最近 surrogate / recommendation 批次
SELECT id, created_at, candidate_count
FROM surrogate_snapshots
WHERE campaign_id = ?
ORDER BY created_at DESC
LIMIT 5;
```

---

## 10. v25 的具体执行顺序

下面给出一条建议执行序列。

### 10.1 Step A — 4x64 锚点 sweep

```bash
uv run python framework/index.py sweep \
  --train-script domains/floorplan_checker/train.py \
  --campaign v25_floorplan_real_001 \
  --run-id v25-floorplan-real-001 \
  --search-space domains/floorplan_checker/manifest/search_space.json \
  --objective-profile domains/floorplan_checker/manifest/objective_profile.json \
  --stage-policy domains/floorplan_checker/manifest/stage_policy.json \
  --stage A \
  --axis num_res_blocks=4 \
  --axis num_filters=64 \
  --axis image_resolution=160,224,256 \
  --axis batch_size=16,32 \
  --axis learning_rate=5e-5,1e-4,5e-4 \
  --axis loss_weight_bed=1.0 \
  --axis loss_weight_bath=1.0 \
  --axis loss_weight_park=1.0 \
  --seeds 42
```

### 10.2 Step B — 对 4x64 工作带做 B 阶段复核

1. 从 A 阶段中选出 4–6 个点
2. 用 2 个 seed 重跑
3. 排除 unstable mirage

### 10.3 Step C — 结构阶梯 sweep

用 Phase 1 确定的最佳工作带固定：

1. `learning_rate`
2. `batch_size`
3. `image_resolution`

然后只扫结构轴：

1. `2x32`
2. `4x32`
3. `4x64`
4. `6x64`
5. `4x128`
6. `6x128`

### 10.4 Step D — 分析 front 与推荐下一步

```bash
uv run python framework/index.py analyze \
  --db output/floorplan_checker/v25-floorplan-real-001/tracker.db \
  --pareto \
  --campaign v25_floorplan_real_001 \
  --metric-source run_metrics \
  --objective-profile domains/floorplan_checker/manifest/objective_profile.json \
  --plot

uv run python framework/index.py analyze \
  --db output/floorplan_checker/v25-floorplan-real-001/tracker.db \
  --recommend-next v25_floorplan_real_001 \
  --format json
```

### 10.5 Step E — 继续执行 recommendation 批次

只接受：

1. 真正补 gap 的 `new_point`
2. front 上缺 seed 的 `seed_recheck`
3. 已经接近 knee 的 `continue_branch`

---

## 11. 如何制作“meaningful”的 Pareto 图片

### 11.1 先承认当前框架的限制

当前 `analyze --pareto --plot` 的内置图只会：

1. 取 **第一个 maximize 轴** 作为 y
2. 取 **第一个 minimize 轴** 作为 x
3. 输出：
   - `overview`
   - `front_only`
   - `knee_zoom`
   - `front.json/csv/md`

这已经够用，但还不够“meaningful”。

所以 v25 不应该只做一张图，而应该做**图集**。

### 11.2 v25 最少要产出的 Pareto 图集

**图 A — 主决策图**

1. x 轴：`wall_time_s`
2. y 轴：`val_acc_macro`
3. 用途：看“总体质量 vs 训练成本”

**图 B — 平衡性图**

1. x 轴：`wall_time_s`
2. y 轴：`val_acc_min_head`
3. 用途：看“最弱 head 是否值得这个成本”

**图 C — 部署图**

1. x 轴：`inference_latency_ms`
2. y 轴：`val_acc_macro`
3. 用途：看部署时延代价

**图 D — 显存图**

1. x 轴：`peak_memory_mb`
2. y 轴：`val_acc_macro`
3. 用途：看容量扩张是否只是吃显存

### 11.3 图的生成方式

v25 推荐两层产物：

1. **框架内置图**
   - 使用 `analyze --pareto --plot`
   - 产出 `overview/front_only/knee_zoom`

2. **图集层**
   - 对同一个 campaign 重复输出不同轴顺序的 Pareto 结果
   - 或从 `front.json/csv` 进一步做专题图

### 11.4 什么样的 Pareto 图才算 meaningful

meaningful 的标准不是“图上点很多”，而是：

1. dominated 区域能一眼看出
2. front 线不是一团点云
3. knee 点能被明确标出来
4. 注释能看出架构身份，例如 `4x64 @ 224`
5. 人能回答：
   - 再多花 20% 时间，值得吗？
   - 再多花 1GB 显存，换到了什么？
   - 是哪个 head 在主导这次 trade-off？

### 11.5 v25 图表纪律

每次关键 frontier 更新，至少保留：

1. `overview.png`
2. `front_only.png`
3. `knee_zoom.png`
4. `front.csv`
5. `front.md`
6. findings 中的人类解释

---

## 12. v25 的数据解读方法

### 12.1 五层阅读顺序

所有结果都按下面顺序阅读：

1. **约束层**
   - 是否 benchmark-clean
   - 是否 NaN / OOM
2. **平衡层**
   - `val_acc_min_head`
   - `val_head_acc_gap`
3. **总体层**
   - `val_acc_macro`
4. **成本层**
   - `wall_time_s`
   - `inference_latency_ms`
   - `peak_memory_mb`
5. **seed 稳定层**
   - 同 candidate 在不同 seed 下是否稳

### 12.2 不同类型点的解释模板

**cheap baseline**

1. 成本低
2. 质量不一定最强
3. 但作为系统基线有价值

**balanced knee**

1. `val_acc_macro` 不差
2. `val_acc_min_head` 不差
3. 成本上升尚未失控
4. 这是最优先继续投资的点

**expensive specialist**

1. 某个头明显更强
2. 成本显著更高
3. 是否值得要看这个头在业务中的真实重要性

**unstable mirage**

1. 单次很好
2. 多 seed 不稳
3. 不可直接当结论

**dominated overbuild**

1. 更大
2. 更慢
3. 但没有带来等价质量提升

### 12.3 v25 最重要的解释句式

v25 findings 不应只写“最好点是 X”，而应写成：

1. `4x64` 在 `224 / batch=16 / lr=1e-4` 附近形成了第一代工作带
2. `6x128` 虽然在某头上更强，但整体是 expensive specialist，不是 knee
3. 当前 knee 位于 `4x64` 邻域，说明第一次真实探索没有要求我们立刻上更大模型
4. 当前 front 主要由 resolution trade-off 还是 capacity trade-off 形成

---

## 13. v25 的成功标准

如果 v25 执行得好，最后至少应得到以下 8 件东西：

1. 一个真正的 `floorplan_checker` 真实研究 campaign
2. 一组围绕 `4x64` 的工作带结论
3. 一条 capacity frontier
4. 一组多头平衡 vs 成本的 Pareto 图
5. 一个被明确标注的 knee point
6. 至少一轮 recommendation 真正驱动的下一步实验
7. 数据库中完整的 frontier / surrogate / recommendation 留痕
8. 一份人类可读的 findings，能解释为什么这个 knee 值得继续

---

## 14. v25 的风险与防错

### 14.1 最大风险

1. 继续把单头高分误判成整体最优
2. 继续把单 seed 幻觉误判成 frontier
3. 让 built-in Pareto 图只输出一张“好看但没法决策”的图
4. 过早进入 branch，而不是先把 frontier 看清楚

### 14.2 防错规则

1. 没有至少 2 个 seed 的候选，不允许宣称“稳定 front 点”
2. `val_head_acc_gap` 很大时，不允许只看 macro 分
3. 没有 `frontier_snapshot` 的 front，不算正式 front
4. 没有 `surrogate_snapshot` 的 recommendation，不算正式 recommendation

---

## 15. 总结

v25 的本质，不是再跑一轮 floorplan，而是：

> **第一次让 autoresearch 在真实世界数据上，像它在 Gomoku 中那样，系统性地寻找“哪里值得继续投算力”。**

这次我们不追求“立刻得到最强模型”，而追求：

1. 一个可信 benchmark
2. 一个可解释的 front
3. 一个真正有意义的 knee
4. 一套可以持续迭代下去的研究流程

如果 v25 执行成功，那么 `floorplan_checker` 将不再只是一个“已接入的新 domain”，而会成为 autoresearch 第一批真正完成 **现实任务 Pareto 探索闭环** 的研究域。
