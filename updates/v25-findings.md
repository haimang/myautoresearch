# v25 Findings — Phase 1 `4x64` 锚点探索结果与分析

> 2026-04-26  
> 前置：`updates/v25-study.md`、`updates/v24.1-findings.md`  
> 说明：本文件记录 v25 第一轮真实世界 autoresearch 探索中，**Phase 1 — `4x64` 锚点工作带搜索** 的实际执行、实验记录、结果分析与下一步决策。

---

## 1. Phase 1 的具体执行计划

本轮严格按 `v25-study.md` 的 Phase 1 思路执行，但把它收敛成一个真正可跑的短预算 campaign：

### 1.1 固定项

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `loss_weight_bed = 1.0`
4. `loss_weight_bath = 1.0`
5. `loss_weight_park = 1.0`

### 1.2 探索轴

1. `image_resolution ∈ {160, 224, 256}`
2. `batch_size ∈ {16, 32}`
3. `learning_rate ∈ {5e-5, 1e-4, 5e-4}`

总点数：

```text
3 × 2 × 3 = 18 个点
```

### 1.3 预算策略

这次是 **快速过一遍**，因此全部走 Stage A：

1. `stage = A`
2. `time_budget = 60s`（来自当前 stage policy）
3. `seed_count = 1`

用户要求每个点不超过 5 分钟；本轮所有点都显著低于这个上限。

### 1.4 本轮真实执行命令

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

---

## 2. 过程监控与执行情况

### 2.1 过程是否出现需要中断修复的问题

本轮运行过程中重点监控了：

1. manifest / dataset contract 相关失败
2. NaN loss
3. OOM / 显存崩溃
4. workspace / DB / campaign ledger 异常
5. batch-size / resolution 拉高后是否出现明显 runtime 错误

结论：

> **本轮 18 个点全部顺利完成，没有出现需要中断修复再重跑的 runtime 问题。**

也就是说：

1. 数据链路稳定
2. 训练脚本稳定
3. DB 账本稳定
4. 这次 Phase 1 的主要“问题”不是运行崩溃，而是**研究层面的硬约束收缩**——也就是显存预算把可行工作带切得很窄

### 2.2 总体执行结果

1. `18 / 18` 成功
2. 总耗时：`391s`
3. 单点耗时范围：`8s ~ 43s`

单点都没有接近 5 分钟限制。

---

## 3. 数据库账本结果

本轮 experiment / campaign 账本如下：

| 项目 | 数值 |
|---|---:|
| `experiment_run_id` | `v25-floorplan-real-001` |
| `campaign` | `v25_floorplan_real_001` |
| `campaign_runs` | 18 |
| `stage_A_runs` | 18 |
| `frontier_snapshots` | 5 |
| `surrogate_snapshots` | 1 |
| `recommendation_batches` | 1 |

Campaign summary 结果：

1. `runs = 18`
2. `protocol drift = none`
3. `status = completed: 18`

Stage summary 结果：

1. `Stage A: runs = 18`
2. `candidates = 18`
3. `promotions = 0`

这说明：

> **v25 的第一轮真实探索，已经不只是“本地脚本跑了一遍”，而是完整进入了 autoresearch 的 campaign ledger。**

---

## 4. Phase 1 的核心结果

### 4.1 重要修正：`12GB` 不是机器上限，而是本轮 objective profile 里的**实验硬约束**

这里先纠正我上一轮分析里最容易误导的地方。

我之前把：

```text
peak_memory_mb <= 12000
```

说成了像是“机器只有 12GB 可用显存”，这是不准确的。

真实情况是：

1. 当前机器在 run 账本里记录的是：`memory_gb = 128`
2. 本轮 Phase 1 的真实峰值内存已经跑到：
   - `max_peak_memory_mb = 29185`
   - 约 `29.2 GB`
3. 也就是说，**机器完全不止 12GB 可用**
4. 当前 `12GB` 只是我在 `domains/floorplan_checker/manifest/objective_profile.json` 里写死的一个**实验预算红线**

因此，后面所有“只剩 3 个 feasible 点”的结论，都必须理解为：

> **是在“12GB 部署/预算视角”下成立，不是在“这台 128GB 机器的物理上限”下成立。**

### 4.2 第一结论：在当前 `12GB` 预算视角下，工作带确实被切得很窄

当前 objective profile 的硬约束之一是：

```text
peak_memory_mb <= 12000
```

而这 18 个点里，真正满足全部 hard constraints 的只有 **3 个**：

1. `160 / batch16 / lr=5e-5`
2. `160 / batch16 / lr=1e-4`
3. `160 / batch16 / lr=5e-4`

也就是说：

> **对当前 `4x64` 结构而言，Phase 1 的可行工作带几乎被压缩成了 `image_resolution=160 + batch_size=16`。**

### 4.3 第二结论：`224 / 256` 不是“跑不动”，而是在当前 `12GB` policy 下被判为 infeasible

修正后的说法应当是：

1. `224`、`256` **都能正常跑完**
2. `batch32` 的高内存点也都**能正常跑完**
3. 它们不是 runtime 崩溃点
4. 它们是**被当前 objective profile 的硬约束排除**

换句话说：

> **这些点不是硬件不支持，而是被当前研究配置定义为“不接受”。**

### 4.4 第三结论：224/256 在“研究视角”下值得保留，但在“12GB 部署预算视角”下应被判出局

虽然 `224`、`256` 上很多点的 `val_acc_macro = 1.0`，但它们在当前 benchmark 约束下并不算真正可行，因为：

1. `224 / batch16` 的峰值显存已经约 `12.56 GB`
2. `256 / batch16` 的峰值显存约 `14.59 GB`
3. `224 / batch32` 约 `22.35 GB`
4. `256 / batch32` 约 `29.19 GB`

所以 Phase 1 给出的不是“224/256 更强”的结论，而是：

> **在当前 `4x64` 容量和 12GB 预算下，更高分辨率首先体现为资源越界，而不是可接受的 frontier 改善。**

但如果换成“机器探索视角”而不是“12GB 部署视角”，结论就应该更温和：

1. `224 / 16` 与 `256 / 16` 应该继续保留在后续探索池里
2. 因为它们确实提供了不同的 cost / memory 区间
3. 只是它们不应和“12GB 预算 front”混在同一张图里解释

### 4.5 第四结论：batch 32 在当前结构下更像高成本探索分支，而不是当前主战区

即使在 `160` 分辨率下：

1. `batch32` 已经把显存推到 `12.83 GB`
2. 已经越过 hard constraint
3. 并且 wall time 也明显高于 `batch16`

因此：

> **当前阶段没必要继续把 `batch32` 当 Phase 2 主战轴。**

但这里也要注意修正语义：

1. 这不是说 `batch32` 完全没价值
2. 而是说在当前第一次真实探索里，它更适合作为“高成本探针”
3. 不是最应该继续加 seed 的主线

### 4.6 第五结论：在 `12GB` 预算视角下，`lr=1e-4` 是当前 knee

可行的 3 个点对比如下：

| 配置 | `val_acc_macro` | `val_acc_min_head` | `wall_time_s` | `latency_ms` | 结论 |
|---|---:|---:|---:|---:|---|
| `160 / 16 / 5e-5` | 0.9375 | 0.8125 | 8.02 | 3.74 | 轻微欠收敛，bedroom 头没跟上 |
| `160 / 16 / 1e-4` | 1.0000 | 1.0000 | 7.89 | 3.72 | **当前最优 knee** |
| `160 / 16 / 5e-4` | 1.0000 | 1.0000 | 7.94 | 3.81 | 与 1e-4 基本同质，但略慢、略高延迟 |

因此本轮真实结论很清楚：

> **`160 / 16 / 1e-4` 是当前 `4x64` 锚点下最干净的第一代 knee。**

但这个 knee 必须带限定语：

> **它是“12GB 部署预算视角”的 knee，不是“整台机器全局探索视角”的唯一 knee。**

---

## 5. 分组汇总：resolution × batch 的结构性信号

按 `resolution × batch` 聚合后，结果如下：

| 分组 | avg macro | avg min-head | avg wall_s | avg latency_ms | peak mem MB | 解释 |
|---|---:|---:|---:|---:|---:|---|
| `160 × 16` | 0.9792 | 0.9375 | 7.95 | 3.76 | 6619 | **唯一稳定可行工作带** |
| `160 × 32` | 0.8889 | 0.6667 | 15.54 | 4.02 | 12826 | 显存越界，且有一个明显坏点 |
| `224 × 16` | 1.0000 | 1.0000 | 15.87 | 8.01 | 12564 | 质量高，但已越过显存红线 |
| `224 × 32` | 1.0000 | 1.0000 | 28.89 | 7.44 | 22349 | 成本高且严重越界 |
| `256 × 16` | 0.9993 | 0.9979 | 19.46 | 10.09 | 14593 | 质量高，但越界更明显 |
| `256 × 32` | 1.0000 | 1.0000 | 41.86 | 10.95 | 29185 | 明显超预算，不具 Phase 1 继续价值 |

这个表说明：

1. 当前 Phase 1 不是“越大越好”
2. 而是**不同 budget 视角会切出不同的 front**
3. `12GB` 视角下，front 只剩 `160 × 16`
4. 更高机器预算视角下，`224 × 16`、`256 × 16` 依然值得保留
5. 当前真正的问题不是“没有好点”，而是**Stage A 对质量区分度不够**

---

## 6. 18 个点的完整实验记录

| res | batch | lr | bed | bath | park | macro | min_head | wall_s | latency_ms | peak_mem_mb | feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 160 | 16 | 5e-05 | 0.8125 | 1.0000 | 1.0000 | 0.9375 | 0.8125 | 8.02 | 3.74 | 6619 | Y |
| 160 | 16 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7.89 | 3.72 | 6619 | Y |
| 160 | 16 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7.94 | 3.81 | 6619 | Y |
| 160 | 32 | 5e-05 | 0.0000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 15.39 | 3.93 | 12826 | N |
| 160 | 32 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.14 | 4.04 | 12826 | N |
| 160 | 32 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.09 | 4.10 | 12826 | N |
| 224 | 16 | 5e-05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.43 | 8.55 | 12564 | N |
| 224 | 16 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.93 | 7.74 | 12564 | N |
| 224 | 16 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.25 | 7.73 | 12564 | N |
| 224 | 32 | 5e-05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 29.37 | 7.47 | 22349 | N |
| 224 | 32 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28.69 | 7.35 | 22349 | N |
| 224 | 32 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28.61 | 7.50 | 22349 | N |
| 256 | 16 | 5e-05 | 0.9938 | 1.0000 | 1.0000 | 0.9979 | 0.9938 | 19.15 | 9.90 | 14593 | N |
| 256 | 16 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19.51 | 10.14 | 14593 | N |
| 256 | 16 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19.72 | 10.24 | 14593 | N |
| 256 | 32 | 5e-05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40.43 | 10.66 | 29185 | N |
| 256 | 32 | 0.0001 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 42.16 | 10.94 | 29185 | N |
| 256 | 32 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 42.99 | 11.23 | 29185 | N |

---

## 7. Pareto 与图表产物

### 7.1 自动 Pareto 结果

自动 Pareto 分析结果：

1. `Feasible = 3`
2. `Infeasible = 15`
3. `Front = 1`
4. `Dominated = 2`
5. Knee：
   - `160 / batch16 / lr=1e-4`

这和人工分析一致，但它只代表：

> **在当前 `objective_profile.json` 定义的 12GB 硬约束下的 Pareto。**

### 7.1.1 为什么图片里“根本没有曲线”

你指出得对，`overview.png` 的视觉上确实不是一条 curve，而基本就是一个点。

我重新检查图片和 `overview_front.json` 之后，可以明确说：

1. 这不是绘图代码“画坏了”
2. 也不是图片生成失败
3. 而是因为当前 front 在严格 hard constraints 下只有 **1 个点**
4. 所以图上自然不会长出一条可解释的 frontier 曲线

更重要的是：

> **就算把 `12GB` 约束拿掉，这一轮 Stage A 也依然很难长出真正有信息量的曲线。**

原因是：

1. 18 个点里绝大多数都已经达到：
   - `val_acc_macro = 1.0`
   - `val_acc_min_head = 1.0`
2. 也就是说，质量轴几乎饱和
3. 一旦质量轴饱和，Pareto 排序就会自然退化成：
   - 谁更快
   - 谁更省
4. 在这种情况下，最快的那个点会直接支配其他点

所以“没有曲线”这件事的真正含义是：

> **Phase 1 的 Stage A 还不足以提供一个有分辨率的质量景观。**

### 7.1.2 这轮更像 legality / throughput sweep，不像真正的 frontier-confirm sweep

从这批结果看，Stage A 在 floorplan 上更像是在回答：

1. 哪些点合法
2. 哪些点会超出当前 budget
3. 哪些点 wall time / latency / peak memory 差别巨大

而不是在回答：

1. 哪些点在质量上形成细致 trade-off
2. 哪些点之间存在真正的 knee curve

所以当前 `overview.png` 其实已经诚实地暴露了一个研究事实：

> **Phase 1 目前只能拿来找 working band，不能拿来宣称“已经看到了漂亮的 front 曲线”。**

### 7.2 本轮生成的 meaningful 图集

产物目录：

```text
output/floorplan_checker/v25-floorplan-real-001/campaigns/v25_floorplan_real_001/pareto/
```

本轮额外生成了 3 套更有解释性的 Phase 1 图：

1. `macro_vs_wall.png`
   - 总体质量 vs 训练成本
2. `minhead_vs_wall.png`
   - 最弱头表现 vs 训练成本
3. `macro_vs_latency.png`
   - 总体质量 vs 推理延迟

重新分析后，本轮又补生成了 3 张 **raw scatter** 图，比 front-only 图更有解释力：

4. `phase1_raw_macro_vs_wall.png`
   - 显示全部 18 个点
   - 颜色映射 `peak_memory_mb`
   - 可直接看出不同 resolution/batch 的耗时簇
5. `phase1_raw_minhead_vs_wall.png`
   - 用来观察是否有“最弱头没跟上”的点
6. `phase1_raw_macro_vs_memory.png`
   - 直接把 `12GB` 红线画出来
   - 一眼能看出哪些点只是被 policy 卡掉，而不是机器跑不动

同时保留了每套图对应的：

1. `front.json`
2. `front.csv`
3. `front.md`
4. `front_only.png`
5. `knee_zoom.png`

### 7.3 对图的解释

重新看图之后，解释应该更新为：

1. `overview.png` / `macro_vs_wall.png` 的 front-only 视角确实不够好看，也不够有信息量
2. `phase1_raw_macro_vs_wall.png` 才真正揭示了这轮 18 个点的结构：
   - `160 × 16` 在最左侧
   - `224 × 16` 聚成第二簇
   - `256 × 16` 聚成第三簇
   - `224 × 32`、`256 × 32` 继续向右展开
3. `phase1_raw_macro_vs_memory.png` 则直接说明：
   - 12GB 红线只是当前 policy
   - 不是机器实际可用内存上限
4. 当前真正有意义的结论不是“front 是一条线”，而是：
   - **质量轴已经过早饱和**
   - **成本/内存轴已经开始明显分层**

这正是 autoresearch 应该做的事情。

---

## 8. Recommendation 输出与如何解读

本轮推荐系统输出了：

1. `seed_recheck`：
   - `160 / 16 / 1e-4`
   - `160 / 16 / 5e-4`
2. 一个额外的 `seed_recheck`：
   - `160 / 32 / 1e-4`
3. `new_point`：
   - 在 `160 / 16 / 1e-4` 周围尝试 `loss_weight_bath = 2.0`
   - 或 `loss_weight_bath = 0.5`

### 8.1 哪些推荐值得接受

当前最值得接受的是：

1. `160 / 16 / 1e-4` 的 `seed_recheck`
2. `160 / 16 / 5e-4` 的 `seed_recheck`

理由：

1. 它们在当前 hard constraints 下是合法点
2. 已经位于 front 邻域
3. 下一步最缺的是 seed 稳定性，而不是继续大面积瞎扫

### 8.2 一个值得记录的系统现象

推荐结果里还出现了：

1. `160 / 32 / 1e-4` 的 `seed_recheck`

但它在 Phase 1 中已经因为：

```text
peak_memory_mb = 12826 > 12000
```

而被判为 infeasible。

这说明当前 recommendation 层虽然能给出有价值方向，但：

> **对 hard constraint 的过滤还不够强，下一阶段不能无脑接受所有推荐。**

这不是本轮运行崩溃的问题，但它是一个需要记住的研究系统 caveat。

---

## 9. Phase 1 后的决策

根据这 18 个点的结果，下一步不应继续平均用力，而应做收缩决策。

### 9.1 先修正：下一步不应把 `12GB` 视角误当成“唯一研究视角”

重新分析后，下一步必须区分两种 front：

1. **deployment-budget front**
   - 继续保留 `peak_memory_mb <= 12000`
   - 适合回答“在严格预算下什么最值”

2. **exploration front**
   - 不把 `12GB` 当唯一红线
   - 可以把内存改成：
     - 更宽松 hard cap（例如 32GB / 48GB）
     - 或直接作为 minimize 轴参与前沿

如果不做这个区分，后面的 Pareto 图还会继续塌成一个点。

### 9.2 已经决定冻结的项（deployment-budget 视角）

Phase 2 开始前，建议先冻结：

1. `batch_size = 16`
2. `image_resolution = 160`

理由：

1. 这是当前唯一稳定处于预算内的工作带
2. 其余分辨率 / batch 组合在当前 `4x64` 下已经没有继续扫的研究意义

### 9.3 研究视角下，应该继续保留的点

如果后续目标是“在这台 128GB 机器上做真实 frontier 探索”，而不是只做 12GB 部署预算筛选，那么建议继续保留：

1. `224 / 16 / 1e-4`
2. `224 / 16 / 5e-4`
3. `256 / 16 / 1e-4`

原因：

1. 它们都正常运行
2. 都代表了更高质量/更高成本的真实簇
3. 只是当前 Stage A 的质量分辨率不够，暂时还看不出它们之间的细差

### 9.4 已经决定保留的 LR 候选

保留：

1. `1e-4`（主锚点）
2. `5e-4`（次锚点）

弱化：

1. `5e-5`

理由：

1. `5e-5` 在 `160 / 16` 下出现了 bedroom 头轻微欠收敛
2. `1e-4` 和 `5e-4` 已经到达完美平衡
3. `1e-4` 略快、延迟略低，因此更适合作为主锚点

### 9.5 对 Phase 2 的推荐动作

因此，Phase 2 不建议再继续用当前这套 `60s + 160 eval sample` 的轻量设置横向乱扫，而应转向：

1. **deployment-budget 线**
   - 先做 `160 / 16 / 1e-4` 与 `160 / 16 / 5e-4` 的 seed recheck

2. **exploration 线**
   - 保留 `224 / 16`、`256 / 16` 进入更长预算阶段
   - 让它们在更严格评估下真正拉开质量差距

3. **统一建议**
   - 下一阶段必须提高质量分辨率：
     - 更长 budget
     - 更大 eval sample
     - 多 seed
   - 否则图还会继续塌成一个点

换句话说：

> **Phase 1 的任务已经完成：它不是选出最终模型，而是先把“policy 预算 front”和“机器探索 front”分开，并证明 Stage A 只适合找工作带，不适合直接宣称真实曲线。**

---

## 10. 本轮经验总结

### 10.1 经验一：Phase 1 的价值在于快速砍空间，而不是追最好分

这次 18 点最重要的收获不是“1.0 macro 很多”，而是：

1. 224 和 256 在当前预算下没资格进入主战区
2. batch32 也基本退出主战区
3. 真正的探索面被快速收缩成 `160 / 16 / {1e-4, 5e-4}`

### 10.2 经验二：当前 `12GB` budget 比 accuracy 更先成为部署视角下的决策边界

在当前 `4x64` 下：

1. 质量不是主要瓶颈
2. 显存预算才是主要瓶颈

这说明下一阶段的结构探索，必须始终带着“12GB 是否过线”的意识进行。

但同样重要的是：

> **这不等于机器只能用 12GB。**

本轮真实数据已经证明：

1. run ledger 记录机器是 `128GB`
2. 这批实验最高已用到 `29.2GB`
3. 所以下一阶段完全可以在更高研究预算下继续做 front 探索

### 10.3 经验三：autoresearch 这次真正开始发挥作用了

这次已经不是人工直觉选点，而是：

1. campaign 统一记录
2. run_metrics 统一落账
3. Pareto front 自动生成
4. recommendation 自动给出下一步候选

虽然 recommendation 对 hard constraint 过滤还不够强，但闭环已经真实跑起来了。

---

## 11. 结论

Phase 1 的最终结论可以浓缩成一句话：

> **在当前 clean benchmark、当前 `4x64` 结构下，Phase 1 已经明确区分了两件事：一是 `12GB` policy 视角下的最优工作带为 `160 / 16 / (1e-4 ~ 5e-4)`，其中 `1e-4` 是当前部署预算 knee；二是这台 `128GB` 机器并不受 12GB 物理上限限制，当前图之所以没有长成 curve，根本原因是 Stage A 过早让质量轴饱和了。**

这意味着：

1. v25 的第一次真实世界 autoresearch 探索已经成功拿到基本实验数据
2. 当前不应再把 front-only 单图当成主要解释依据
3. 下一步应转向：
   - 双视角前沿（deployment vs exploration）
   - seed recheck
   - 更高分辨率 budget 下的结构阶梯比较
   - 更严格评估下的 frontier 精细化探索

---

## 12. Append — Phase 2 `6x64` 笛卡尔积快扫结果与新分析

> 2026-04-26  
> 对应计划已写回 `updates/v25-study.md` 的修订版 Phase 2。  
> 本轮不是继续扫多结构，而是先把容量锚点上移到 `6x64`，再用与 Phase 1 同构的平面快扫，看平面几何形状是否发生实质变化。

### 12.1 本轮实际执行计划

固定：

1. `num_res_blocks = 6`
2. `num_filters = 64`
3. `loss_weight_bed = 1.0`
4. `loss_weight_bath = 1.0`
5. `loss_weight_park = 1.0`

探索轴：

1. `image_resolution ∈ {160, 224, 256}`
2. `batch_size ∈ {16, 32}`
3. `learning_rate ∈ {5e-5, 1e-4, 5e-4}`

总点数：

```text
3 × 2 × 3 = 18 个点
```

执行命令：

```bash
uv run python framework/index.py sweep \
  --train-script domains/floorplan_checker/train.py \
  --campaign v25_floorplan_real_002 \
  --run-id v25-floorplan-real-002 \
  --search-space domains/floorplan_checker/manifest/search_space.json \
  --objective-profile domains/floorplan_checker/manifest/objective_profile.json \
  --stage-policy domains/floorplan_checker/manifest/stage_policy.json \
  --stage A \
  --axis num_res_blocks=6 \
  --axis num_filters=64 \
  --axis image_resolution=160,224,256 \
  --axis batch_size=16,32 \
  --axis learning_rate=5e-5,1e-4,5e-4 \
  --axis loss_weight_bed=1.0 \
  --axis loss_weight_bath=1.0 \
  --axis loss_weight_park=1.0 \
  --seeds 42
```

### 12.2 执行情况

结果：

1. `18 / 18` 成功
2. 总耗时：`547.48s`
3. 单点耗时：`10.86s ~ 58.55s`
4. 单点仍全部低于 5 分钟
5. 无 NaN、无 dataset contract 异常、无 runtime 崩溃

Campaign / Pareto 摘要：

1. `campaign = v25_floorplan_real_002`
2. `experiment_run_id = v25-floorplan-real-002`
3. `Feasible = 3`
4. `Infeasible = 15`
5. `Front = 2`
6. `Dominated = 1`
7. 当前自动 knee：
   - `160 / 16 / 5e-5`

这里要立刻强调：

> **这个 knee 只是当前 12GB policy 约束下、且质量轴几乎完全饱和时的自动选择结果；它不是“6x64 明显优于 4x64”的证据。**

### 12.3 先看图：为什么 `overview.png` 还是不构成真正的 curve

`overview.png` 这次依然只显示出极窄的 front，而且你也已经看到了：

1. front-only 视角依旧几乎没有曲线
2. 这是因为严格 feasible 区仍然只有 `160 / 16` 这一簇
3. 同时在 `6x64` 下，Stage A 更容易把质量轴直接压到：
   - `val_acc_macro ≈ 1.0`
   - `val_acc_min_head ≈ 1.0`

所以这次的正确解读不是“图坏了”，而是：

> **`6x64` 把 Phase 2 的 Stage A 进一步推向了“质量全饱和、只剩成本分层”的形态。**

本轮更有解释力的图是：

1. `phase2_raw_macro_vs_wall.png`
2. `phase2_raw_macro_vs_memory.png`
3. `phase1_vs_phase2_macro_vs_wall.png`

这些图共同说明：

1. `6x64` 平面没有被“拉开”
2. 而是整体向：
   - 更高 wall time
   - 更高 latency
   - 更高 peak memory
   方向平移

### 12.4 Phase 2 分组汇总：`6x64` 平面本身长什么样

| 分组 | avg macro | avg min-head | avg wall_s | avg latency_ms | peak mem MB | feasible |
|---|---:|---:|---:|---:|---:|---:|
| `160 × 16` | 1.0000 | 1.0000 | 10.88 | 5.46 | 8354 | 3 |
| `160 × 32` | 1.0000 | 1.0000 | 22.26 | 5.83 | 16061 | 0 |
| `224 × 16` | 0.9993 | 0.9979 | 21.18 | 10.88 | 15738 | 0 |
| `224 × 32` | 1.0000 | 1.0000 | 41.64 | 10.79 | 31071 | 0 |
| `256 × 16` | 1.0000 | 1.0000 | 28.17 | 14.74 | 20281 | 0 |
| `256 × 32` | 1.0000 | 1.0000 | 58.37 | 15.13 | 40578 | 0 |

这里最重要的事实有 4 个：

1. **12GB policy 下仍然只有 `160 × 16` 合法**
2. `6x64` 的最大峰值内存已经到 `40.58 GB`
3. `224 × 16` 并没有形成比 `160 × 16` 更强的质量簇
4. `256 × 16`、`224 × 32`、`256 × 32` 只是把成本继续向右上推

### 12.5 `4x64` vs `6x64`：容量上移到底换来了什么

参数量变化：

1. `4x64 = 300,430`
2. `6x64 = 449,166`
3. 增幅约 **49.5%**

按 `resolution × batch` 对照：

| 分组 | macro delta | min-head delta | wall delta_s | latency delta_ms | mem delta_mb | 解释 |
|---|---:|---:|---:|---:|---:|---|
| `160 × 16` | +0.0208 | +0.0625 | +2.93 | +1.70 | +1735 | 补齐了 Phase 1 的轻微欠收敛，但成本明显上升 |
| `160 × 32` | +0.1111 | +0.3333 | +6.72 | +1.81 | +3235 | 质量改善明显，但仍然不满足 12GB policy |
| `224 × 16` | -0.0007 | -0.0021 | +5.31 | +2.87 | +3173 | **略差于 4x64**，说明同样 60s 下更大模型反而更容易欠训练 |
| `224 × 32` | +0.0000 | +0.0000 | +12.75 | +3.34 | +8723 | 质量不变，只是更贵 |
| `256 × 16` | +0.0007 | +0.0021 | +8.71 | +4.65 | +5688 | 提升几乎不可感知，成本显著恶化 |
| `256 × 32` | +0.0000 | +0.0000 | +16.51 | +4.19 | +11393 | 纯成本放大 |

这张表给出的核心判断非常明确：

> **在当前 `60s + Stage A` 设定下，`6x64` 没有带来新的 frontier 结构，只是把成本面整体抬高。**

### 12.6 最关键的新判断

#### 12.6.1 `6x64` 没有改变 deployment-budget 结论

在 `peak_memory_mb <= 12000` 的当前 policy 下：

1. 合法区仍然只有 `160 / 16`
2. Phase 1 的部署视角结论没有被推翻
3. `4x64 / 160 / 16 / 1e-4` 依然是更可信的 deployment-budget 主锚点

也就是说：

> **把容量从 `4x64` 提到 `6x64`，并没有打开新的 12GB 内工作带。**

#### 12.6.2 `6x64` 也没有改善 exploration 视角下的“曲线问题”

如果站在机器探索视角看，问题也依然存在：

1. 大多数点还是很快到 `macro ≈ 1.0`
2. 这导致质量轴区分度接近消失
3. 于是图像只剩 wall / latency / memory 的平移差异

所以：

> **Phase 2 证明的不是“6x64 更强”，而是“继续用当前浅预算横扫，已经无法从质量轴获得新信息”。**

#### 12.6.3 `224 × 16` 的表现尤其重要

这组结果是本轮最值得记住的信号：

1. 在 `4x64` 下，`224 × 16` 平均是 `macro = 1.0 / min_head = 1.0`
2. 到了 `6x64`，`224 × 16` 反而变成：
   - `avg_macro = 0.9993`
   - `avg_min_head = 0.9979`
3. 同时 wall / latency / memory 都明显增加

这强烈暗示：

> **对于更大容量，`60s` 的 Stage A 已经开始出现“模型更大，但预算不够，所以看起来不升反降”的欠训练现象。**

这也是为什么后面不能继续靠 Stage A 横扫来判断容量优劣。

### 12.7 基于这轮新数据，对后续实验计划的建议

这次 `6x64` Phase 2 跑完后，下一步我不建议继续做第三轮大面积笛卡尔积快扫。  
更合理的是转向 **Phase 2.5 / Stage B-C 风格验证**。

#### 12.7.1 deployment-budget 线

继续保留：

1. `4x64 / 160 / 16 / 1e-4`
2. `4x64 / 160 / 16 / 5e-4`

原因：

1. 它们已经满足 12GB policy
2. `6x64` 没有给出更优的 12GB 内替代点

#### 12.7.2 exploration 线

建议做更严格、但更窄的容量对照，而不是再横扫：

1. `4x64 / 160 / 16 / 1e-4`
2. `6x64 / 160 / 16 / 1e-4`
3. `4x64 / 224 / 16 / 1e-4`
4. `6x64 / 224 / 16 / 1e-4`
5. 备选补点：
   - `4x64 / 160 / 16 / 5e-4`
   - `6x64 / 160 / 16 / 5e-4`

并且必须同时提升：

1. budget
2. eval sample
3. seed 数

否则：

> **我们只会继续得到“更大模型更贵，但图还是一条横线”的结果。**

### 12.8 Phase 2 append 结论

Phase 2 `6x64` 快扫可以浓缩成一句话：

> **`6x64` 没有把 frontier 拉开，只是把成本整体抬高了；在当前 `60s` Stage A 设定下，它既没有改变 12GB policy 下的工作带，也没有提供足够的质量分辨率来支持更大的容量比较。**

因此，v25 的下一步最合理方向已经进一步收敛为：

1. 不再继续大面积浅预算横扫
2. 进入更窄、更深的 `4x64 vs 6x64` 对照验证
3. 用更严格 budget / eval / seeds 去判断容量上移到底有没有真实价值
