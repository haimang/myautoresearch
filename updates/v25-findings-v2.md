# v25 Findings v2 — 目标校正后的第一批验证结果

> 2026-04-26  
> 前置：`updates/v25-study-v2.md`

---

## 1. 本轮执行的目的

v2 的目标不是继续沿用旧的“单头准确率 vs wall time”语义，而是验证：

> **当我们把目标改成 “min-head / head-gap / macro / cost” 之后，结构与分辨率的比较是否会出现新的真实信号。**

因此，本轮执行的是 `v25-study-v2.md` 中约定的 **Batch-1: Anchor Confirmation**。

---

## 2. 本轮实际执行配置

固定：

1. `batch_size = 16`
2. `learning_rate = 1e-4`
3. `loss_weight_bed = 1.0`
4. `loss_weight_bath = 1.0`
5. `loss_weight_park = 1.0`
6. `seed = 42`

变化轴：

1. `num_res_blocks ∈ {4, 6}`
2. `num_filters = 64`
3. `image_resolution ∈ {160, 224}`

共 4 个点：

1. `4x64 / 160 / 16 / 1e-4`
2. `6x64 / 160 / 16 / 1e-4`
3. `4x64 / 224 / 16 / 1e-4`
4. `6x64 / 224 / 16 / 1e-4`

与旧实验不同，本轮显式固定了采样预算：

1. `time_budget = 300`
2. `max_train_samples = 8192`
3. `max_eval_samples = 2048`
4. `max_test_samples = 2048`

这意味着本轮结果已经不再受：

1. `time_budget < 600` 自动把 `eval/test` 压到 `batch*10`
2. 过低评估采样导致的伪饱和

的直接影响。

---

## 3. 本轮 objective 与图表语义

本轮使用的新目标定义为：

1. maximize:
   - `val_acc_min_head`
   - `val_acc_macro`
   - `val_acc_bedroom`
   - `val_acc_bathroom`
   - `val_acc_parking`
2. minimize:
   - `val_head_acc_gap`
   - `wall_time_s`
   - `peak_memory_mb`
   - `inference_latency_ms`

对应配置：

1. `domains/floorplan_checker/manifest/objective_profile_v2.json`

因此，本轮 Pareto 主图已经被纠正为：

1. `x = val_head_acc_gap`
2. `y = val_acc_min_head`

而不再是旧语义的单头主图。

---

## 4. 执行结果总览

结果：

1. `4 / 4` 成功
2. 无 NaN
3. 所有 dataset contract 指标均为 0
4. 自动 Pareto：
   - `Feasible = 4`
   - `Front = 1`
   - `Dominated = 3`
   - `Knee = 4x64 / 160 / 16 / 1e-4`

产物位置：

1. `output/floorplan_checker/v25-floorplan-v2-001/pareto/v25_v2_balance_overview.png`
2. `output/floorplan_checker/v25-floorplan-v2-001/pareto/v25_v2_balance_overview_front.csv`
3. `output/floorplan_checker/v25-floorplan-v2-001/pareto/v25_v2_balance_plane.png`
4. `output/floorplan_checker/v25-floorplan-v2-001/pareto/v25_v2_cost_plane.png`
5. `output/floorplan_checker/v25-floorplan-v2-001/v25_v2_focus_001_summary.json`

---

## 5. 4 个点的完整结果

| 配置 | val min-head | val gap | val macro | test min-head | wall_s | peak mem MB | latency_ms | deployment-feasible |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `4x64 / 160 / 16 / 1e-4` | 0.5786 | 0.2905 | 0.7723 | 0.4546 | 166.86 | 8914.97 | 3.80 | Y |
| `6x64 / 160 / 16 / 1e-4` | 0.5786 | 0.2905 | 0.7723 | 0.4546 | 243.93 | 10647.04 | 5.93 | Y |
| `4x64 / 224 / 16 / 1e-4` | 0.5786 | 0.2905 | 0.7723 | 0.4546 | 333.36 | 17024.71 | 7.87 | N |
| `6x64 / 224 / 16 / 1e-4` | 0.5786 | 0.2905 | 0.7723 | 0.4546 | 349.62 | 20236.45 | 11.75 | N |

更关键的是，4 个点的 per-head 验证准确率完全一致：

1. `val_acc_bedroom = 0.8691`
2. `val_acc_bathroom = 0.5786`
3. `val_acc_parking = 0.8691`

也就是说：

> **本轮 4 个点在 “最弱头、头间差距、整体平均” 这三个最关键的 v2 指标上，完全没有拉开差异。**

---

## 6. 本轮最重要的发现

### 6.1 目标校正是对的，但它暴露出更强的问题

v2 的目标校正是有效的：

1. 主图已经从“单头 vs 时间”切换成了“min-head vs head-gap”
2. 这让我们真正看到了 balanced knee 想看的东西

但 Batch-1 也同时暴露出一个更强的事实：

> **问题已经不再是“图画错了”，而是这 4 个点在质量空间里根本没有分开。**

这说明：

1. 前两轮的“没有 curve”并不只是因为旧图语义不对
2. 在更高评估采样下，结构 / 分辨率差异仍然没有体现在 argmax 级别的质量指标上

### 6.2 当前结构比较已经被“同一预测区间”锁死

4 个点同时出现：

1. 相同的 `val_acc_min_head`
2. 相同的 `val_head_acc_gap`
3. 相同的 `val_acc_macro`
4. 相同的 `test_acc_min_head`

这非常强烈地暗示：

> **当前这些配置在 Batch-1 预算下，仍然停留在同一个预测区间里。**

更直白地说：

1. 换更大模型没有改变最终 argmax 行为
2. 换更高分辨率也没有改变最终 argmax 行为
3. 变化只体现在：
   - loss
   - time
   - memory
   - latency

而没有体现在我们真正关心的 balanced quality 上。

### 6.3 当前 weakest head 不是 capacity 问题，而是 bathroom 问题

这批结果里最显眼的结构不是模型大小，而是头部失衡：

1. bedroom = `0.8691`
2. bathroom = `0.5786`
3. parking = `0.8691`

因此：

1. `val_acc_min_head` 完全由 bathroom 头决定
2. `val_head_acc_gap = 0.2905` 也是 bathroom 被拉开的结果
3. 无论 `4x64 -> 6x64`
4. 还是 `160 -> 224`
5. 都没有改变这一点

这意味着：

> **当前最主要的研究瓶颈不是容量不足，而是 bathroom 头长期掉队。**

### 6.4 这轮直接判掉了两条旧方向

基于这批结果，可以直接判掉两条优先级过高的旧方向：

1. **继续优先加大结构**
   - `6x64` 在 balanced quality 上没有任何收益
   - 但 wall / memory / latency 都显著上升

2. **继续优先抬高分辨率**
   - `224` 没有带来任何 min-head / gap / macro 改善
   - 但直接把成本抬到更高区间

因此在 v2 视角下：

> **`6x64` 和 `224` 现在都更像纯成本放大器，而不是新的 balanced knee 候选。**

---

## 7. 为什么 Pareto 只剩一个 front 点

在 v2 profile 下，Pareto 仍然只剩一个 front 点，原因现在已经很清楚：

1. 所有点在：
   - `val_acc_min_head`
   - `val_head_acc_gap`
   - `val_acc_macro`
   上完全一样
2. 一旦质量完全一样，Pareto 排序自然退化成成本排序
3. 此时最便宜的点会支配其余所有点

所以这次的单点 front 不是分析失败，而是：

> **在“质量完全相同”的前提下，`4x64 / 160 / 16 / 1e-4` 作为最低成本点自然支配全场。**

这也是为什么自动 knee 会稳定落在它身上。

---

## 8. v2 Batch-1 对实验路线的真正修正

原本 v2 设计里，结构对照只是第一步。  
现在 Batch-1 已经足够说明：**继续优先比较结构/分辨率不会给我们带来新的研究信息。**

### 8.1 deployment 线

deployment 线现在更加稳定：

1. `4x64 / 160 / 16 / 1e-4`
   - 仍然是最强 deployment anchor
2. `6x64 / 160 / 16 / 1e-4`
   - 虽然也在 12GB 内
   - 但质量完全不变，成本更高
   - 因此没有继续投入的价值

### 8.2 research 线

research 线也被重新校正：

1. 下一个最值得探索的方向不再是更大结构
2. 也不再是更高分辨率
3. 而应该是：
   - **loss balance**
   - 特别是 **bathroom-first** 的权重补偿

也就是说，下一阶段应该优先测试：

1. 固定 `4x64 / 160 / 16 / 1e-4`
2. 只扫：
   - `loss_weight_bath`
3. 并观察：
   - `val_acc_min_head`
   - `val_head_acc_gap`
   - `val_acc_macro`

这会比再加大模型更符合 v2 的真正目标。

---

## 9. 建议的下一批参数空间

基于 Batch-1，建议下一批直接切到 **bathroom-first balance sweep**：

固定：

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `image_resolution = 160`
4. `batch_size = 16`
5. `learning_rate = 1e-4`

探索轴：

1. `loss_weight_bath ∈ {1.0, 1.5, 2.0, 3.0}`
2. 可选：
   - `loss_weight_bed ∈ {1.0, 0.75}`
   - `loss_weight_park ∈ {1.0, 0.75}`

最小可执行版本：

1. `(bed=1.0, bath=1.0, park=1.0)`
2. `(bed=1.0, bath=1.5, park=1.0)`
3. `(bed=1.0, bath=2.0, park=1.0)`
4. `(bed=1.0, bath=3.0, park=1.0)`

如果这条线仍然无法改善 `min_head / gap`，再考虑：

1. 更长 budget
2. 多 seed
3. class-balance / label-distribution 方向的更深分析

---

## 10. 本轮结论

v25 v2 Batch-1 的核心结论可以浓缩成一句话：

> **在目标被校正到 “min-head / head-gap / macro / cost” 之后，第一批 4 点实验表明：当前最主要的问题不是容量和分辨率，而是 bathroom 头持续掉队；结构与分辨率的增加没有带来任何 balanced quality 改善，只是单纯抬高了成本。**

因此，v25 的下一步不应继续优先探索：

1. 更大模型
2. 更高分辨率

而应优先探索：

1. `4x64 / 160 / 16 / 1e-4` 下的 **bathroom-first loss balance**
2. 在这个方向上继续寻找真正的 balanced knee
