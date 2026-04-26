# v25.4 Update — Floorplan-Checker Search Space 重划分 / Multi-Surface Truth Search

> 2026-04-26  
> 前置：`updates/v25-findings.md`、`updates/v25-findings-v2.md`、`updates/v25-findings-v3.md`、`updates/v25-findings-v4.md`  
> 定位：在 v25 到 v25-v4 已经确认“参数种类不够”之后，不再继续把 `floorplan_checker` 当成一个单一大平面来扫，而是把 search space 拆成多个 **独立假设面 / 独立 truth surface**，分别验证不同失败机理，再根据各 surface 的结果决定后续集成与放大路线。

---

## 1. v25.4 一句话目标

> **把当前混合、低信息量、容易把成本分层误当质量 frontier 的大 search space，重构成多套彼此独立的局部 search space：每一套只服务一个明确假设、产出一张明确 truth surface，并用这些局部真相面替代旧的“全局乱扫”。**

v25.4 不做：

1. 不再继续以 `num_res_blocks × num_filters × resolution × batch` 作为主战大平面。
2. 不再继续把 `loss_weight_bath` 当成值得扩充的主轴。
3. 不再继续用 bedroom / parking 的 easy-head 表现定义 knee。
4. 不再继续把 raw Pareto front 直接当作人类最终决策面。
5. 不在还没有出现真实 bathroom 质量移动前就重开大规模 Bayes。

v25.4 要做：

1. **冻结共同锚点 baseline**，让所有新 surface 都在同一控制组上比较。
2. **把 search space 拆成多个独立方案**，每个方案只验证一种失败假设。
3. **为每个方案单独定义 truth surface**，避免不同机理混在一张图里。
4. **只比较同一 surface 内的候选**，不把不同 surface 的 front 直接混合解释。
5. **等局部 surface 出现真实质量信号后，再做组合与结构放大。**

---

## 2. 为什么必须重划分 search space

v25 到 v25-v4 的实验已经非常清楚地说明，旧 search space 的主要问题不是“值还没扫够”，而是“不同类型的问题被混在一起了”。

### 2.1 旧 search space 把不同机理混在一个大平面里

旧平面主要包含：

1. `num_res_blocks`
2. `num_filters`
3. `image_resolution`
4. `batch_size`
5. `learning_rate`
6. `loss_weight_bed / bath / park`

但这些参数其实在控制不同层面的东西：

1. 全局容量
2. 训练吞吐 / 显存预算
3. 图像信息密度
4. 优化步幅
5. 多头 loss 总量配比

它们不是一个单一问题的不同数值，而是**不同问题的不同旋钮**。

### 2.2 旧 search space 没有对应到当前真正的主矛盾

v2 已证明：

1. `4x64 / 6x64`
2. `160 / 224`

在更严格评估下并没有改变：

1. `val_acc_bathroom`
2. `val_acc_min_head`
3. `val_head_acc_gap`

v3 又证明：

1. `loss_weight_bath = 1.0 -> 4.0`

也没有改变 bathroom 的 argmax 行为。

因此，继续在旧轴上扩数值，几乎只会继续得到：

> **成本平移、质量不动、Pareto 失真。**

### 2.3 我们真正需要的是多个“局部真相面”

从 v25-v4 的判断出发，当前最可能的失败机理至少有 4 类：

1. **loss geometry 错位**
   - count / ordinal 任务被当成无序分类做。
2. **class / hard-example 梯度分配不合理**
   - bathroom 内部的 minority / hard examples 梯度不够。
3. **sample exposure 不足**
   - 在固定 sample cap 下，bathroom 关键样本看到得不够。
4. **head 表达能力不足**
   - 共享 backbone + 对称 linear head 对 bathroom 不够有针对性。

这些机理必须分别探索，分别出图，分别判断。

---

## 3. v25.4 的总原则：固定公共锚点，拆分局部 surface

### 3.1 所有 surface 的共同锚点

除非特别说明，v25.4 所有方案都默认固定：

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `image_resolution = 160`
4. `batch_size = 16`
5. `learning_rate = 1e-4`
6. `loss_weight_bed = 1.0`
7. `loss_weight_bath = 1.0`
8. `loss_weight_park = 1.0`
9. `max_train_samples = 8192`
10. `max_eval_samples = 2048`
11. `max_test_samples = 2048`
12. `seed = 42`

这不是说它一定是最终最好模型，而是说：

> **它是目前最稳定、最低成本、能稳定暴露 bathroom 弱点的控制锚点。**

### 3.2 从全局主轴退役的参数

下列参数不再作为 v25.4 的主搜索轴：

1. `num_res_blocks`
2. `num_filters`
3. `image_resolution > 160`
4. `batch_size > 16`
5. `loss_weight_bath` 的密集数值扫描

它们只在后续“放大验证”阶段重新打开，不再作为第一层 truth search 的入口。

### 3.3 v25.4 的 search space 文件规划

建议把单一 search space 拆成多份 manifest：

1. `domains/floorplan_checker/manifest/search_space_v25_4_loss_geometry.json`
2. `domains/floorplan_checker/manifest/search_space_v25_4_gradient_balance.json`
3. `domains/floorplan_checker/manifest/search_space_v25_4_sampler_exposure.json`
4. `domains/floorplan_checker/manifest/search_space_v25_4_head_capacity.json`
5. `domains/floorplan_checker/manifest/search_space_v25_4_compound.json`
6. `domains/floorplan_checker/manifest/search_space_v25_4_amplification.json`

每个 search space 只应该包含：

1. 公共锚点固定轴
2. 当前方案真正要验证的 2~4 个局部轴

这样做的好处是：

1. 候选池更纯净
2. Bayes / recommendation 更聚焦
3. 产出的 front 与 truth surface 更可解释
4. 不同方案之间不会互相污染

---

## 4. 共同指标与共同纪律

### 4.1 v25.4 之前必须补齐的指标

在执行任何 v25.4 方案前，应先补齐并记录：

1. `val_acc_balanced_bathroom`
2. `test_acc_balanced_bathroom`
3. `val_recall_bathroom_1`
4. `val_recall_bathroom_2`
5. `val_recall_bathroom_3`
6. `val_recall_bathroom_4plus`
7. `test_recall_bathroom_1`
8. `test_recall_bathroom_2`
9. `test_recall_bathroom_3`
10. `test_recall_bathroom_4plus`
11. `val_mae_bathroom_count`
12. `test_mae_bathroom_count`
13. `val_confusion_bathroom`
14. `test_confusion_bathroom`

### 4.2 所有 surface 的共同硬约束

1. `nan_loss_count == 0`
2. `dataset_missing_file_count == 0`
3. `dataset_missing_field_count == 0`
4. `dataset_leakage_count == 0`
5. `dataset_summary_mismatch_count == 0`

### 4.3 v25.4 的共同判定基线

当前 baseline：

| 指标 | 基线 |
|---|---:|
| `val_acc_bathroom` | 0.5786 |
| `val_acc_min_head` | 0.5786 |
| `val_head_acc_gap` | 0.2905 |
| `test_acc_bathroom` | 0.4546 |
| `test_acc_min_head` | 0.4546 |
| `val_acc_macro` | 0.7723 |

v25.4 第一层成功定义：

1. `bathroom` 相关质量指标真实上升
2. `min_head` 真实上升
3. `head_gap` 真实下降
4. bedroom / parking 不被明显打坏

---

## 5. 方案 A — Loss Geometry Surface

### 5.1 要验证的假设

> **bathroom 弱头的主因不是样本量不够，而是当前 CE 把 count/ordinal 任务当成了无序分类，导致错误几何结构不对。**

### 5.2 独立 search space

固定公共锚点，只开放以下轴：

1. `loss_family_bath ∈ {cross_entropy, ordinal_ce, distance_weighted_ce, expected_count_loss}`
2. `ordinal_loss_weight_bath ∈ {0.25, 0.5, 1.0}`
3. `ordinal_loss_scope ∈ {bath_only, all_heads}`
4. `near_class_smoothing_bath ∈ {off, ordinal_neighbor}`
5. `label_smoothing_bath ∈ {0.0, 0.03, 0.05}`

### 5.3 这套 surface 的真相面

**主 truth surface：**

1. `x = test_mae_bathroom_count`
2. `y = test_acc_balanced_bathroom`
3. 颜色 = `test_head_acc_gap`

**辅助 truth surface：**

1. `x = test_acc_bathroom`
2. `y = test_acc_min_head`
3. 颜色 = `val_acc_macro - test_acc_macro`

这套 surface 回答的问题是：

1. bathroom 的错误是否更接近“相邻类错误”而非“完全错类”
2. ordinal-aware loss 是否能改善 count distance
3. bathroom balanced accuracy 是否会真正跟着动

### 5.4 这套 surface 的成功信号

1. `test_mae_bathroom_count` 下降
2. `test_acc_balanced_bathroom` 上升
3. `test_acc_min_head` 上升
4. `test_head_acc_gap` 下降

### 5.5 失败信号

如果：

1. count MAE 不动
2. balanced bathroom acc 不动
3. 只有 soft loss 在动

则说明：

> **loss geometry 不是当前第一主因。**

---

## 6. 方案 B — Gradient Balance Surface

### 6.1 要验证的假设

> **bathroom 弱头来自 bathroom 内部不同类别、不同难度样本的梯度分配不合理，而不是整体 head 权重不足。**

### 6.2 独立 search space

固定公共锚点，只开放：

1. `class_weighting_bath ∈ {none, inverse_freq, effective_num}`
2. `effective_num_beta_bath ∈ {0.99, 0.999, 0.9999}`
3. `focal_gamma_bath ∈ {0.0, 1.0, 2.0, 2.5}`
4. `focal_alpha_mode_bath ∈ {none, class_balanced}`

注意：

1. 这套面不混入 sampler。
2. 这套面不混入 head-specific 架构。
3. 它只测试“同样的数据、同样的 head、不同梯度分配”。

### 6.3 这套 surface 的真相面

**主 truth surface：**

1. `x = test_recall_bathroom_4plus`
2. `y = test_acc_balanced_bathroom`
3. 颜色 = `test_acc_min_head`

**辅助 truth surface：**

1. `x = test_recall_bathroom_3`
2. `y = test_recall_bathroom_4plus`
3. 颜色 = `test_head_acc_gap`

这套 surface 回答的问题是：

1. minority 类是否真的被救起来了
2. hard-example-focused 梯度是否能改变 bathroom 内部类别分布表现
3. 改善是否只是向某个单一类偏斜

### 6.4 成功信号

1. `test_recall_bathroom_3 / 4plus` 上升
2. `test_acc_balanced_bathroom` 上升
3. `test_acc_min_head` 上升
4. `test_head_acc_gap` 不恶化

### 6.5 失败信号

如果：

1. minority recall 不动
2. balanced bathroom acc 不动
3. 只出现大类倾斜

则说明：

> **单纯梯度重分配不是主解法。**

---

## 7. 方案 C — Sampler Exposure Surface

### 7.1 要验证的假设

> **bathroom 弱头的主要问题是固定 sample cap 下关键样本曝光不足，因此 sampler 比 loss 更接近根因。**

### 7.2 独立 search space

固定公共锚点，只开放：

1. `train_sampler ∈ {uniform, bathroom_balanced, bathroom_minority_boost}`
2. `bathroom_sampler_alpha ∈ {0.25, 0.5, 1.0}`
3. `sampler_scope ∈ {bath_only, all_heads}`

可选补轴，但只在 second pass 使用：

1. `max_train_samples ∈ {8192, 16384}`

第一轮不建议直接把 sample cap 混入主轴，避免同时验证两个问题。

### 7.3 这套 surface 的真相面

**主 truth surface：**

1. `x = test_recall_bathroom_4plus`
2. `y = test_acc_min_head`
3. 颜色 = `wall_time_s`

**辅助 truth surface：**

1. `x = test_acc_balanced_bathroom`
2. `y = test_head_acc_gap`
3. 颜色 = `peak_memory_mb`

这套 surface 回答的问题是：

1. 如果不改 loss，仅改样本暴露，bathroom 是否会动
2. minority recall 是否对最弱头改善有直接贡献
3. sampler 是否带来明显吞吐/稳定性代价

### 7.4 成功信号

1. `test_recall_bathroom_4plus` 上升
2. `test_acc_min_head` 上升
3. `test_acc_balanced_bathroom` 上升
4. wall time 没有不可接受地恶化

### 7.5 失败信号

如果 sampler 只能带来：

1. 训练更慢
2. 指标不动

则说明：

> **样本暴露并不是第一主因。**

---

## 8. 方案 D — Head Capacity Surface

### 8.1 要验证的假设

> **bathroom 弱头来自 head-specific 表达能力不足；共享 backbone + 对称 linear head 无法承载 bathroom 的额外语义复杂度。**

### 8.2 独立 search space

固定公共锚点，只开放：

1. `head_arch ∈ {linear, bath_mlp, mlp_shared}`
2. `bath_head_hidden ∈ {32, 64, 128}`
3. `bath_head_dropout ∈ {0.0, 0.1}`
4. `head_adapter ∈ {none, bath_residual_adapter}`

第一轮建议拆成两个局部子面分别跑：

1. `head_arch + bath_head_hidden + bath_head_dropout`
2. `head_adapter`

避免一开始把 MLP 与 adapter 混成一个过大的候选池。

### 8.3 这套 surface 的真相面

**主 truth surface：**

1. `x = num_params`
2. `y = test_acc_balanced_bathroom`
3. 颜色 = `test_acc_min_head`

**辅助 truth surface：**

1. `x = inference_latency_ms`
2. `y = test_head_acc_gap`
3. 颜色 = `test_acc_bathroom`

这套 surface 回答的问题是：

1. bathroom 是否需要更多 head-specific 非线性表达
2. 这种局部容量提升是否比扩大 backbone 更有效率
3. 改善是否值得其复杂度成本

### 8.4 成功信号

1. `test_acc_balanced_bathroom` 上升
2. `test_acc_min_head` 上升
3. `test_head_acc_gap` 下降
4. 成本增量明显小于 `4x64 -> 6x64`

### 8.5 失败信号

如果：

1. 参数量增加
2. latency 增加
3. bathroom 质量不动

则说明：

> **head expression 不是第一主因，至少不是最小改动解。**

---

## 9. 方案 E — Compound Surface

### 9.1 要验证的假设

> **真正的有效解不是单一参数种类，而是“一个有效的 loss family + 一个有效的 sampler / head trick”的组合。**

### 9.2 进入条件

只有在 A/B/C/D 至少出现一个“有效 family”后，才进入 E。

有效 family 的定义：

1. `test_acc_min_head` 明显上升
2. 或 `test_acc_balanced_bathroom` 明显上升
3. 或 `test_mae_bathroom_count` 明显下降

### 9.3 独立 search space

这套面不是全排列暴力组合，而是只组合 winner family：

1. `winner_loss_family`
2. `winner_gradient_family`
3. `winner_sampler_family`
4. `winner_head_family`

每次最多只组合两个 family，例如：

1. `ordinal_loss + bathroom_balanced_sampler`
2. `focal/effective_num + bathroom_balanced_sampler`
3. `ordinal_loss + bath_mlp`
4. `bathroom_balanced_sampler + bath_mlp`

### 9.4 这套 surface 的真相面

**主 truth surface：**

1. `x = test_head_acc_gap`
2. `y = test_acc_min_head`
3. 颜色 = `peak_memory_mb`

**辅助 truth surface：**

1. `x = complexity_penalty_rank`
2. `y = test_acc_balanced_bathroom`
3. 颜色 = `test_acc_macro`

这是第一张真正接近“候选决策面”的 truth surface。

### 9.5 成功信号

1. 组合方案进一步拉高 `test_acc_min_head`
2. 压低 `test_head_acc_gap`
3. 不显著打坏 bedroom / parking
4. 成本仍在可接受范围内

---

## 10. 方案 F — Amplification Surface

### 10.1 要验证的假设

> **如果某个新参数种类已经有效，那么重新打开结构/分辨率，可能会产生新的正交增益。**

### 10.2 进入条件

只有当 E 或单一 surface 中已经找到稳定有效方法后，才重新打开：

1. `num_res_blocks ∈ {4, 6}`
2. `image_resolution ∈ {160, 224}`
3. 必要时 `learning_rate ∈ {1e-4, 5e-4}`

### 10.3 独立 search space

固定 winner family，只开放：

1. `num_res_blocks`
2. `image_resolution`
3. 必要时 `learning_rate`

这时结构轴不再是“裸容量比较”，而是：

> **在已知有效方法上做 amplification test。**

### 10.4 这套 surface 的真相面

**主 truth surface：**

1. `x = peak_memory_mb`
2. `y = test_acc_min_head`
3. 颜色 = `test_acc_balanced_bathroom`

**辅助 truth surface：**

1. `x = wall_time_s`
2. `y = test_head_acc_gap`
3. 颜色 = `num_params`

### 10.5 这套 surface 的意义

它不再回答“结构本身是否有用”，而是回答：

> **有效方法出现后，更大结构 / 更高分辨率是否值得放大。**

---

## 11. 多 surface 的执行顺序与关键路径

### 11.1 Phase 0 — 指标补齐

必须先做：

1. bathroom balanced accuracy
2. bathroom per-class recall
3. bathroom confusion
4. bathroom count MAE

没有这些指标，v25.4 的 truth surface 不成立。

### 11.2 Phase 1 — 四个单一机理 surface 并行或半并行执行

优先顺序建议：

1. **A — Loss Geometry**
2. **B — Gradient Balance**
3. **C — Sampler Exposure**
4. **D — Head Capacity**

理由：

1. A/B/C 改动较轻，先判断 loss/梯度/样本暴露是否已经足够。
2. D 结构入侵更大，应该在 A/B/C 后执行。

### 11.3 Phase 2 — 只放大 winner family

如果 A/B/C/D 中出现成功 family：

1. 进入 E — Compound Surface
2. 只组合 winner，不重新暴力开全排列

### 11.4 Phase 3 — 结构放大验证

只有 E 或单 surface winner 已稳定，才进入 F：

1. `4x64 / 160`
2. `6x64 / 160`
3. `4x64 / 224`
4. `6x64 / 224`

---

## 12. Campaign / DB / 产物组织方式

### 12.1 每个 surface 单独 campaign

建议命名：

1. `v25_4_loss_geometry_001`
2. `v25_4_gradient_balance_001`
3. `v25_4_sampler_exposure_001`
4. `v25_4_head_capacity_001`
5. `v25_4_compound_001`
6. `v25_4_amplification_001`

### 12.2 每个 surface 单独 objective profile

建议配套：

1. `objective_profile_v25_4_loss_geometry.json`
2. `objective_profile_v25_4_gradient_balance.json`
3. `objective_profile_v25_4_sampler_exposure.json`
4. `objective_profile_v25_4_head_capacity.json`
5. `objective_profile_v25_4_compound.json`
6. `objective_profile_v25_4_amplification.json`

原因是：

1. 各 surface 关注的指标不同
2. 不应使用同一 objective profile 硬解释全部问题
3. 各 surface 的 front 只在各自问题内有意义

### 12.3 不同 surface 的 front 不直接混合

v25.4 的一个重要纪律是：

> **不同 surface 的 Pareto / knee / recommendation 不做横向直接比较。**

先比较的是：

1. 哪个 surface 先让 bathroom 质量动起来
2. 哪个 surface 的成功最稳
3. 哪个 surface 的代价最小

只有在 E / F 阶段，才进入跨 surface 的集成比较。

---

## 13. v25.4 的成功定义

v25.4 不是要立刻找到最终最佳模型，而是要产出下面 3 层成果。

### 13.1 第一层成功

至少有一个 surface 出现：

1. `bathroom balanced accuracy` 上升
2. `bathroom count MAE` 下降
3. `test_acc_min_head` 上升
4. `test_head_acc_gap` 下降

### 13.2 第二层成功

我们能明确判断：

1. 主因更像 loss geometry
2. 或更像 gradient allocation
3. 或更像 sample exposure
4. 或更像 head expressivity

也就是：

> **哪一类机制最值得继续投预算。**

### 13.3 第三层成功

在 E / F 阶段形成真正的候选决策面：

1. `balanced bathroom repair knee`
2. `deployment-feasible candidate`
3. `research-best candidate`

---

## 14. 当前建议的首批执行批次

如果按风险最低、信息密度最高排序，建议首批执行：

1. **A1**
   - `loss_family_bath=ordinal_ce`
2. **A2**
   - `loss_family_bath=distance_weighted_ce`
3. **B1**
   - `class_weighting_bath=effective_num, beta=0.999`
4. **B2**
   - `focal_gamma_bath=2.0, focal_alpha_mode_bath=class_balanced`
5. **C1**
   - `train_sampler=bathroom_balanced, bathroom_sampler_alpha=0.5`
6. **D1**
   - `head_arch=bath_mlp, bath_head_hidden=64`

这 6 个点加 baseline 就能组成 v25.4 的第一轮最小 truth probe。

它们的意义不是要一轮结束问题，而是：

1. 快速看 4 类机理里谁先动
2. 决定后续哪套 surface 值得扩张

---

## 15. v25.4 最终结论

v25.4 的核心不是“再写一个更大的 search space”，而是：

1. 把旧的大平面退役
2. 把 search 重构成多个局部真相面
3. 每个面只验证一个假设
4. 只有在局部面里真的出现 bathroom 修复信号之后，才重新谈组合、Bayes 和结构放大

可以把本次 update 的结论浓缩成一句话：

> **从 v25.4 开始，`floorplan_checker` 的 autoresearch 不再围绕“统一大平面”的假想 frontier 展开，而是围绕多个独立 hypothesis surface 展开；我们要先找出“哪一类机制能让 bathroom 真正动起来”，再谈最终的 balanced bathroom repair knee。**

---

## 16. v25.4 执行日志（预留，待实现后回填）

本章节预留给后续真实执行后的回填：

1. 指标补齐日志
2. 各 surface 的实际 campaign / run_id
3. 产物路径
4. 哪些 surface 有效 / 无效
5. 最终进入 E / F 的 winner family

