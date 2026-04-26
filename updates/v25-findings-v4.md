# v25 Findings v4 — 参数种类、退役边界与下一阶段 knee 定义

> 2026-04-26  
> 前置：`updates/v25-findings.md`、`updates/v25-findings-v2.md`、`updates/v25-findings-v3.md`

---

## 1. 为什么需要 v4

v25 到 v3 的探索已经给出一个比“哪组超参更好”更重要的结论：

> **当前 floorplan_checker 的主要问题不是参数值没扫够，而是参数种类不足。**

但这个结论还需要进一步拆开，否则下一阶段仍然容易继续犯同样的错误：

1. 把不该继续作为主轴的参数继续扫下去。
2. 把只适合做诊断的平面误认为最终选型平面。
3. 把 bedroom / parking 这种容易头的表现误认为真正的 knee。
4. 把 Pareto front 的形状当成答案，而不是先问 front 是否在有意义的目标空间里。

因此 v4 要回答 3 个问题：

1. 当前哪些参数应该保留，哪些应该退役。
2. 新增哪些参数种类最有希望真正改变结果。
3. 新参数加入后，我们究竟要找什么结果，knee point 应该如何重新定义。

---

## 2. 过去全部 v25 实验的共同总结

### 2.1 v25 Phase 1：`4x64` 快扫的真实价值

Phase 1 扫描了：

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `image_resolution ∈ {160, 224, 256}`
4. `batch_size ∈ {16, 32}`
5. `learning_rate ∈ {5e-5, 1e-4, 5e-4}`

核心发现不是“找到了最终模型”，而是：

1. `12GB` 是 objective profile 的 policy 约束，不是机器硬件上限。
2. 当前机器是 `128GB` 内存环境，Phase 1 已经跑到约 `29.2GB`。
3. 在 `12GB` deployment-budget 视角下，合法工作带几乎只剩：
   - `image_resolution = 160`
   - `batch_size = 16`
4. `224 / 256` 不是跑不动，而是超出当前部署预算。
5. Stage A 的质量轴过早饱和，front-only 图自然塌成单点。

所以 Phase 1 的真正贡献是：

> **找到 working band 与预算边界，而不是找到真实 quality frontier。**

### 2.2 Phase 2：`6x64` 没有打开新 frontier

Phase 2 把容量上移到：

1. `num_res_blocks = 6`
2. `num_filters = 64`

并复刻 Phase 1 的 `resolution × batch × lr` 平面。

核心发现：

1. `6x64` 没有打开新的 12GB 内工作带。
2. `6x64` 没有让质量曲线变得更有解释力。
3. 它主要把：
   - `wall_time_s`
   - `peak_memory_mb`
   - `inference_latency_ms`
   整体抬高。
4. 在浅预算下，更大模型还可能出现“更大但欠训练”的假象。

因此：

> **`6x64` 目前不是主战参数，只能保留为后续验证哨兵。**

### 2.3 v2：目标校正后，结构/分辨率仍然不动质量

v2 把目标从旧的单头/成本视角校正为：

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. 成本

并用更严格采样重新对照：

1. `4x64 / 160 / 16 / 1e-4`
2. `6x64 / 160 / 16 / 1e-4`
3. `4x64 / 224 / 16 / 1e-4`
4. `6x64 / 224 / 16 / 1e-4`

显式采样预算：

1. `max_train_samples = 8192`
2. `max_eval_samples = 2048`
3. `max_test_samples = 2048`

结果最关键的地方是 4 个点的 balanced-quality 指标完全相同：

| 指标 | 数值 |
|---|---:|
| `val_acc_bedroom` | 0.8691 |
| `val_acc_bathroom` | 0.5786 |
| `val_acc_parking` | 0.8691 |
| `val_acc_min_head` | 0.5786 |
| `val_head_acc_gap` | 0.2905 |
| `test_acc_min_head` | 0.4546 |

这说明：

> **结构和分辨率没有改变当前错误模式；真正掉队的是 bathroom。**

### 2.4 v3：简单 bathroom loss reweighting 也失败

v3 固定：

1. `4x64 / 160 / 16 / 1e-4`
2. `max_train_samples = 8192`
3. `max_eval_samples = 2048`
4. `max_test_samples = 2048`

扫描：

1. `(bed=1.0, bath=1.0, park=1.0)`
2. `(bed=1.0, bath=1.5, park=1.0)`
3. `(bed=1.0, bath=2.0, park=1.0)`
4. `(bed=1.0, bath=3.0, park=1.0)`
5. `(bed=1.0, bath=4.0, park=1.0)`
6. `(bed=0.75, bath=3.0, park=0.75)`

结果：

| 配置 | val bath acc | val min-head | val gap | val bath loss | test bath acc | test bath loss |
|---|---:|---:|---:|---:|---:|---:|
| `(1.0, 1.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | **2.6824** | 0.4546 | **3.4614** |
| `(1.0, 1.5, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 3.2878 | 0.4546 | 4.2422 |
| `(1.0, 2.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 2.9983 | 0.4546 | 3.8684 |
| `(1.0, 3.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 4.0395 | 0.4546 | 5.2250 |
| `(1.0, 4.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 3.5734 | 0.4546 | 4.6157 |
| `(0.75, 3.0, 0.75)` | 0.5786 | 0.5786 | 0.2905 | 3.1831 | 0.4546 | 4.1220 |

这批实验的价值不是“找到了 bathroom 训练 knee”，而是：

> **证明 simple loss reweighting 不是当前修复 weakest-head 的有效杠杆。**

更准确地说，v3 是一个必要但笨重的否定实验：

1. 它证明 bathroom loss 权重能让 soft loss 数字变化。
2. 但它没有改变 argmax 分类行为。
3. baseline 反而拥有最低 bathroom loss。
4. 所以继续在 `loss_weight_bath` 上做更密集数值搜索没有意义。

### 2.5 v3 的重要局限

v3 的诊断平面是：

1. `x = val_loss_bathroom`
2. `y = val_acc_min_head`
3. 颜色 = `val_head_acc_gap`

这个平面适合回答：

> **某个 bathroom-directed 参数有没有让 weakest-head 动起来。**

但它不适合回答：

> **最终人类应该选择哪个 floorplan checker。**

原因是：

1. `val_loss_bathroom` 是诊断指标，不是最终产品目标。
2. bedroom / parking 已经明显不是当前瓶颈，不应该围绕容易头找 knee。
3. 如果 accuracy 完全不动，Pareto 可能从“单点 front”跳到“全 front”，这两者都不是可靠的人类决策集。
4. 原始多轴 Pareto 对几十 MB 的内存抖动过于敏感，会把噪声当成 trade-off。

所以 v4 必须把 v3 的结论改写为：

> **v3 证明了旧参数种类不足，但 v3 不是下一阶段最终 knee 的定义。**

---

## 3. 当前参数：保留、降级与退役

### 3.1 应该保留的参数

| 参数 | 保留方式 | 原因 |
|---|---|---|
| `num_res_blocks = 4` | 主锚点 | `4x64` 是当前最低成本、最稳定的控制组。 |
| `num_filters = 64` | 主锚点 | `4x64` 已足够暴露 bathroom 问题；继续加宽不是当前主矛盾。 |
| `image_resolution = 160` | 主锚点 | 12GB policy 内最稳定，且 v2 证明 `224` 没有改善 balanced quality。 |
| `batch_size = 16` | 主锚点 | 当前唯一稳定的 deployment-budget 工作带。 |
| `learning_rate = 1e-4` | 主锚点 | 在多轮实验中最稳定，作为所有新参数种类的 baseline。 |
| `learning_rate = 5e-4` | 次级验证点 | 可作为优化鲁棒性补点，但不再作为大平面主轴。 |
| `6x64` | 验证哨兵 | 只在新参数种类已经有效后，用来判断容量上移是否放大收益。 |
| `224 / batch16` | 后期研究预算验证点 | 只在新方法有效后判断高分辨率是否进一步提升，不再先扫。 |

保留的核心原则是：

> **先固定一个低成本、可复现、能暴露 bathroom 弱点的 anchor，再测试新参数种类。**

### 3.2 应该降级的参数

| 参数 | 降级后角色 | 原因 |
|---|---|---|
| `num_res_blocks ∈ {6, 8}` | Stage C 验证 | 只有当新参数种类已经让 quality 动起来后，才值得看容量是否放大收益。 |
| `num_filters ∈ {32, 128}` | 暂缓 | `32` 可能过弱，`128` 大概率先放大成本；当前不是主问题。 |
| `image_resolution ∈ {224, 256}` | 暂缓到后期 | v2 已经说明 `224` 没有改善 weakest-head；`256` 成本更高。 |
| `batch_size ∈ {32, 64}` | throughput / memory 探针 | 当前不是 quality 修复杠杆，且容易越过 12GB policy。 |
| `loss_weight_bed / bath / park` | 只保留 baseline 与极少数 sanity check | v3 已经判定 broad reweighting 不值得继续扫。 |

### 3.3 应该退役的参数使用方式

以下不是说参数永远不能出现，而是说它们应该从“主搜索轴”退役：

1. **大面积结构横扫**
   - `num_res_blocks × num_filters × resolution × batch` 的宽扫已经被证明主要得到成本分层。

2. **以单头准确率作为主 knee**
   - 尤其不能围绕 bedroom / parking 找 knee。
   - 它们是容易头，不是当前瓶颈。

3. **浅预算 Stage A 直接宣称真实 frontier**
   - Stage A 适合找 legality / throughput / working band。
   - 不适合确认最终 quality knee。

4. **继续密集扫描 `loss_weight_bath`**
   - v3 已经测试到 `1.5 / 2.0 / 3.0 / 4.0`。
   - accuracy 不动，loss 还不优于 baseline。

5. **原始多轴 Pareto 直接给人类决策**
   - 在质量不动时，Pareto 会退化成成本排序。
   - 在成本微抖且 soft-loss 微动时，Pareto 又会变成全 front。
   - 下一阶段必须用更聚焦的 utility 和分层图解释。

---

## 4. 新参数种类：什么最值得加

### 4.1 第一优先级：count / ordinal-aware loss

floorplan 的三个头本质上都是数量预测：

1. bedroom: `1, 2, 3, 4, 5+`
2. bathroom: `1, 2, 3, 4+`
3. parking: `0, 1, 2, 3, 4+`

当前 cross entropy 把类别当作无序分类：

1. 把 `1` 预测成 `2`
2. 把 `1` 预测成 `4+`

在 CE 里都是同类错误，没有距离概念。

但对 bathroom 来说，真实难点很可能就是边界模糊：

1. `2` vs `3`
2. `3` vs `4+`

因此最值得加入的新参数种类是：

1. `loss_family_bath`
   - `cross_entropy`
   - `ordinal_ce`
   - `distance_weighted_ce`
   - `expected_count_loss`
2. `ordinal_loss_weight_bath`
   - `{0.25, 0.5, 1.0}`
3. `ordinal_loss_scope`
   - `bath_only`
   - `all_heads`

预期作用：

> **不是简单让 bathroom loss 更重，而是让 bathroom 错误的几何结构更接近真实任务。**

这是比 `loss_weight_bath` 更有希望的方向。

### 4.2 第二优先级：bathroom class-balanced / focal loss

clean train manifest 中 bathroom label 分布为：

| bathroom 类 | train count |
|---|---:|
| `1` | 122,294 |
| `2` | 130,882 |
| `3` | 69,690 |
| `4+` | 25,749 |

这不是极端稀缺，但 `4+` 与 `3` 确实明显少。

因此应该加入：

1. `class_weighting_bath`
   - `none`
   - `inverse_freq`
   - `effective_num`
2. `effective_num_beta_bath`
   - `{0.99, 0.999, 0.9999}`
3. `focal_gamma_bath`
   - `{0.0, 1.0, 2.0}`
4. `focal_alpha_mode_bath`
   - `none`
   - `class_balanced`

这和 v3 的 `loss_weight_bath` 不同：

1. v3 是把整个 bathroom head 同乘一个权重。
2. class-balanced / focal 是改变 bathroom 内部不同类别、不同难度样本的梯度分布。

如果 bathroom 弱点来自少数类或 hard examples，这类参数最可能让 argmax 行为真正变化。

### 4.3 第三优先级：bathroom-focused sampler

当前 `FloorplanDataset` 只做普通 shuffle。

下一阶段应该加入采样策略参数：

1. `train_sampler`
   - `uniform`
   - `bathroom_balanced`
   - `bathroom_minority_boost`
2. `bathroom_sampler_alpha`
   - `{0.5, 1.0}`

目标不是无限 oversample bathroom，而是让每个训练 epoch / sample cap 下：

1. `bathroom=3`
2. `bathroom=4+`

获得更充分曝光。

这对当前实验尤其重要，因为 v2/v3 使用固定 `8192` train samples；如果样本 cap 下 minority 类覆盖不足，模型可能根本没有足够机会调整 bathroom 决策边界。

### 4.4 第四优先级：head-specific MLP / adapter

当前模型是：

```text
shared ResNetBackbone -> shared features -> three symmetric Linear heads
```

对应代码：

1. `FloorplanNet.backbone`
2. `MultiHeadOutput.head_bed`
3. `MultiHeadOutput.head_bath`
4. `MultiHeadOutput.head_park`

这意味着：

1. 三个头共享同一表征。
2. 三个头的输出层结构完全对称。
3. bathroom 没有额外的 head-specific 表达能力。

如果 bathroom 的语义更难，仅仅改变全局 backbone 可能无效；更合理的是加入：

1. `head_arch`
   - `linear`
   - `mlp_shared`
   - `bath_mlp`
2. `bath_head_hidden`
   - `{32, 64, 128}`
3. `bath_head_dropout`
   - `{0.0, 0.1}`
4. `head_adapter`
   - `none`
   - `bath_residual_adapter`

优先级低于 loss/sampler 的原因是：

1. 它会增加结构复杂度。
2. 它可能带来更多成本和过拟合风险。
3. 应该先用 loss/sampler 判断问题是否来自优化与数据暴露。

但如果 loss/sampler 仍然不能移动 bathroom，它就是下一条最合理路线。

### 4.5 第五优先级：noise-robust / smoothing 参数

如果 bathroom 标签边界确实更模糊，那么强行用 hard CE 可能不合适。

可以加入：

1. `label_smoothing_bath`
   - `{0.0, 0.03, 0.05, 0.1}`
2. `near_class_smoothing_bath`
   - `off`
   - `ordinal_neighbor`

`ordinal_neighbor` 的含义是：

1. `2` 的少量概率给到 `1` 和 `3`
2. `3` 的少量概率给到 `2` 和 `4+`

这比普通 label smoothing 更符合 count label 的结构。

---

## 5. 新指标：没有这些指标，下一阶段仍然容易误判

v2/v3 主要看：

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `val_loss_bathroom`

但如果要判断新参数种类是否真的修复 bathroom，下一阶段必须增加更细指标：

| 新指标 | 目的 |
|---|---|
| `val_acc_balanced_bathroom` | 防止大类准确率掩盖小类失败。 |
| `test_acc_balanced_bathroom` | 检查 bathroom 修复是否泛化。 |
| `val_recall_bathroom_1/2/3/4plus` | 看具体是哪类仍然掉队。 |
| `test_recall_bathroom_1/2/3/4plus` | 避免只在 val 上修复。 |
| `val_confusion_bathroom` | 判断错误是否集中在相邻数量。 |
| `val_mae_bathroom_count` | 用数量距离衡量错误严重度。 |
| `test_mae_bathroom_count` | 检查数量距离是否泛化改善。 |

尤其是 `bathroom balanced accuracy` 与 `bathroom count MAE` 很关键。

原因是：

> **如果模型只是更会预测大类 bathroom=1/2，却继续放弃 3/4+，普通 accuracy 可能看起来改善，但这不是我们想要的 floorplan checker。**

---

## 6. 新 knee point 应该如何定义

### 6.1 不能再定义的 knee

下一阶段不应该再找这些 knee：

1. **bedroom knee**
   - bedroom 已经不是瓶颈。
   - 围绕 bedroom 训练或选择 knee 没有研究价值。

2. **parking knee**
   - 与 bedroom 类似，当前不是主矛盾。

3. **loss-only knee**
   - v3 已经证明 bathroom loss 会动，但 argmax 行为不动。
   - loss 可以做诊断，不能单独做最终选型。

4. **cost-only knee**
   - 当质量完全相同时，最低成本点当然支配其他点。
   - 这只是 deployment baseline，不是研究突破。

5. **raw Pareto knee**
   - 多轴 Pareto 对目标选择极其敏感。
   - 如果目标空间错了，knee 只会稳定地给出错误答案。

### 6.2 v4 的目标：balanced bathroom repair knee

下一阶段真正要找的是：

> **在不显著伤害 bedroom / parking、不明显增加部署成本的前提下，最有效修复 bathroom weakest-head 的最小复杂度配置。**

这个 knee 不是“bathroom 最高分点”，而是三者的平衡：

1. bathroom 真的变好。
2. 头间差距真的缩小。
3. 复杂度和成本没有明显跳升。

### 6.3 v4 knee 的硬门槛

一个候选点必须先满足：

1. `nan_loss_count = 0`
2. `dataset_missing_file_count = 0`
3. `dataset_missing_field_count = 0`
4. `dataset_leakage_count = 0`
5. `dataset_summary_mismatch_count = 0`
6. 与 baseline 相同的显式 sample cap 和 seed 设置。
7. `peak_memory_mb` 在 deployment 线中仍以 `12000` 为硬约束；在 research 线中只作为成本轴。

### 6.4 v4 knee 的质量门槛

以当前强 baseline：

```text
4x64 / 160 / batch16 / lr=1e-4 / CE / uniform sampler / linear heads
```

为对照，第一阶段成功门槛应定义为：

| 指标 | baseline | v4 第一成功门槛 |
|---|---:|---:|
| `val_acc_bathroom` | 0.5786 | `>= 0.63` |
| `val_acc_min_head` | 0.5786 | `>= 0.63` |
| `val_head_acc_gap` | 0.2905 | `<= 0.24` |
| `test_acc_bathroom` | 0.4546 | `>= 0.50` |
| `test_acc_min_head` | 0.4546 | `>= 0.50` |
| `val_acc_macro` | 0.7723 | 不下降超过 `0.02` |
| `val_acc_bedroom / parking` | 0.8691 / 0.8691 | 单头不下降超过 `0.03` |

如果新增 balanced metrics 后，应再加：

1. `val_acc_balanced_bathroom` 明显上升。
2. `test_acc_balanced_bathroom` 明显上升。
3. `val_mae_bathroom_count` 下降。
4. `test_mae_bathroom_count` 下降。

### 6.5 v4 knee 的排序规则

当多个点都达到质量门槛后，排序应采用 lexicographic + utility，而不是原始多轴 Pareto：

1. 先过滤数据完整性与 NaN。
2. 再过滤是否满足最低质量门槛。
3. 再按主 utility 排序：

```text
utility =
  + 1.00 * Δtest_acc_min_head
  + 0.80 * Δtest_acc_balanced_bathroom
  + 0.60 * Δtest_acc_bathroom
  - 0.70 * Δtest_head_acc_gap
  - 0.50 * Δtest_mae_bathroom_count
  - cost_penalty
  - complexity_penalty
```

其中：

1. `cost_penalty` 只在质量差异达到门槛后才进入排序。
2. `complexity_penalty` 用来惩罚更复杂的结构：
   - sampler < loss family < loss family + sampler < head MLP < adapter + larger backbone
3. 如果两个点质量接近，选择更简单、更便宜的点。

这才是下一阶段真正的人类可权衡选择集：

> **不是“哪个点 Pareto 不被支配”，而是“哪个最小改动真正修复了 bathroom，同时没有破坏 floorplan checker 的整体平衡”。**

---

## 7. 下一阶段实验方法

### 7.1 Stage v4-A：先补指标，再做单因子参数种类探针

先实现并记录新指标：

1. bathroom per-class recall
2. bathroom balanced accuracy
3. bathroom confusion matrix
4. bathroom count MAE

然后固定 baseline：

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `image_resolution = 160`
4. `batch_size = 16`
5. `learning_rate = 1e-4`
6. `max_train_samples = 8192`
7. `max_eval_samples = 2048`
8. `max_test_samples = 2048`
9. `seed = 42`

第一批只做单因子或近似单因子探针：

| 点 | 新参数种类 | 配置 |
|---|---|---|
| A0 | baseline | `CE + uniform sampler + linear heads` |
| A1 | ordinal loss | `loss_family_bath=ordinal_ce` |
| A2 | distance loss | `loss_family_bath=distance_weighted_ce` |
| A3 | class-balanced loss | `class_weighting_bath=effective_num, beta=0.999` |
| A4 | focal loss | `focal_gamma_bath=2.0` |
| A5 | focal + class-balanced | `focal_gamma_bath=2.0, alpha=class_balanced` |
| A6 | sampler | `train_sampler=bathroom_balanced` |
| A7 | sampler + focal | `train_sampler=bathroom_balanced, focal_gamma_bath=2.0` |
| A8 | bath MLP head | `head_arch=bath_mlp, bath_head_hidden=64` |

这一批的目的不是马上找最终 knee，而是回答：

> **哪一种参数种类能让 bathroom 的 argmax、balanced accuracy 或 count MAE 真正动起来。**

### 7.2 Stage v4-B：只放大有效参数种类

如果 v4-A 中某个 family 满足以下任一条件：

1. `val_acc_bathroom` 提升 `>= 0.03`
2. `val_acc_balanced_bathroom` 提升 `>= 0.03`
3. `val_mae_bathroom_count` 下降明显
4. `val_head_acc_gap` 下降 `>= 0.03`

则进入 v4-B。

v4-B 只围绕有效 family 做小范围细化，例如：

1. `focal_gamma_bath ∈ {1.0, 1.5, 2.0, 2.5}`
2. `effective_num_beta_bath ∈ {0.99, 0.999, 0.9999}`
3. `bathroom_sampler_alpha ∈ {0.5, 1.0}`
4. `ordinal_loss_weight_bath ∈ {0.25, 0.5, 1.0}`

此时可以让 Bayesian surrogate 介入，但只在这个局部 family 内使用。

### 7.3 Stage v4-C：多 seed 与更大样本验证

只有 v4-B 的候选进入 v4-C。

v4-C 要做：

1. seeds:
   - `{42, 43, 44}`
2. sample caps:
   - `max_train_samples = 32768`
   - `max_eval_samples = 8192`
   - `max_test_samples = 8192`
3. 对照点：
   - baseline
   - top-2 new parameter candidates

v4-C 的判断标准：

1. test 指标必须同步改善。
2. 多 seed 平均改善必须存在。
3. 改善不能只来自单一 seed。
4. bedroom / parking 不能被明显牺牲。

### 7.4 Stage v4-D：新参数有效后，再回头验证结构

只有当某个新参数 family 已经能修 bathroom，才重新打开结构轴：

1. `4x64 / 160`
2. `6x64 / 160`
3. `4x64 / 224`
4. `6x64 / 224`

目的不是重新做大平面，而是回答：

> **新方法是否与容量/分辨率有正交增益。**

如果没有，最终 deployment knee 仍应保留在 `4x64 / 160 / 16`。

---

## 8. 下一阶段验证目标

下一阶段的目标不是“跑出一张更漂亮的 Pareto 图”，而是验证下面 4 个命题。

### 8.1 命题一：bathroom 弱点是否来自无序 CE 与 count 任务不匹配

验证方式：

1. ordinal-aware loss
2. distance-weighted CE
3. bathroom count MAE

成功信号：

1. `bathroom count MAE` 下降。
2. 相邻类混淆减少。
3. `test_acc_bathroom` 上升。

### 8.2 命题二：bathroom 弱点是否来自 minority / hard examples 梯度不足

验证方式：

1. class-balanced loss
2. focal loss
3. focal + class-balanced

成功信号：

1. `bathroom=3 / 4+` recall 上升。
2. `bathroom balanced accuracy` 上升。
3. 不以牺牲 `bathroom=1 / 2` 为代价。

### 8.3 命题三：bathroom 弱点是否来自样本暴露不足

验证方式：

1. bathroom-balanced sampler
2. bathroom-minority boost sampler

成功信号：

1. 在相同 `8192` train sample cap 下，bathroom minority recall 上升。
2. 多 seed 后仍保持。

### 8.4 命题四：bathroom 弱点是否来自 head 表达能力不足

验证方式：

1. bath-specific MLP head
2. bath residual adapter

成功信号：

1. loss/sampler 无效时，head-specific architecture 能让 bathroom 指标移动。
2. 成本增量小于 backbone 扩大。

---

## 9. v4 最终结论

v25 到 v3 的完整结论现在可以收束为：

1. `4x64 / 160 / 16 / 1e-4` 应保留为主 baseline。
2. `6x64`、`224`、`batch32` 不应继续作为主搜索轴。
3. `loss_weight_bath` 的密集数值搜索应退役。
4. 单头 easy-head knee、loss-only knee、raw Pareto knee 都不应作为下一阶段决策目标。
5. 下一阶段最值得新增的参数种类依次是：
   - count / ordinal-aware loss
   - class-balanced / focal bathroom loss
   - bathroom-focused sampler
   - head-specific MLP / adapter
   - noise-robust smoothing
6. 新 knee 应定义为 **balanced bathroom repair knee**：
   - 修复 bathroom weakest-head
   - 压低 head gap
   - 不显著牺牲 bedroom / parking
   - 不显著增加部署成本
   - 在 test 与多 seed 上成立

最重要的一句话是：

> **我们不再寻找 bedroom、parking 或单纯成本意义上的 knee；下一阶段要寻找的是“最小新增复杂度下，能够稳定修复 bathroom 弱头的 floorplan balanced knee”。**

