# v25 Findings v3 — 从实验失败中重构搜索方向

> 2026-04-26  
> 前置：`updates/v25-findings.md`、`updates/v25-findings-v2.md`

---

## 1. 为什么要写 v3

v25 到目前为止已经做了三组关键实验：

1. `4x64` 的全局快扫
2. `6x64` 的同构快扫
3. v2 的结构/分辨率诊断对照

它们共同指向同一个事实：

> **当前的问题已经不再是“主图该怎么画”，而是“我们一直在扫一组主要改变成本、却不改变 weakest-head 错误模式的参数”。**

这意味着 v3 必须完成两件事：

1. 把前面几组实验的共同发现系统化写清楚
2. 把“Pareto / Bayesian 快速逼近应该如何介入”这件事重新定义

---

## 2. 前几组实验共同给出的发现

### 2.1 第一发现：全局结构/分辨率轴当前主要放大成本

无论是：

1. `4x64 → 6x64`
2. `160 → 224 → 256`

在当前 floorplan 任务上，它们主要带来的都是：

1. 更高 `wall_time_s`
2. 更高 `peak_memory_mb`
3. 更高 `inference_latency_ms`

而不是稳定、可解释的 balanced-quality 提升。

### 2.2 第二发现：v2 已经证明问题不只是“旧平面画错了”

v2 把主平面改成：

1. `x = val_head_acc_gap`
2. `y = val_acc_min_head`

但 4 个对照点仍然在质量空间中完全重合。

因此：

> **不是只要换一张图就能得到有价值的可权衡选择集。**

### 2.3 第三发现：当前主矛盾是 bathroom 头

v2 的 4 个点显示：

1. `val_acc_bedroom = 0.8691`
2. `val_acc_bathroom = 0.5786`
3. `val_acc_parking = 0.8691`

这说明：

1. `val_acc_min_head` 完全由 bathroom 决定
2. `val_head_acc_gap` 也是 bathroom 被拉开的结果

因此当前真正要修的是：

> **bathroom 头为什么学不好**

而不是：

> “模型再大一点会不会更好”

### 2.4 第四发现：bathroom 的 raw label 分布并非极端稀缺

按 clean train manifest 统计：

1. bathroom `1`: 122,294
2. bathroom `2`: 130,882
3. bathroom `3`: 69,690
4. bathroom `4+`: 25,749

这说明 bathroom 确实有不均衡，但不是那种“极端少样本”到完全解释当前弱头现象的程度。

所以 v3 当前更合理的判断是：

1. bathroom 更弱
2. 但根因未必只是类别频次
3. 还可能包含：
   - 语义难度更高
   - 标签边界更模糊
   - 共享 backbone + 对称线性头没有提供足够针对性

### 2.5 第五发现：当前 search space 的参数种类不够

当前主要探索的都是：

1. `num_res_blocks`
2. `num_filters`
3. `image_resolution`
4. `batch_size`
5. `learning_rate`
6. `loss_weight_*`

但如果 bathroom 问题不是纯容量问题，那么真正缺的可能是：

1. head-specific loss family
2. class/sample weighting
3. bathroom-focused sampler
4. head-specific MLP / hidden layer

也就是说，v3 必须承认：

> **我们现在面对的不只是“参数值选择错误”，还有“参数种类可能不够”的问题。**

---

## 3. 对笛卡尔积平面的重新定义

v3 的结论是：

> **当前不应该继续在“结构 × 分辨率 × batch”这样的全局平面上期待可权衡选择集。**

### 3.1 旧平面的局限

旧平面的问题不是数学形式，而是变量家族不对：

1. 自变量主要控制全局容量
2. 因变量主要关心 bathroom 弱头
3. 这两者当前没有形成足够灵敏的映射

### 3.2 v3 的新平面

因此 v3 选择把主平面换成：

1. `x = val_loss_bathroom`
2. `y = val_acc_min_head`
3. 颜色 = `val_head_acc_gap`

这个平面回答的问题是：

1. bathroom loss 降了没有
2. weakest head 是否真的被拉起来了
3. 头间失衡有没有被压缩

换句话说，v3 的平面不再主要服务“人类最终选型”，而是先服务：

> **人类诊断系统到底哪里没学会**

---

## 4. 对 Pareto 与 Bayesian 快速逼近的重新理解

### 4.1 Pareto 为什么之前总塌成一个点

因为：

1. 质量指标不动
2. 成本指标在动

于是 Pareto 排序自然退化成：

1. 谁更便宜
2. 谁更快

这不是 Pareto 错，而是数据面没有给 Pareto 足够的质量差异。

### 4.2 当前框架里的 Bayesian-style 模块说明了什么

框架中已有：

1. `framework/services/bayes/surrogate.py`
   - bootstrap ridge ensemble
   - 产出 `mu` / `sigma`
2. `framework/services/bayes/acquisition.py`
   - 基于 `p_feas * (mu + 2.75*sigma + priors) - penalty`
   - 对 unseen candidates 做快速排序

这说明：

> **框架具备“快速贝叶斯逼近 Pareto frontier”的基础骨架。**

### 4.3 但为什么之前不该急着上 Bayes

因为在全局结构空间里：

1. accuracy 几乎不动
2. surrogate 学到的主要是 cost ordering
3. 这样即使加了 uncertainty，也还是很难给人类提供新选择

所以前面的真正问题不是“没有 Bayes”，而是：

> **Bayes 用在了一个当前不会动 quality 的空间上。**

### 4.4 v3 对 Bayes 的新定位

v3 认为 Bayes 仍然有价值，但必须满足两个条件：

1. 先在局部空间里有一批会动质量的 seed 点
2. surrogate 的 utility 必须围绕 weakest-head 诊断来定义

也就是说，v3 对 Bayes 的重新定位是：

> **不是拿来替代全局网格搜索，而是拿来在局部 diagnosis subspace 中快速逼近一条有意义的“bathroom 修复 frontier”。**

---

## 5. 当前最合理的 v3 假设

到此为止，最合理的工作假设是：

1. 结构/分辨率不是主矛盾
2. bathroom 头是主矛盾
3. 单纯的全局容量扩展对 bathroom 没帮助
4. 下一步应该先测：
   - bathroom 权重
   - 以及它是否真的改变 weakest-head 指标

因此，v3 的方向不再是：

1. `4x64 vs 6x64`
2. `160 vs 224`

而是：

1. 固定 `4x64 / 160 / 16 / 1e-4`
2. 先做 **bathroom-first diagnosis sweep**

---

## 6. v3 当前要执行的实验

对应研究计划已经写入：

1. `updates/v25-study-v3.md`

本轮立即执行的配置是：

1. 固定：
   - `4x64`
   - `160`
   - `batch=16`
   - `lr=1e-4`
2. 显式采样预算：
   - `8192 / 2048 / 2048`
3. 扫描：
   - `(1.0, 1.0, 1.0)`
   - `(1.0, 1.5, 1.0)`
   - `(1.0, 2.0, 1.0)`
   - `(1.0, 3.0, 1.0)`
   - `(1.0, 4.0, 1.0)`
   - `(0.75, 3.0, 0.75)`

下面在本文件后半段回填实际结果。

---

## 7. Append — v3 Bathroom-first diagnosis sweep 实际结果

### 7.1 实际执行配置

本轮固定：

1. `4x64`
2. `160`
3. `batch = 16`
4. `learning_rate = 1e-4`
5. `max_train_samples = 8192`
6. `max_eval_samples = 2048`
7. `max_test_samples = 2048`
8. `seed = 42`

实际扫描的 6 个点：

1. `(bed=1.0, bath=1.0, park=1.0)`
2. `(bed=1.0, bath=1.5, park=1.0)`
3. `(bed=1.0, bath=2.0, park=1.0)`
4. `(bed=1.0, bath=3.0, park=1.0)`
5. `(bed=1.0, bath=4.0, park=1.0)`
6. `(bed=0.75, bath=3.0, park=0.75)`

全部运行成功：

1. `6 / 6` 成功
2. 无 NaN
3. dataset contract 指标全部为 0

产物目录：

1. `output/floorplan_checker/v25-floorplan-v3-001/pareto/v25_v3_diagnosis_overview.png`
2. `output/floorplan_checker/v25-floorplan-v3-001/pareto/v25_v3_diagnosis_plane.png`
3. `output/floorplan_checker/v25-floorplan-v3-001/pareto/v25_v3_balance_plane.png`
4. `output/floorplan_checker/v25-floorplan-v3-001/pareto/v25_v3_bath_weight_vs_loss.png`
5. `output/floorplan_checker/v25-floorplan-v3-001/v25_v3_bath_001_summary.json`

### 7.2 实验结果表

| 配置 | val bath acc | val min-head | val gap | val bath loss | test bath acc | test bath loss | wall_s | mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `(1.0, 1.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | **2.6824** | 0.4546 | **3.4614** | **166.91** | 8919.72 |
| `(1.0, 1.5, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 3.2878 | 0.4546 | 4.2422 | 171.76 | 8895.97 |
| `(1.0, 2.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 2.9983 | 0.4546 | 3.8684 | 187.77 | 8881.72 |
| `(1.0, 3.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 4.0395 | 0.4546 | 5.2250 | 180.90 | 8881.72 |
| `(1.0, 4.0, 1.0)` | 0.5786 | 0.5786 | 0.2905 | 3.5734 | 0.4546 | 4.6157 | 178.46 | 8884.69 |
| `(0.75, 3.0, 0.75)` | 0.5786 | 0.5786 | 0.2905 | 3.1831 | 0.4546 | 4.1220 | 178.11 | 8886.47 |

### 7.3 最重要的结果：bathroom 权重让 loss 动了，但没有让 weakest head 动

这是 v3 到目前为止最重要的一条新信息：

1. `val_acc_bathroom`
2. `val_acc_min_head`
3. `val_head_acc_gap`
4. `test_acc_bathroom`
5. `test_acc_min_head`

在 6 个点上 **全部完全相同**。

但与此同时：

1. `val_loss_bathroom` 明显变化
2. `test_loss_bathroom` 也明显变化

这意味着：

> **v3 终于找到了一个会动的“软质量轴”，但它揭示出的结论不是“bath 重权有用”，而是“bath 重权在当前设置下并没有改变最终分类行为”。**

### 7.4 更强的负结论：simple reweighting 甚至在 soft 指标上也没有改善

如果只看 bathroom loss：

1. baseline `(1.0, 1.0, 1.0)` 反而是：
   - 最低 `val_loss_bathroom = 2.6824`
   - 最低 `test_loss_bathroom = 3.4614`
2. 提高 `loss_weight_bath` 到 `1.5 / 2.0 / 3.0 / 4.0`
   - 没有提高 bathroom accuracy
   - 也没有降低 bathroom loss
3. 把 `bed/park` 降到 `0.75` 且 `bath = 3.0`
   - 仍然没有优于 baseline

也就是说：

> **单纯提高 bathroom loss 权重，不仅没把 weakest head 拉起来，甚至连 bathroom 的 soft loss 也没有真正改善。**

### 7.5 这说明了什么

这批结果把问题进一步收紧成了下面这个判断：

1. 结构/分辨率不是主矛盾
2. 单纯的 head loss reweight 也不是足够强的杠杆
3. 当前错误模式更像是：
   - 表征没有抓到 bathroom 的关键区分特征
   - 或标签边界本身更模糊
   - 或优化方式不足以改变 bathroom 决策边界

因此，v3 已经把问题从：

1. “参数值该怎么调”

推进到了：

2. “我们是不是需要新的参数种类”

### 7.6 Pareto 在 v3 上为什么又变得“全 front”

在 `objective_profile_v3.json` 下，本轮 Pareto 的结果是：

1. `Feasible = 6`
2. `Front = 6`
3. `Dominated = 0`

这看起来像是与前面“单点 front”相反的极端，但其实原因很清楚：

1. 所有 accuracy 指标完全一样
2. `val_loss_bathroom` 与 `wall_time_s`、`peak_memory_mb` 之间又出现了细小 trade-off
3. 尤其是 `peak_memory_mb` 只有几十 MB 的微小抖动
4. 这些微小差异足以阻止严格 domination

所以：

> **v3 证明了另一件事：当 accuracy 完全不动、cost 和 soft-loss 只有细小浮动时，原始 Pareto 会从“单点 front”跳到“全 front”，同样不适合直接给人类做最终决策。**

这进一步说明，v3 之后的 Bayes / Pareto 不能直接照搬原始多轴，而要：

1. 使用更聚焦的 utility
2. 对噪声级成本轴做去敏感或分层解释

### 7.7 对 Bayesian 快速逼近的新修正

本轮数据让 v3 对 Bayes 的看法进一步具体化：

1. 好消息：
   - `val_loss_bathroom` 终于会动
   - 这意味着局部 surrogate 不再完全没有学习对象
2. 坏消息：
   - `val_acc_min_head` / `val_head_acc_gap` 仍完全不动
   - 所以如果 Bayes 仍只盯 accuracy，它还是学不到新东西

因此，v3 之后的 Bayesian 快速逼近应当改成：

1. **局部空间**
   - 只在 bathroom diagnosis subspace 上跑
2. **局部 utility**
   - 优先围绕：
     - `val_loss_bathroom`
     - `val_acc_min_head`
     - `val_head_acc_gap`
   - 而不是继续混入全局结构空间
3. **局部目标**
   - 寻找“在不恶化 min-head / gap 的前提下，bathroom loss 是否还能继续下降”的点

换句话说：

> **Bayes 现在仍然有价值，但它下一步也应该服务 diagnosis，不是服务 global frontier。**

### 7.8 v3 到这里的最终结论

v25 到 v3 这一串实验，已经把问题逼近到了一个非常窄的核心：

1. 扩大结构无效
2. 提高分辨率无效
3. 简单提高 bathroom 权重也无效

因此，下一步最合理的方向已经不再是继续扩展当前数值空间，而是扩展**参数种类**：

1. focal loss / class-balanced loss
2. bathroom-focused sampler
3. head-specific hidden MLP
4. head-specific feature adapter
5. 更长 budget + 多 seed 作为验证层，而不是主诊断手段

可以把 v3 当前的最终判断浓缩成一句话：

> **我们已经证明，当前 search space 的主要问题不是“值选错了”，而是“参数种类不够”；结构、分辨率和简单 loss reweight 都没有改变 bathroom 这个 weakest head 的分类行为。**
