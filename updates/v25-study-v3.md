# v25 Study v3 — 从 Frontier Search 转向 Diagnosis Search

> 2026-04-26  
> 前置：`updates/v25-findings.md`、`updates/v25-findings-v2.md`

---

## 1. v3 的问题定义

v25 到目前为止已经做了三类实验：

1. `4x64` 的宽而浅笛卡尔积快扫
2. `6x64` 的同构快扫
3. v2 的目标校正后结构/分辨率对照

这些实验共同给出了一个清晰结论：

> **当前问题已经不是“哪个结构更好”，而是“为什么 bathroom 头始终是 weakest head”。**

因此，v3 的研究目标不再是继续做全局 frontier search，而是：

> **先用诊断型参数空间让 `val_acc_min_head`、`val_head_acc_gap`、`val_loss_bathroom` 动起来，再谈真正的 Pareto 与 knee。**

---

## 2. 为什么要换方向

### 2.1 前两轮全局平面已经证明不够有信息量

在 `4x64` 与 `6x64` 的宽扫里：

1. 质量轴很快饱和
2. front 经常塌成一个点
3. 更大结构、更高分辨率主要抬高成本

### 2.2 v2 的结构/分辨率对照进一步证明问题不在全局容量

v2 Batch-1 的 4 个点：

1. `4x64 / 160 / 16 / 1e-4`
2. `6x64 / 160 / 16 / 1e-4`
3. `4x64 / 224 / 16 / 1e-4`
4. `6x64 / 224 / 16 / 1e-4`

在以下指标上完全一致：

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `test_acc_min_head`

所以：

> **结构和分辨率并没有改变 balanced quality，只是在放大成本。**

### 2.3 数据分布并不支持“纯粹是 bathroom 类别极度稀缺”这一简单解释

按当前 clean manifest 统计：

1. `bathroom_head` 的 4 个类在 train split 中分别约为：
   - `1`: 122,294
   - `2`: 130,882
   - `3`: 69,690
   - `4+`: 25,749

这说明 bathroom 确实不是完全均匀，但也不是那种“少到几乎学不到”的极端稀缺分布。

因此 v3 的当前判断是：

1. bathroom 头更弱
2. 但根因未必只是 raw class imbalance
3. 更可能是：
   - 语义更难
   - 标签边界更模糊
   - 当前共享 backbone + 对称线性头的配置没给 bathroom 头足够针对性

---

## 3. v3 的新笛卡尔积平面

v3 不再把“人类可权衡选择集”建立在全局结构/分辨率平面上，而是先建立在**diagnostic plane**上。

### 3.1 主平面：bathroom diagnosis plane

坐标：

1. `x = val_loss_bathroom`
2. `y = val_acc_min_head`
3. 颜色 = `val_head_acc_gap`

原因：

1. 当前 `val_acc_min_head` 基本由 bathroom 头决定
2. 如果 bathroom loss 下降但 `min_head` 不升，说明只是更自信地犯同样的错
3. 如果 bathroom loss 下降且 `min_head` 上升，说明这类参数真的在修 weakest head

### 3.2 辅助平面：balance plane

坐标：

1. `x = val_head_acc_gap`
2. `y = val_acc_min_head`
3. 颜色 = `wall_time_s`

用途：

1. 判断某个参数是否真的让模型更平衡
2. 同时看到为了平衡付出了多少成本

### 3.3 成本平面

坐标：

1. `x = peak_memory_mb`
2. `y = wall_time_s`
3. 颜色 = `val_acc_min_head`

用途：

1. 避免再次把纯成本放大点误认为 frontier 候选

---

## 4. v3 的 Pareto / Bayesian 快速逼近思路

### 4.1 Pareto 不该在“错误的 space”上硬逼

前面的失败不是 Pareto 方法错了，而是：

1. 质量轴没有动
2. 于是 Pareto 只剩成本排序

所以 v3 的真正前提是：

> **先让诊断型参数把质量轴拉开，然后 Pareto 才有意义。**

### 4.2 框架里现有的 Bayesian-style 模块能做什么

当前框架中已经有一个轻量级的 Bayesian-style surrogate：

1. `framework/services/bayes/surrogate.py`
   - 用 bootstrap ridge ensemble 拟合 surrogate
   - 输出 `mu` 与 `sigma`
2. `framework/services/bayes/acquisition.py`
   - 用 `p_feas * (mu + 2.75 * sigma + priors) - penalty`
   - 对 unseen candidates 做快速排序

这说明：

> **框架已经具备“快速贝叶斯逼近”的基础骨架。**

### 4.3 但 v3 不应该立刻在全局空间上跑 Bayes

原因：

1. 当前全局空间里的许多点在 accuracy 上完全同值
2. 这种情况下 surrogate 学到的主要会是：
   - cost ordering
   - priors
3. 它不会给人类提供真正新的质量权衡

因此 v3 的正确做法是：

1. 先在一个**局部、诊断型、会动 quality 的空间**里做 seed grid
2. 再在这个局部空间上让 Bayesian surrogate 做快速逼近

### 4.4 v3 的局部 Bayesian 子空间

建议的局部空间：

1. 固定：
   - `4x64`
   - `160`
   - `batch=16`
2. 主要探索：
   - `loss_weight_bath`
3. 次级探索：
   - `loss_weight_bed`
   - `loss_weight_park`
   - `learning_rate`

这个空间的好处：

1. 维度低
2. 变量直接对应 bathroom 头
3. 更容易在少量点上学出可用 surrogate

---

## 5. v3 的参数 space

### 5.1 全局固定

1. `num_res_blocks = 4`
2. `num_filters = 64`
3. `image_resolution = 160`
4. `batch_size = 16`

### 5.2 v3 Local Diagnostic Space

主轴：

1. `loss_weight_bath ∈ {1.0, 1.5, 2.0, 3.0, 4.0}`

次轴：

1. `loss_weight_bed ∈ {1.0, 0.75}`
2. `loss_weight_park ∈ {1.0, 0.75}`
3. `learning_rate ∈ {1e-4}`

后续如需要再开：

1. `learning_rate ∈ {5e-5, 1e-4}`
2. `time_budget ∈ {300, 600}`

---

## 6. v3 当前要执行的第一批实验

本轮立即执行的是 **Batch-1: Bathroom-first diagnosis sweep**。

固定：

1. `4x64 / 160 / 16 / 1e-4`
2. `max_train_samples = 8192`
3. `max_eval_samples = 2048`
4. `max_test_samples = 2048`
5. `seed = 42`

扫描点：

1. `(bed=1.0, bath=1.0, park=1.0)`
2. `(bed=1.0, bath=1.5, park=1.0)`
3. `(bed=1.0, bath=2.0, park=1.0)`
4. `(bed=1.0, bath=3.0, park=1.0)`
5. `(bed=1.0, bath=4.0, park=1.0)`
6. `(bed=0.75, bath=3.0, park=0.75)`

这一批的目的不是找最终模型，而是回答：

1. `loss_weight_bath` 能不能让 `val_acc_min_head` 动起来
2. `val_loss_bathroom` 与 `val_acc_min_head` 是否开始出现 trade-off
3. `head_gap` 是否能被显著压缩

---

## 7. v3 的输出与解读

### 7.1 主输出指标

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `val_acc_bathroom`
5. `val_loss_bathroom`
6. `test_acc_min_head`
7. `test_head_acc_gap`

### 7.2 成本指标

1. `wall_time_s`
2. `peak_memory_mb`
3. `inference_latency_ms`

### 7.3 解读顺序

1. 先看 `val_acc_min_head` 是否上升
2. 再看 `val_head_acc_gap` 是否下降
3. 再看 `val_loss_bathroom` 是否同步下降
4. 最后再看成本

### 7.4 成功标准

若以下任一出现，就说明 v3 的方向是对的：

1. `val_acc_min_head` 比 baseline 明显上升
2. `val_head_acc_gap` 明显下降
3. `val_acc_bathroom` 上升且没有把另外两个头明显打坏

如果这 3 条都不出现，则说明：

1. 单纯的 loss 权重还不够
2. 下一步要扩展参数种类，而不是继续扩展数值范围

---

## 8. v3 之后的分叉

### 情形 A：bathroom 权重有效

则下一步：

1. 在局部空间上做更细的 Bayes refine
2. 补 seed
3. 提高 budget

### 情形 B：bathroom 权重无效

则下一步：

1. 引入新参数种类：
   - focal loss
   - class/sample weighting
   - bathroom-focused sampler
   - head-specific hidden layer / MLP head
2. 也就是说，问题将从“参数值”升级为“参数种类”
