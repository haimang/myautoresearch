# v25 Study v2 — Floorplan-Checker 目标校正后的探索设计

> 2026-04-26  
> 前置：`updates/v25-study.md`、`updates/v25-findings.md`

---

## 1. 这次 v2 要纠正什么

前两轮 `v25` 已经给出两个重要事实：

1. 当前默认 `overview.png` 的主图语义并不等于我们真正想找的平衡
2. 当前 `60s` Stage A 还会把 `eval/test` 采样压到 `batch_size * 10`
3. 因此图像会同时受到：
   - **目标错位**
   - **评估分辨率过低**
   两个问题影响

具体来说，旧目标存在 3 个混淆：

1. 把 **部署预算问题** 和 **研究探索问题** 混在一个 knee 里
2. 把 **单头指标** 混进默认主图，导致图的语义偏向 `bedroom`
3. 把 **Stage A legality / throughput screening** 误当成了真正的 frontier / knee 确认

所以 v2 的任务不是“继续跑更多点”，而是先把：

> **我们到底想在什么平面上找什么平衡**

说清楚。

---

## 2. v2 的核心定义：我们真正想找什么

对 floorplan 这种多头分类任务，真正值得找的不是“某一头最高”，而是：

> **最弱头已经站稳、头间差距不过大、总体准确率够高，同时额外成本还没有陡增的 balanced knee。**

因此，v2 的决策优先级必须重排为：

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `wall_time_s`
5. `peak_memory_mb`
6. `inference_latency_ms`

换句话说：

1. 先看最差头有没有掉队
2. 再看头间是否失衡
3. 再看整体平均是不是足够好
4. 最后才看代价

---

## 3. v2 的三张平面

v2 不再试图让“一张图解决所有问题”，而是明确拆成三张平面。

### 3.1 平面 A — Feasibility / throughput plane

**用途**：快速看哪些点值得继续投入。

坐标：

1. `x = peak_memory_mb`
2. `y = wall_time_s`
3. 颜色 = `val_acc_min_head`

这张图回答：

1. 点是否过贵
2. 点是否明显越出预算
3. 在成本接近时，哪一点的最弱头更好

### 3.2 平面 B — Deployment knee plane

**用途**：只回答“12GB 内什么最值”。

过滤：

1. `peak_memory_mb <= 12000`

坐标：

1. `x = wall_time_s`
2. `y = val_acc_min_head`
3. 颜色 = `val_acc_macro`

这张图只服务部署视角，不服务研究视角。

### 3.3 平面 C — Research balance plane

**用途**：回答“更大模型 / 更高分辨率是否真的换来了更平衡的质量提升”。

坐标：

1. `x = val_head_acc_gap`
2. `y = val_acc_min_head`
3. 颜色 = `wall_time_s` 或 `peak_memory_mb`

这张图是 v2 最重要的主平面。

原因：

1. `val_acc_min_head` 直接代表最弱头底线
2. `val_head_acc_gap` 直接代表平衡性
3. 只有当点同时向：
   - 更高 `min_head`
   - 更低 `head_gap`
   方向移动时，我们才会说它更接近 balanced knee

---

## 4. v2 的 objective profile

为服务上面的定义，v2 不再沿用旧 profile 的主排序方式。

新 profile：

```text
hard constraints:
  nan_loss_count == 0
  dataset_missing_file_count == 0
  dataset_missing_field_count == 0
  dataset_leakage_count == 0
  dataset_summary_mismatch_count == 0

maximize:
  1. val_acc_min_head
  2. val_acc_macro
  3. val_acc_bedroom
  4. val_acc_bathroom
  5. val_acc_parking

minimize:
  1. val_head_acc_gap
  2. wall_time_s
  3. peak_memory_mb
  4. inference_latency_ms
```

这意味着：

1. 默认 Pareto 主图会变成：
   - `y = val_acc_min_head`
   - `x = val_head_acc_gap`
2. 这比旧的“单头准确率 vs wall time”更接近真实研究目标

对应配置文件：

1. `domains/floorplan_checker/manifest/objective_profile_v2.json`

---

## 5. v2 的实验路线预测

### 5.1 为什么不再继续大面积浅预算横扫

前两轮已经证明：

1. `60s` Stage A 过于容易让质量轴饱和
2. `6x64` 快扫没有带来新的质量景观，只是抬高了成本
3. 所以继续大面积横扫，只会继续得到“图上还是一条横线”的结果

因此，v2 的路线应该从：

> **宽而浅的平面扫描**

改成：

> **窄而深的锚点对照 + 更严格评估**

### 5.2 v2 Batch-1：Anchor Confirmation

这是 v2 的第一批实际执行实验。

目的：

1. 在更高评估分辨率下，重新判断 `4x64` 与 `6x64`
2. 验证 `160` 与 `224` 的差异是否真实存在
3. 避免再被 `batch32` 和高 LR 噪声干扰

Batch-1 参数空间：

| 轴 | 取值 |
|---|---|
| `num_res_blocks` | `{4, 6}` |
| `num_filters` | `{64}` |
| `image_resolution` | `{160, 224}` |
| `batch_size` | `{16}` |
| `learning_rate` | `{1e-4}` |
| `loss_weight_bed` | `{1.0}` |
| `loss_weight_bath` | `{1.0}` |
| `loss_weight_park` | `{1.0}` |
| `seed` | `{42}` |

总点数：

```text
2 × 1 × 2 × 1 × 1 = 4 points
```

执行预算：

1. `time_budget = 300`
2. 显式指定：
   - `max_train_samples = 8192`
   - `max_eval_samples = 2048`
   - `max_test_samples = 2048`

为什么这样设定：

1. 不再让 `<600s` 自动把评估样本压到 `160`
2. 仍然把每点成本控制在可接受范围
3. 让 v2 的第一批实验更像“受控验证”而不是“粗糙 smoke”

### 5.3 对 Batch-1 的预期

如果 `6x64` 真的更值得继续研究，那么在 v2 Batch-1 下应该出现至少一种信号：

1. `val_acc_min_head` 明显高于 `4x64`
2. `val_head_acc_gap` 更低或至少不更差
3. 额外 wall / memory 成本仍然能解释

如果这些信号没有出现，那么更可能说明：

1. 当前任务对 `6x64` 的收益并不明显
2. 或者当前提升容量不如提升 loss strategy / data protocol 有价值

### 5.4 Batch-1 之后的路线预测

#### 情形 A：`6x64` 显著改善 balanced quality

则进入：

1. `6x64` 局部 LR 验证
   - `lr ∈ {1e-4, 5e-4}`
2. seed recheck
3. 研究视角 front 细化

#### 情形 B：`4x64` 与 `6x64` 差别极小

则进入：

1. 固定 `4x64`
2. 转向 loss balance 子空间
3. 或转向 deployment-budget 线的 seed confirm

#### 情形 C：`224` 仍然没有形成质量优势

则进入：

1. 暂时冻结 `resolution = 160`
2. 不再把更高分辨率当优先方向

---

## 6. v2 的输出参数

v2 约定，每批实验至少输出以下参数：

### 6.1 质量参数

1. `val_acc_min_head`
2. `val_head_acc_gap`
3. `val_acc_macro`
4. `val_acc_bedroom`
5. `val_acc_bathroom`
6. `val_acc_parking`

### 6.2 成本参数

1. `wall_time_s`
2. `peak_memory_mb`
3. `inference_latency_ms`
4. `num_params`

### 6.3 完整性参数

1. `nan_loss_count`
2. `dataset_missing_file_count`
3. `dataset_missing_field_count`
4. `dataset_leakage_count`
5. `dataset_summary_mismatch_count`

---

## 7. v2 的解读方法

v2 不用单一分数，而用分层规则解释。

### 7.1 先过滤

以下任一不满足，直接判为无效点：

1. `nan_loss_count != 0`
2. 任一 dataset contract 指标非 0

### 7.2 再判断 balanced quality

优先顺序：

1. `val_acc_min_head` 更高者优先
2. 若差异极小，再看 `val_head_acc_gap`
3. 若仍接近，再看 `val_acc_macro`

### 7.3 最后看成本是否值得

当两个点质量接近时：

1. wall 更低者优先
2. memory 更低者优先
3. latency 更低者优先

### 7.4 实用阈值约定

为避免把浮动当结论，v2 暂时采用以下经验阈值：

1. `val_acc_min_head` 提升 `< 0.002`
   - 视为基本同级
2. `val_head_acc_gap` 恶化 `> 0.003`
   - 视为平衡性明显变差
3. wall 或 memory 增长 `> 50%` 且 `min_head` 没有明确提升
   - 视为不值得继续投资

---

## 8. 本次立刻执行的内容

本次用户要求中，立即执行的是：

1. 编写本文件
2. 创建新的 objective profile
3. 运行 Batch-1 四点对照实验
4. 将实际结果写入 `updates/v25-findings-v2.md`

这意味着 v2 不是只写计划，而是已经开始第一批真实验证。
