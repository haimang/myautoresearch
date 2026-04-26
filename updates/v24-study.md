# v24 Study — Floorplan-Checker Domain / Multi-Head MLX Training & Pareto Optimization

> 2026-04-25  
> 前置：`updates/v22-study.md`、当前 `framework/` 代码  
> 目标：评估在 `myautoresearch` 框架下引入一个全新的计算机视觉领域（Floorplan-Checker），并使用 Apple MLX 训练多头残差神经网络（Multi-head ResNet），最终通过贝叶斯算法逼近多目标 Pareto 前沿的可行性与实施路线。

---

## 1. 一句话结论

> **完全可行，并且这是验证框架从“单一标量奖励 (Gomoku Win Rate)”走向“多任务/多约束深度学习模型”泛化能力的绝佳实证场。正确做法是：在 MLX 中构建共享主干的多头网络，将各个头的损失/准确率与计算资源消耗解耦，全面接入 v22 的 `run_metrics` 与 `objective_profiles`，利用受约束的贝叶斯获取函数（Bayesian Acquisition）在“训练成本、任务 A 性能、任务 B 性能”的高维空间中寻找非支配前沿（Pareto Frontier）。**

这意味着 v24 的核心挑战不是“写一个分类器”，而是：
1. **多目标对齐 (Multi-objective alignment)**：不同预测头（如房间分类、尺寸回归、拓扑关系）的 Loss 尺度和重要性不同。
2. **资源与精度的博弈**：主干网络的容量（Filters, Blocks）直接决定了特征表达能力，但也线性增加了显存 (OOM 风险) 和推理延迟。
3. **架构级的 Search Space**：不仅是调整 Learning Rate，还包括多头损失权重 (Loss Weights) 和分支结构的动态配置。

---

## 2. 当前框架能直接复用什么

依托 v22 已经完成的 `run_metrics` 和 `objective_profiles` 泛化，框架层无需为多头网络做核心逻辑的硬编码修改。概念映射如下：

| 当前 Framework 概念 | Floorplan-Checker Domain 映射 |
|---|---|
| `campaign` | 针对特定的户型数据集、特定的算力预算上限开展的一轮系统性多头网络搜索。 |
| `search_space.json` | 可枚举的网络深度（ResBlocks）、宽度（Filters）、各预测头的 Loss 权重、学习率及批大小。 |
| `stage_policy.json` | Fidelity 分层：Stage 0 (微小数据集冒烟测试)，Stage 1 (少 epoch 截断训练看收敛趋势)，Stage 2 (Full epoch 完整验证)。 |
| `branch_policy.json` | 策略局部分叉：例如冻结主干网络，仅对某些难以收敛的特定 Head 增加学习率或调整结构进行 Continuation 训练。 |
| `selector.py` & `acquisition.py` | 基于已完成的训练轨迹，利用 Surrogate Model 预测未探索超参的验证集表现，并结合 UCB/EI 生成下一批推荐超参。 |
| `run_metrics` ledger | 记录 `val_loss_cls`, `val_acc_cls`, `val_mae_reg`, `wall_time_s`, `peak_memory_mb`。 |

---

## 3. Floorplan-Checker 领域的核心建模

### 3.1 状态与指标变量 (Metrics)

对于一个多头的 Floorplan-Checker，训练结束或 Checkpoint 阶段必须收集以下多维状态：

1. **成本与资源指标 (Cost/Resource)**
   - `wall_time_s`: 训练总挂钟时间。
   - `peak_memory_mb`: MLX 训练过程中的显存峰值（硬约束预警）。
   - `inference_ms`: 单张户型图的前向传播延迟。

2. **真理指标 (Truth/Quality)**
   - `val_loss_total`: 总验证损失。
   - `val_acc_room_type`: 房间类型分类准确率（Head 1）。
   - `val_iou_segmentation`: 房间区域分割 IoU 或回归 MAE（Head 2）。

### 3.2 动作变量 (Actions / Hyperparameters)

Agent 可以调整的搜索空间（Search Space）包括：
- **Architecture**: `num_res_blocks`, `num_filters`
- **Optimization**: `learning_rate`, `batch_size`, `weight_decay`
- **Multi-task Balancing**: `loss_weight_head1`, `loss_weight_head2`

### 3.3 红线约束 (Hard Constraints)

如同 v22 FX 领域的流动性红线，深度学习训练也有硬约束：
1. `out_of_memory (OOM)`: 如果超参组合导致 MLX 分配显存失败，直接标记为 Infeasible，Acquisition 必须惩罚该区域。
2. `loss_divergence (NaN)`: 训练中途 Loss 爆炸，立即早停，记录为无效策略。
3. `min_acc_threshold`: 如果分类头准确率甚至不如随机瞎猜（例如 < 10%），拒绝晋升。

---

## 4. Pareto 前沿的定义与多目标优化

### 4.1 目标定义 (Objective Profile)

我们不能简单地“最大化准确率”，因为这会导致无脑堆叠极其庞大的模型。合理的 Objective Profile 应该是：

```json
{
  "name": "floorplan_balanced_profile",
  "objectives": {
    "maximize": ["val_acc_room_type", "val_iou_segmentation"],
    "minimize": ["wall_time_s", "inference_ms", "peak_memory_mb"]
  },
  "hard_constraints": {
    "oom_crashes": 0,
    "nan_losses": 0
  }
}
```

### 4.2 Bayesian Acquisition 的重构重点

为了在多目标下逼近前沿，我们的 Surrogate Model（代理模型）需要同时预测多个目标：
- 预测器 A：预测 `val_acc_room_type`
- 预测器 B：预测 `val_iou_segmentation`
- 分类器 C：预测 `is_feasible` (是否会 OOM)

Acquisition Score 可以采用 **Constrained qEHVI** (Expected Hypervolume Improvement) 思想，或者更简单的加权标量化（Scalarized UCB）：

```text
score =
  w1 * predicted_acc
  + w2 * predicted_iou
  + uncertainty_bonus
  + frontier_gap_bonus
  - time_penalty
  - memory_penalty
  - infeasibility_penalty (极大值)
```

---

## 5. 建议的目录结构

在 `domains/floorplan_checker/` 下建立独立的生态：

```text
domains/floorplan_checker/
├── train.py                         # 核心入口，接受 dynamic candidate JSON，执行 MLX 训练循环
├── models/
│   ├── backbone.py                  # ResNet 共享主干网络
│   └── heads.py                     # 分类头、回归头、分割头
├── data_loader.py                   # 户型图数据集加载与增强
├── metrics_reporter.py              # 将多头 Loss 和性能转换为 run_metrics 写入 tracker.db
├── search_space.json                # v24 搜索空间定义
├── objective_profile.json           # 多目标与约束定义
├── stage_policy.json                # 多阶段训练截断策略
├── branch_policy.json               # Head 级微调分叉规则
└── selector_policy.json             # 基于贝叶斯的候选生成策略
```

---

## 6. v24 阶段推进计划

### Stage 1: 基础设施与冒烟测试 (Infrastructure & Smoke)
**目标**：打通 MLX 多头前向传播、反向传播与指标入库。
- 编写 `models/` 与 `train.py`。
- 使用合成的随机 Tensor 作为假图片跑通循环，验证显存增长情况（防止 MLX 显存泄漏）。
- 确保 `val_acc_room_type` 等指标成功写入 `tracker.db` 的 `run_metrics` 表。

### Stage 2: 局部快速探索 (Local Grid/Random Sweep)
**目标**：建立初始的性能景观。
- 启动 `index.py sweep`，使用较小的 budget（如 `--time-budget 300`）。
- 在 `search_space.json` 范围内随机采样 20-30 组参数。
- 观察不同 `loss_weight` 组合对两个 Head 性能的拉扯效应。

### Stage 3: 贝叶斯驱动的 Pareto 前沿逼近 (Bayesian Frontier Search)
**目标**：高效寻找 Knee Point。
- 使用 `index.py analyze --recommend-next` 生成新候选。
- 系统根据前期的 30 个数据点，推断出哪些参数区域既能保持高精度又不至于 OOM。
- 自动执行推荐的候选，持续多轮迭代，直到 `analyze --pareto` 输出的边界稳定。

### Stage 4: 人类选点与微调 (Human Decision & Branching)
**目标**：决策与定型。
- 渲染 3D Pareto 散点图（X: 精度A, Y: 精度B, Color/Size: 延迟/显存）。
- 人类研究员挑选一个位于 Pareto 拐点（最优 Trade-off）的运行配置。
- 使用 `index.py branch` 针对该配置进行完整数据集的大预算长时间训练。

---

## 7. 第一版 Search Space 草案

```json
{
  "domain": "floorplan_checker",
  "name": "multi_head_resnet_v1",
  "version": "0.1",
  "axes": {
    "num_res_blocks": {
      "type": "int",
      "values": [4, 6, 8, 12, 16],
      "scale": "linear",
      "role": "structure"
    },
    "num_filters": {
      "type": "int",
      "values": [32, 64, 128, 256],
      "scale": "linear",
      "role": "structure"
    },
    "learning_rate": {
      "type": "float",
      "values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
      "scale": "log",
      "role": "training"
    },
    "loss_weight_head1": {
      "type": "float",
      "values": [0.1, 0.5, 1.0, 2.0, 5.0],
      "scale": "linear",
      "role": "strategy"
    },
    "loss_weight_head2": {
      "type": "float",
      "values": [0.1, 0.5, 1.0, 2.0, 5.0],
      "scale": "linear",
      "role": "strategy"
    }
  }
}
```

---

## 8. 第一版 Stage Policy 草案

与 Gomoku 类似，我们需要规避将昂贵的算力浪费在极差的超参组合上：

```text
Stage A — Synthetic Smoke (Budget: 30s)
  目标：验证不 OOM，Loss 不为 NaN。
  动作：在小批合成数据上跑 1 个 epoch。

Stage B — Early Truncation (Budget: 300s)
  目标：排除收敛极慢或分类器坍缩（全部预测为某一类）的配置。
  动作：在真实数据子集上跑 5 个 epoch，检查 val_acc 是否明显提升。

Stage C — Full Validation (Budget: 3600s)
  目标：获取真实的 Pareto 指标用于 Bayesian Acquisition。
  动作：完整数据集训练至早停。
```

---

## 9. 总结

引入 `floorplan_checker` 不是为了重造轮子，而是验证 myautoresearch 系统作为 **Domain-Agnostic AI Scientist** 的纯度。当我们在 v24 成功找到多头网络在资源和多任务精度上的 Pareto 前沿时，就证明了这套从数据入库、特征采样到贝叶斯推荐的闭环，真正具备了解决一般化工程研究问题的能力。