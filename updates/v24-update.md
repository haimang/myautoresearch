# v24 Update — Floorplan-Checker Multi-Head MLX 执行计划

> 2026-04-25  
> 前置：`updates/v24-study.md`、`updates/v22-update.md`  
> 定位：在 `myautoresearch` 框架下落地第一个计算机视觉领域：**基于多头残差神经网络（Multi-head ResNet）的户型图自动化分类与参数预测系统**。

---

## 1. v24 一句话目标

> **利用 v22 建立的 `run_metrics` 泛化体系，在 Apple MLX 框架上训练一个共享 ResNet 主干、同时预测卧室/浴室/车位数量的多头模型，并通过贝叶斯优化（Bayesian Optimization）在“模型复杂度、训练延迟、多任务准确率”之间寻找最优的 Pareto 前沿。**

v24 不做：
1. 不涉及复杂的图像分割或目标检测（本阶段仅做分类/回归头）。
2. 不处理非户型图的通用图像数据。
3. 不在非 Apple Silicon 平台上进行训练。
4. 不把图像特征抽取强行塞进 Gomoku 的 MCTS 逻辑。

v24 要做：
1. **数据集接入**：完成对 `syd2` 同步过来的 40.5 万张户型图（17GB）的高效加载与预处理。
2. **多头模型构建**：实现一个具有 3 个输出头（Bedrooms, Bathrooms, Parking）的 MLX ResNet。
3. **领域策略配置**：编写 `floorplan_checker` 的 `search_space.json` 与 `objective_profile.json`。
4. **Pareto 搜索**：运行一轮完整的 Campaign，找出在显存限制下的最佳网络宽度（Filters）、深度（Blocks）和多任务 Loss 权重。
5. **验证泛化能力**：证明框架不需要修改核心代码，仅通过 JSON 配置即可适配 CV 领域的多目标优化任务。

---

## 2. 领域目录结构与文件分工 (Directory Tree & Responsibilities)

为了确保 Floorplan-Checker 领域逻辑的内聚，并完全符合 myautoresearch 框架的 subprocess 契约，我们将在 `domains/floorplan_checker/` 下建立如下深度的文件结构：

```text
domains/floorplan_checker/
├── __init__.py
├── train.py                         # [核心] framework subprocess 入口；负责解析 JSON 参数、启动 MLX 训练循环、调度数据和模型，并写入 run_metrics
├── data_loader.py                   # [数据] 负责解析 manifests/*.jsonl，多线程读取图像，执行归一化，组装 batch (images, labels_bed, labels_bath, labels_park)
├── models/                          # [架构] MLX 模型定义目录
│   ├── __init__.py
│   ├── backbone.py                  # ResNet 共享主干网络 (ResBlocks, Convolutional layers)
│   └── heads.py                     # 分支输出头 (Linear layers + Softmax)
├── utils/                           # [工具]
│   ├── __init__.py
│   ├── transforms.py                # 图像增广与预处理 (Resize, Normalize)
│   └── memory_monitor.py            # MLX 显存峰值监控，防止 OOM 导致进程静默崩溃
├── search_space.json                # [策略] v24 超参搜索空间 (结构、学习率、Loss 权重)
├── objective_profile.json           # [策略] Pareto 多目标与硬约束定义 (maximize acc, minimize memory/time)
├── stage_policy.json                # [策略] 训练预算的 Fidelity 分层 (Smoke, Early Truncation, Full Validation)
├── branch_policy.json               # [策略] 策略局部分叉规则 (例如调整特定 Head 的学习率)
├── selector_policy.json             # [策略] 候选生成策略
└── acquisition_policy.json          # [策略] 受约束的贝叶斯获取函数权重 (Constrained qEHVI / UCB)
```

### 2.1 核心文件详细分工

- **`train.py`**: 
  - 扮演框架与具体领域的桥梁。它不直接包含网络结构定义，而是作为“实验编排者”。
  - 接收 `--candidate-json`、`--time-budget`、`--campaign-id`。
  - 初始化 `data_loader` 和 `models`。
  - 执行 `mlx.core.eval` 训练步。
  - 将 `val_acc_bedroom`, `val_acc_bathroom`, `val_acc_parking`, `peak_memory_mb`, `wall_time_s` 等聚合后写入 `tracker.db` 的 `run_metrics` 表。
- **`data_loader.py`**:
  - 处理 `dataset/manifests/` 下的元数据。
  - 由于有 40.5 万个小文件（17GB），如果采用同步 I/O 会导致 GPU 饥饿。这里需要实现一个具备 Prefetch 队列的异步 Batch Generator。
- **`models/backbone.py`**:
  - 提供参数化的 ResNet 变体，允许从 `search_space` 传入 `num_res_blocks` 和 `num_filters` 动态构建层。
- **`models/heads.py`**:
  - 分离三个分类任务的 Linear 映射层，并提供统一的 `forward(features)` 接口。
- **`objective_profile.json`**:
  - 将告诉框架层 `analyze.py --pareto` 如何绘制前沿，哪些是需要最大化的（三个 Accuracy），哪些是硬性约束（显存 < 12GB，不能 NaN）。

---

## 3. 训练数据组织与任务映射

数据集已经通过 rsync 物理隔离并附带了 manifests：

```text
domains/floorplan_checker/dataset/
├── train/                  # 34.8万文件
├── eval/                   # 3.6万文件
├── test/                   # 2.1万文件
└── manifests/
    ├── summary.json        # 统计元数据
    ├── train.jsonl         # 每行一条样本 JSON，包含标签和路径
    └── eval.jsonl / test.jsonl
```

**任务级映射 (Targets)**：
根据 `train.jsonl` 中的元数据，我们需要从每个样本中提取：
- `bedroom_head`: 分类目标 `[0, 1, 2, 3, 4, "5+"]` (6 类)。
- `bathroom_head`: 分类目标 `[1, 2, "3+"]` (3 类)。
- `parking_head`: 分类目标 `[0, 1, "2+"]` (3 类)。
- `image`: 统一 resize 到 `256x256` 或 `512x512`（可作为超参）。

---

## 4. 第一版 Search Space 草案

为了让贝叶斯引擎在架构和多任务平衡间找到最优解，我们将以下参数暴露为 `search_space.json` 中的 Axes：

```json
{
  "domain": "floorplan_checker",
  "name": "multi_head_resnet_v1",
  "version": "0.1",
  "axes": {
    "num_res_blocks": {
      "type": "int",
      "values": [4, 6, 8, 12],
      "scale": "linear",
      "role": "structure"
    },
    "num_filters": {
      "type": "int",
      "values": [32, 64, 128],
      "scale": "linear",
      "role": "structure"
    },
    "learning_rate": {
      "type": "float",
      "values": [1e-5, 5e-5, 1e-4, 5e-4],
      "scale": "log",
      "role": "training"
    },
    "loss_weight_bed": {
      "type": "float",
      "values": [0.5, 1.0, 2.0],
      "scale": "linear",
      "role": "strategy"
    },
    "loss_weight_bath": {
      "type": "float",
      "values": [0.5, 1.0, 2.0],
      "scale": "linear",
      "role": "strategy"
    },
    "loss_weight_park": {
      "type": "float",
      "values": [0.5, 1.0, 2.0],
      "scale": "linear",
      "role": "strategy"
    },
    "image_resolution": {
      "type": "categorical",
      "values": [256, 384, 512],
      "role": "execution"
    }
  }
}
```

---

## 5. v24 内部 Phase 实施计划

| Phase | 名称 | 工作内容与验收标准 |
|---|---|---|
| **Phase 1** | 数据流与预处理 | 编写 `data_loader.py` 与 `utils/transforms.py`。<br>**验收**：能够从 `manifests` 构建数据集对象，迭代读取速度稳定在 > 300 img/s。 |
| **Phase 2** | MLX 模型与前向传播 | 编写 `models/backbone.py` 和 `models/heads.py`。<br>**验收**：给定一个随机的 Batch Tensor，能够无报错地通过 ResNet 并输出三个头的 logits，无显存泄漏。 |
| **Phase 3** | 训练循环与指标上报 | 编写 `train.py`。对接框架 `run_metrics` 协议。<br>**验收**：运行单次训练，成功写入 `val_acc_bedroom`, `peak_memory_mb`, `wall_time_s` 等字段到 `tracker.db`。 |
| **Phase 4** | 策略文件与冒烟测试 | 编写 `search_space.json`, `objective_profile.json` 等策略配置。<br>**验收**：使用 `framework/sweep.py` 成功执行一个基于 JSON 候选的短期冒烟训练，证明框架能够跨域驱动 CV 训练。 |
| **Phase 5** | 局部快速探索 (Grid Sweep) | 使用 `sweep.py` 对较小的参数空间进行撒网（约 15-20 个 run）。<br>**验收**：观察不同网络容量（blocks/filters）与多任务权重的初步性能景观。 |
| **Phase 6** | 贝叶斯逼近 Pareto 前沿 | 使用 `analyze.py --recommend-next` 调用 Bayesian Acquisition 引擎，迭代寻找最佳 trade-off。<br>**验收**：在不超显存红线的情况下，找到精度最高且成本合理的拐点（Knee Point）。 |
| **Phase 7** | 报告与可视化 | 使用 `analyze.py --pareto` 与报告生成工具。<br>**验收**：输出展示三个准确率与训练时间、显存消耗关系的 2D/3D 散点图。 |

---

## 6. 数据与硬件红线 (Constraints & Disclaimers)

1. **显存红线 (OOM Risk)**：
   由于 CV 模型特征图较大，如果 `num_res_blocks=12`, `num_filters=128`, `image_resolution=512` 组合碰在一起，极易导致 Apple Unified Memory 爆显存。`train.py` 必须捕获分配异常，如果触发 OOM，立刻上报 `peak_memory_mb=MAX` 及 `nan_loss_count=1`，确保 Bayesian 模型将其识别为 Infeasible 区域。
2. **多目标竞争 (Negative Transfer)**：
   卧室、浴室和车位的特征层级可能不同（车位可能依赖外部轮廓，浴室依赖局部纹理）。当多任务同时反向传播时，可能出现互相干扰。这是将 `loss_weight` 放入 Search Space 的根本原因。
3. **框架侵入性隔离**：
   所有特定于 `floorplan_checker` 的逻辑（包括图像读取库如 PIL / OpenCV，或特殊的 MLX 算子）必须严格限制在 `domains/floorplan_checker/` 内部。`framework/` 绝对不允许增加任何与 "Image", "Pixels", "ResNet" 相关的硬编码。

---

## 7. 验收与收口判断

当 v24 完成时，系统必须能回答以下问题：

> **“在显存不超过 12GB、单 Epoch 训练时间低于 10 分钟的前提下，为了让卧室预测准确率超过 85% 同时浴室准确率超过 90%，我们应该使用多深的网络？多分辨率的图像？以及如何分配三个任务的 Loss 权重？”**

并且，回答这个问题的过程完全是由 `myautoresearch` 框架的 **Generic Objective Profile + Bayesian Acquisition** 自动完成的，而非人类手动调参。这将彻底坐实框架作为通用 AI Scientist 的能力。
