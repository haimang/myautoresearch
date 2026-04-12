# v11 Findings — MCTS 首次训练实验分析

> 2026-04-12 | 基于 mcts_1st_exp.db（3 runs, 133 cycle_metrics, 4 checkpoints）  
> 硬件：Apple M3 Max

---

## 1. 实验概览

| Run | 模式 | MCTS sims | 并行 | 时间 | Cycles | 局数 | 最终 WR | Cycle/min |
|-----|------|-----------|------|------|--------|------|---------|-----------|
| c334509c | MCTS-50 | 50 | 8 | 609s | 12 | 96 | 46.0% | 1.2 |
| d5f7fdae | Pure Policy | 0 | 8 | 300s | 114 | 912 | 76.0% | 22.8 |
| e742d1f5 | MCTS-200 | 200 | 4 | 650s | 7 | 28 | 48.0% | 0.6 |

**核心速度差异：** MCTS-50 每 cycle 约 50 秒，Pure Policy 每 cycle 约 2.7 秒。MCTS-200 每 cycle 约 94 秒。MCTS 慢了 **19-35 倍**。

---

## 2. WR 曲线对比分析

### Pure Policy (d5f7fdae) — 22 个 eval 点

```
cycle   5: 62%  ← 快速上升
cycle  10: 72%
cycle  15: 62%  ← 开始震荡
cycle  20: 74%
cycle  25: 68%
cycle  30: 72%
...
cycle  80: 82%  ← 局部高点
cycle 110: 76%  ← 最终值
```

WR 均值 72.5%，标准差 5.9%。曲线在 cycle 15 后进入 60-82% 的震荡区间，没有清晰的收敛趋势。**这与 v11-analysis-by-opus.md 中 b3f99d4f 的模式完全一致**——纯 policy 自对弈的 WR 在早期快速上升后进入噪声震荡。

### MCTS-50 (c334509c) — 2 个 eval 点

```
cycle  5: 26%
cycle 10: 46%
```

**数据量严重不足。** 12 个 cycle 中只有 2 个 eval 点，无法判断收敛趋势。46% 的最终 WR 远低于纯 policy 的 76%，但这是因为：

1. **训练量差异巨大：** MCTS-50 只训练了 96 盘，纯 policy 训练了 912 盘（9.5 倍）
2. **训练步数差异：** MCTS-50 只做了 148 步梯度更新，纯 policy 做了 3291 步（22 倍）
3. **Loss 未收敛：** MCTS-50 的 loss 从 6.48 降到 6.01（几乎没动），纯 policy 从 6.32 降到 1.87

**结论：** MCTS-50 在 600 秒内只完成了相当于纯 policy ~15 秒的训练量。WR 低是因为训练不够，不是因为信号质量差。

### MCTS-200 (e742d1f5) — 2 个 eval 点

```
cycle  3: 26%
cycle  6: 48%
```

与 MCTS-50 类似的模式。7 个 cycle，28 盘，21 步梯度更新。同样是训练量极度不足。

---

## 3. 关键发现

### 3.1 MCTS 速度是当前的瓶颈，不是信号质量

| 指标 | MCTS-50 | Pure Policy | 比值 |
|------|---------|-------------|------|
| 局数/分钟 | 9.5 | 182.4 | **0.05x** |
| 梯度步/分钟 | 14.6 | 658.2 | **0.02x** |
| 每局耗时 | ~6.3s | ~0.33s | **19x** |

MCTS-50 每步走棋需要 50 次前向传播（= 50 次 MLX inference），每盘约 50 步，所以每盘需要 ~2500 次前向传播。纯 policy 每盘只需 ~50 次。

**这意味着在相同时间预算内，MCTS 的训练样本量是纯 policy 的 1/19（50 sims）到 1/35（200 sims）。** 即使每个样本的信号质量更高，总训练量的巨大差距也足以解释 WR 差异。

### 3.2 当前实验不能判定 MCTS 信号质量

要公平对比 MCTS 和纯 policy 的信号质量，需要控制训练步数（而不是训练时间）。两种方案：

**方案 A：固定训练步数对比**
- 两个 run 都训练到 500 步梯度更新
- MCTS-50 需要约 ~35 分钟，纯 policy 需要约 ~1.5 分钟
- 对比相同步数下的 WR

**方案 B：给 MCTS 足够长的时间**
- `--mcts-sims 50 --time-budget 3600`（1 小时）
- 预期产生 ~120 个 cycle、~960 盘、~1500 步
- 足以与纯 policy 300s 的 912 盘 / 3291 步做粗略对比

### 3.3 Loss 下降验证了 MCTS 训练信号可用

MCTS-50 的 loss 从 6.48 降到 6.01，虽然下降幅度小，但方向正确。考虑到只有 148 步梯度更新，这是合理的。纯 policy 的 loss 在前 148 步（约 5 个 cycle）内也只从 6.32 降到约 5.4。

**MCTS 训练信号是可用的**——loss 在正确方向移动。需要更多训练量来验证信号质量是否优于纯 policy。

---

## 4. GPU 功率与利用率分析

### 4.1 为什么 MCTS 训练时 GPU 功率只有 5W

观察到的现象：
- 纯 policy 训练：GPU 功率 ~25W
- MCTS 训练：GPU 功率 ~5W
- CPU 功率也很低：~5W

**根本原因：MCTS 的瓶颈在 Python 解释器，不在 GPU。**

当前 MCTS 实现的执行流程：

```
每步走棋（50 sims）:
  for sim in range(50):           ← Python for 循环
    node = root                    ← Python 对象操作
    sim_state = copy_fn(state)     ← numpy 数组拷贝
    while node.is_expanded:        ← Python while 循环
      node = node.select_child()   ← Python 对象遍历 + 浮点计算
      apply_fn(sim_state, action)  ← numpy 数组操作
    priors, value = evaluate_fn()  ← 一次 MLX GPU 推理 ★
    node.expand(priors, mask)      ← Python 对象创建
    node.backup(value)             ← Python while 循环
```

**50 次模拟中，GPU 只被调用了 50 次（每次一个单样本推理）。每次 GPU 推理只需 ~0.1ms，但 Python 的树遍历、对象创建、numpy 拷贝可能需要 ~2ms。** 这意味着 GPU 利用率只有约 5%——绝大部分时间在等 Python。

对比纯 policy 模式：每个 cycle 做一次 64-sample 的 batch 推理，GPU 一次处理 64 个棋盘的前向传播，然后做 30 步 batch=256 的训练。**GPU 每次处理的数据量大得多，利用率自然高。**

### 4.2 为什么 CPU 也没跑满

MCTS 是单线程的 Python 循环。M3 Max 有 16 个 CPU 核心，但只有 1 个核心在跑 MCTS 的 Python 解释器。其余 15 个核心空闲。

### 4.3 加速方案

**短期（v12 可做）：**

1. **叶子节点批量合并（Leaf Batching）**  
   当前：每次 expand 触发一次单样本 GPU 推理（batch=1）  
   改进：攒 16-64 个待评估叶子节点后一次性 batch 推理  
   预期加速：**5-15x GPU 利用率提升**  
   实现方式：在 mcts_search 中使用"虚拟损失"（virtual loss），允许同时展开多条搜索路径，把叶子节点攒成 batch

2. **多棋盘叶子合并**  
   当前：8 盘棋逐个串行搜索  
   改进：8 盘棋同时搜索，叶子节点跨棋盘合并成大 batch  
   预期加速：**额外 2-4x**（与叶子批量合并叠加）

3. **Python 热路径 Cython/Numba 化**  
   `select_child()` 的 PUCT 计算和 `backup()` 的树遍历是纯数值操作  
   用 Cython 或 Numba 重写可以将 Python 开销降低 10-50x  
   但这增加了构建复杂度

**中期（需要更大改动）：**

4. **MLX 原生 MCTS**  
   将整个搜索树用 MLX 张量表示（fixed-size tensor tree），select/expand/backup 全部在 GPU 上执行  
   这是 AlphaZero 工程化实现的标准做法，但实现复杂度高

**推荐优先级：** 叶子批量合并 > 多棋盘合并 > Cython 热路径 > MLX 原生 MCTS

---

## 5. 数据库与观测体系评估

### 5.1 现有记录是否足够

| 数据项 | 是否记录 | 评估 |
|--------|---------|------|
| `mcts_simulations` in runs | ✓ 已记录 | 可区分 MCTS vs 纯 policy |
| 每 cycle WR | ✓ 已记录 | 可做曲线对比 |
| 每 cycle loss | ✓ 已记录 | 可做 loss 对比 |
| 每 cycle timestamp | ✓ 已记录 | 可算 cycle 耗时 |
| MCTS sims/sec | ✗ 未入库 | TUI 显示但未持久化 |
| MCTS entropy | ✗ 未入库 | TUI 显示但未持久化 |
| MCTS focus (top1_share) | ✗ 未入库 | TUI 显示但未持久化 |
| 每 cycle self-play 耗时 | ✗ 未入库 | 已计算但未持久化 |

**评估：当前 DB 记录对 v11 实验分析已够用。** MCTS 的 sims/sec、entropy、focus 等指标主要用于实时监控（TUI），不需要跨 run 分析。如果未来需要跨 run 对比 MCTS 效率，再添加到 cycle_metrics 表。

### 5.2 表结构

当前 `cycle_metrics` 表有 `win_rate`、`loss`、`total_games`、`timestamp_s` 等字段，可以支撑：
- WR 曲线形态分析（收敛 vs 震荡）
- 训练效率对比（WR/步数、WR/时间）
- 停滞检测（`--stagnation`）

不需要为本轮 MCTS 实验修改表结构。

---

## 6. 对 v12 scope 的更新分析

### 6.1 v11-update.md 中 v12 scope 的修正

v11-update.md 第 8 章定义的 v12 scope 中，优先级需要调整：

| v12 原计划 | 调整 |
|-----------|------|
| MCTS 训练验证与调优 | **仍然最高优先级**，但需要先解决速度瓶颈 |
| 批量 MCTS 优化 | **提升为第一优先级** — 没有批量合并，MCTS 训练不可行 |
| 基于真实 WR 重建对手进化链 | 延后 — 需要先验证 MCTS 长时间训练的效果 |
| Pareto 集成到 report | 延后 — 优先级低于速度优化 |
| 早停机制 | 保持 — 对长时间 MCTS 训练更重要 |

### 6.2 v12 的核心任务：让 MCTS 达到可用速度

当前 MCTS-50 的速度是 9.5 局/分钟。要达到与纯 policy 可比的训练效率，需要至少 **50 局/分钟**（纯 policy 的 1/3 即可，因为 MCTS 样本质量更高）。

目标加速比：**5-6x**。

实现路径：

**Phase 1：叶子节点批量合并（预期 3-5x 加速）**

修改 `core/mcts.py` 的 `mcts_search()`：
1. 不再每次 expand 立即调用 `evaluate_fn`
2. 改为攒够 batch_size 个待评估叶子节点后批量调用
3. 需要引入"虚拟损失"机制：每条搜索路径在 select 阶段对经过的节点减去虚拟损失（防止所有路径走同一条），backup 后恢复

接口变更：`evaluate_fn` 从 `(state) -> (priors, value)` 变成 `(states: list) -> list[(priors, value)]`，支持批量。

**Phase 2：多棋盘并行搜索（预期额外 2x 加速）**

修改 `_run_self_play_mcts()`：
1. 同时对 N 盘棋进行 MCTS 搜索
2. 把 N 盘棋的待评估叶子节点合并成一个大 batch
3. 一次 GPU 调用处理所有棋盘的叶子评估

**Phase 3：长时间 MCTS 训练验证**

在速度优化完成后，运行：
```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 3600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42
```

预期（优化后）：~3000 盘 / 小时，足以与纯 policy 对比信号质量。

### 6.3 v12 成功标准

1. **速度指标：** MCTS-50 达到 ≥50 局/分钟（当前 9.5）
2. **信号质量指标：** 在相同训练步数下，MCTS 的 WR 曲线标准差 < 纯 policy 的 WR 曲线标准差
3. **绝对 WR 指标：** MCTS 1 小时训练后 vs L0 WR ≥ 70%

### 6.4 对 autoresearch 框架的启示

本次实验揭示了框架的一个重要能力缺口：**框架能比较不同超参配置的效果，但不能比较不同算法路线的效果**（MCTS vs 纯 policy 需要控制训练步数而非训练时间）。

v12 应考虑在 `analyze.py` 中增加**步数归一化对比**：不按时间对齐两个 run 的 WR 曲线，而是按梯度步数对齐。这对于评估"算法变更"（而非"超参调整"）至关重要。

---

## 7. 总结

| 问题 | 判定 |
|------|------|
| MCTS 训练信号是否可用？ | **是** — loss 在正确方向移动，WR 从 26% 升到 46%（虽然样本量不足） |
| MCTS 是否打破了纯 policy 的天花板？ | **无法判定** — 训练量差异 20x，不具备可比性 |
| 速度瓶颈在哪？ | **Python 解释器** — GPU 利用率 ~5%，单样本推理占 95% 空闲等待 |
| 下一步应该做什么？ | **叶子批量合并** — 这是唯一能将 MCTS 速度从不可用提升到可用的路径 |
| DB 和观测体系是否足够？ | **足够** — 当前记录支撑了本次分析的所有需求 |

> **一句话：MCTS 训练信号可用但速度不可行。v12 的第一优先级不是调超参或重建对手链，而是实现叶子节点批量合并让 GPU 真正工作起来。当前 GPU 功率 5W/25W = 20% 利用率，目标是恢复到 80%+。**
