# v12 Update — Action Plan

> 2026-04-12  
> 前置：[v11-findings.md](./v11-findings.md) (MCTS 速度瓶颈实证)、[v11-update.md](./v11-update.md) (v12 scope 原始定义)、[v11-analysis-by-opus.md](./v11-analysis-by-opus.md) (训练信号分析)、[pareto-frontier.md](./pareto-frontier.md) (加速机制设计)  
> 数据：[mcts_1st_exp.db](./mcts_1st_exp.db) (3 runs, M3 Max)

---

## 1. 版本目标

**一句话：让 MCTS 训练从"信号可用但速度不可行"变成"信号可用且速度可行"。**

v11 证明了：
- MCTS 训练信号有效（loss 下降方向正确，WR 26%→46%）
- 但速度不可行（9.5 局/分钟 vs 纯 policy 的 182 局/分钟，GPU 利用率仅 5%）
- 瓶颈在 Python 解释器（95% 时间等 Python 遍历树，GPU 空闲）

v12 的任务是把 MCTS-50 从 **9.5 局/分钟** 提升到 **≥50 局/分钟**（5x 加速），恢复 GPU 利用率到 **≥60%**。

---

## 2. Phase 总览

| Phase | 内容 | 改动文件 | 预期加速 | 累计 |
|-------|------|---------|---------|------|
| 1 | 叶子节点批量合并 | `core/mcts.py` + `train.py` adapter | 5-10x | 5-10x |
| 2 | 多棋盘并行搜索 | `train.py` (`_run_self_play_mcts`) | 2-4x | 10-20x |
| 3 | 训练循环早停 | `train.py` (训练主循环) | N/A（节省浪费） | — |
| 4 | 步数归一化对比 | `analyze.py` | N/A（分析能力） | — |

Phase 1+2 是速度核心。Phase 3+4 是框架增强。

---

## 3. Phase 1：叶子节点批量合并（Leaf Batching）

### 3.1 原理

当前 `mcts_search()` 的 50 次模拟是串行的：

```
sim 1: select → expand → evaluate(batch=1) → backup
sim 2: select → expand → evaluate(batch=1) → backup
...
sim 50: select → expand → evaluate(batch=1) → backup
```

GPU 被调用 50 次，每次只处理 1 个样本。M3 Max 的 Neural Engine / GPU 处理 batch=1 和 batch=64 的时间几乎相同（~0.1ms），这意味着 98% 的 GPU 算力被浪费。

**批量合并的思路：** 不再每次 expand 立即 evaluate，而是攒够一批叶子节点后一次性 batch evaluate。用"虚拟损失"（virtual loss）让多条搜索路径不会走到同一个节点。

```
批量搜索 (batch=8):
  path 1: select → 到达 leaf A (标记虚拟损失)
  path 2: select → 到达 leaf B (路径被虚拟损失引导到不同节点)
  ...
  path 8: select → 到达 leaf H
  batch evaluate([A, B, ..., H])  ← 一次 GPU 调用处理 8 个样本
  expand + backup 所有 8 条路径（撤销虚拟损失）
```

50 次模拟 / 8 并发 = ~6 次 GPU 调用（vs 当前 50 次）。GPU 每次处理 batch=8 vs batch=1 = **~8x 效率提升**。

### 3.2 改动：`framework/core/mcts.py`

**新增 `mcts_search_batched()` 函数**，保留原 `mcts_search()` 不变（向后兼容）。

```python
def mcts_search_batched(
    root_state: Any,
    evaluate_batch_fn: Callable[[list[Any]], list[tuple[np.ndarray, float]]],
    copy_fn, legal_mask_fn, apply_fn, terminal_fn, terminal_value_fn,
    action_size: int,
    num_simulations: int,
    batch_size: int = 8,       # 每批并发搜索路径数
    virtual_loss: float = 3.0, # 虚拟损失值
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_frac: float = 0.25,
) -> np.ndarray:
```

**关键变更点：**

1. `evaluate_fn` → `evaluate_batch_fn`：接受 `list[state]`，返回 `list[(priors, value)]`
2. 新增 `virtual_loss` 参数：在 select 阶段对选中节点施加虚拟损失，防止并发路径聚集
3. 搜索循环从 `for _ in range(N)` 变为 `for batch in range(N // batch_size)`

**MCTSNode 新增方法：**

```python
def apply_virtual_loss(self, vl: float):
    """Select 阶段施加虚拟损失，让并发路径避开此节点。"""
    self.visit_count += 1
    self.value_sum -= vl  # 降低 Q，让其他路径不选这个节点

def revert_virtual_loss(self, vl: float):
    """Backup 前撤销虚拟损失。"""
    self.visit_count -= 1
    self.value_sum += vl
```

**搜索流程：**

```python
remaining = num_simulations
while remaining > 0:
    batch_n = min(batch_size, remaining)
    paths = []          # [(node, sim_state)] per path
    leaf_nodes = []     # 需要 evaluate 的叶子节点
    leaf_states = []    # 对应的 state
    terminal_results = [] # 已到终态的 (node, value) 对

    # 1. Select batch_n 条路径（带虚拟损失）
    for _ in range(batch_n):
        node = root
        sim_state = copy_fn(root_state)
        search_path = [node]
        while node.is_expanded and not terminal_fn(sim_state):
            node.apply_virtual_loss(virtual_loss)
            node = node.select_child(c_puct)
            apply_fn(sim_state, node.action)
            search_path.append(node)
        node.apply_virtual_loss(virtual_loss)
        
        if terminal_fn(sim_state):
            terminal_results.append((search_path, terminal_value_fn(sim_state)))
        else:
            leaf_nodes.append(node)
            leaf_states.append(sim_state)
            paths.append(search_path)

    # 2. Batch evaluate 所有非终态叶子
    if leaf_states:
        results = evaluate_batch_fn(leaf_states)
        for (node, sim_state, path), (priors, value) in zip(
                zip(leaf_nodes, leaf_states, paths), results):
            lm = legal_mask_fn(sim_state)
            node.expand(priors, lm)
            leaf_value = -value  # negate: NN value is current player, we want parent's
            # 撤销虚拟损失并 backup
            for n in path:
                n.revert_virtual_loss(virtual_loss)
            node.backup(leaf_value)

    # 3. 处理终态路径
    for path, tv in terminal_results:
        for n in path:
            n.revert_virtual_loss(virtual_loss)
        path[-1].backup(tv)

    remaining -= batch_n
```

### 3.3 改动：`domains/gomoku/train.py` adapter

更新 `mcts_search()` 包装函数，提供 `evaluate_batch_fn`：

```python
def _evaluate_batch(states):
    """批量评估多个棋盘状态。一次 MLX 前向传播。"""
    encodings = np.stack([s.encode() for s in states])  # [B, 3, 15, 15]
    enc_mx = mx.array(encodings)
    logits, values = model(enc_mx)
    mx.eval(logits, values)
    priors_batch = np.array(mx.softmax(logits, axis=-1))  # [B, 225]
    values_np = np.array(values).flatten()                 # [B]
    return [(priors_batch[i], float(values_np[i])) for i in range(len(states))]
```

### 3.4 测试命令

```bash
# Phase 1 验证：叶子批量合并 + 50 sims
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 对比基线（v11 的串行 MCTS 已被替换，用纯 policy 做对照）
uv run python domains/gomoku/train.py \
  --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42
```

**Phase 1 成功标准：**

| 指标 | v11 MCTS-50 | Phase 1 目标 |
|------|-------------|-------------|
| 局/分钟 | 9.5 | ≥40 |
| Cycle 耗时 | 50s | ≤12s |
| GPU 功率 | ~5W | ≥15W |
| TUI Sim/s | ~500 | ≥3000 |

---

## 4. Phase 2：多棋盘并行搜索

### 4.1 原理

Phase 1 解决了单盘搜索内的 GPU 利用率。但 `_run_self_play_mcts()` 仍然是 8 盘棋逐个串行搜索。

Phase 2 让 8 盘棋同时搜索，每盘各自走 select 到叶子，**跨棋盘合并叶子节点**成一个更大的 batch。

```
Board 0: select → leaf A
Board 1: select → leaf B
Board 2: select → leaf C
...
Board 7: select → leaf H
batch evaluate([A, B, C, ..., H])  ← 一次 GPU 调用处理 8 个不同棋盘的叶子
```

与 Phase 1 叠加：如果每盘并发 4 条路径 × 8 盘 = batch=32 的 GPU 调用。

### 4.2 改动：`domains/gomoku/train.py`

修改 `_run_self_play_mcts()` 主循环：

- 从 `for game_idx in range(num_games)` 逐盘串行
- 改为维护 N 个活跃 Board 对象，每个周期同时推进所有未结束的棋盘
- 共享 `evaluate_batch_fn`，跨棋盘合并叶子

### 4.3 测试命令

```bash
# Phase 2 验证：多棋盘并行 + 叶子批量
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 16 --time-budget 600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42
```

**Phase 2 成功标准：**

| 指标 | Phase 1 预期 | Phase 2 目标 |
|------|-------------|-------------|
| 局/分钟 | ≥40 | ≥80 |
| 总训练盘数/10min | ~400 | ~800 |
| GPU 功率 | ≥15W | ≥20W |

---

## 5. Phase 3：训练循环早停

### 5.1 原理

v11 的 `--stagnation` 是分析工具（事后检测）。Phase 3 将其嵌入训练循环：当 WR 停滞时自动停止训练，避免 b3f99d4f 式的 57,600 盘浪费。

### 5.2 改动：`domains/gomoku/train.py` 训练循环

在 probe evaluation 后增加停滞检查：

```python
# 在 probe eval 之后，检查最近 N 个 eval 的趋势
if len(wr_history) >= stagnation_window:
    recent = wr_history[-stagnation_window:]
    xs = list(range(len(recent)))
    slope, _, r2 = _linear_regression(xs, recent)
    wr_std = (sum((w - sum(recent)/len(recent))**2 for w in recent) / len(recent))**0.5
    expected = abs(slope * len(recent))
    if r2 < 0.15 and expected < wr_std:
        _log_event(f"⚠ Stagnation detected: WR plateau for {stagnation_window} evals")
        if auto_stop_on_stagnation:
            stop_reason = "stagnation"
            break
```

### 5.3 CLI flag

```
--stagnation-window N  (默认 10，检测窗口)
--auto-stop-stagnation (默认关闭，启用后自动停止)
```

### 5.4 测试命令

```bash
# 纯 policy 长训练 + 早停
uv run python domains/gomoku/train.py \
  --parallel-games 8 --time-budget 3600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation --stagnation-window 10 --seed 42

# MCTS 长训练 + 早停
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 3600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation --stagnation-window 10 --seed 42
```

---

## 6. Phase 4：步数归一化对比

### 6.1 原理

v11-findings 揭示：按时间对比 MCTS vs 纯 policy 是不公平的（MCTS 每步耗时 19x）。需要按梯度步数对齐 WR 曲线。

### 6.2 改动：`framework/analyze.py`

新增 `--compare-by-steps RUN_A RUN_B` 命令：

- 从 `cycle_metrics` 读取 `(total_steps, win_rate)` 序列
- 按 `total_steps` 对齐两个 run 的 WR 曲线
- 输出对齐后的对比表

### 6.3 测试命令

```bash
# 步数归一化对比 MCTS vs 纯 policy
python3 framework/analyze.py --compare-by-steps <mcts_run> <pure_run>
```

---

## 7. In-scope / Out-of-scope

### In-scope

1. `core/mcts.py` 新增 `mcts_search_batched()` + 虚拟损失机制
2. `train.py` adapter 新增 `evaluate_batch_fn`（MLX batch forward）
3. `train.py` `_run_self_play_mcts()` 改为多棋盘并行
4. `train.py` 训练循环早停（`--auto-stop-stagnation`）
5. `analyze.py` 步数归一化对比（`--compare-by-steps`）
6. 向后兼容：`mcts_search()` 保留，`MCTS_SIMULATIONS=0` 行为不变

### Out-of-scope

1. **Cython/Numba 热路径优化** — 如果 Phase 1+2 达标则不需要
2. **MLX 原生 MCTS（张量树）** — 工程复杂度过高，Phase 1+2 足够
3. **对手进化链重建** — 依赖速度优化后的长训练验证
4. **Pareto 集成到 report** — 独立命令已可用，集成延后
5. **自适应采样 / 渐进精化** — 属于 sweep 层面的优化，非 MCTS 关键路径

---

## 8. 文件改动汇总

| 文件 | Phase | 改动类型 | 预估 |
|------|-------|---------|------|
| `framework/core/mcts.py` | 1 | 新增 `mcts_search_batched()` + 虚拟损失方法 | +120 行 |
| `domains/gomoku/train.py` | 1,2,3 | adapter 更新 + 多棋盘并行 + 早停 | +100 行改写 |
| `framework/analyze.py` | 4 | 新增 `--compare-by-steps` | +60 行 |

总改动量：~280 行新增/改写。核心逻辑集中在 `mcts_search_batched()` 的 ~80 行。

---

## 9. 执行顺序与验证流程

### 开发机（Linux，无 MLX）

```bash
# 1. Phase 1: 实现 mcts_search_batched + 虚拟损失
# 2. 纯 Python 单元测试（mock evaluate_batch_fn）
python3 -c "
from framework.core.mcts import mcts_search_batched, MCTSNode
# ... 验证 visit 分布在一步必胜局面上集中
"

# 3. Phase 3: 训练循环早停
# 4. Phase 4: 步数归一化对比
# 5. AST 语法检查全部文件
# 6. commit + push
```

### macOS Apple Silicon（M3 Max）

```bash
git pull origin main && uv sync

# ===== Phase 1 验证 =====

# 测试 1: MCTS-50 批量合并 (5 min)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 测试 2: 纯 policy 对照 (5 min)
uv run python domains/gomoku/train.py \
  --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 查看速度对比
uv run python framework/analyze.py --runs

# ===== Phase 2 验证（如果 Phase 1 达标）=====

# 测试 3: MCTS-50 多棋盘并行 (10 min)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 16 --time-budget 600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# ===== 长训练验证（如果 Phase 1+2 达标）=====

# 测试 4: MCTS-50 1 小时训练
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 3600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation --seed 42

# 测试 5: MCTS-200 1 小时训练
uv run python domains/gomoku/train.py \
  --mcts-sims 200 --parallel-games 8 --time-budget 3600 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation --seed 42

# ===== 分析 =====

# 步数归一化对比
uv run python framework/analyze.py --compare-by-steps <mcts_run> <pure_run>

# 停滞检测
uv run python framework/analyze.py --stagnation <run_id>

# Pareto 更新
uv run python framework/analyze.py --pareto

# 带回 DB
cp output/tracker.db /path/to/share/v12_test.db
```

---

## 10. 成功标准

### Phase 1+2 速度

| 指标 | v11 实测 | v12 目标 | 判定 |
|------|---------|---------|------|
| MCTS-50 局/分钟 | 9.5 | ≥50 | TUI `Gm/s` × 60 |
| MCTS-50 Cycle 耗时 | 50s | ≤10s | TUI 观察 |
| TUI Sim/s | ~500 | ≥3000 | TUI MCTS 行 |
| GPU 功率 | 5W | ≥15W | macOS Activity Monitor |

### 信号质量（长训练后判定）

| 指标 | 纯 policy 基线 | MCTS 目标 |
|------|---------------|----------|
| WR 曲线形态 | 震荡（std ~6%） | 收敛（std < 4%） |
| 1h 后 vs L0 WR | ~76% | ≥70%（步数归一化后可比） |
| Entropy 趋势 | N/A | 下降 |
| 停滞检测 | 触发 | 不触发 |

### 整体 v12 verdict

v12 成功 = Phase 1 速度达标 + 长训练 WR 曲线呈收敛趋势。  
如果达成，v13 可以开始重建对手进化链和 Pareto 集成。  
如果 Phase 1 速度不达标，考虑 Cython 热路径作为后备。

---

## 11. 工作日志

> 执行者：Claude Opus 4.6  
> 执行日期：2026-04-12  
> 环境：Linux 开发机（无 MLX，逻辑测试 + mock 验证）

### 11.1 Phase 1：叶子批量合并

**framework/core/mcts.py** (167 → 294 行, +127 行)

新增内容：

1. `MCTSNode.apply_virtual_loss(vl)` / `revert_virtual_loss(vl)` — 虚拟损失施加/撤销
2. `mcts_search_batched()` 函数 (~110 行) — 完整的批量 MCTS 搜索：
   - 每轮收集 `batch_size` 条搜索路径（带虚拟损失防止路径聚集）
   - 区分终态路径和需要 evaluate 的叶子节点
   - 一次 `evaluate_batch_fn` 调用处理所有叶子
   - 撤销虚拟损失后执行真实 backup
3. 保留原 `mcts_search()` 不变（向后兼容）

**测试结果：**
- 200 sims, batch_size=16: **13 次 GPU batch 调用** vs 串行 200 次 (15x 减少)
- 500 sims, batch_size=16: **18 次 GPU batch 调用** vs 串行 500 次 (28x 减少)
- 一步必胜局面：两个获胜走法合计获得 55% visit（与串行版 59% 接近）

### 11.2 Phase 1 续：Gomoku adapter 更新

**domains/gomoku/train.py** 适配改动：

1. 新增 `MCTS_BATCH_SIZE = 8` 和 `MCTS_VIRTUAL_LOSS = 3.0` 常量
2. `mcts_search()` 从调用 `_mcts_search_generic` 改为调用 `_mcts_search_batched`
3. 新增 `_evaluate_batch(states)` — 将多个 Board 编码成 numpy batch，一次 MLX 前向传播
4. import 更新：添加 `mcts_search_batched as _mcts_search_batched`

### 11.3 Phase 2：多棋盘并行自对弈

**`_run_self_play_mcts()` 完全重写：**

- 旧版：`for game_idx in range(num_games)` 逐盘串行
- 新版：所有 N 盘棋同时维护，每轮推进所有活跃棋盘各一步
- 每轮结束后调用 `mx.clear_cache()` 清理 Metal 缓存
- 棋局完成后立即标记 finished，不再参与后续搜索
- MCTS stats（sims/sec, entropy, focus）汇总方式不变

### 11.4 Phase 3：训练循环早停

1. 新增 CLI flags：`--auto-stop-stagnation` (默认关闭) + `--stagnation-window N` (默认 10)
2. 在 probe eval 之后检测最近 N 个 eval 的 WR 趋势
3. 判定逻辑：对窗口内 WR 做线性回归，如果 `expected_change < wr_std` 且 `wr_std > 1%`，判定为停滞
4. 停滞时 `stop_reason = "stagnation"`，触发正常的 run 结束流程

### 11.5 Phase 4：步数归一化对比

**framework/analyze.py** (+100 行)

1. `cmd_compare_by_steps(conn, run_a, run_b)` — 按 `total_steps` 对齐两个 run 的 WR 曲线
2. 使用线性插值在公共步数范围内采样 WR
3. 输出对齐后的 WR 对比表 + 汇总（A 赢 / B 赢统计）
4. CLI：`--compare-by-steps RUN_A RUN_B`

**对 mcts_1st_exp.db 测试结果：**
```
MCTS-50 vs Pure: 0-123 步范围内，Pure 领先所有 5 个采样点
但差距从 36% 缩小到 22%（MCTS 信号质量的间接证据）
```

### 11.6 最终文件变化

| 文件 | 原行数 | 新行数 | 变化 |
|------|--------|--------|------|
| `framework/core/mcts.py` | 167 | 294 | +127 |
| `domains/gomoku/train.py` | 1786 | 1835 | +49 |
| `framework/analyze.py` | 1195 | 1313 | +118 |
| **合计** | 3148 | 3442 | **+294** |

### 11.7 验证清单

- [x] 12 个 Python 文件全部通过 AST 语法检查
- [x] `mcts_search_batched()` mock 测试通过（200 sims → 13 GPU calls）
- [x] `--compare-by-steps` 对 mcts_1st_exp.db 运行正确
- [x] `--auto-stop-stagnation` 和 `--stagnation-window` CLI 解析正确
- [x] `MCTS_SIMULATIONS=0` 行为不变（向后兼容）
- [x] 原 `mcts_search()` 函数保留（供 play.py 等非训练场景使用）
- [ ] Apple Silicon 上的实际速度验证（待 Mac 测试）
- [ ] GPU 功率验证（待 Mac 测试）
