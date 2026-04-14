# v15 Findings — mcts_12 实测复盘 + v15.2 batched eval 修复

> 2026-04-13 晚 | 基于 `updates/mcts_12th_exp.db` (run `3c638319`) + `updates/mcts_12th.png`
> 前置：v15-update.md §11-13（v15 + v15.1 hotfix）

---

## 1. 一句话结论

> **mcts_12 暴露了 v15 设计里一个我没考虑到的根本性 bottleneck：`_in_process_eval` 是 sequential single-board 的，每次 NN 移动都做一次 batch=1 MLX forward。Mac M3 Max 上 batch=1 MLX 的 dispatch latency 是 ~250-500 ms（vs Linux fake-NN 测试里的 ~0 ms），所以 1800 个 NN 调用 × 250 ms = 7+ 分钟，而不是我预测的 13 秒。修复方法是 v15.2 里的 batched eval —— 把所有并行 game 的 NN 调用合并成一次 forward。同时 mcts_12 还触发了 3 个次生问题（`eval_submitted_cycle` 没真的落库；auto-promote 没在 Ctrl+C 路径上跑通；用户对晋升机制的预期与实际不符）。本章把 4 个问题和它们的修复一次性记下来。**

---

## 2. mcts_12 的事实清单

### 2.1 Run 元数据（`runs` 表）

| 字段 | 值 |
|------|-----|
| Run ID | `3c638319-2f7f-41ba-8882-883c7838602c` |
| Resumed from | `6c9c8bdd-67bd-4405-97b9-0533970ec12a` (mcts_10 final S1v2) |
| 架构 | 8×128 (2.49M params) |
| MCTS sims | **800**（DB 确认；TUI 显示 "MCTS 80" 是因为日志行被 box 宽度截断了 "0sims [C-native] | minimax [c]" 后缀）|
| Parallel games | 16 |
| LR / steps / buffer | 3e-4 / 50 / 100,000 |
| time_budget | **12,000 s ≈ 3.33 h**（用户实际用的预算，不是我们建议的 21,600）|
| eval_level | 2 |
| Status | **`running`**（未调到 `finish_run`！见 §4） |
| Seed | 42 |

### 2.2 实际跑了多少

从 `cycle_metrics`：

| 指标 | 值 |
|------|-----|
| Cycles 范围 | 177 → 192（resume 之后 16 个 cycle）|
| 总 cycle 行数 | 18（含 2 个 probe row 和 16 个普通 cycle row）|
| 总 game 数 | 256（最后 cycle 里的 `total_games`）|
| 总 train steps | 257 |
| Wall clock | ~55 分钟（从 `Started run` 18:39:05 到 `Training interrupted` 19:34:19）|

### 2.3 Probe eval 时间线（来自 TUI 事件日志）

```
18:39:05 Started run 3c638319 | 2490.0K params | budget 12000s | MCTS 80...
19:11:03 Probe c184: 55.3% (150g vs L2, 32u/16o, 1120.3s)   ← 第 1 个 probe
19:33:36 Probe c192: 94.0% (avg:74.7%) (150g vs L2, 32u/16o, 657.9s)   ← 第 2 个 probe
19:34:19 Training interrupted by user
```

**Probe 实测耗时：**

| Probe | Cycle | Games | WR | unique | wall time |
|-------|-------|-------|-----|--------|-----------|
| 第 1 次 | 184 (resume +8) | 150 | 55.3% | 32/16 | **1120 s ≈ 18.7 分钟** |
| 第 2 次 | 192 (resume +16) | 150 | 94.0% | 32/16 | **658 s ≈ 11.0 分钟** |

**v15-update.md §13.7 的预测是 ~13 秒**（80 games）→ ~25 秒（150 games）。**实测比预测慢 25-45 倍。**

### 2.4 Checkpoint 表

```
=== CHECKPOINTS (run 3c638319) ===
  (空)
```

**0 个 checkpoint。** 没有任何 `wr085_c*` / `wr090_c*` / `wr095_c*` / `wr100_c*` 被保存。

### 2.5 用户的两个直接问题

> Q1：为什么 eval 占用近 20 分钟？这和 v15 的设计指标差别太大。
>
> Q2：第二次 probe 已经达到 90% vs L2，为什么对手没有自动晋级？

下面 §3 / §4 / §5 分别回答。

---

## 3. Q1 根因 — `_in_process_eval` 是 sequential single-board，Mac MLX batch=1 latency 杀手

### 3.1 我在 Linux 上测的是什么

v15-update.md §11.2 报告的 "80-game probe vs L2 ~13 秒" 是这样测出来的：

```python
# fake "NN": always picks the cell closest to center
def fake_nn_move(board):
    return min(board.get_legal_moves(), key=lambda rc: abs(rc[0]-7)+abs(rc[1]-7))

# Then run _in_process_eval-style game loop with this fake NN + real C minimax
```

**这个 fake NN 完全没用 MLX。** 它只是 numpy 操作，每次走子大约 100 微秒。所以我的 13 秒里**不包含任何真实 NN forward 的 dispatch 开销**——只有 Linux 上 C minimax 的 ~15 ms/call。

### 3.2 真实运行时是什么样

`_in_process_eval` 的真实 NN 路径（v15.1 sync 版本，未修）：

```python
for game_i in range(n_games):                          # ← sequential
    while not board.is_terminal():
        if board.current_player == nn_player:
            encoded = board.encode()                    # numpy [3,15,15]
            x = mx.array(encoded[np.newaxis, ...])      # MLX batch=1
            policy_logits, _ = model(x)                 # MLX forward, batch=1
            ...
            action = int(mx.argmax(masked).item())      # SYNC POINT
            ...
        elif use_nn_opponent:
            ...
        else:
            row, col = opponent_fn(board)               # C minimax
        board.place(row, col)
```

每个游戏每次 NN 移动都做一次 `model(x)` with batch shape `(1, 3, 15, 15)`. 这个 batch=1 的 MLX 调用在 Mac M3 Max 上有几个固定的 latency 来源：

1. **Metal kernel dispatch overhead**：MLX 对小 batch 的 GPU 命令编码开销 30-100 ms 量级
2. **8x128 ResNet on batch=1 是大材小用**：GPU 90% 时间都在等命令排队，10% 在算
3. **`.item()` 强制同步**：每次 argmax 都要等 GPU 全部完成
4. **`mx.array(numpy)` host→device 拷贝**：每次都新建一个 1×3×15×15 数组

我估计 Mac 上每次 batch=1 NN call 是 **150-400 ms**（具体值取决于 MLX 版本和 chip 状态）。

### 3.3 算笔账，看看是否对得上

mcts_12 probe 1：150 games × ~25 moves avg = 3750 总 moves，其中 NN 走 ~1875，opponent 走 ~1875。

- 假设 NN per call = 250 ms，minimax per call = 50 ms（C 后端）
- NN 总时间：1875 × 0.25 = **469 sec**
- Minimax 总时间：1875 × 0.05 = **94 sec**
- 两者加起来：**~563 sec = 9.4 min**

probe 1 实测 18.7 分钟，比上面估计还高一倍。可能 NN per call 实际是 ~400 ms，或者一些初始化开销。但**数量级完全对得上**：sequential batch=1 是 8-15 分钟级别的，不是 13 秒级别的。

probe 2 = 11 分钟 < probe 1 = 18.7 分钟 也符合预期：第一次 probe MLX 还要 JIT-compile 一些 kernel，第二次预热好了。

### 3.4 为什么 Linux 测试没暴露这个

**因为 Linux dev 机没有 MLX**——我在 Linux 上根本无法跑真的 GPU forward，只能用 numpy fake_eval 替代。**fake_eval 没有 MLX dispatch 开销，所以测出来的"13 秒" 完全没考虑这部分**。

这是 v15 留下的最大盲区。**任何依赖 MLX 性能的修改都必须 Mac 实机验证才算 land**——v15.1 hotfix 已经记录过一次同样的教训（threading + Metal command buffer assertion），但 v15.2 这次又一次因为同样的原因栽跟头：Linux 测试给了虚假的安全感。

### 3.5 修复 — v15.2 batched `_in_process_eval`

把 sequential single-board loop 改成 wave-based concurrent loop：

```python
boards = [Board() for _ in range(n_games)]
nn_player_of = [BLACK if i < n_games//2 else WHITE for i in range(n_games)]
finished = [False] * n_games
# apply opening seeds in bulk

while not all(finished):
    # Collect boards where NN is to move RIGHT NOW
    nn_turn_idx = [i for i in range(n_games) if not finished[i]
                                              and not boards[i].is_terminal()
                                              and boards[i].current_player == nn_player_of[i]]

    if nn_turn_idx:
        # ── ONE batched MLX forward for all of them ──
        encodings = np.stack([boards[i].encode() for i in nn_turn_idx])
        x = mx.array(encodings)  # batch ~ n_games/2 = 75
        policy_logits, _ = model(x)
        mx.eval(policy_logits)
        policy_np = np.array(policy_logits)
        for k, i in enumerate(nn_turn_idx):
            # Apply argmax → place stone on boards[i]
            ...

    # Sequential opponent moves (C minimax — already fast)
    for i in opp_turn_idx:
        row, col = opponent_fn(boards[i])
        boards[i].place(row, col)

    # Tick / check finished
    for i in range(n_games):
        if not finished[i] and boards[i].is_terminal():
            finished[i] = True
```

**预期效果（Mac M3 Max）**：

| 配置 | v15.1 sequential | **v15.2 batched** | 加速 |
|------|------------------|------------------|------|
| 80-game probe vs L2 | ~10 min（实测推算）| **~30-60 秒** | **10-20×** |
| 150-game probe vs L2 | ~18 min（mcts_12 实测）| **~60-120 秒** | **9-18×** |
| 200-game full eval vs L2 | ~25 min（实测推算）| **~80-160 秒** | **9-19×** |

加速来源：把 1875 次 batch=1 forward → ~25 次 batch=75 forward。每次 forward 的 dispatch 开销固定 30-100 ms，因此从 1875 × 250 ms = 469 sec → 25 × 100 ms = 2.5 sec。**MLX dispatch 开销几乎消失**。

C minimax 部分不变，仍是 ~50-100 sec for 1875 calls. 这部分以后可以也批量化或多线程（C 释放 GIL）—— v16 的事。

### 3.6 v15.2 修复已落地

- `domains/gomoku/train.py` 的 `_in_process_eval` 已经改写成 wave-based batched 版本
- 接口（参数 + 返回 dict）100% 向后兼容，调用方无需修改
- 新增返回字段 `nn_batch_calls`（诊断：本次 eval 总共做了几次 batched forward），可写入未来版本的 `cycle_metrics`
- Linux 上 27/27 测试仍然通过（test 不涉及 MLX 路径）
- **必须在 Mac 上实机验证一次才算真正 land**——Linux 测试无法保证

### 3.7 Mac 实机验证步骤

在用户的 Mac 上跑一次 1 分钟 smoke test：

```bash
cd /path/to/project
git pull origin main
cd domains/gomoku && bash build_native.sh && cd ../..
cd framework/core && bash build_native.sh && cd ../..

# 验证 backend：第一行应该出现 "minimax [c]"
uv run python -c "
import sys; sys.path.insert(0, 'framework'); sys.path.insert(0, 'domains/gomoku')
import prepare
print('minimax backend:', prepare.MINIMAX_BACKEND)
"

# 1 分钟 smoke probe — 用真实模型跑 32 局 vs L2
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 600 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 32 \
  --full-eval-games 64 --eval-openings 16 \
  --resume 6c9c8bdd --seed 42
```

**期望看到的事**：
1. 启动事件日志包含 `minimax [c]`（不是 `[python]`）
2. 第一个 probe（cycle 181）的事件日志：`Probe c181: X% (32g vs L2, ..., **TT.Ts**)` 其中 `TT.T` 应该是 **20-60 秒**之间，不是 5+ 分钟
3. 如果 `TT.T` 仍然 > 3 分钟，停下来，把日志贴回来

如果 batched eval 不够快，下一步是 v15.3：**也批量化 minimax**（每个 wave 的所有 opponent 移动并行，C 函数内部多线程 + 释放 GIL）。但先看 v15.2 的效果。

---

## 4. Q2 根因 — auto-promote 没运行的三层原因

用户问 "第二次 probe 90% 为什么没有 auto-promote 到 S2"。这里有 **三个独立的问题**叠加：

### 4.1 第一层：smoothed WR 没越过最低阈值 0.85

mcts_12 跑了 2 个 probe：
- 第 1 个：raw WR = 55.3%，smoothed (1 sample) = 55.3%
- 第 2 个：raw WR = 94.0%，smoothed (2 samples avg) = **74.65%**

`crossed_threshold(effective_wr, last_ckpt_wr)` 检查 smoothed WR 是否越过 `CHECKPOINT_THRESHOLDS = [0.85, 0.90, 0.95, 1.00]`。第 2 个 probe 的 smoothed 是 74.65%，**没有越过 0.85**——所以**没有任何 checkpoint 被触发**。

注意：smoothed WR 用的是 sliding average of recent N probes (`probe_window = 5`)。两个 probe 的 average 才两个样本，单次 94% 被前一次 55% 拉平。这是**故意的设计**——单次 probe 噪声大，需要 smoothing。

→ 用户期望"94% 单次 probe 应当触发晋升"，但 v15 的设计不是这样的。**checkpoint 触发用 smoothed WR**，需要至少 3-5 个 probe 才能有可信的 average。

### 4.2 第二层：threshold 0.85 vs `can_promote` 阈值 0.60 (L2) 是两件不同的事

我在 v15 同时引入了两套阈值，它们 **不是同一个东西**：

| 名字 | 在哪 | 用途 | 值（vs L2）|
|------|------|------|-----------|
| `CHECKPOINT_THRESHOLDS` | `framework/core/db.py` | 触发 checkpoint **保存** | `[0.85, 0.90, 0.95, 1.00]` |
| `PROMOTION_WR_THRESHOLD_BY_LEVEL[2]` | `framework/core/db.py:can_promote` | 决定一个**已存在**的 checkpoint 是否 `eligible` | **0.60** |

晋升流程是 **两步串联**：

1. 训练中 smoothed WR 越过 `CHECKPOINT_THRESHOLDS` 中的某个值 → **触发 full eval → 保存一份 checkpoint**
2. 训练结束（或 Ctrl+C 后的 finalize）→ 遍历该 run 的 **所有 checkpoint**，对每一个调 `can_promote()`：
   - L2 时只要 `win_rate ≥ 0.60` + unique ≥ 16 + avg_len ∈ [12,60] + 最近 5 个 smoothed WR 跨度 ≤ 15% → eligible
3. 选最新 (cycle 最大) 的 eligible checkpoint，复制到 `output/opponents/<alias>/`，写 `opponents` 表

**关键点：第 2 步只对第 1 步保存下来的 checkpoint 起作用。** mcts_12 第 1 步从未触发（因为 smoothed 没到 0.85），所以第 2 步**没有任何东西可以晋升**。`can_promote` 的 0.60 阈值根本没机会生效。

这是 **设计上的两层 gate**：先要"看起来够强了"才会保存，然后再"严格审一遍才允许晋升"。但用户观察到的是中间一个被卡住了——v15 的 `CHECKPOINT_THRESHOLDS=[0.85,0.90,...]` 对 vs L2 训练来说**第一道门就太紧了**，因为 L2 真实 60% 已经算强了，等到 smoothed 到 85% 可能要训练好几小时。

**修复方向（不在 v15.2 里做，留给后续讨论）：** 让 `CHECKPOINT_THRESHOLDS` 也按 level 分级。例如：

```python
CHECKPOINT_THRESHOLDS_BY_LEVEL = {
    0: [0.85, 0.90, 0.95, 1.00],  # vs L0: 维持 v15 默认
    1: [0.80, 0.90, 0.95, 1.00],
    2: [0.55, 0.65, 0.75, 0.85],  # ← vs L2 大幅下调
    3: [0.40, 0.50, 0.60, 0.70],  # ← vs L3 进一步下调
}
```

这样 mcts_12 在 cycle 192 (smoothed 75%) 就会触发 `wr065_c0192` checkpoint。配合 `can_promote` 的 L2=0.60 阈值，自然就 eligible 了。**这是下一次 v15.3 或 v16 的 0.5 天工作。**

### 4.3 第三层：用户 Ctrl+C 之后 `finish_run` 没执行

DB 里 `runs.status = 'running'`，确认 `finish_run()` 从未被调用。看 train.py 的 try/except/finally 结构：

```python
try:
    while True:  # main training loop
        ...
except KeyboardInterrupt:
    stop_reason = "interrupted"
finally:
    eval_executor.shutdown(...)
    if use_tui: print final TUI

# ──────── try/except/finally 出来之后 ────────
# 1. save_model
# 2. final_eval (BLOCKS for ~10-20 min on Mac due to Q1 bug)
# 3. finish_run                      ← THIS IS WHAT WRITES status='completed'
# 4. --auto-promote-to handling      ← THIS RUNS ONLY IF #3 RAN
```

时间线推断：
1. 19:34:19 用户按 Ctrl+C → `KeyboardInterrupt` 被捕获，stop_reason 设为 "interrupted"
2. finally 块运行（关 executor、重画 TUI），完成
3. 进入 final_eval：开始跑 200 局 vs L2，**因为 v15 sequential eval 慢，这要 ~25 分钟**
4. 用户（合理地）等不及 → **再按 Ctrl+C** → 这次 `KeyboardInterrupt` 没有 try/except 接住，直接传上去 → 进程死掉
5. `finish_run` 从未跑 → status 留在 'running'
6. `--auto-promote-to S2` 也从未跑

**修复（已落地 v15.2）**：把 final_eval 包在 try/except 里，第二次 Ctrl+C 触发时打 warning 并跳过 checkpoint 保存，但**继续往下走 finish_run 和 auto-promote**。

```python
try:
    result = _in_process_eval(model, eval_level, full_eval_games, ...)
except KeyboardInterrupt:
    print("⚠ Final eval interrupted by second Ctrl+C — "
          "skipping checkpoint save, proceeding to finalize.")
    stop_reason = "interrupted"
    result = None
```

这样即使第二次 Ctrl+C，run 也会被正确标记 `status='interrupted'`，并且 `auto-promote-to` 的代码会跑——虽然在 mcts_12 的情况下它会打印 `⚠ no checkpoint is promotion_eligible` 因为没有任何 checkpoint。

### 4.4 Q2 总结的因果链

```
mcts_12 第 2 个 probe 94% raw / 75% smoothed
  → smoothed < 0.85（CHECKPOINT_THRESHOLDS 最低阈值）
    → 没有触发 full eval
      → 没有 checkpoint 被保存
        → 即使训练正常完成 + auto-promote 跑了，也找不到 eligible checkpoint
          → 用户观察到 "auto-promote 没起作用"

平行因果链：
mcts_12 用户 Ctrl+C
  → 进入 final_eval（200 局 × ~10 min on Mac）
    → 用户再次 Ctrl+C
      → KeyboardInterrupt 没人接，进程死
        → finish_run 没跑，status='running'
        → auto-promote-to 那段代码也没跑

两条链合在一起：即使 (a) checkpoint 阈值已经被合理设置 + (b) 用户耐心等完 final_eval，
auto-promote 仍然会因为"没有 eligible checkpoint"而打 warning。
```

### 4.5 Q2 的修复需要 3 件事

| # | 修复 | 状态 |
|---|------|------|
| 1 | `CHECKPOINT_THRESHOLDS` 按 level 分级（vs L2 用 [0.55, 0.65, 0.75, 0.85]）| **延迟到 v15.3 / v16，本次没改** |
| 2 | Final eval try/except KeyboardInterrupt → 让 finish_run + auto-promote 仍然跑 | **v15.2 已落地** |
| 3 | 文档清楚说明 "checkpoint threshold" 和 "promotion gate" 是两层独立的 gate | **本文 §4.2 已写** |

---

## 5. 顺手发现的次生 bug

### 5.1 `eval_submitted_cycle` 列没真的写入

v15 schema migration 加了 `cycle_metrics.eval_submitted_cycle` 列，并且 `_integrate_probe_result` 在传给 `save_cycle_metric` 的 dict 里包含了这个 key——**但 `save_cycle_metric` 的 INSERT 语句没有这个列**。结果是 mcts_12 的 2 个 probe row 这个字段都是 NULL，尽管 dict 里传了正确的值。

**修复（v15.2 已落地）：** `framework/core/db.py:save_cycle_metric` 的 INSERT 列表加上 `eval_submitted_cycle`。

### 5.2 启动日志缺少 `minimax [backend]`

v15.1 启动日志只显示 MCTS 的 backend (`MCTS 800sims [C-native]`)，但**没显示 minimax 的 backend**。Mac 用户没办法在启动时确认 C minimax 是真的加载了 vs 静默回退到 Python。如果用户忘了跑 `bash build_native.sh`，`prepare.MINIMAX_BACKEND` 会是 `'python'`，eval 会慢 ~40×，但 v15.1 不会告诉你。

**修复（v15.2 已落地）：** 启动日志现在包含 `minimax [c]` 或 `minimax [python]`。例如：

```
[18:39:05] Started run 3c638319 | 2490.0K params | budget 12000s | MCTS 800sims [C-native] | minimax [c]
```

### 5.3 TUI 事件日志被 box width 截断

mcts_12 截图里的事件日志显示 `Started run 3c638319 | 2490.0K params | budget 12000s | MCTS 80`——在 "MCTS 80" 之后被截断，所以看不到 "0sims [C-native]" 后缀，差点让我误以为用户实际跑的是 80 sims。

这是 TUI 渲染的固有限制（box 内宽 76 字符），**不算 bug**，但启动日志的内容应该更紧凑。v15.2 把启动日志改成更紧凑的格式，加上 `minimax [c]` 后总长度仍能放下。

---

## 6. mcts_12 的"积极一面"——训练本身没问题

虽然 eval 慢得不能忍，**训练循环本身是健康的**：

| 指标 | 值 | 评价 |
|------|----|------|
| Total cycles (resumed) | 16 (cycle 177→192) | 在 ~55 分钟里走了 16 cycle |
| Self-play 的 Gm/s | 0.1 | 与 mcts_10 / mcts_11 同量级 |
| MCTS Sim/s | 4524 | 与 mcts_10 (5131) 同量级 |
| Focus | 9% | 偏低，但和 resume 起点的策略稀薄一致 |
| Entropy | 4.14 | 高，但仍在改善路径上 |
| P-Loss | 4.542 → 减到 4.18 区间 | 在缓慢下降 |
| V-Loss | 0.197 → 0.36 区间 | 稳定在 [0.2, 0.4] |
| **Probe c184 raw WR** | **55.3% vs L2** | resume 立即就达到 50%+，比 mcts_11 同时点更强 |
| **Probe c192 raw WR** | **94.0% vs L2** | 4 个 cycle 之后就拉到 94%！ |

这是一个 **非常强的训练信号**。如果 eval 不慢、checkpoint 阈值合理，mcts_12 可能在第 3-4 个 probe (cycle 200-208) 就稳定 90%+ vs L2 — 即在 ~1.5 小时里达到 v14-findings §8.9 定的 "smoothed ≥ 60% vs L2" 判据，远早于 3 小时预算。

**模型在快速学习，框架在拖后腿。** 这是 v15 应该解决的问题，v15.2 是对的方向。

---

## 7. v15.2 落地清单

| # | 文件 | 改动 | 验证 |
|---|------|------|------|
| 1 | `domains/gomoku/train.py:_in_process_eval` | 完全重写为 wave-based batched 实现，所有 NN forward 合并成一次 forward | Linux 27/27 测试通过；Mac 实机待 §3.7 |
| 2 | `framework/core/db.py:save_cycle_metric` | INSERT 列表加上 `eval_submitted_cycle` | Linux DB roundtrip 测试 |
| 3 | `domains/gomoku/train.py:Started run` 日志 | 加上 ` | minimax [c]` 后缀 | 视觉检查 |
| 4 | `domains/gomoku/train.py:final_eval` | 包在 try/except KeyboardInterrupt 里，让 finish_run + auto-promote 仍能跑 | 代码审阅 |

**没有改动的（留给 v15.3 / v16）：**

- `CHECKPOINT_THRESHOLDS` 按 level 分级
- minimax 调用本身的并行化 / multiprocessing eval worker
- v15 (B 系) 的 ThreadPoolExecutor / async eval scaffold（保留 dormant，不删）

---

## 8. 给用户的下一次 6 小时训练命令

**v15.2 修复后再跑 mcts_13。完整命令：**

```bash
# 0. 必做的前置
cd /path/to/project
git pull origin main
cd domains/gomoku && bash build_native.sh && cd ../..
cd framework/core && bash build_native.sh && cd ../..

# 1. 验证 backend（必看 minimax [c] 三个字）
uv run python -c "
import sys; sys.path.insert(0, 'framework'); sys.path.insert(0, 'domains/gomoku')
import prepare
print('minimax backend:', prepare.MINIMAX_BACKEND)
"

# 2. 1 分钟 smoke probe（强烈建议跑一次再上 6h）
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 600 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 32 \
  --full-eval-games 64 --eval-openings 16 \
  --resume 6c9c8bdd --seed 42
# 期望：第一个 probe 事件日志的耗时 < 1 分钟（vs mcts_12 的 18.7 min）
# 如果不是，停机，回贴日志

# 3. 6 小时正式训练
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 21600 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --auto-stop-stagnation --stagnation-window 15 \
  --auto-promote-to S2 \
  --resume 6c9c8bdd --seed 42
```

**与 mcts_12 的差异：**

| 参数 | mcts_12 用户实际值 | mcts_13 推荐值 | 理由 |
|------|------|---------|------|
| `--time-budget` | 12000 (3.3 h) | **21600 (6 h)** | 给训练充分时间稳定 smoothed WR |
| `--probe-games` | 150 | **80** | smoke test 验证后可以再调；80 是 v14-findings §8.9 一直用的标准 |
| `--full-eval-games` | (隐含 200?) | **200** | 标准 |
| `--eval-interval` | 8 (推断) | **5** | 更密的 probe → smoothed WR 收敛更快 |
| `--auto-promote-to` | 没传 | **S2** | 让 v15 E5 真的有机会跑 |
| `--auto-stop-stagnation` | 没传 | **on** | 训练若早早达标，自动停 |

**重要预警：** 即使 v15.2 batched eval 起作用，**checkpoint 仍可能不被保存**——因为 `CHECKPOINT_THRESHOLDS` 的 0.85 vs L2 仍然偏紧。如果你想让 mcts_13 一定能 auto-promote 到 S2，建议：

**在跑 mcts_13 之前手动改一行**（v15.3 应该自动化但本版本没做）：

```python
# framework/core/db.py 顶部
# 临时改成 (vs L2 训练专用)：
CHECKPOINT_THRESHOLDS = [0.55, 0.65, 0.75, 0.85]
```

这样 smoothed WR 一旦达到 55% vs L2 就会触发 checkpoint，然后 `can_promote()` 在 60% 阈值上判定为 eligible，auto-promote 就有东西可选了。

**或者**，等 v15.3 把这个改成 per-level 配置，那次再跑。看你倾向哪种节奏。

---

## 9. v15 整体的诚实自评（截至 v15.2）

| 系列 | v15-update 计划 | 实际 land | mcts_12 实测验证 |
|------|----------------|----------|------------------|
| A | bug 修复 + checkpoint 策略 | ✅ 全部 land | A1 跨 level resume **无法验证**（mcts_12 没跑到 checkpoint 触发的点）；A2 4 档阈值 land 但 §4.2 显示对 vs L2 太紧 |
| B | async eval | ❌ 完全失败两次（v15 thread→Metal assertion；v15.2 改回 sync 但又遇到 batch=1 latency）| 需要彻底换设计——multiprocess 或 sync+batched |
| C | minimax C 化 | ✅ Land 且 Linux 测过 | **Mac 上工作正常**——L2 minimax part 不是 mcts_12 的瓶颈，瓶颈是 NN forward |
| E | 观察 + 晋升门禁 | ✅ 全部 land | mcts_12 没跑到产生 checkpoint 的点，所以 E1/E2/E5 **未在 Mac 上验证**；E6/E7 (analyze.py) 已经独立测过 |
| F | v16 预留 | ✅ Land | 未测试 |
| G | README | ✅ Land | 文本审阅通过 |
| **v15.2 hotfix** | (新) batched eval | ✅ Land | **Mac 实机待验证（§3.7）** |

**最大教训**：v15 用 Linux + fake-NN 测试上线是不够的。Linux 测试只能验证算法正确性，**不能验证 MLX 性能特性**。任何涉及"主循环里 model(x) 的次数变化"的设计都必须在 Mac 实机跑一次 smoke test 才能信。

v16 应该把这个变成纪律：**任何性能-relevant 的修改必须有配套的 Mac 1 分钟 smoke test，并在 v{N}-findings 里贴出 Mac 实测的数字**。

---

## 10. 一句话总结（v15.2 时点）

> **mcts_12 的 18 分钟 probe 不是训练问题，是 v15 sequential eval 的 batch=1 MLX dispatch latency 暴露——v15.2 batched eval 应当把它压到 ~1 分钟。Auto-promote 没运行有三个独立原因（smoothed WR 没到 0.85 阈值；阈值表对 vs L2 太紧；用户 Ctrl+C 两次让 finish_run 跳过）——v15.2 修了第三个，前两个留给 v15.3 / v16。同时 mcts_12 的训练数据（55% → 94% 在 4 cycles 内）证明 S1v2 → S2 的训练路径完全有效，瓶颈 100% 在框架，模型本身在快速学习。**

---

## 11. v15.3 findings（2026-04-13）— sys.path bug + 对手强度重估 + 训练效率分析

> 基于 run `7b41a00e`（v15.3 后首个训练）、run `6c9c8bdd`（parent）、Mac M3 Max 128GB 实机分析。
> 前置：v15-update §14（v15.3 hotfix）。

### 11.1 一句话结论

> **v15.3 修了 sys.path 排序 bug（eval 从 213s 降到 2s），但这个修复同时揭露了一个更大的事实：v15.0 以来所有历史训练的 eval 都在用 `framework/prepare.py` 的纯 Python minimax 对手，而非 `domains/gomoku/prepare.py` 的 C minimax 对手。Python minimax 比 C minimax 弱得不是一个数量级——C-L1（depth 2）100 战全胜 Python-L2（depth 4）。这意味着此前所有 "100% vs L1"、"90% vs L2" 的 WR 数据都是对着错误的对手测的，模型的真实水平远低于历史 WR 显示的数字。同时，训练效率分析表明 MCTS 800 sims 自对弈是当前训练的绝对瓶颈（占 99.8% 周期时间），MLX 框架下的优化空间约 15-25%。**

### 11.2 v15.3 修复回顾

详见 v15-update §14。`sys.path.insert(0, framework/)` 把域目录从 `[0]` 推到 `[1]`，导致 `from prepare import OPPONENTS` 始终解析到 `framework/prepare.py`（纯 Python minimax），而非 `domains/gomoku/prepare.py`（C minimax）。

修复后 32-game probe vs L2：213s → 2.0s（109×）。

### 11.3 核心发现：历史 WR 数据全部失效

#### 11.3.1 对手强度对比实测

在 Mac M3 Max 上实测，每组 100 局（双方各执黑白 50 局）：

| 对局 | 胜者 | 比分 | 含义 |
|------|------|------|------|
| Domain L1 (C, depth 2) vs Framework L1 (Python, depth 2) | Domain L1 | **100-0** | C 的评估函数远优于 Python |
| Domain L1 (C, depth 2) vs Framework L2 (Python, depth 4) | Domain L1 | **40-0** (40 局) | C depth 2 强于 Python depth 4 |
| Framework L1 (Python, depth 2) vs Domain L0 (random) | Framework L1 | **100-0** | Python L1 ≈ 比随机稍强 |
| Domain L1 (C, depth 2) vs Domain L2 (C, depth 4) | Domain L2 | **76-22-2** | L2 明显强于 L1 |

**结论：Python minimax 的对手强度约等于 random ~ L0 水平。** C minimax 使用了复杂的棋型评估（威胁模式识别、连子计分），而 Python minimax 是基础的位置权重评估。同样的 depth 2，C 版本的棋力是质的飞跃。

#### 11.3.2 历史 WR 重标定

| Run | 历史记录 | 实际对手 | 真实棋力评估 |
|------|---------|---------|-------------|
| `789730e3` | "100% vs L1" (457 cycles) | Python L1 ≈ random+ | 能赢随机，不能赢 C-L1 |
| `6c9c8bdd` | "100% vs L1" (176 cycles) | Python L1 ≈ random+ | 同上 |
| `eeb43077` | "90.4% vs L2" (70 cycles) | Python L2 < C-L1 | 能赢 Python L2，不能赢 C-L1 |
| `d59fba4e` | "34.4% vs L2" (4 cycles) | Python L2 < C-L1 | 部分赢 Python L2 |

**验证：** 加载 parent model（`6c9c8bdd` c176）通过 `_in_process_eval` 测试：
- vs L0（random）：**100%** ✓
- vs C-L1（depth 2）：**0%** ✗
- vs C-L2（depth 4）：**0%** ✗

v15-findings §10 原文 "mcts_12 的训练数据（55% → 94% 在 4 cycles 内）证明 S1v2 → S2 的训练路径完全有效" —— 这里的 94% 是对 Python L2 的，对 C L2 可能接近 0%。

#### 11.3.3 对 §9 v15 自评的修正

| 原评估 | 修正 |
|--------|------|
| C 系 minimax C 化 "Mac 上工作正常" | **C 编译正常但从未在 eval 中被调用过**——sys.path bug 导致所有 eval 走 Python 路径 |
| B 系 batched eval "应当把 probe 压到 ~1 分钟" | batched eval 正确实现，但 213s 的 probe 有 210s+ 是 Python minimax 贡献的，不是 NN batch 问题 |
| "瓶颈 100% 在框架，模型本身在快速学习" | **模型在学习，但学习目标太低——只是在学赢 Python minimax（约等于随机）** |

### 11.4 Run 7b41a00e 分析 — 为什么 60 cycles 内 WR 始终为 0%

#### 11.4.1 事实

- 配置：MCTS 800 sims, 16 parallel games, 8 blocks, 128 filters (2.49M params)
- Resume from `6c9c8bdd` c176, eval_level 2
- 60 cycles, 944 games, 2283 training steps, 85 分钟
- 10 次 probe（每 6 cycles），**全部 0.0% vs C-L2**
- Loss: 5.83 → 4.79（policy 4.58 → 4.34, value 0.41 → 0.36）

#### 11.4.2 对比 parent 的 L1 学习曲线

| 指标 | Parent 6c9c8bdd (vs Python-L1) | 7b41a00e (vs C-L2) |
|------|------|------|
| 首次 WR > 0% | cycle 50 (500 games, 1959 steps) | **未达到**（cycle 236, 944 games, 2283 steps） |
| Loss 在首次 WR > 0% 时 | 5.179 | 4.792（更低但 WR 仍为 0） |
| 对手实际强度 | Python L1 ≈ random+ | C-L2 (depth 4, pattern scoring) |

#### 11.4.3 为什么 0%

1. **起点就不够高**：parent model 只能赢 random（100% vs L0），连 C-L1 都赢不了。从 "赢 random" 到 "赢 C-L2" 跨度极大。
2. **训练数据不够**：2283 training steps 大约等于 parent 在 cycle 50 时的水平（1959 steps）。Parent 在 cycle 50 时刚刚开始赢 Python-L1，而 C-L2 比 Python-L1 强好几个级别。
3. **自对弈信号弱**：MCTS 800 sims 的自对弈产生的策略目标质量受限于模型自身水平。模型目前水平约 L0+，自对弈产生的数据不包含击败 C-L2 所需的深层战术模式。
4. **Loss 还在下降**：4.79 的 policy loss 对 15×15 棋盘（ln(225) ≈ 5.42）来说仍然很高，模型的策略还很分散。正常训练好的模型 policy loss 应在 1.5-2.5。

#### 11.4.4 预期

按 parent 的学习曲线外推（假设 C-L2 的难度约为 Python-L1 的 5-10 倍），模型可能需要：
- **5000-10000+ training steps**（当前 2283）才能开始对 C-L1 产生胜率
- **20000+ training steps** 才能对 C-L2 产生胜率
- 按当前速度（~38 steps/min），约 2-4 小时训练才能看到 vs C-L1 的首胜，8-12 小时才能看到 vs C-L2 的首胜

### 11.5 训练效率分析 — 当前瓶颈与 MLX 优化空间

#### 11.5.1 周期时间分解

对 16 games × MCTS 800 sims 的单周期实测（Mac M3 Max）：

| 组件 | 耗时 | 占比 |
|------|------|------|
| MCTS 搜索（含 NN forward） | 82.1s | **99.8%** |
| 数据打包 + 训练步骤 | 0.1s | 0.2% |
| **总计** | **82.2s** | 100% |

MCTS 搜索内部（4650 sims/sec）：
- NN forward pass 是主要成本：16 棵树 × 8 leaves/round = ~128 batch, 100 rounds/move
- Batch=128 MLX forward: 18.6 ms/call
- ~477 total moves × 100 rounds = 47,700 NN forward 调用
- 理论 NN 时间: ~47,700 × ~1.5ms (考虑 batch 大小变化) ≈ 70s

#### 11.5.2 MLX 优化机会

| 优化 | 预期加速 | 复杂度 | 备注 |
|------|---------|--------|------|
| **`mx.compile()` JIT** | 15-20% NN 提速 | 低（1 行） | batch=128: 18.6ms → 15.7ms (实测) |
| **增大 MCTS_BATCH_SIZE** | 负效果 | — | 实测 batch=8 (60s) < batch=16 (71s) < batch=32 (92s)，更大 batch 反而更慢（virtual loss 退化） |
| **Board 操作 C 化** | ~20-30% 总提速 | 高 (1-2天) | `_check_win` 和 `_fast_copy` 被调用 ~960K 次/cycle，GIL-bound |
| **混合精度 (fp16)** | ~10% 带宽提速 | 中 | MLX 支持但需验证精度 |

**综合估计：可实现的最大优化约 1.3-1.5×**（周期时间从 82s 降到 55-63s）。

#### 11.5.3 关键结论

> **训练效率不是当前的核心问题。** 在 MCTS 800 sims 下，82s/cycle 已经接近 MLX GPU throughput 的物理极限（理论下限 ~60s）。真正的瓶颈是：
> 1. 模型从未真正学会击败有意义的对手（历史 WR 全部失效）
> 2. 需要的训练时长远超之前的预期（不是 1.5 小时，而是 8-12 小时）
> 3. 可能需要课程学习策略：先练 C-L0→C-L1，再练 C-L1→C-L2

### 11.6 教训

1. **对手强度验证必须端到端**。v15 花了大量精力优化 minimax C 化和 eval batching，但从未验证过 "eval 到底在用哪个 minimax"。一个简单的 `print(prepare.__file__)` 就能在第一天发现问题。
2. **同名模块是架构债**。`prepare.py` 同时存在于两个目录，依赖 `sys.path` 顺序来决定加载哪个。这是定时炸弹。**v16 应该重命名或用 package import 消除歧义。**
3. **WR 指标需要绝对锚点**。如果只看相对 WR 曲线（"从 0% 涨到 100%"），看不出对手强度是否正确。应该有一个已知棋力的 baseline（例如 "C-L1 vs C-L2 = 24% WR"）作为锚点。
4. **第四次 "Linux 测试给了虚假安全感"**：v15.0（Metal threading）、v15.2（MLX batch latency）、v15.3（sys.path + 对手强度）。每一次都是在 Mac 实机上才暴露问题。

### 11.7 后续建议

| 优先级 | 建议 | 理由 |
|--------|------|------|
| P0 | **课程训练：从 C-L0 开始**，先验证模型能 100% 赢 random，再 eval-level 1 对 C-L1 训练 | 跳级到 C-L2 信号太稀疏，模型学不到有效梯度 |
| P0 | **消除同名 prepare.py**，给 framework 版本加前缀或改用 package import | 防止 sys.path bug 复发 |
| P1 | **添加 `mx.compile()` 到 MCTS evaluate_batch** | 零风险的 15-20% 训练加速 |
| P1 | **添加 eval 对手强度验证到启动检查**：打印 `prepare.__file__` 和 `MINIMAX_BACKEND` | 一秒钟的检查能避免几天的无效训练 |
| P2 | **Board 热路径 C 化**（_check_win, _fast_copy） | 额外 20-30% 训练加速，但开发成本高 |

### 11.8 v15.3 findings 一句话总结

> **v15.3 sys.path 修复揭露了一个比 eval 性能更严重的问题：v15.0 以来所有训练的 eval 对手都是错误的。C minimax L1（depth 2）对 Python minimax L2（depth 4）100 战全胜。模型此前 "100% vs L1" 的成绩等价于 "100% vs random+"。Run 7b41a00e 的 0% WR vs C-L2 不是训练 bug，而是模型从未达到过真正的 L1 水平。训练效率方面，MCTS 800 sims 的 82s/cycle 已接近 MLX GPU 极限，优化空间约 15-30%。核心问题不是速度，而是需要重新从 C-L0/L1 开始课程训练。**

---

## §12 v15.4 findings — 2 小时 C-L1 训练零胜率根因分析

### 12.1 实验概况

| 项目 | 值 |
|------|----|
| Run ID | `440f81ae` |
| 对手 | C-L1（minimax depth 2 + C 模式评估） |
| 模型 | 8 blocks × 128 filters = 2,490,035 参数 |
| MCTS | 800 sims, batch=8, C-native |
| Cycles | 105 |
| 自对弈游戏 | 1,680 |
| 训练步数 | 3,004（batch_size=256） |
| Wall time | 7,218 秒（~2 小时） |
| 所有 21 次 probe WR | **0.0%**（80 games/probe） |
| 最终 200 局 eval | **0W / 200L / 0D** |

### 12.2 Loss 趋势：模型在学习，但不够

| 阶段 | Total Loss | Policy Loss | Value Loss |
|------|-----------|-------------|------------|
| 前 10 cycles | 6.159 | 5.410 | 0.721 |
| 后 10 cycles | 4.908 | 4.441 | 0.378 |

- **Policy loss 初始值 5.41 ≈ ln(225) = 5.416**，即完全随机策略
- 降至 4.44 → 有效困惑度从 224 降至 85（从 225 个等概动作收窄到 ~85 个）
- Value loss 从 0.72 降至 0.38，说明 value head 也在学习
- **Loss 持续下降、未平台化 → 模型远未收敛，需要更多训练**

### 12.3 模型行为诊断

**空棋盘**：
- Policy 熵 = 4.572 / 5.416（max），占 84.4% — 仍非常接近均匀分布
- Top-1 概率仅 4.88%（随机 = 0.44%）
- Value head 输出 -0.37（不合理，空棋盘应接近 0）

**4 连子威胁检测**（Black 在 (7,0)-(7,3)，White 必须堵 (7,4)）：

| 方法 | 堵住 (7,4) 的概率 |
|------|-------------------|
| 随机策略 | 0.44% |
| **NN raw policy** | **2.70%** |
| MCTS 800 sims | **66.0%** |

**关键发现**：MCTS 800 sims 能正确识别威胁（66% 概率），但 NN 本身几乎不会堵（2.7%）。这意味着 **训练目标质量良好（MCTS 提供强信号），但模型未能充分学习这些目标**。

**自对弈游戏长度**（20 局样本）：
- 最短 11 步，最长 119 步，中位数 40.5 步
- 短游戏说明双方频繁漏看威胁 → 数据质量受限于模型自身水平

**vs C-L1 实测**（10 局 argmax）：
- 0 胜 10 负，平均 14-16 步被杀 — 模型完全无法识别基本战术

### 12.4 根因：训练量严重不足

这是核心问题。以下数据说明一切：

| 指标 | 当前运行 | AlphaGo Zero | 差距 |
|------|---------|-------------|------|
| 模型参数 | 2.49M | 13M | 0.19× |
| 总训练步数 | 3,004 | 700,000 | 0.4% |
| 总梯度更新（steps × batch） | 769,024 | 1,433,600,000 | 0.05% |
| **更新次数 / 参数数** | **0.31** | **110** | **0.3%** |

**每个参数只被更新了 0.31 次**。在深度学习中，这相当于几乎没有训练。

#### 为什么 `steps_per_cycle=30` 是瓶颈

训练步数还受 replay buffer 大小限制：`effective_steps = min(steps_per_cycle, buffer_size // batch_size)`。

| Cycle | Buffer | Max Steps | 实际执行 |
|-------|--------|-----------|---------|
| 1 | 885 | 3 | **3** |
| 2 | 1,766 | 6 | **6** |
| 5 | 4,598 | 17 | **17** |
| 10 | 9,199 | 35 | **30** (受限) |
| 30 | 22,166 | 86 | **30** (受限) |
| 60 | 34,905 | 136 | **30** (受限) |
| 105 | 47,423 | 185 | **30** (受限) |

从 cycle 10 起，模型每 cycle 都浪费了可以多训练 5-6× 的机会。

#### `steps_per_cycle=200` 的效果预估

同样 105 cycles、1680 games 的数据：

| 配置 | 总训练步数 | 更新/参数 | 额外时间 |
|------|-----------|----------|---------|
| spc=30 (当前) | 3,004 | 0.31 | — |
| **spc=200** | **11,969** | **1.23** | **+22 分钟** |

**4× 训练量，仅多 22 分钟**（训练每步 ~150ms，自对弈 ~65s/cycle 是绝对瓶颈）。

### 12.5 辅助问题

#### 问题 1：C-L1 对完全随机是 100% 胜率

```
C-L0 (random) vs C-L1: 0/100 (0%)
Random NN (argmax) vs C-L1: 0/20 (0%)
```

C-L1 对任何不具备基本战术能力的对手都是碾压。eval 使用二值 win/lose 指标，即使模型内部进步了（loss 在降），只要还没跨过战术门槛，WR 就一直是 0%。这不代表模型没在学习，只是 **"0% → 首胜" 是一个阶跃函数，需要足够的训练量触发**。

#### 问题 2：D4 对称增强有效但被低估

训练中每步随机应用 D4 变换（8 种对称），有效增加数据多样性 8×。但 `steps_per_cycle=30` 时，buffer 中大量数据未被用到：

- Buffer 47K 条样本，每 cycle 只采 30 × 256 = 7,680 条
- 即使算 D4 增强，有效利用率 = 7680 / (47423 × 8) ≈ **2%**

#### 问题 3：Policy loss 下降速率预示需要的额外训练

Policy loss 从 5.41 降至 4.44，速率约 0.009/cycle。要达到可能打赢 C-L1 的水平（估计 policy loss ~3.5-4.0），按当前速率还需 50-100 个 cycle。如果提高 `steps_per_cycle`，收敛会更快。

### 12.6 与历史运行对比

| Run | 对手 | 模型 | Games | Steps | spc | WR | 时间 |
|-----|------|------|-------|-------|-----|-----|------|
| 7b41a00e | C-L2 | 8×128 | 944 | 2,283 | 30 | 0% | 85m |
| **440f81ae** | **C-L1** | **8×128** | **1,680** | **3,004** | **30** | **0%** | **120m** |
| eeb43077 (旧) | Py-L2 | 8×128 | ~5000 | ~4000 | 30 | 90% | ~5h |

旧的 eeb43077 在 ~5000 games 时达到了 90% WR，但那是对 Python minimax（极弱对手）。C-L1 要求的模型能力比 Py-L2 高得多。

### 12.7 结论与建议

**诊断结论**：模型在学习（loss 持续下降），但 **`steps_per_cycle=30` 导致训练量严重不足**（0.31 更新/参数）。2.49M 参数的模型需要至少 5-10× 的梯度更新才能发展出基本战术能力。

**P0 修复：提高 `--steps-per-cycle` 到 200**

这是最高 ROI 的改动：4× 训练量，仅多 15% 时间。

**推荐训练命令**：

```bash
PYTHONUNBUFFERED=1 uv run python domains/gomoku/train.py \
    --eval-level 1 \
    --mcts-sims 800 \
    --parallel-games 16 \
    --num-blocks 8 --num-filters 128 \
    --eval-interval 5 \
    --probe-games 32 \
    --steps-per-cycle 200 \
    --target-win-rate 0.55 \
    --time-budget 7200
```

**预期**：
- 2 小时内达到 ~12,000 步（vs 当前 3,004），更新/参数从 0.31 → 1.23
- Policy loss 应更快收敛至 ~3.5
- 首胜可能出现在 cycle 30-60（1-2 小时内）

**备选方案**（如 spc=200 后 2h 仍无突破）：
1. 使用更小的模型：4 blocks × 64 filters（564K 参数），同等训练量下更新/参数 = 5.46
2. 进一步提高 spc 到 500（buffer 足够大时生效）
3. 先从 eval-level 0 训练，确认模型能在 L0 上快速达到 90%+ 后再切 L1

### 12.8 v15.4 findings 一句话总结

> **Run 440f81ae（2h、1680 games vs C-L1）的 0% WR 根因是 `steps_per_cycle=30` 导致训练量严重不足——2.49M 参数模型仅获得 0.31 次/参数的梯度更新，相当于几乎没训练。Loss 持续下降证明模型在学习，但 MCTS 产生的高质量训练目标（威胁检测准确率 66%）未能充分传递给 NN（仅 2.7%）。将 `steps_per_cycle` 提高到 200 可在不增加自对弈成本的情况下获得 4× 训练量，是最高优先级修复。**