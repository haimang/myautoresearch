# v14 Findings — C 原生 MCTS 首测 + 关键 Bug 修复

> 2026-04-12 | 基于 mcts_7th_exp.db + mcts_7th.png (M3 Max)

---

## 1. 实验结果

### de11c51e: 8x64, MCTS-800, resume S0, vs L1

| 指标 | 值 |
|------|-----|
| 架构 | 8x64 (713K params) |
| MCTS | 800 sims [C-native] |
| Resume | d6c6bce4 (S0, 99% vs L0) |
| 时间 | 1815s / 1800s budget |
| Cycles | 109 (cycle 66-174, resumed from 65) |
| 总局数 | 872 |
| 训练步数 | 2,763 |
| 速度 | 28.8 局/分钟, 16.6s/cycle |
| **Probe WR vs L1** | **100%（cycle 75 起全部 100%）** |
| **Full Eval WR vs L1** | **0%（200 games subprocess eval）** |

---

## 2. 关键 Bug：Probe 100% vs Full Eval 0% 的矛盾

### 现象

TUI 截图显示 probe eval 一直是 100%：
```
[00:04:24] Probe: 100.0% (avg:100.0%) (50 games vs L1)
[00:05:10] Probe: 100.0% (avg:100.0%) (50 games vs L1)
...（全部 100%）
```

但最终 full eval（subprocess, 200 games）返回 0%：
```
Running final evaluation vs L1 (200 games)...
Final win_rate: 0.0%
```

### 根因

**subprocess 的 `load_model` 加载了错误的模型架构。**

执行链：
```
subprocess -c "import train; ..."
  → train.py 顶层: from prepare import TIME_BUDGET
    → import prepare
      → prepare.py: from train import load_model  ← 绑定到 ORIGINAL load_model
  → monkey-patch: train.load_model = patched_version  ← 太晚了！
  → prepare.evaluate_win_rate(...) → prepare.load_model(path)  ← 用的是 ORIGINAL
```

`prepare.py` 在 `import train` 阶段就通过循环导入绑定了原始 `load_model`（默认 6x64）。monkey-patch 发生在此之后，无法覆盖 `prepare.py` 已经持有的引用。

结果：subprocess 把 8x64 (713K) 的权重加载到 6x64 (564K) 的架构中 → 权重不匹配 → 模型输出垃圾 → 0% WR。

### 修复

同时 patch `train.load_model` 和 `prepare.load_model`：

```python
patch = (
    "import train; import prepare; "
    "_orig = train.load_model; "
    "_patched = lambda p, **kw: _orig(p, num_blocks=8, num_filters=64); "
    "train.load_model = _patched; "
    "prepare.load_model = _patched; "  # ← 修复：覆盖 prepare 已绑定的引用
)
```

**已在 train.py 中修复。**

---

## 3. TUI 截图分析

从 mcts_7th.png 读取的关键指标：

| 指标 | 值 | 解读 |
|------|-----|------|
| Focus | **27%** | 比 50 sims 的 10% 大幅提升。800 sims 能让搜索聚焦 |
| Entropy | **3.16** | 中等偏高，但比 50 sims 有改善 |
| Sim/s | **11,210** | C 原生搜索速率 |
| SP time | **15.1s** | 每 cycle 自对弈耗时 |
| Loss | **0.00** | 最后一个 cycle 被时间截断，无训练步执行 |
| Gm/s | **0.5** | 800 sims 下每秒 0.5 盘（合理） |
| AvgLen | **26.4** | 游戏平均 26 步——比之前 35 步更短，说明模型更快结束游戏 |

### Focus 趋势判断

**Focus 27% = 搜索有效。** 800 sims 在 225 位置上给最高位置分配了 ~216 次访问（27%×800），其余分散在其他位置。对比：
- 50 sims / Focus 10% = 最高位置只有 5 次访问 → 接近随机
- 800 sims / Focus 27% = 最高位置 216 次 → 有明确偏好

Entropy 3.16 仍偏高（理想 <2.5），说明模型先验不够聚焦。但这是从 S0（vs L0 训练）resume 来的——S0 的先验本来就不指向战术位置。随着 vs L1 训练继续，entropy 应该下降。

### WR Sparkline

WR 图表显示从 cycle 75 起持续 100%。但这是 probe eval 的结果，受 subprocess bug 影响，**真实 vs L1 WR 未知**。修复 bug 后需要重新训练验证。

---

## 4. 速度分析

### C 原生 vs Python 实测对比

| 配置 | Python (v12) | C 原生 (v14) | 加速 |
|------|-------------|-------------|------|
| 8×50 sims, 8x64, pg=10 | 122 gm/min | — | — |
| 8×800 sims, 8x64, pg=8 | ~5 gm/min (估) | **29 gm/min** | **~6x** |

C 原生在 800 sims 下达到 29 gm/min。这比 Python 50 sims 的速度（122 gm/min）慢，但 800 sims 的搜索量是 50 sims 的 16 倍。**按每 sim 效率算，C 比 Python 快约 6x。**

### Wall Time 分析

```
总 wall time:    1815s (≈预算 1800s)
109 cycles × 16.6s/cycle = 1809s
剩余：~6s 初始化 + 结尾
```

wall time ≈ 预算是正常的——训练一直运行到时间耗尽。16.6s/cycle 的分解：
- MCTS 搜索（8 boards × ~50 moves × 800 sims tree ops）: ~10s
- Python board.copy() + place()（每 sim round 8 copies + ~5 places）: ~4s
- GPU evaluate（~50 batch calls per search）: ~1s
- Training step（30 gradient steps）: ~1.5s

**进一步降低 cycle 时间的方法：**
1. `--parallel-games 4`（4 盘代替 8 盘，cycle 从 16s 降到 ~9s，但局数产出减半）
2. 把 Board.copy() + place() 也写成 C（预期额外 2x，但 game.py 是 READ-ONLY）
3. `--eval-interval 10`（减少 probe eval 频率，省 ~5% 时间）

---

## 5. 训练信号评估

### Probe WR 可信度

Probe eval 显示 100% vs L1，但 subprocess full eval 显示 0%。由于 subprocess bug，**真实 WR 未知**。

但 probe eval 的 `_in_process_eval` 代码逻辑是正确的——它加载内存中的正确模型、使用 `OPPONENTS[1]`（minimax L1）、以 argmax 方式走棋。100% WR 如果是真实的，说明 **MCTS-800 resume S0 的训练产生了能赢 L1 的模型。**

### Loss 轨迹

```
起始 (cycle 66): 5.65 (from S0 checkpoint)
最低: 4.54 (cycle 168)
下降: 1.11 (在 109 cycles 内)
```

Loss 持续下降，说明模型在有效学习 MCTS-800 的 policy target。最终 loss=0.00 是因为最后一个 cycle 时间截断，未执行训练步。

---

## 6. 下一阶段分析与建议

### 6.1 首先：修复 bug 后重新验证

**必须先修复 subprocess eval bug，再做任何训练决策。** 当前的 probe WR 100% 需要被 full eval 验证。

```bash
git pull origin main

# 编译 C 扩展
cd framework/core && bash build_native.sh && cd ../..

# 重新训练（修复后的代码），800 sims，30 分钟
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 8 --mcts-batch 16 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --resume d6c6bce4 --seed 42
```

如果 full eval 也显示 >80% vs L1 → 进入 6.2。如果 full eval 仍然很低 → 需要更长训练。

### 6.2 升级到 8x128 长训练

**参数选择分析：**

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `--num-blocks` | 8 | 8 | 8 层已足够。v11 数据证明 >10 层边际收益为负 |
| `--num-filters` | 64 | 128 | 容量从 713K → 2.49M。更多 filter = 更强特征表示 |
| `--mcts-sims` | 800 | 800 | 已验证 Focus 27%，搜索有效 |
| `--learning-rate` | 5e-4 | **2e-4** | 大模型需要更小学习率防止训练不稳定 |
| `--steps-per-cycle` | 30 | **50** | 更大模型需要更多步数来消化每 cycle 的数据 |
| `--buffer-size` | 50000 | **100000** | 2.49M 模型需要更多数据防止过拟合 |
| `--mcts-batch` | 16 | 16 | 保持不变 |
| `--parallel-games` | 8 | 8 | 8x128 模型更大，8 盘已够 |
| `--eval-level` | 1 | 1 | 先征服 L1，再考虑 L2 |

**为什么要调 learning rate：**

之前没有调是因为 6x64 (564K) 模型较小，默认的 5e-4 足够稳定。但 8x128 (2.49M) 模型大 4.4 倍，相同 learning rate 的有效步长更大，容易导致训练不稳定（v13 的 8x128 vs S0 训练出现剧烈震荡，std 22.7%）。降到 2e-4 能稳定训练。

**为什么要调 steps-per-cycle：**

默认 30 步/cycle。800 sims 每 cycle 产出 8 盘 × ~26 步 = ~208 个训练样本（D4 augmentation 后 ×8 = ~1664 有效样本）。30 步 × batch=256 = 7680 个样本被训练。比率 7680/1664 = 4.6x 意味着每个样本平均被看 4.6 次/cycle。对 2.49M 模型来说，50 步（7.7x）更合适，确保模型充分消化每 cycle 的数据。

**建议命令（6 小时，vs L1）：**

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 8 --mcts-batch 16 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 2e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 21600 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation --stagnation-window 15 \
  --seed 42
```

### 6.3 什么时候从 L1 升级到 L2

**L1 (minimax depth 2) → L2 (minimax depth 4) 的升级条件：**

- WR vs L1 ≥ 80% 且稳定（后半段 std < 5%）
- 注册为 S1 对手
- 然后 resume S1 继续训练 vs L2

```bash
# 注册 S1
uv run python domains/gomoku/train.py \
  --register-opponent S1 --from-run <run_id> --from-tag <tag> \
  --description "8x128 MCTS-800, 80%+ vs L1"

# Stage 2: vs L2
uv run python domains/gomoku/train.py \
  --mcts-sims 800 --parallel-games 8 --mcts-batch 16 \
  --num-blocks 8 --num-filters 128 \
  --learning-rate 2e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 21600 \
  --eval-level 2 --no-eval-opponent \
  --resume <S1_run> --target-win-rate 0.60 \
  --auto-stop-stagnation --seed 42
```

**注意：** vs L2（minimax depth 4）会显著更难。L2 能看到 4 步远，可以做双活三等高级战术。模型可能需要更长训练时间。

### 6.4 不建议现在做的事

| 不做 | 理由 |
|------|------|
| 直接跳到 L2 | 先验证 vs L1 是否真的能赢（修复 bug 后确认） |
| 增加 mcts-sims > 800 | Focus 27% 已足够，更多 sims 只增加时间不增加质量 |
| 多进程 worker | C 扩展已提供 6x 加速，够用 |
| 修改 c_puct | 1.5 是标准值，无数据支持修改 |

---

## 7. mcts_9th 复盘 — 6 小时长训，100% WR 的真实性审查

> 2026-04-13 | 基于 updates/mcts_9th_exp.db + updates/mcts_9th.png

### 7.1 run 概要（789730e3）

| 项目 | 值 |
|------|-----|
| 架构 | 8×128（2.49M params）|
| MCTS | 800 sims（C 原生）|
| Resume | 无（从零训练）|
| LR / steps / buffer | 2e-4 / 50 / 100,000（按 §6.2 建议）|
| Parallel / batch | 10 / 256 |
| time_budget | 21600s（6 h）|
| Cycles / games / steps | 457 / 4,570 / 22,135 |
| Final loss（TUI）| 0.00 |
| Final WR vs L1（full eval 200 局）| **100% (200/200)** |
| Probe WR 轨迹 | c20=0%, c60=50%, c80=100%, 260 起稳定 100% |

### 7.2 结论先行

**final_c0457 是对 L1（minimax depth 2）的一个专用解，而不是一个普适 5×5 连珠棋手。** 200 局 full eval 的统计效力实际上只有 **2 局**；loss 在整个训练中几乎没下降；模型学到的是"两条针对确定性 L1 开局的必胜线"，而不是稳健的 policy。下面是证据。

### 7.3 致命发现 1：所谓 200 局 eval，本质只是 2 局

- `_in_process_eval`（train.py:1543）中 NN 走子用 `mx.argmax`（行 1574），对手用 `OPPONENTS[level]`。
- `opponent_l1` → `minimax_move(depth=2, move_order_fn=_move_order_basic)`（prepare.py:348）。
- `_move_order_basic`（prepare.py:241）的注释写 "then random shuffle"，但**实现里根本没有 shuffle**，只按中心距离稳定排序。
- 评估期间棋盘随机性为 0：NN 和 minimax 都是确定性函数，初始盘面也确定。所以：
  - NN 作黑的 100 局 → 完全相同的 1 局，复制 100 次
  - NN 作白的 100 局 → 完全相同的 1 局，复制 100 次
  - **任何一次 full eval 的合法取值只能是 {0.0, 0.5, 1.0}**

数据印证：cycle_metrics 表 22 次 probe（120 局）的 win_rate 全部落在 {0.0, 0.5, 1.0}；checkpoint 表的 200 局 full eval 只见 0.5 和 1.0。这不是巧合，是一个结构性问题。

**因此 "100% (200/200) vs L1" 并不代表 "对 L1 几乎不败"，它代表的是 "模型碰巧找到了两个确定性开局都能走赢的分支"。**

### 7.4 致命发现 2：loss 曲线显示模型几乎没学东西

从 cycle_metrics 抽样：

| cycle | loss | probe WR |
|-------|------|----------|
| 1 | 6.363 | — |
| 20 | 5.633 | 0% |
| 80 | 4.814 | 100%（开始）|
| 140 | 4.626 | 100% |
| 260 | 4.403 | 100%（稳定）|
| 440 | 4.273 | 100% |
| 456 | 4.253 | — |
| 457 | **0.000** | — |

- 6 小时 22k 步，loss 从 6.36 降到 4.25，总下降 **2.11 nats**。
- 策略+价值联合 loss，考虑到 225 个动作的 uniform 基线 ≈ log(225) = 5.42，模型最终的 policy CE 粗估仍在 **~4.0 左右**，只比随机好一点点。
- TUI 截图里的 **Entropy = 3.03**（远高于理想的 ≤2.5）与此一致：先验仍是"十几个位置几乎等概率"。
- "Final loss 0.00" 是 cycle 457 那一行的记账伪影（时间预算耗尽，当 cycle 没有任何 training step，loss 写成 0）。这会污染 analyze.py 的最终指标，应当被视为 **显示 bug**。

**100% WR + 基本没学的 loss** → 唯一解释是：高 MCTS 访问数（800 sims）直接把对 L1 的两条致胜线"背"进了访问分布，NN 只是被驯化到能在这两条线上给出 argmax。泛化能力未被任何证据支持。

### 7.5 致命发现 3：avg_game_length 的塌缩指向同一件事

| checkpoint | avg_game_length |
|------------|------------------|
| wr050_c0100 | 28.5 |
| wr080_c0140 | 13.5 |
| wr090_c0260 | 15.5 |
| wr100_c0300 | 14.5 |
| **final_c0457** | **10.5** |

10.5 步结束一局意味着黑方用 5-6 手就连成五。这是**单一开局 + 单一速赢路线**反复重演的典型特征，不是"普遍强"。真正的稳健棋手面对不同对手，局长分布应当是有方差的。这里每个 checkpoint 的 avg 完全看不到方差——因为本来就没有方差，只有 2 局。

### 7.6 final_c0457 checkpoint 是否可用？

| 用途 | 结论 |
|------|------|
| 作为 "80% vs L1 的 S1"，晋升到 L2 训练 | **不可用**。100% 是伪的，换成 L2 就没有先验保护。 |
| 注册成 web 对手和人类对弈 | **勉强可用，但会很弱**。人类不会走 L1 的确定序列，模型大概率暴露未学会棋型。 |
| 作为 self-play 初始化（resume） | **不推荐**。作为 prior 它偏向单一开局，会给后续训练带来模式崩塌风险。 |
| 作为 "L1 专用快速开局库" | 可用，但这不是我们要的东西。 |

**短结论：** final_c0457 不是一个可信的 stage 1 里程碑。不要注册为 S1，也不要基于它 resume 到 L2。

### 7.7 参数评价（§6.2 的建议在实际数据上如何？）

| 参数 | 值 | 评价 |
|------|-----|------|
| `num_blocks / num_filters` | 8 / 128 | 合理，2.49M 不算过参数化 |
| `learning_rate` | 2e-4 | **偏低**。loss 下降 2.11 nats / 22k steps 说明学习太慢，可考虑 3e-4 或加 warmup 后到 5e-4 |
| `steps_per_cycle` | 50 | 合理，但在 SP 产出过窄时会放大同分布过拟合 |
| `buffer_size` | 100,000 | 合理，后期 buffer 全满 |
| `mcts_sims` | 800 | **不是问题**，但配上确定性对手会让 SP 数据分布极窄 |
| `parallel_games` | 10 | 合理 |
| `time_budget` | 21600 | 时间花对了，但数据生成路径有结构性缺陷（见 7.8），更多时间不能解决 |

**真正的问题不在超参数上，而在 self-play 和 evaluation 的数据分布太窄。**

### 7.8 框架 / game 代码应当调整的地方

下列改动才是让下一轮训练有意义的前提，优先级由高到低：

1. **给 minimax 对手加随机性（prepare.py）**
   - `opponent_l1/l2/l3` 在得分近似相等的 top-k 动作中 `random.choice`；或者在最外层 move ordering 真的 shuffle。
   - 没有这个，任何 "N 局 eval" 的统计效力都等于 "2 局 eval"。这是**阻塞级**问题。

2. **评估开局多样化（train.py: `_in_process_eval`）**
   - 在 200 局里均匀分配 2–4 手的随机合法开局（或从一个开局库里采样），再交给双方落子。
   - 这样即便双方都是确定性的，eval 里也能看到真实的分布，而不是 2 份拷贝。

3. **Self-play 加温度和 Dirichlet noise 的正确性复核（train.py）**
   - 若 temp_threshold=30 但 SP 产出的棋局长度已经掉到 ~10 步，那整局 SP 从头到尾都在 argmax 区间，根本没有探索。需要确认 root-Dirichlet 是否开启、ε 是否合理。
   - 这是"loss 不降 + 开局单一"的另一可能根因。

4. **Loss 拆分到 policy / value 两路并入库**
   - 目前 cycle_metrics 只有一个合计 loss，看不出 "模型学不到 policy" 还是 "学不到 value"。加两列 `policy_loss` / `value_loss` 就能一眼看出。

5. **Full eval "200 局" 语义修正**
   - 在当前实现下 200 局 = 2 局，这条信息应在 TUI 和 DB 里显式标注（例如 `eval_unique_games`），否则 analyze.py 的报告会误导 agent。

6. **cycle 457 的 loss=0 写入应改成 NULL 或继承前一 cycle**
   - 当前会让 `final_loss` 字段变成 0，污染 runs 表。

**game.py 本身看起来没有问题** — board 引擎是确定性是应该的，不需要改。问题完全在 "对手 + 评估协议" 层。

### 7.9 Web 对弈可行性确认

可以，但有两个前提：

1. **运行环境必须是 Apple Silicon + MLX**。`play_service.py:_load_cached_model` 强依赖 `mlx.core`，当前这台 Linux 开发机无法启动 web_app。
2. **final_c0457 尚未注册为命名对手**。`opponents` 表现在只有 S0（8×64）和 S1（来自 run c70162b9 的 1 小时 8×128，WR 82%），**并不包含 mcts_9th 的 final_c0457**。

要在 Mac 上通过浏览器和这个 checkpoint 对弈，步骤是：

```bash
# 1. 注册 checkpoint 为命名对手（在 Mac 的项目根目录）
uv run python domains/gomoku/train.py \
  --register-opponent MCTS9 \
  --from-run 789730e3-822f-482c-bba3-53a2da692f2d \
  --from-tag final_c0457 \
  --description "8x128 MCTS-800, 6h, final (⚠ 对 L1 过拟合)"

# 2. 启动 web 服务
uv run python domains/gomoku/web/web_app.py
# 然后浏览器打开 http://127.0.0.1:8000
```

前端 `/api/opponents` 会自动把 `MCTS9` 作为一个 NN 对手选项列出（见 play_service.py:149 `get_frontend_opponents`），选它 + 人类执黑/白即可下棋。

**但请管理预期：** 基于 7.3–7.5 的分析，这个模型面对人类（非确定性、非 L1 风格）的开局极可能很快走出"学过的棋谱"外，表现比 TUI 上的 100% 弱很多。把这次对弈当做"验证伪 100%"的手段，而不是"展示 stage 1 成果"。

### 7.10 下一步建议

优先级顺序：

1. **先修数据分布，再谈训练** — 落地 7.8 里的 1、2、3 项改动；
2. **用修好的协议重跑 200 局 eval 对 final_c0457**，得到第一份真实 WR 数字；
3. 如果真实 WR < 60%（高度可能），**丢弃 final_c0457**，在修好的 self-play 协议下从 S0 重新 resume，仍用 8×128 / 800 sims，但 LR 上调到 3e-4；
4. 只有当某个 checkpoint 在"随机化 L1 对手 + 多样化开局"下稳定 ≥80% vs L1，才注册成 S1 并进入 L2 阶段。

**v14 到此为止的教训可以一句话总结：** 在评估协议有确定性坍缩的前提下，任何 100% 胜率都不可信，任何由它推出来的训练决策都不可信。先把尺子校准，再谈长度。
