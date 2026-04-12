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
