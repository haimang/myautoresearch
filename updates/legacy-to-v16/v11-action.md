# v11 MCTS 测试行动指南

> 目标读者：在 macOS Apple Silicon 上执行测试的操作者  
> 前置条件：`uv sync` 已完成，项目在 main 分支最新状态

---

## 1. 测试目标

验证 MCTS 训练信号是否真正打破纯 policy 自对弈的停滞天花板。

**具体量化标准：**

| 指标 | 纯 policy 基线 | MCTS 预期 |
|------|---------------|----------|
| WR 曲线形态 | 震荡不收敛（±15% 随机波动） | 单调上升或平稳收敛 |
| 300s 内 vs L0 WR | ~70-85%（历史数据） | ≥70%（信号质量验证，不追求更高） |
| 训练效率 WR/千局 | 前 200 局 >4.0，之后急降到 <0.5 | 前 200 局可能较低（MCTS 慢），但持续有效 |
| TUI 中 Entropy 趋势 | N/A | 应从高值逐渐下降（搜索越来越聚焦） |
| TUI 中 Focus (top1_share) | N/A | 应从低值逐渐上升（搜索越来越确定） |

**核心判断：** 如果 MCTS 300s 训练的 WR 曲线呈现收敛趋势（而非历史数据中的随机震荡），即为成功。绝对 WR 数值不是重点——MCTS 每盘耗时更长，局数更少。

---

## 2. 测试命令

### 测试 A：MCTS 基础验证（最重要）

```bash
# 50 sims × 8 并行盘 × 300 秒，vs 随机对手
uv run python domains/gomoku/train.py \
  --mcts-sims 50 \
  --parallel-games 8 \
  --time-budget 300 \
  --no-eval-opponent \
  --eval-interval 5 \
  --probe-games 50 \
  --seed 42
```

**预期：**
- 每 cycle 自对弈约 5-15 秒（vs 纯 policy 的 <1 秒）
- 每 cycle 产生 8 盘 × ~50 步 = ~400 个训练样本
- TUI 中 `MCTS 50sims` 行出现，显示 Sim/s 和 Focus
- 300s 内约 20-60 个 cycle（vs 纯 policy 的 ~200 个 cycle）
- WR 评估每 5 个 cycle 进行一次

### 测试 B：纯 policy 对照组

```bash
# 相同条件但无 MCTS，作为基线对照
uv run python domains/gomoku/train.py \
  --parallel-games 8 \
  --time-budget 300 \
  --no-eval-opponent \
  --eval-interval 5 \
  --probe-games 50 \
  --seed 42
```

### 测试 C：MCTS 200 sims（如果测试 A 成功）

```bash
# 更强搜索，更慢但信号更好
uv run python domains/gomoku/train.py \
  --mcts-sims 200 \
  --parallel-games 4 \
  --time-budget 600 \
  --no-eval-opponent \
  --eval-interval 3 \
  --probe-games 50 \
  --seed 42
```

---

## 3. TUI 中看什么

训练启动后，TUI 仪表盘分为以下区块：

### 3.1 头部信息
```
╭──────────────────────────────────────────────────────────────────────────────╮
│ Run: a1b2c3d4   Apple M3 Max   564.5K params   [exploratory]               │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━───────────────────────── 52% 2:36 / 5:00         │
```
- 确认 run ID 和硬件
- 进度条显示时间预算消耗

### 3.2 核心指标
```
│  Cycle       12   │   Loss       4.2031   │   Games        96               │
│  Steps       36   │   Buffer        384   │   WR   62% avg:58%              │
│  Gm/s      0.3   │   AvgLen      48.2    │   vs L0                         │
```
- **Gm/s** — MCTS 模式下会远低于纯 policy（预期 0.1-0.5 vs 纯 policy 的 5-20）
- **AvgLen** — 平均每盘步数。MCTS 棋局应更长更有质量
- **WR** — 关键指标。看趋势，不看绝对值

### 3.3 MCTS 专属行（仅 MCTS 启用时显示）
```
│╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌│
│  MCTS      50sims │   Sim/s       1250   │   SP time  12.3s                │
│  Focus     34%    │   Entropy      2.85   │   c_puct  1.5                  │
```

**关键 MCTS 指标解读：**

| 指标 | 含义 | 健康范围 | 异常信号 |
|------|------|---------|---------|
| **Sim/s** | 每秒完成的 MCTS 模拟次数 | 500-5000 | <100 = 可能有 Metal buffer 问题 |
| **SP time** | 本 cycle 自对弈总耗时 | 5-30s | >60s = 搜索太慢，降低 sims |
| **Focus** | 最高 visit 占比 (top1_share) | 20-80% | <10% = 搜索没有聚焦；>95% = 可能退化为贪心 |
| **Entropy** | visit 分布的信息熵 | 1.0-4.0 | >5.0 = 接近随机（搜索无用）；<0.5 = 过度确定 |
| **c_puct** | 探索常数 | 固定 1.5 | 仅展示，不应变化 |

### 3.4 趋势图（Sparkline）

**WR 曲线（4 行高度）：**
```
│  Win Rate                                          ▅▆▇  62%                 │
│                                               ▃▄▅█                          │
│                                          ▃▅▆█                               │
│                                     ▂▃▅█                                    │
```
- 纯 policy：看到锯齿形震荡 → 不好
- MCTS：看到持续上升 → 好

**Entropy 曲线（2 行高度，MCTS 专属）：**
```
│  Entropy                                    █▇▆▅▄▃▂▁  2.85                 │
│                                             ██████▇▆                        │
```
- 从高到低下降 → 好（模型学到了东西，搜索越来越聚焦）
- 平坦不变 → 模型没有学到新知识

### 3.5 事件日志
```
│  1:23  Probe: 62% (avg:58%) (50 games vs L0)                               │
│  2:15  ⚠ Checkpoint: wr060_c0010 → WR=62.0% (200 games, L0)               │
```

---

## 4. 测试后需要带回的文件

在 Mac 上测试完成后，将以下文件拷贝回开发机进行分析：

```bash
# 必须带回
cp output/tracker.db /path/to/dev/updates/mcts_test.db

# 可选（如果想查看具体棋局）
# cp -r output/<run-uuid>/recordings /path/to/dev/updates/mcts_recordings/
```

**tracker.db 中已记录的完整数据：**

| 表 | 内容 | 用途 |
|----|------|------|
| `runs` | 每个 run 的超参快照（含 `mcts_simulations`）| 对比 MCTS vs 纯 policy |
| `cycle_metrics` | 每 cycle 的 loss、WR、games、buffer | WR 曲线形态分析 |
| `checkpoints` | 里程碑模型的 WR、eval 详情 | 最佳模型定位 |
| `recordings` | 棋局文件元数据 | 棋谱质量分析 |

### 开发机上的分析命令

```bash
# 把 mcts_test.db 放到 output/tracker.db
cp updates/mcts_test.db output/tracker.db

# 查看所有 run（对比 MCTS vs 非 MCTS）
python3 framework/analyze.py --runs

# MCTS run 的停滞检测（应该没有停滞）
python3 framework/analyze.py --stagnation <mcts_run_id>

# 对照组的停滞检测（可能检测到停滞）
python3 framework/analyze.py --stagnation <baseline_run_id>

# 对比两个 run
python3 framework/analyze.py --compare <mcts_run_id> <baseline_run_id>

# Pareto 分析
python3 framework/analyze.py --pareto

# 查看 WR 曲线细节
python3 -c "
import sqlite3
conn = sqlite3.connect('output/tracker.db')
rows = conn.execute('''
    SELECT r.mcts_simulations, m.cycle, m.win_rate, m.loss, m.total_games
    FROM cycle_metrics m JOIN runs r ON m.run_id = r.id
    WHERE m.win_rate IS NOT NULL
    ORDER BY r.mcts_simulations, m.cycle
''').fetchall()
for sims, cycle, wr, loss, games in rows:
    tag = 'MCTS' if sims and sims > 0 else 'PURE'
    print(f'{tag:5s}  cycle {cycle:>4d}  WR {wr:.1%}  loss {loss:.3f}  games {games}')
"
```

---

## 5. 成功/失败判定

### 成功标准

测试 A 运行完成后，以下 **任意两条** 成立即为 MCTS 验证成功：

1. **WR 曲线形态改善：** MCTS run 的后半段 WR 标准差 < 纯 policy run 的后半段 WR 标准差
2. **停滞检测差异：** 纯 policy run 触发停滞警告，MCTS run 不触发
3. **Entropy 下降趋势：** MCTS entropy 曲线在训练过程中呈下降趋势（搜索越来越聚焦）

### 失败信号

- MCTS run 的 WR 曲线和纯 policy 一样震荡 → MCTS 搜索深度不够或 value head 太差
- Sim/s < 100 → Metal buffer 问题，需要调整 `mx.clear_cache()` 频率
- 训练 crash（Metal resource limit exceeded）→ 降低 `--parallel-games` 或 `--mcts-sims`
- Entropy 始终 >4.0 不下降 → 搜索接近随机，50 sims 不够

### 失败时的后备方案

| 问题 | 后备 |
|------|------|
| Metal crash | `--mcts-sims 25 --parallel-games 4` |
| 太慢（<5 cycles/min） | `--mcts-sims 25 --parallel-games 4` |
| WR 不收敛 | 延长到 `--time-budget 600` 再观察 |
| Entropy 不下降 | 增加到 `--mcts-sims 200 --parallel-games 2` |

---

## 6. 完整测试流程

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 安装依赖
uv sync

# 3. 运行 MCTS 测试（~5 分钟）
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 4. 运行纯 policy 对照（~5 分钟）
uv run python domains/gomoku/train.py \
  --parallel-games 8 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 5. 快速查看结果
uv run python framework/analyze.py --runs

# 6. 拷贝 DB 回开发机
cp output/tracker.db /path/to/share/mcts_test.db
```

总耗时：~12 分钟（两个 5 分钟 run + 操作时间）。
