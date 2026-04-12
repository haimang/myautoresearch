# v13 Findings (2) — vs L1 训练失败分析与修复

> 2026-04-12 | 基于 mcts_6th_exp.db + mcts_6th_exp.png

---

## 1. Bug 修复：subprocess eval 路径冲突

### 错误

```
Running final evaluation vs L1 (200 games)...
Evaluation error (exit 1):
TypeError: evaluate_win_rate() got an unexpected keyword argument 'tag'
```

### 根因

`_subprocess_eval()` 启动子进程时只设置 `PYTHONPATH=domains/gomoku/`，但 `import train` 触发 train.py 的路径设置将 `framework/` 也加入 `sys.path`。

`framework/prepare.py`（模板）有一个同名函数 `evaluate_win_rate(model_path, level, n_games, record_games, experiment_id)` —— **没有 `tag` 和 `run_id` 参数**。

子进程的 `from prepare import evaluate_win_rate` 在某些情况下会解析到框架模板而非 `domains/gomoku/prepare.py`，导致参数不匹配。

### 修复

在 `_subprocess_eval()` 中，显式设置 `PYTHONPATH=domains/gomoku:framework`，确保 domain 目录优先于 framework 目录：

```python
env = {**os.environ, "PYTHONPATH": f"{src_dir}:{fw_dir}", "PYTHONUNBUFFERED": "1"}
```

**已在 train.py 中修复。**

---

## 2. 训练数据分析

### b23b088a (8x64, MCTS-50, vs L1, 从零开始)

| 阶段 | Cycle | WR vs L1 | 说明 |
|------|-------|----------|------|
| 阶段 1 | 5-50 | **0.0%** | 完全无法赢 L1 |
| 阶段 2 | 55-110 | **50.0%** | 卡在精确 50% |
| 阶段 3 | 110 | 0.0% | 偶尔崩回 |

TUI 截图关键指标：
- **Focus 10%** — MCTS 搜索完全没聚焦（50 sims 分散在 225 个位置上）
- **Entropy 2.85** — 接近均匀分布
- **AvgLen 35.0** — 棋局很短（L1 快速获胜）
- **Loss 从 6.14 降到 4.90** — 模型在学，但学的是垃圾

### 为什么 WR 卡在 0% 和 50%

- **0%**：模型完全随机，L1 的 depth-2 搜索轻松赢每盘
- **50%**：probe eval 50 盘（25 盘执黑，25 盘执白）。模型可能学会了利用先手优势偶尔赢，但只赢了一半 = 执黑全赢执白全输，或某种固定模式

### 核心问题：鸡与蛋困境

```
┌───────────────────────────────────────────────┐
│  模型从零开始 → 网络输出 ≈ 均匀分布            │
│       ↓                                        │
│  MCTS-50 搜索 → 50 sims / 225 位置 = 每位置 0.2 次 │
│       ↓                                        │
│  搜索结果 ≈ 均匀分布 → policy target 无信息     │
│       ↓                                        │
│  训练后模型仍然 ≈ 均匀分布 → 回到起点           │
└───────────────────────────────────────────────┘
```

**50 sims 在 225 位置的棋盘上不够用。** AlphaZero 用 800 sims 在 19x19 的围棋上。我们 50 sims 在 15x15 上，每个位置平均只有 0.2 次访问。搜索完全无法突破随机先验。

---

## 3. 可行方案

### 方案 A：从 S0 resume（不从零开始）

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --resume d6c6bce4 --seed 42
```

S0 (d6c6bce4) 虽然只是赢随机的水平，但它的网络先验比随机好。MCTS 搜索基于非均匀先验，能产生更聚焦的 policy target。

### 方案 B：大幅增加 MCTS sims

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 200 --parallel-games 8 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --seed 42
```

200 sims 每位置平均 ~1 次访问，搜索能开始发现 1-2 步的战术。代价：速度降到 ~35 局/分钟。

### 方案 C（推荐）：从 S0 resume + 200 sims

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 200 --parallel-games 8 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --resume d6c6bce4 --seed 42
```

组合两个优势：
1. S0 提供非均匀先验 → MCTS 搜索更有效
2. 200 sims → 搜索深度足以发现基本战术
3. vs L1 eval → 提供真实的进步衡量

---

## 4. 从此次失败中学到的

| 教训 | 数据 |
|------|------|
| 50 sims 在 225 位置上不够发现战术 | Focus 10%，entropy 2.85 |
| 从零开始 vs 强对手 = 完全学不会 | 0% WR 持续 50 cycles |
| MCTS 放大网络质量，不创造新知识 | 随机网络 + MCTS = 随机搜索 |
| 必须有非随机起点才能 bootstrap | S0 resume 是关键 |
| subprocess eval 路径需要显式设置 | framework/prepare.py 阴影了 domain 版本 |
