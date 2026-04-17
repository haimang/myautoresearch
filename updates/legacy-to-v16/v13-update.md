# v13 Update — Action Plan

> 2026-04-12  
> 前置：[v12-findings.md](./v12-findings.md)（MCTS 验证成功，12.9x 加速）  
> 状态：MCTS-50 vs L0 random 达到 99.2% WR，122 局/分钟

---

## 1. 版本目标

**一句话：把 MCTS 从"赢随机对手"推进到"赢 minimax 对手"，同时最大化硬件利用率。**

v12 证明了 MCTS 训练信号有效。但 v12 的全部实验都是 vs L0 (random)——这是最弱的对手。v13 要回答：MCTS 训练能否产生一个真正会下棋的模型？

---

## 2. Phase 总览

| Phase | 内容 | 关键产出 |
|-------|------|---------|
| 1 | 对手阶段晋升 (L0→L1→L2→L3) | 验证 MCTS 能否打败非随机对手 |
| 2 | 模型容量提升 | 更大模型匹配更强对手 |
| 3 | 系统利用率优化 | 利用 M3 Max 的 16 核 CPU + 128GB 内存 |
| 4 | README 全面更新 | 覆盖所有现有命令和 MCTS 训练流程 |

---

## 3. Phase 1：对手阶段晋升

### 3.1 晋升路径

```
Stage 0: vs L0 (random)       → WR > 95%  → 注册为 S0, 晋升到 L1
Stage 1: vs L1 (minimax depth 2) → WR > 80%  → 注册为 S1, 晋升到 L2
Stage 2: vs L2 (minimax depth 4) → WR > 60%  → 注册为 S2, 晋升到 L3
Stage 3: vs L3 (minimax depth 6) → 持续优化
```

### 3.2 Stage 0 已完成

v12 mcts_4th_exp.db 中 30min run 达到 99.2% vs L0。可以直接注册为 S0 并开始 Stage 1。

### 3.3 训练命令

**Stage 0 → 注册 S0 对手：**

```bash
# 查看 v12 训练 run 的 checkpoints
uv run python framework/analyze.py --runs
# 找到 30min MCTS run 的 ID（例如 13f7f395）

# 查看该 run 的 checkpoints
sqlite3 -header -column output/tracker.db \
  "SELECT tag, cycle, win_rate FROM checkpoints WHERE run_id LIKE '13f7f395%' ORDER BY cycle DESC"

# 注册最佳 checkpoint 为对手 S0
uv run python domains/gomoku/train.py \
  --register-opponent S0 \
  --from-run 13f7f395 \
  --from-tag wr099_c0224 \
  --description "MCTS-50 trained, 99% vs L0 random"

# 验证注册成功
uv run python domains/gomoku/play.py --list-opponents
```

`--register-opponent` 的全部参数：

| 参数 | 必需 | 说明 |
|------|------|------|
| `--register-opponent ALIAS` | 是 | 对手别名（如 S0, S1） |
| `--from-run RUN_ID` | 是 | 来源 run UUID（支持短前缀） |
| `--from-tag TAG` | 是 | 来源 checkpoint 标签（如 wr099_c0224） |
| `--description TEXT` | 否 | 描述文本 |

**Stage 1 训练 (vs minimax L1)：**

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --target-win-rate 0.80 --seed 42
```

**Stage 2 训练 (vs minimax L2)：**

```bash
# 先注册 Stage 1 最佳模型为 S1
uv run python domains/gomoku/train.py \
  --register-opponent S1 --from-run <stage1_run> --from-tag <best_tag> \
  --description "MCTS-50 trained, 80%+ vs minimax L1"

# 用 S1 作为 resume 起点，对战 L2
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 3600 \
  --eval-level 2 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --resume <stage1_run> --target-win-rate 0.60 --seed 42
```

**设置默认 eval 对手：**

`--eval-opponent ALIAS` 让 probe evaluation 使用注册的 NN 对手（而非 minimax）：

```bash
# 对战注册的 NN 对手 S0 做 probe eval
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 1800 \
  --eval-opponent S0 --eval-level 1 \
  --eval-interval 5 --probe-games 50 --seed 42
```

注意：`--eval-level` 控制 minimax checkpoint full eval 的对手等级，`--eval-opponent` 控制 probe eval 的 NN 对手。两者可同时使用。

### 3.4 预期与风险

| Stage | 预期难度 | 风险 |
|-------|---------|------|
| 0→1 (vs minimax L1 depth 2) | 中 | minimax L1 有基本战术意识，需要 MCTS 搜索发现威胁 |
| 1→2 (vs minimax L2 depth 4) | 高 | minimax L2 搜索深度 4，可能需要更大模型 |
| 2→3 (vs minimax L3 depth 6) | 极高 | minimax L3 是强对手，可能需要 200 sims + 更大模型 |

---

## 4. Phase 2：模型容量提升

### 4.1 当前模型

| 参数 | 当前值 | 参数量 |
|------|--------|--------|
| 6x64 (6 blocks, 64 filters) | 默认 | 564K |

### 4.2 推荐的容量阶梯

基于 v11 Pareto 分析（pareto-frontier.md 第 14.2 节）的数据：

| 架构 | 参数量 | v11 最佳 WR (vs L0) | 建议用于 |
|------|--------|---------------------|---------|
| 6x64 | 564K | 71% (纯 policy) | Stage 0-1 |
| 8x64 | 713K | 85% (纯 policy) | Stage 1-2 |
| 10x64 | 862K | 85.5% (纯 policy) | Stage 2-3 |
| 8x128 | 2.6M | 未测 | Stage 3+ (如果需要) |
| 10x128 | 3.2M | 未测 | 上限探索 |

**建议：**

- Stage 1 继续用 6x64 (564K)——v12 已证明 MCTS-50 + 6x64 可以到 99% vs L0
- Stage 2 升级到 8x64 (713K)——模型容量 +26%，适配更强对手
- Stage 3 如果 8x64 不够，升级到 10x64 (862K) 或 8x128 (2.6M)

**M3 Max 128GB 可以轻松支持到 10x128 (3.2M)**——模型权重 <50MB，完全在 GPU 内存范围内。不要用 16x64 或 12x64——v11 数据已证明在同 filter 下增加 block 超过 10 层后边际收益为负（12x64 花了 84x 训练量但 WR 更低）。**增加 filters 比增加 blocks 更有效。**

**容量提升命令示例：**

```bash
# 8x64 (713K) — Stage 2
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 --eval-level 2 --no-eval-opponent \
  --resume <stage1_run> --seed 42

# 8x128 (2.6M) — Stage 3 探索
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 8 \
  --num-blocks 8 --num-filters 128 \
  --time-budget 3600 --eval-level 3 --no-eval-opponent \
  --resume <stage2_run> --seed 42
```

---

## 5. Phase 3：系统利用率优化

### 5.1 当前利用率诊断

M3 Max 硬件：16 核 CPU、40 核 GPU、128GB 统一内存。

当前 MCTS 训练的利用率：

| 资源 | 利用率 | 原因 |
|------|--------|------|
| CPU | ~1 核 / 16 核 = **6%** | Python GIL，单线程 MCTS 搜索 |
| GPU | ~5W / 25W = **20%** | 99% 时间等 Python，batch=29 太小 |
| 内存 | <2GB / 128GB = **1.5%** | 模型 564K params = 2.2MB |

### 5.2 可优化方向

#### 方向 A：多进程并行自对弈（最高 ROI）

**原理：** Python GIL 限制单进程只能用 1 个 CPU 核。但可以启动多个进程，每个进程独立运行 MCTS 自对弈，共享同一个训练循环。

**方案：** 将自对弈拆分为 `--workers N` 个子进程：

```
主进程:  训练循环（梯度更新 + checkpoint + TUI）
Worker 0: MCTS 自对弈（4 盘棋）→ 产出 training data → 发送到主进程
Worker 1: MCTS 自对弈（4 盘棋）→ 产出 training data → 发送到主进程
Worker 2: MCTS 自对弈（4 盘棋）→ 产出 training data → 发送到主进程
Worker 3: MCTS 自对弈（4 盘棋）→ 产出 training data → 发送到主进程
```

- 4 个 worker × 4 盘/worker = 16 盘并行，每个 worker 用 1 个 CPU 核
- CPU 利用率：4-8 核 / 16 核 = 25-50%
- 训练数据产出速率：4x
- 预期整体加速：**3-4x**（不是线性，因为 GPU 训练步在主进程串行）

**实现复杂度：** 高。需要 `multiprocessing` + 共享 replay buffer + 模型权重同步。

**CLI：**
```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 16 --workers 4 \
  --time-budget 3600 --eval-level 1 --seed 42
```

#### 方向 B：增大 batch 维度（中等 ROI）

**原理：** 当前 `sims_per_round=4` × `parallel_games=8-10` = GPU batch ~32。M3 Max GPU 处理 batch=32 和 batch=256 几乎同速。增大 batch 不增加 GPU 时间但增加每次调用的有效工作量。

**方案：**
- 增大 `--parallel-games` 到 20-32（直接增大 GPU batch）
- 增大 `sims_per_round` 到 8（每次 GPU 调用处理更多叶子）

```bash
# 32 并行盘 × sims_per_round=8 = batch ~256
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 32 \
  --time-budget 600 --no-eval-opponent --seed 42
```

**预期：** GPU batch 从 29 升到 ~256，GPU 功率可能从 5W 升到 10-15W。但 Python 瓶颈仍然限制总加速到 ~1.5x。

#### 方向 C：评估并行化（低 ROI 但简单）

**原理：** `evaluate_win_rate()` 当前是逐盘串行的（200 盘 × 单线程）。可以改为批量评估。

**方案：** 评估函数内部使用 `BatchBoards` 并行评估多盘棋。

**预期：** 评估时间从 ~30s 降到 ~5s（仅影响 checkpoint full eval，不影响训练循环）。

#### 方向 D：调度优化 — 训练与自对弈流水线化

**原理：** 当前循环是串行的：自对弈 → 训练 → 评估 → 自对弈 → ...  
如果让自对弈和训练重叠执行（自对弈在 CPU 上跑的同时，GPU 在做训练步），可以隐藏 Python 等待时间。

**实现：** 需要双缓冲 replay buffer + 线程/进程分离。复杂度较高。

### 5.3 推荐优先级

| 方向 | 预期加速 | 实现复杂度 | v13 建议 |
|------|---------|-----------|---------|
| B: 增大 parallel-games | ~1.5x | 零（CLI 参数） | **立即尝试** |
| A: 多进程 workers | ~3-4x | 高 | v14 考虑 |
| C: 评估并行化 | 仅影响 eval | 中 | 按需 |
| D: 训练/自对弈流水线 | ~2x | 高 | v14 考虑 |

**v13 立即可做：** `--parallel-games 32` + `sims_per_round=8`，零代码改动，看是否提升 GPU 利用率。

---

## 6. Phase 4：README 更新

README.md 需要覆盖以下缺失内容：

1. **MCTS 训练** — `--mcts-sims`, `--c-puct`, `--dirichlet-alpha` 说明和示例
2. **新 analyze 命令** — `--stagnation`, `--pareto`, `--compare-by-steps`
3. **对手注册完整流程** — register → list → eval-opponent → train-opponent
4. **框架目录结构更新** — `framework/core/` 层的说明
5. **早停机制** — `--auto-stop-stagnation`, `--stagnation-window`

---

## 7. In-scope / Out-of-scope

### In-scope

1. 对手阶段晋升训练（L0→L1→L2）+ 对手注册
2. 模型容量提升（8x64 → 8x128 探索）
3. `--parallel-games 32` 利用率测试
4. README.md 全面更新
5. `sims_per_round` 可配置化（CLI flag `--mcts-batch`）

### Out-of-scope

1. **多进程 worker 并行** — 实现复杂度高，留给 v14
2. **训练/自对弈流水线** — 需要架构重构，留给 v14
3. **Cython MCTS 热路径** — 122 局/分已足够
4. **MLX 原生张量树** — 工程复杂度极高
5. **跨领域泛化** — 仍限制在 Gomoku 域内

---

## 8. 测试命令汇总

### 基础速度测试（增大 parallel-games）

```bash
# 测试 1: pg=32, 验证 GPU 利用率是否提升
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 32 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42

# 对比: pg=10 (v12 实测基线)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 300 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 --seed 42
```

### Stage 1 训练 (vs minimax L1)

```bash
# 注册 S0 (从 v12 最佳模型)
uv run python domains/gomoku/train.py \
  --register-opponent S0 --from-run <run_id> --from-tag <tag> \
  --description "MCTS-50, 99%+ vs L0"

# Stage 1 训练 (30min)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 1800 \
  --eval-level 1 --no-eval-opponent \
  --eval-interval 5 --probe-games 50 \
  --target-win-rate 0.80 --auto-stop-stagnation --seed 42
```

### Stage 2 训练 (vs minimax L2, 升级模型)

```bash
# 注册 S1 + 升级到 8x64
uv run python domains/gomoku/train.py \
  --register-opponent S1 --from-run <stage1_run> --from-tag <tag> \
  --description "MCTS-50, 80%+ vs minimax L1"

uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 3600 --eval-level 2 --no-eval-opponent \
  --resume <stage1_run> --target-win-rate 0.60 \
  --auto-stop-stagnation --seed 42
```

### 分析

```bash
# 全部 run 概览
uv run python framework/analyze.py --runs

# 停滞检测
uv run python framework/analyze.py --stagnation <run_id>

# Pareto 前沿（含 MCTS runs）
uv run python framework/analyze.py --pareto

# 步数对比 (MCTS vs MCTS 不同 stage)
uv run python framework/analyze.py --compare-by-steps <stage0_run> <stage1_run>

# 带回 DB
cp output/tracker.db <共享路径>/v13_test.db
```

---

## 9. 成功标准

| 指标 | 目标 |
|------|------|
| Stage 1: vs L1 WR | ≥80% |
| Stage 2: vs L2 WR | ≥60% |
| pg=32 GPU 功率 | ≥10W（vs 当前 ~5W） |
| README 覆盖率 | 100% 已实现命令 |

---

## 10. 文件改动预估

| 文件 | 改动 |
|------|------|
| `README.md` | 全面重写（MCTS、新 analyze 命令、框架结构） |
| `domains/gomoku/train.py` | 小改（`--mcts-batch` CLI flag，默认 sims_per_round 调整） |
| `updates/v13-update.md` | 本文件 + 工作日志回填 |

主要工作量在训练执行（Mac 上运行）和 README 文档编写。代码改动量极小。

---

## 11. 工作日志

> 执行者：Claude Opus 4.6  
> 执行日期：2026-04-12

### 11.1 代码改动

**domains/gomoku/train.py** (+10 行)：
1. 新增 `--mcts-batch N` CLI flag — 控制 `sims_per_round`（每棵树每 GPU 轮的模拟数）
2. MCTS 启用时自动设置 `MCTS_BATCH_SIZE = min(8, mcts_sims)`（之前硬编码 4）
3. 用户可通过 `--mcts-batch 16` 手动覆盖，增大 GPU batch（32 盘 x 16 sims = batch 512）

**README.md**（全面重写，268 行）：
- 新增：MCTS 训练完整指南（命令、参数、示例）
- 新增：对手注册流程（register → list → eval → train）
- 新增：所有 analyze 命令（--stagnation, --pareto, --compare-by-steps）
- 新增：模型容量参考表（6x64 → 8x128）
- 新增：早停机制说明
- 更新：目录结构（framework/core/ 层）
- 更新：参数参考表（含所有 MCTS 参数）

### 11.2 Mac 测试命令

**命令 1：创建 8x64 MCTS 模型（900 秒，pg=32）**

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 32 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 900 \
  --no-eval-opponent --eval-interval 5 --probe-games 50 \
  --seed 42
```

说明：
- `--mcts-sims 50`：每步 50 次 MCTS 搜索
- `--parallel-games 32`：32 盘并行（GPU batch = 8x32=256，最大化 GPU 利用率）
- `--mcts-batch 8`：每棵树每轮 8 次模拟（比默认更激进，减少 GPU 调用次数）
- `--num-blocks 8 --num-filters 64`：8x64 架构（713K 参数）
- `--time-budget 900`：15 分钟训练
- `--no-eval-opponent`：仅用 minimax L0 做 probe eval
- `--seed 42`：可复现

**命令 2：注册该模型为 S0 对手**

训练完成后，查看 run ID 和最佳 checkpoint：
```bash
uv run python framework/analyze.py --runs
# 找到刚完成的 run ID（最新的一条）

# 查看该 run 的 checkpoints
uv run python framework/analyze.py --best
```

注册：
```bash
uv run python domains/gomoku/train.py \
  --register-opponent S0 \
  --from-run <run_id> \
  --from-tag <最佳checkpoint的tag> \
  --description "8x64 MCTS-50, vs L0"
```

验证：
```bash
uv run python domains/gomoku/play.py --list-opponents
```

**命令 3：使用 S0 作为 eval 对手，进行新 MCTS 训练**

```bash
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 32 --mcts-batch 8 \
  --num-blocks 8 --num-filters 64 \
  --time-budget 1800 \
  --eval-opponent S0 --eval-level 1 \
  --eval-interval 5 --probe-games 50 \
  --auto-stop-stagnation \
  --seed 42
```

说明：
- `--eval-opponent S0`：probe eval 使用 S0（注册的 NN 对手）
- `--eval-level 1`：checkpoint full eval 使用 minimax L1（depth 2）
- `--auto-stop-stagnation`：WR 停滞时自动停止
- 其余参数同命令 1

### 11.3 分析命令

```bash
# 训练后查看所有 run
uv run python framework/analyze.py --runs

# 对比两个 run（按步数归一化）
uv run python framework/analyze.py --compare-by-steps <run1> <run2>

# 停滞检测
uv run python framework/analyze.py --stagnation <run_id>

# Pareto 前沿
uv run python framework/analyze.py --pareto

# 带回 DB 到开发机分析
cp output/tracker.db <共享路径>/v13_test.db
```
