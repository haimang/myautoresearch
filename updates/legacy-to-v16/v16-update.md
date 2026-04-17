# v16 Update — S vs S 自我对弈训练（草稿骨架）

> 占位文档：v15 已为 v16 预留了入口（`--initial-opponent` + `--opponent-mix 1.0` + auto-promote）。本文件 v16 开工前会被实质性扩写。

---

## 1. v16 主题

> **基于 v15 留下的 S2 对手，开启 "S2 vs S2 从零训练" 范式。
> 主网络从 S2 权重开始，对手永远固定为 S2，直到训练出明显更强的 S3。**

---

## 2. 与 v14/v15 训练范式的区别

| 维度 | v14/v15（S0→S1→S2 链）| v16（S vs S 进化） |
|------|---------------------|-------------------|
| 起点 | random init 或 resume | `--initial-opponent S2` 加载 S2 权重 |
| 对手 | minimax (`--eval-level`) | 固定 S2 NN 权重（冻结） |
| Self-play 占比 | 100%（默认）| 0%（`--opponent-mix 1.0`） |
| Reward 信号 | self-play 终局胜负 | 对 S2 的对弈胜负 |
| 晋升路径 | 越过 minimax level | 当对 S2 胜率 ≥ 60% 时晋升为 S3 |
| Risk | minimax 上限会卡住 | "陪练对手永远不变"可能导致策略局部最优 |

v15 的 `train_opponent_model` 在加载后会调用 `model.eval()` 并冻结权重，所以"对手不更新" 是开箱即用的行为。

---

## 3. v16 的 in-scope 工作（待开工）

- [ ] 验证 `--initial-opponent S2 --opponent-mix 1.0` 能跑通完整训练循环
- [ ] 在 `run_opponent_play` 路径里加入 v15 §4.3 的开局多样化和轨迹指纹（确保 self-as-opponent 的训练数据也是多样的，不会重蹈 mcts_9th 的统计坍缩）
- [ ] 决定 `eval-level` 的语义：S2 vs S2 训练时 probe 是 vs L2（外部 truth）还是 vs S2 本身（内部 progress）—— 默认两者都跑
- [ ] 周期性 opponent snapshot 升级：每 N cycles 把当前主网络快照"作为新对手"，实现 AlphaZero 式自我进化（可选高阶）
- [ ] 收敛判据：smoothed WR vs S2 ≥ 60% 持续 stagnation_window 个 probe → 自动 `--auto-promote-to S3`
- [ ] v16 实测命令模板（见 §5）

## 4. v16 的 out-of-scope（继续推到 v17+）

- Board 核心路径 C 化（v15 已经识别为 P2，未做）
- 多进程 self-play worker
- minimax transposition table / killer move 高级优化
- analyze.py 的 SVG/PNG 可视化导出
- web 前端的 async eval 状态推送

## 5. v16 占位训练命令

```bash
# 前置：v15 已经产出了 S2 对手（来自 mcts_12 训练 + auto-promote）
uv run python framework/analyze.py --opponents | grep S2  # 确认存在

# v16 训练：S2 vs S2 从 S2 权重起步，纯对弈（无 self-play）
uv run python domains/gomoku/train.py \
  --initial-opponent S2 \
  --train-opponent S2 \
  --opponent-mix 1.0 \
  --num-blocks 8 --num-filters 128 \
  --mcts-sims 800 --parallel-games 16 --mcts-batch 32 \
  --learning-rate 3e-4 --steps-per-cycle 50 --buffer-size 100000 \
  --time-budget 21600 \
  --eval-level 2 \
  --eval-interval 5 --probe-games 80 \
  --full-eval-games 200 --eval-openings 16 \
  --auto-stop-stagnation --stagnation-window 20 \
  --auto-promote-to S3 \
  --seed 42
```

**v16 启动前必须验证：**

1. v15 `mcts_12` 的训练 land 并 auto-promote 出 `S2`
2. `framework/analyze.py --promotion-chain` 能看到 `S0 → S1v2 → S2`
3. `_run_self_play_mcts` 的 opening diversification 能覆盖 `run_opponent_play` 路径
