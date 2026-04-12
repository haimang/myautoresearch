# MAG Gomoku

Train a Gomoku (五子棋) AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent reads experiment reports, forms hypotheses, modifies the training code, and drives the research loop — all autonomously.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Architecture

The project is split into a **domain-agnostic framework** and **domain-specific implementations**:

- **`framework/`** — Reusable autoresearch infrastructure: experiment tracking, analysis, sweep, TUI, plus root templates (`train.py`, `prepare.py`, `program.md`) that serve as starting points for any new domain.
- **`domains/gomoku/`** — The Gomoku-specific instantiation: game engine, minimax opponents, neural network training, play/replay tools, and browser UI.

To add a new domain, copy the framework templates to `domains/<name>/` and fill in domain-specific logic.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python domains/gomoku/train.py                  # run one training experiment
uv run python framework/analyze.py --runs              # view all training runs
uv run python domains/gomoku/play.py --list            # list saved checkpoints
uv run python domains/gomoku/play.py --list-opponents  # list registered NN opponents
uv run python domains/gomoku/play.py                   # play against the trained AI
```

## Project structure

```
mag-gomoku/
├── framework/                   # ═══ autoresearch framework (domain-agnostic) ═══
│   ├── tracker.py               #   SQLite experiment tracking              [READ-ONLY]
│   ├── analyze.py               #   experiment reports + training analysis  [READ-ONLY]
│   ├── sweep.py                 #   batch hyperparameter sweep              [READ-ONLY]
│   ├── tui.py                   #   TUI rendering helpers                   [READ-ONLY]
│   ├── train.py                 #   training loop template (root copy)      [TEMPLATE]
│   ├── prepare.py               #   evaluation harness template (root copy) [TEMPLATE]
│   └── program.md               #   experiment protocol template            [TEMPLATE]
├── domains/
│   └── gomoku/                  # ═══ Gomoku domain ═══
│       ├── game.py              #   board engine, renderer, batch self-play [READ-ONLY]
│       ├── prepare.py           #   minimax opponents L0-L3, eval harness   [READ-ONLY]
│       ├── train.py             #   NN, self-play, training loop, TUI, CLI  [AGENT-EDITABLE]
│       ├── program.md           #   Gomoku experiment protocol (agent reads this)
│       ├── play.py              #   human vs AI / AI vs AI gameplay         [READ-ONLY]
│       ├── play_service.py      #   shared gameplay services                [READ-ONLY]
│       ├── replay.py            #   replay recorded games, export frames    [READ-ONLY]
│       └── web/                 #   browser-based UI
│           ├── web_app.py       #     FastAPI backend
│           ├── index.html
│           ├── app.js
│           └── styles.css
├── docs/                        # documentation
│   ├── program.md               #   experiment protocol template            [TEMPLATE]
│   └── caveats.md               #   pitfall records & troubleshooting
├── output/                      # generated artifacts (gitignored)
│   ├── tracker.db               #   SQLite database — global index across all runs
│   ├── opponents/               #   registered NN opponents
│   │   └── <alias>/model.safetensors
│   └── <uuid>/                  #   per-run output directory
│       ├── model.safetensors    #     final trained model
│       ├── checkpoints/         #     model snapshots at win-rate milestones
│       └── recordings/games/    #     full game records (JSON, bound to checkpoints)
├── updates/                     # experiment archives & design documents
│   └── gomoku_1st_exp.db        #   first Gomoku experiment archive (SQLite)
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

## Training

The training script features a plain-text dashboard with sparkline charts, automatic checkpoint export at win-rate milestones, and full experiment tracking via SQLite.

```bash
# 默认训练（无时间限制时自动设 300 秒安全限）
uv run python domains/gomoku/train.py

# 训练到 80% 胜率停止（无时间限制）
uv run python domains/gomoku/train.py --target-win-rate 0.80

# 10 分钟快速训练，每 5 cycle 评估一次
uv run python domains/gomoku/train.py --time-budget 600 --eval-interval 5

# 长时间训练到 95% 胜率，详细评估
uv run python domains/gomoku/train.py --target-win-rate 0.95 --time-budget 7200 --probe-games 100

# 低负载持久训练（16 并行对局，降低 GPU 占用）
uv run python domains/gomoku/train.py --target-win-rate 0.80 --parallel-games 16

# 对战 minimax depth-2 对手训练
uv run python domains/gomoku/train.py --eval-level 1 --target-win-rate 0.80

# 对战注册的 NN 对手训练
uv run python domains/gomoku/train.py --eval-opponent L0 --target-win-rate 0.80

# 混合对手训练（20% 对局 vs 注册对手，80% 自对弈）
uv run python domains/gomoku/train.py --eval-opponent L0 --train-opponent L0 --opponent-mix 0.2 --target-win-rate 0.80

# 使用更大模型（8 残差块 × 64 通道，~713K 参数）
uv run python domains/gomoku/train.py --num-blocks 8 --num-filters 64 --target-win-rate 0.80

# 使用更大 replay buffer
uv run python domains/gomoku/train.py --buffer-size 100000 --target-win-rate 0.80

# 调整学习率和每 cycle 训练步数
uv run python domains/gomoku/train.py --learning-rate 3e-4 --steps-per-cycle 50 --target-win-rate 0.80

# 可复现实验（固定随机种子）
uv run python domains/gomoku/train.py --seed 42 --time-budget 300

# 从上一次训练断点续训（支持短 UUID 前缀）
uv run python domains/gomoku/train.py --resume c8d815ac --time-budget 600

# 后台运行并记录日志
PYTHONUNBUFFERED=1 uv run python domains/gomoku/train.py --target-win-rate 0.80 > output/train.log 2>&1 &
```

Each run creates its own directory under `output/<uuid>/` with isolated model, checkpoints, and recordings.

### Parameter reference

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-budget` | 无 (可选) | 最大训练时间（秒），不设则无限运行直到达标 |
| `--target-win-rate` | 无 (可选) | 达到此平滑胜率后停止 |
| `--target-games` | 无 (可选) | 达到此局数后停止 |
| `--parallel-games` | 64 | 并行自对弈局数（降低可减少 GPU 占用） |
| `--num-blocks` | 6 | 残差块数量（模型深度） |
| `--num-filters` | 64 | 卷积通道数（模型宽度） |
| `--learning-rate` | 5e-4 | 学习率 |
| `--steps-per-cycle` | 30 | 每个 cycle 的训练步数 |
| `--seed` | 无 (可选) | 随机种子（random + numpy + MLX），用于可复现实验 |
| `--buffer-size` | 50000 | Replay buffer 最大样本数 |
| `--eval-level` | 0 | 对手: 0=random, 1=minimax2, 2=minimax4, 3=minimax6 |
| `--eval-opponent` | 无 (可选) | 对战注册的 NN 对手（别名），与 eval-level 互不影响 |
| `--train-opponent` | 无 (可选) | 混合训练时的 NN 对手（别名） |
| `--opponent-mix` | 0.25 | 混合训练对手对局占比（0.0–1.0） |
| `--eval-interval` | 15 | 每 N 个 cycle 做一次 probe 评估 |
| `--probe-games` | 50 | probe 评估的游戏数 |
| `--probe-window` | 3 | 平滑胜率的滑动窗口大小 |
| `--full-eval-games` | 200 | checkpoint 完整评估的游戏数 |
| `--resume` | — | 从指定 UUID 的最新 checkpoint 续训（支持短前缀） |

### Opponent management

Register a trained checkpoint as a named NN opponent for future training:

```bash
# Register a checkpoint as opponent with alias
uv run python domains/gomoku/train.py --register-opponent L0 --from-run 374d567f --from-tag wr065_c0310 --description "5000-game baseline"

# List registered opponents
uv run python domains/gomoku/play.py --list-opponents

# Train against registered opponent
uv run python domains/gomoku/train.py --eval-opponent L0 --target-win-rate 0.80
```

Probe evaluations use the NN opponent for fast feedback; checkpoint full evaluations still use minimax for consistent cross-run comparison.

### Checkpoint milestones

Checkpoints are automatically saved when win rate crosses these thresholds:
- **< 80%**: every 5% (50%, 55%, 60%, 65%, 70%, 75%)
- **80–90%**: every 2% (80%, 82%, 84%, 86%, 88%)
- **> 90%**: every 1% (90%, 91%, ..., 99%, 100%)

Each checkpoint includes: model weights, 200-game full evaluation, and all game recordings.

### Resuming training

If a training run is interrupted, or you want to continue from a previous experiment:

```bash
# List past runs
sqlite3 -header -column output/tracker.db "SELECT substr(id,1,8) AS run, status, total_cycles, final_win_rate FROM runs ORDER BY created_at DESC"

# Resume from the last checkpoint of a specific run
uv run python domains/gomoku/train.py --resume <uuid>
```

Resume creates a **new run** with its own UUID directory. The model weights are loaded from the previous run's last checkpoint, and training continues from the saved cycle number.

## How it works

```
┌─────────────────────────────────────────────────────────┐
│  AI Agent (Claude Code / Cursor / Copilot CLI)          │
│  = Research Director                                     │
│                                                         │
│  LOOP:                                                  │
│    1. Read report (analyze.py --report --format json)   │
│    2. Form hypothesis from signals + data               │
│    3. Edit domains/gomoku/train.py                      │
│    4. git commit                                        │
│    5. Run training (train.py --time-budget 300)         │
│    6. Read new report → compare → keep or revert        │
│                                                         │
└─────────────────────────────────────────────────────────┘
         ↑ reads output              ↓ modifies code
┌─────────────────────────────────────────────────────────┐
│  MAG-Gomoku Codebase                                    │
│                                                         │
│  domains/gomoku/train.py   ← agent edits (only mutable)│
│  framework/analyze.py      ← experiment reports         │
│  framework/sweep.py        ← batch hyperparam search    │
│  output/tracker.db         ← experiment history         │
│  domains/gomoku/prepare.py ← fixed benchmark (judge)    │
└─────────────────────────────────────────────────────────┘
```

The single metric is **win_rate** against fixed minimax opponents. The agent promotes through increasingly strong opponents:

| Stage | Opponent | Promotion |
|---|---|---|
| 0 | Random | win_rate > 95% |
| 1 | Minimax depth 2 | win_rate > 80% |
| 2 | Minimax depth 4 | win_rate > 60% |
| 3 | Minimax depth 6 | Keep optimizing |

## Running the autoresearch loop

```bash
git checkout -b autoresearch/gomoku-v1
uv run python domains/gomoku/train.py   # establish baseline
# Point Claude Code (or any coding agent) at domains/gomoku/program.md and let it run
```

The agent reads `domains/gomoku/program.md`, runs `analyze.py --report --format json` to understand the current state, then enters the experiment loop: hypothesize → edit `train.py` → train → evaluate → keep/revert → repeat.

Each experiment takes ~8 minutes (5 min training + eval). Expect ~7 experiments/hour, ~70 overnight.

### Experiment reports

The agent consumes structured experiment reports before each iteration:

```bash
uv run python framework/analyze.py --report                    # Chinese markdown (human-readable)
uv run python framework/analyze.py --report --format json      # Structured JSON (agent-consumable)
uv run python framework/analyze.py --report --recent 10        # More history context
```

Reports include: recent runs with full hyperparameters, best checkpoint, win-rate frontier, training stability, opponent registry, stage assessment, and auto-generated signals with suggested actions.

## Playing against the AI

```bash
uv run python domains/gomoku/play.py                              # vs latest model
uv run python domains/gomoku/play.py --checkpoint best            # vs best archived model
uv run python domains/gomoku/play.py --black stage0 --white best  # watch two AIs play
uv run python domains/gomoku/play.py --level 2                    # vs minimax (no NN)
uv run python domains/gomoku/play.py --list                       # list all checkpoints
uv run python domains/gomoku/play.py --list-opponents             # list registered NN opponents
```

## Recording and replay

Training automatically records all evaluation games at each checkpoint. Recordings are bound to checkpoints in the SQLite database.

```bash
# Replay a specific game
uv run python domains/gomoku/replay.py output/<uuid>/recordings/games/wr070_c0045_game003.json

# Export frames for video
uv run python domains/gomoku/replay.py output/<uuid>/recordings/games/wr070_c0045_game003.json --export

# Growth montage
uv run python domains/gomoku/replay.py --montage
```

## Analysis tool

`analyze.py` provides read-only queries against the tracker database:

```bash
uv run python framework/analyze.py --report                        # experiment report (markdown, Chinese)
uv run python framework/analyze.py --report --format json          # experiment report (JSON, for agent)
uv run python framework/analyze.py --runs                          # list all runs with stats
uv run python framework/analyze.py --best                          # best checkpoint per run
uv run python framework/analyze.py --frontier                      # WR progression frontier
uv run python framework/analyze.py --compare 374d567f 1922e0f0     # side-by-side run comparison
uv run python framework/analyze.py --lineage 1922e0f0              # trace resume chain
uv run python framework/analyze.py --opponents                     # list registered opponents
uv run python framework/analyze.py --stability 374d567f            # training stability report
uv run python framework/analyze.py --matrix sweep1                 # sweep matrix results by tag prefix
```

## Hyperparameter sweep

`sweep.py` runs batch experiments with cartesian product of hyperparameters:

```bash
# Dry-run: preview the experiment matrix
uv run python framework/sweep.py --num-filters 32,64 --learning-rate 3e-4,5e-4 --seeds 42,137 \
  --time-budget 120 --tag arch_search --dry-run

# Run the sweep (2 filter × 2 lr × 2 seeds = 8 experiments)
uv run python framework/sweep.py --num-filters 32,64 --learning-rate 3e-4,5e-4 --seeds 42,137 \
  --time-budget 120 --tag arch_search

# Resume (skip already-completed configs)
uv run python framework/sweep.py --num-filters 32,64 --learning-rate 3e-4,5e-4 --seeds 42,137 \
  --time-budget 120 --tag arch_search --resume

# View results
uv run python framework/analyze.py --matrix arch_search
```

Sweep axes: `--num-blocks`, `--num-filters`, `--learning-rate`, `--steps-per-cycle`, `--buffer-size` (comma-separated values). Fixed params: `--time-budget` (required), `--seeds`, `--tag`. Passthrough: `--eval-opponent`, `--parallel-games`, `--target-win-rate`.

## Querying experiment data

All experiment data is stored in `output/tracker.db` (SQLite):

```bash
# List all training runs
sqlite3 -header -column output/tracker.db "SELECT substr(id,1,8) AS run, status, total_cycles, final_win_rate, wall_time_s, output_dir FROM runs"

# List checkpoints for a specific run
sqlite3 -header -column output/tracker.db "SELECT tag, cycle, win_rate, eval_games FROM checkpoints WHERE run_id LIKE 'c117fa23%' ORDER BY cycle"

# Win rate progression
sqlite3 -header -csv output/tracker.db "SELECT cycle, win_rate FROM cycle_metrics WHERE win_rate IS NOT NULL" > wr_curve.csv

# Export all recordings for a checkpoint
sqlite3 -header -csv output/tracker.db "SELECT game_file, result, total_moves, nn_side, nn_won FROM recordings WHERE checkpoint_id = 1"

# View resume chain
sqlite3 -header -column output/tracker.db "SELECT substr(id,1,8) AS run, substr(resumed_from,1,8) AS parent FROM runs WHERE resumed_from IS NOT NULL"

# List registered opponents
sqlite3 -header -column output/tracker.db "SELECT alias, win_rate, eval_level, substr(source_run,1,8) AS src_run, description FROM opponents"
```

## Hardware

Tested on M3 Max 128GB. The model is tiny (~564K params default) — any Apple Silicon Mac will work. Model architecture is configurable via `--num-blocks` and `--num-filters` (e.g. 8×64 = ~713K, 6×96 = ~1.1M). Default config: 64 parallel self-play games, 6 residual blocks, 64 filters. Training uses D4 board symmetry augmentation (8× effective data), recency-weighted replay buffer, gradual temperature decay, and optional mixed opponent training.

## License

MIT
