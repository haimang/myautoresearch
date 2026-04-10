# MAG Gomoku

Train a Gomoku (五子棋) AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent autonomously modifies the training code, runs experiments, and evolves the neural network — all while you sleep.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python src/train.py                     # run one 5-minute training experiment
uv run python src/play.py --list               # list saved checkpoints
uv run python src/play.py --list-opponents     # list registered NN opponents
uv run python src/play.py                      # play against the trained AI
```

## Project structure

```
mag-gomoku/
├── src/                         # Python source code
│   ├── game.py                  #   board engine, pygame renderer, batched self-play (read-only)
│   ├── prepare.py               #   minimax opponents, evaluation (read-only)
│   ├── train.py                 #   neural network, training loop, plain-text TUI (agent-mutable)
│   ├── tracker.py               #   SQLite experiment tracking (output/tracker.db)
│   ├── play.py                  #   human vs AI / AI vs AI gameplay
│   └── replay.py                #   replay recorded games, export frames for video
├── docs/                        # documentation
│   ├── program.md               #   autonomous experiment protocol (agent reads this)
│   └── caveats.md               #   pitfall records & troubleshooting notes
├── output/                      # generated artifacts (gitignored)
│   ├── tracker.db               #   SQLite database — global index across all runs
│   ├── opponents/               #   registered NN opponents
│   │   └── <alias>/model.safetensors
│   └── <uuid>/                  #   per-run output directory
│       ├── model.safetensors    #     final trained model
│       ├── checkpoints/         #     model snapshots at win-rate milestones
│       └── recordings/games/    #     full game records (JSON, bound to checkpoints)
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

## Training

The training script features a plain-text dashboard with sparkline charts, automatic checkpoint export at win-rate milestones, and full experiment tracking via SQLite.

```bash
# 默认训练（无时间限制时自动设 300 秒安全限）
uv run python src/train.py

# 训练到 80% 胜率停止（无时间限制）
uv run python src/train.py --target-win-rate 0.80

# 10 分钟快速训练，每 5 cycle 评估一次
uv run python src/train.py --time-budget 600 --eval-interval 5

# 长时间训练到 95% 胜率，详细评估
uv run python src/train.py --target-win-rate 0.95 --time-budget 7200 --probe-games 100

# 低负载持久训练（16 并行对局，降低 GPU 占用）
uv run python src/train.py --target-win-rate 0.80 --parallel-games 16

# 对战 minimax depth-2 对手训练
uv run python src/train.py --eval-level 1 --target-win-rate 0.80

# 对战注册的 NN 对手训练
uv run python src/train.py --eval-opponent L0 --target-win-rate 0.80

# 从上一次训练断点续训（支持短 UUID 前缀）
uv run python src/train.py --resume c8d815ac --time-budget 600

# 后台运行并记录日志
PYTHONUNBUFFERED=1 uv run python src/train.py --target-win-rate 0.80 > output/train.log 2>&1 &
```

Each run creates its own directory under `output/<uuid>/` with isolated model, checkpoints, and recordings.

### Parameter reference

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-budget` | 无 (可选) | 最大训练时间（秒），不设则无限运行直到达标 |
| `--target-win-rate` | 无 (可选) | 达到此平滑胜率后停止 |
| `--target-games` | 无 (可选) | 达到此局数后停止 |
| `--parallel-games` | 64 | 并行自对弈局数（降低可减少 GPU 占用） |
| `--eval-level` | 0 | 对手: 0=random, 1=minimax2, 2=minimax4, 3=minimax6 |
| `--eval-opponent` | 无 (可选) | 对战注册的 NN 对手（别名），与 eval-level 互不影响 |
| `--eval-interval` | 15 | 每 N 个 cycle 做一次 probe 评估 |
| `--probe-games` | 50 | probe 评估的游戏数 |
| `--probe-window` | 3 | 平滑胜率的滑动窗口大小 |
| `--full-eval-games` | 200 | checkpoint 完整评估的游戏数 |
| `--resume` | — | 从指定 UUID 的最新 checkpoint 续训（支持短前缀） |

### Opponent management

Register a trained checkpoint as a named NN opponent for future training:

```bash
# Register a checkpoint as opponent with alias
uv run python src/train.py --register-opponent L0 --from-run 374d567f --from-tag wr065_c0310 --description "5000-game baseline"

# List registered opponents
uv run python src/play.py --list-opponents

# Train against registered opponent
uv run python src/train.py --eval-opponent L0 --target-win-rate 0.80
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
uv run python src/train.py --resume <uuid>
```

Resume creates a **new run** with its own UUID directory. The model weights are loaded from the previous run's last checkpoint, and training continues from the saved cycle number.

## How it works

```
autoresearch loop (agent modifies src/train.py → self-play + train → evaluate win_rate → keep/revert)
        ↓
  output/<uuid>/checkpoints/*.safetensors  (snapshots at milestones)
  output/tracker.db                        (full experiment history across all runs)
        ↓
  src/play.py (human vs AI game)
  src/replay.py (video production)
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
uv run python src/train.py   # establish baseline
# Point Claude Code at docs/program.md and let it run
```

Each experiment takes ~8 minutes (5 min training + eval). Expect ~7 experiments/hour, ~70 overnight.

## Playing against the AI

```bash
uv run python src/play.py                              # vs latest model
uv run python src/play.py --checkpoint best            # vs best archived model
uv run python src/play.py --black stage0 --white best  # watch two AIs play
uv run python src/play.py --level 2                    # vs minimax (no NN)
uv run python src/play.py --list                       # list all checkpoints
uv run python src/play.py --list-opponents             # list registered NN opponents
```

## Recording and replay

Training automatically records all evaluation games at each checkpoint. Recordings are bound to checkpoints in the SQLite database.

```bash
# Replay a specific game
uv run python src/replay.py output/<uuid>/recordings/games/wr070_c0045_game003.json

# Export frames for video
uv run python src/replay.py output/<uuid>/recordings/games/wr070_c0045_game003.json --export

# Growth montage
uv run python src/replay.py --montage
```

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

Tested on M3 Max 128GB. The model is tiny (~564K params) — any Apple Silicon Mac will work. Default config: 64 parallel self-play games, 6 residual blocks, 64 filters.

## License

MIT
