# MAG Gomoku

Train a Gomoku (五子棋) AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent autonomously modifies the training code, runs experiments, and evolves the neural network — all while you sleep.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python src/train.py                     # run one 5-minute training experiment
uv run python src/play.py --list               # list saved checkpoints
uv run python src/play.py                      # play against the trained AI
```

## Project structure

```
mag-gomoku/
├── src/                         # Python source code
│   ├── game.py                  #   board engine, pygame renderer, batched self-play (read-only)
│   ├── prepare.py               #   minimax opponents, evaluation (read-only)
│   ├── train.py                 #   neural network, training loop, Rich TUI (agent-mutable)
│   ├── tracker.py               #   SQLite experiment tracking (output/tracker.db)
│   ├── play.py                  #   human vs AI / AI vs AI gameplay
│   └── replay.py                #   replay recorded games, export frames for video
├── docs/                        # documentation
│   ├── program.md               #   autonomous experiment protocol (agent reads this)
│   └── action-plan.md           #   project plan, dev log & troubleshooting
├── output/                      # generated artifacts (gitignored)
│   ├── tracker.db               #   SQLite database — global index across all runs
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

The training script features a Rich TUI dashboard, automatic checkpoint export at win-rate milestones, and full experiment tracking via SQLite.

```bash
# Default: 5-minute training run
uv run python src/train.py

# Custom parameters
uv run python src/train.py \
  --time-budget 600 \           # 10 minutes
  --target-win-rate 0.95 \      # stop when 95% win rate reached
  --eval-level 0 \              # opponent: 0=random, 1=minimax2, 2=minimax4, 3=minimax6
  --eval-interval 10 \          # probe eval every 10 cycles
  --probe-games 50 \            # games per probe
  --full-eval-games 200 \       # games per checkpoint evaluation
  --resume <uuid>               # resume from a previous run's last checkpoint
```

Each run creates its own directory under `output/<uuid>/` with isolated model, checkpoints, and recordings.

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
```

## Hardware

Tested on M3 Max 128GB. The model is tiny (~564K params) — any Apple Silicon Mac will work. Default config: 64 parallel self-play games, 6 residual blocks, 64 filters.

## License

MIT
