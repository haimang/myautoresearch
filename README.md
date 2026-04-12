# MAG Gomoku

Train a Gomoku AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent reads experiment reports, forms hypotheses, modifies the training code, and drives the research loop — all autonomously.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Architecture

```
mag-gomoku/
├── framework/                   # ═══ autoresearch framework (domain-agnostic) ═══
│   ├── core/                    #   core infrastructure layer
│   │   ├── db.py                #     SQLite experiment tracking & CRUD        [READ-ONLY]
│   │   ├── tui.py               #     TUI rendering helpers (sparklines)       [READ-ONLY]
│   │   └── mcts.py              #     Domain-agnostic MCTS algorithm           [READ-ONLY]
│   ├── analyze.py               #   experiment reports + training analysis     [READ-ONLY]
│   ├── sweep.py                 #   batch hyperparameter sweep                 [READ-ONLY]
│   ├── train.py                 #   training loop template                     [TEMPLATE]
│   ├── prepare.py               #   evaluation harness template                [TEMPLATE]
│   └── program.md               #   experiment protocol template               [TEMPLATE]
├── domains/
│   └── gomoku/                  # ═══ Gomoku domain ═══
│       ├── game.py              #   board engine, renderer, batch self-play    [READ-ONLY]
│       ├── prepare.py           #   minimax opponents L0-L3, eval harness      [READ-ONLY]
│       ├── train.py             #   NN, MCTS, self-play, training loop, CLI    [AGENT-EDITABLE]
│       ├── program.md           #   Gomoku experiment protocol (agent reads)
│       ├── play.py              #   human vs AI / AI vs AI gameplay            [READ-ONLY]
│       ├── play_service.py      #   shared gameplay services                   [READ-ONLY]
│       ├── replay.py            #   replay recorded games, export frames       [READ-ONLY]
│       └── web/                 #   browser-based UI
├── output/                      # generated artifacts (gitignored)
│   ├── tracker.db               #   SQLite database — global index
│   ├── opponents/               #   registered NN opponents
│   └── <uuid>/                  #   per-run output directory
├── updates/                     # experiment archives & design documents
├── pyproject.toml
└── uv.lock
```

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python domains/gomoku/train.py --mcts-sims 50 --parallel-games 10 --time-budget 300
uv run python framework/analyze.py --runs
uv run python domains/gomoku/play.py
```

## Training

### MCTS training (recommended)

MCTS (Monte Carlo Tree Search) produces high-quality training signal by searching ahead before each move. The policy target becomes the MCTS visit distribution instead of the raw network output.

```bash
# MCTS-50: 50 simulations per move, 10 parallel games, 15 minutes
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 900

# MCTS with larger model (8 residual blocks)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --num-blocks 8 \
  --time-budget 1800

# MCTS with GPU batch tuning (increase sims_per_round for larger GPU batches)
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 32 --mcts-batch 8 \
  --time-budget 600

# MCTS with early stopping on WR plateau
uv run python domains/gomoku/train.py \
  --mcts-sims 50 --parallel-games 10 --time-budget 3600 \
  --auto-stop-stagnation --stagnation-window 10
```

### Pure policy training (no MCTS)

```bash
# Default: batched self-play with softmax sampling
uv run python domains/gomoku/train.py --time-budget 300

# With specific target
uv run python domains/gomoku/train.py --target-win-rate 0.80 --time-budget 600
```

### Training against specific opponents

```bash
# Train against minimax depth-2
uv run python domains/gomoku/train.py --mcts-sims 50 --eval-level 1 --time-budget 1800

# Train against a registered NN opponent
uv run python domains/gomoku/train.py --mcts-sims 50 --eval-opponent S0 --time-budget 1800

# Mixed training: 20% vs NN opponent, 80% self-play
uv run python domains/gomoku/train.py --mcts-sims 50 \
  --train-opponent S0 --opponent-mix 0.2 --time-budget 1800

# Resume from a previous run
uv run python domains/gomoku/train.py --mcts-sims 50 --resume <uuid> --time-budget 1800
```

### Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--time-budget` | None | Training time limit (seconds) |
| `--target-win-rate` | None | Stop when smoothed WR reaches this |
| `--target-games` | None | Stop after N games |
| `--parallel-games` | 64 | Simultaneous self-play games |
| `--num-blocks` | 6 | Residual blocks (model depth) |
| `--num-filters` | 64 | Conv channels (model width) |
| `--learning-rate` | 5e-4 | Learning rate |
| `--steps-per-cycle` | 30 | Gradient steps per self-play cycle |
| `--buffer-size` | 50000 | Replay buffer capacity |
| `--mcts-sims` | 0 | MCTS simulations per move (0=disable) |
| `--mcts-batch` | auto | Sims per tree per GPU batch (auto=min(8, mcts-sims)) |
| `--c-puct` | 1.5 | MCTS exploration constant |
| `--dirichlet-alpha` | 0.03 | Root Dirichlet noise alpha |
| `--eval-level` | 0 | Minimax opponent: 0=random, 1=depth2, 2=depth4, 3=depth6 |
| `--eval-opponent` | L4 | Registered NN opponent for probe eval |
| `--no-eval-opponent` | — | Disable NN opponent, minimax only |
| `--eval-interval` | 15 | Probe eval every N cycles |
| `--probe-games` | 100 | Games per probe eval |
| `--probe-window` | 5 | Smoothed WR window |
| `--full-eval-games` | 200 | Games per checkpoint full eval |
| `--seed` | None | Random seed |
| `--resume` | None | Resume from run UUID |
| `--auto-stop-stagnation` | off | Stop on WR plateau |
| `--stagnation-window` | 10 | Eval points for stagnation detection |

### Model sizes

| Architecture | Parameters | Suggested use |
|-------------|-----------|---------------|
| 6x64 | 564K | Stage 0-1 (vs random, vs minimax L1) |
| 8x64 | 713K | Stage 1-2 (vs minimax L1-L2) |
| 10x64 | 862K | Stage 2-3 (vs minimax L2-L3) |
| 8x128 | 2.6M | Stage 3+ (stronger opponents) |

## Opponent management

Register trained checkpoints as named NN opponents for future training and evaluation.

```bash
# Register a checkpoint as opponent
uv run python domains/gomoku/train.py \
  --register-opponent S0 \
  --from-run <run-uuid> \
  --from-tag wr099_c0224 \
  --description "MCTS-50 trained, 99% vs L0"

# List all registered opponents
uv run python domains/gomoku/play.py --list-opponents

# Train against a registered opponent
uv run python domains/gomoku/train.py --mcts-sims 50 --eval-opponent S0 --time-budget 1800

# Use as mixed training partner
uv run python domains/gomoku/train.py --mcts-sims 50 \
  --train-opponent S0 --opponent-mix 0.2 --time-budget 1800
```

| Flag | Required | Description |
|------|----------|-------------|
| `--register-opponent ALIAS` | Yes | Opponent name (e.g., S0, S1) |
| `--from-run UUID` | Yes | Source run ID (prefix OK) |
| `--from-tag TAG` | Yes | Source checkpoint tag |
| `--description TEXT` | No | Optional description |

### Stage promotion

| Stage | Opponent | Promotion threshold | Action |
|-------|----------|-------------------|--------|
| 0 | L0 (random) | WR > 95% | Register as S0 |
| 1 | L1 (minimax depth 2) | WR > 80% | Register as S1 |
| 2 | L2 (minimax depth 4) | WR > 60% | Register as S2 |
| 3 | L3 (minimax depth 6) | Keep optimizing | — |

## Analysis tools

`analyze.py` provides read-only queries against the tracker database.

```bash
# Experiment report (Chinese markdown / structured JSON)
uv run python framework/analyze.py --report
uv run python framework/analyze.py --report --format json

# List all runs with stats
uv run python framework/analyze.py --runs

# Best checkpoint per run
uv run python framework/analyze.py --best

# Win-rate progression frontier
uv run python framework/analyze.py --frontier

# Side-by-side run comparison
uv run python framework/analyze.py --compare <run_a> <run_b>

# Step-normalized comparison (fair MCTS vs pure-policy comparison)
uv run python framework/analyze.py --compare-by-steps <run_a> <run_b>

# Training stability report
uv run python framework/analyze.py --stability <run_id>

# Stagnation detection (WR plateau)
uv run python framework/analyze.py --stagnation <run_id>

# Pareto frontier (WR vs params vs wall time)
uv run python framework/analyze.py --pareto
uv run python framework/analyze.py --pareto --format json

# Resume chain
uv run python framework/analyze.py --lineage <run_id>

# Registered opponents
uv run python framework/analyze.py --opponents

# Sweep matrix results
uv run python framework/analyze.py --matrix <tag_prefix>
```

## Hyperparameter sweep

```bash
# Preview matrix
uv run python framework/sweep.py --train-script domains/gomoku/train.py \
  --num-filters 32,64 --learning-rate 3e-4,5e-4 --seeds 42,137 \
  --time-budget 120 --tag arch_search --dry-run

# Run sweep
uv run python framework/sweep.py --train-script domains/gomoku/train.py \
  --num-filters 32,64 --learning-rate 3e-4,5e-4 --seeds 42,137 \
  --time-budget 120 --tag arch_search

# View results
uv run python framework/analyze.py --matrix arch_search
```

## Playing against the AI

```bash
uv run python domains/gomoku/play.py                              # vs latest model
uv run python domains/gomoku/play.py --checkpoint best            # vs best model
uv run python domains/gomoku/play.py --black stage0 --white best  # AI vs AI
uv run python domains/gomoku/play.py --level 2                    # vs minimax L2
uv run python domains/gomoku/play.py --list                       # list checkpoints
uv run python domains/gomoku/play.py --list-opponents             # list NN opponents
```

## Recording and replay

```bash
uv run python domains/gomoku/replay.py output/<uuid>/recordings/games/<file>.json
uv run python domains/gomoku/replay.py <file>.json --export       # export frames
uv run python domains/gomoku/replay.py --montage                  # growth montage
```

## Hardware

Tested on M3 Max 128GB. Any Apple Silicon Mac works. Default model is ~564K params. MCTS training at 50 sims achieves ~120 games/min on M3 Max with `--parallel-games 10`.

## License

MIT
