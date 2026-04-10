# mag-gomoku autoresearch

Train a Gomoku (五子棋) AI using the autoresearch experiment loop on Apple Silicon (MLX). The agent autonomously modifies `src/train.py` to discover the best neural network architecture and training strategy for beating increasingly strong opponents.

## Repository layout

```
mag-gomoku/
├── src/
│   ├── game.py        # Board engine, renderer, batch self-play          [READ-ONLY]
│   ├── prepare.py     # Minimax opponents L0-L3, evaluation harness      [READ-ONLY]
│   ├── train.py       # NN, self-play, training loop, TUI, CLI           [AGENT-EDITABLE]
│   ├── tracker.py     # SQLite experiment tracking                       [READ-ONLY]
│   ├── tui.py         # Terminal UI helpers (sparklines, panels)          [READ-ONLY]
│   ├── play.py        # Human vs AI / AI vs AI                           [READ-ONLY]
│   ├── replay.py      # Replay recorded games, export video frames       [READ-ONLY]
│   └── analyze.py     # Query tracker.db for analysis                    [READ-ONLY]
├── docs/
│   ├── program.md     # THIS FILE — agent operating instructions
│   └── caveats.md     # Known pitfalls and troubleshooting
├── output/            # All generated artifacts (gitignored)
│   ├── tracker.db     # SQLite database — global index across all runs
│   ├── opponents/     # Registered NN opponents
│   └── <uuid>/        # Per-run output directory
│       ├── model.safetensors
│       ├── checkpoints/
│       └── recordings/games/
├── pyproject.toml
└── uv.lock
```

## The single rule

**`src/train.py` is the ONLY file you modify.** All other source files are read-only.

## Setup

1. Read the in-scope files:
   - `src/game.py` — board engine (BOARD_SIZE=15, WIN_LENGTH=5), BatchBoards for parallel self-play
   - `src/prepare.py` — minimax opponents, `evaluate_win_rate()` harness
   - `src/train.py` — the file you edit: model architecture, self-play, training loop, hyperparameters
2. Install dependencies: `uv sync`
3. Run one baseline experiment: `uv run python src/train.py --time-budget 300`
4. Confirm baseline win_rate in the output summary, then begin the experiment loop.

## Experimentation

### What you CAN do
- Modify `src/train.py` — everything inside is fair game: model architecture, optimizer, hyperparameters, self-play strategy, batch size, temperature schedule, replay buffer, MCTS, loss function, etc.

### What you CANNOT do
- Modify any file other than `src/train.py`
- Install new packages or add dependencies
- Modify the evaluation harness (`evaluate_win_rate` in `src/prepare.py`)

### Simplicity criterion
Same as upstream autoresearch. Simpler is better. A 1% win_rate improvement that adds ugly complexity? Probably not worth it. A 1% improvement from deleting code? Keep.

## Evaluation metric

The single metric is `win_rate` — the fraction of games won against a fixed opponent.

```
win_rate = wins / n_games   (higher is better, range [0.0, 1.0])
```

Evaluation happens automatically:
- **Probe eval**: Every N cycles during training (lightweight, in-process)
- **Full eval**: At each checkpoint milestone (200 games, subprocess-isolated)
- **Final eval**: At run end (200 games vs minimax)

The evaluation opponent level is controlled by `--eval-level` (default: 0).

## Stage promotion

As the model improves, promote to harder opponents:

| Stage | Opponent | Promotion threshold | Action |
|---|---|---|---|
| 0 | L0 (random) | win_rate > 0.95 | Use `--eval-level 1` in next run |
| 1 | L1 (minimax depth 2) | win_rate > 0.80 | Use `--eval-level 2` in next run |
| 2 | L2 (minimax depth 4) | win_rate > 0.60 | Use `--eval-level 3` in next run |
| 3 | L3 (minimax depth 6) | Keep optimizing | No limit |

Alternatively, register a trained checkpoint as a named NN opponent and train against it with `--eval-opponent <alias>`.

## Canonical benchmark profile

For **comparable** experiments, use the standard benchmark settings:

```bash
uv run python src/train.py --time-budget 300 --eval-level 0 --probe-games 50 --eval-interval 15 --full-eval-games 200
```

Rules for benchmark runs:
- Fixed 5-minute wall-clock budget
- Fixed minimax opponent (no `--eval-opponent`)
- No `--resume` (fresh run only)
- Results are directly comparable across experiments

Exploratory runs (custom time budget, NN opponents, resume) are encouraged for development but should not be used to claim benchmark progress.

## Experiment tracking

All experiment data is stored in `output/tracker.db` (SQLite). Key tables:

- **runs**: One row per training run. UUID, hyperparams, hardware info, status, final metrics.
- **cycle_metrics**: Per-cycle loss, win_rate, buffer size, game counts.
- **checkpoints**: Model snapshots at WR milestones, with full eval results.
- **recordings**: Game records linked to checkpoints.
- **opponents**: Registered NN opponents with alias, source run, and WR.

Each run creates an isolated directory: `output/<uuid>/`. The tracker.db is the global cross-run index.

## Output format

The training script prints a summary at the end:

```
============================================================
Run:        a3f7b2c1 (target_win_rate)
Cycles:     85 (total cycle #85)
Games:      5440
Steps:      4226
Final loss: 1.168
Win rate:   73.0% (vs L0)  [benchmark]
Checkpoints:4
Wall time:  300.2s
Output:     output/a3f7b2c1-.../
Tracker:    output/tracker.db
============================================================
```

Query results from DB:
```bash
sqlite3 -header -column output/tracker.db "SELECT substr(id,1,8) AS run, status, total_cycles, final_win_rate FROM runs ORDER BY created_at DESC LIMIT 10"
```

## The experiment loop

LOOP FOREVER:

1. Read the current state: last run's win_rate, loss trajectory, any regressions
2. Form a hypothesis and modify `src/train.py`
3. `git add src/train.py && git commit -m "experiment: <description>"`
4. Run: `uv run python src/train.py --time-budget 300`
5. Read the output summary (win_rate, loss, cycles)
6. If crashed: `tail -n 50` of terminal output for stack trace
7. If win_rate improved: keep the commit
8. If win_rate same/worse: `git reset --hard <previous kept commit>`
9. Check for stage promotion: if win_rate exceeds threshold, promote in next run

**Timeout**: Each benchmark experiment should take ~6-8 minutes total (5 min training + eval). Kill after 15 minutes.

## Hints for the agent

**Architecture ideas to try:**
- More/fewer residual blocks (4-12)
- Wider/narrower filters (32-128)
- Different activation functions (ReLU → SiLU, GELU)
- Squeeze-and-Excitation blocks
- Attention mechanisms in the residual tower
- Separate learning rates for policy/value heads

**Training strategy ideas:**
- Temperature annealing during self-play
- Prioritized replay (prefer recent games or surprising outcomes)
- Curriculum learning (start with short games, gradually increase)
- Add MCTS during self-play (improves data quality, costs speed)
- Augment training data with board rotations/reflections (8x data for free)
- Adjust policy vs value loss weighting
- Learning rate warmup/decay schedules

**What to watch for:**
- If win_rate flatlines, try a more radical architecture change
- If `avg_game_length` is very short, the model may be playing badly (both sides blunder early)
- If loss is very low but win_rate doesn't improve, the model may be overfitting to the replay buffer
- If games_completed is low, self-play is too slow — reduce model size or parallel games

**NEVER STOP**: Once the loop begins, do NOT pause to ask. Run indefinitely until manually stopped. If stuck, think harder — try data augmentation, different optimizers, architectural innovations. The loop runs until interrupted.
