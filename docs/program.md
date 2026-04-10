# mag-gomoku autoresearch

Train a Gomoku (五子棋) AI using the autoresearch experiment loop on Apple Silicon (MLX). The agent autonomously modifies `src/train.py` to discover the best neural network architecture and training strategy for beating increasingly strong opponents.

**Monorepo note:** This project lives inside `autoresearch-mlx/mag-gomoku/`. Always stage only `autoresearch-mlx/mag-gomoku/` paths. Never use blind `git add -A`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `src/game.py` — board engine, rendering, batch self-play. Do not modify.
   - `src/prepare.py` — minimax opponents (L0-L3), evaluation, checkpoint archive. Do not modify.
   - `src/train.py` — the file you modify. NN architecture, self-play, training loop.
4. **Install dependencies**: `cd autoresearch-mlx/mag-gomoku && uv sync`
5. **Initialize data/results.tsv**: Run `uv run python src/train.py` once to establish YOUR baseline on this hardware.
6. **Confirm and go**.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). Launch: `uv run python src/train.py`.

**What you CAN do:**
- Modify `src/train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, self-play strategy, batch size, temperature schedule, replay buffer, MCTS, loss function, etc.

**What you CANNOT do:**
- Modify `src/prepare.py` or `src/game.py`. They are read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness (`evaluate_win_rate` in `src/prepare.py`).

**The goal is simple: get the highest win_rate against the current evaluation opponent.** Since the time budget is fixed, you don't need to worry about training time. Everything is fair game as long as the code runs and finishes within budget.

**Simplicity criterion**: Same as upstream autoresearch. Simpler is better. A 1% win_rate improvement that adds ugly complexity? Probably not worth it. A 1% improvement from deleting code? Keep.

## Evaluation metric

The single metric is `win_rate` — the fraction of games won against a fixed minimax opponent at the current evaluation level.

```
win_rate = wins / n_games   (higher is better, range [0.0, 1.0])
```

Evaluation runs automatically at the end of `src/train.py`. The evaluation opponent level is controlled by `EVAL_LEVEL` in train.py (default: 0, starting with random opponent).

## Stage promotion

As the agent improves, promote to harder opponents:

| Stage | Opponent | Promotion threshold | Action |
|---|---|---|---|
| 0 | L0 (random) | win_rate > 0.95 | Set EVAL_LEVEL = 1 |
| 1 | L1 (minimax depth 2) | win_rate > 0.80 | Set EVAL_LEVEL = 2 |
| 2 | L2 (minimax depth 4) | win_rate > 0.60 | Set EVAL_LEVEL = 3 |
| 3 | L3 (minimax depth 6) | Keep optimizing | No limit |

When promoting:
1. **Archive the current model**: call `prepare.archive_checkpoint()` with a descriptive tag
2. **Update EVAL_LEVEL** in train.py
3. **Record the new baseline** in data/results.tsv
4. **Continue the loop** — the new baseline win_rate will be lower, and you climb again

## Output format

The script prints a summary at the end:

```
---
training_seconds: 300.0
total_seconds:    452.3
peak_vram_mb:     1842.0
num_params_K:     876.5
total_games:      12000
total_train_steps: 1750
final_loss:       0.8234

---
win_rate:         0.7300
eval_level:       2
wins:             146
losses:           38
draws:            16
avg_game_length:  47.2
```

Read results:
```
grep "^win_rate:" output/run.log
```

## Logging results

Log to `data/data/results.tsv` (tab-separated):

```
commit	win_rate	eval_level	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. win_rate achieved (e.g. 0.7300) — use 0.0000 for crashes
3. eval_level (0-3)
4. peak memory in GB (rough estimate from output)
5. status: `keep`, `discard`, or `crash`
6. short text description

Example:
```
commit	win_rate	eval_level	memory_gb	status	description
a1b2c3d	0.4200	0	1.2	keep	baseline: 6 res blocks, 64 filters
e5f6g7h	0.6800	0	1.2	keep	increase parallel games to 128
i9j0k1l	0.9700	0	1.2	keep	temperature annealing + lower LR
m2n3o4p	0.8200	1	1.2	keep	promoted to L1, new baseline
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `src/train.py` with an experimental idea
3. `git add src/train.py && git commit -m "experiment: <description>"`
4. Run: `uv run python src/train.py > output/run.log 2>&1`
5. Read results: `grep "^win_rate:\|^eval_level:" output/run.log`
6. If grep is empty, run crashed. `tail -n 50 output/run.log` for stack trace.
7. Record in data/results.tsv
8. If win_rate improved: `git add data/results.tsv && git commit --amend --no-edit`
9. If win_rate same/worse: `git reset --hard <previous kept commit>`
10. **Check for stage promotion**: if win_rate exceeds promotion threshold, promote

**Timeout**: Each experiment should take ~7-8 minutes total. Kill after 15 minutes.

**Stage promotion workflow**: When promoting to a harder opponent:
```bash
# 1. Archive checkpoint
python3 -c "from prepare import archive_checkpoint; archive_checkpoint('model.safetensors', 'stage1_beat_random', {'win_rate': 0.97, 'eval_level': 0, 'experiment': 12})"

# 2. Update EVAL_LEVEL in train.py to the next level
# 3. Commit and continue the loop
```

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
