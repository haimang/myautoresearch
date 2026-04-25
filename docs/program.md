# Autoresearch Experiment Protocol — Template

Train a **<<<DOMAIN>>>** AI using the autoresearch experiment loop on Apple Silicon (MLX). The agent autonomously modifies `domains/<<<domain>>>/train.py` to discover the best neural network architecture and training strategy.

> **This is模板文件。** 复制到 `domains/<name>/program.md`，
> 填入领域特定细节（对手等级、棋盘维度等）后使用。
> 参考实现见 `domains/gomoku/program.md`。

## Repository layout

```
project/
├── framework/               # ═══ autoresearch framework (domain-agnostic) ═══
│   ├── index.py             # Unified CLI entrypoint                    [READ-ONLY]
│   ├── core/                # DB, TUI, shared runtime helpers          [READ-ONLY]
│   ├── facade/              # CLI facades                              [READ-ONLY]
│   ├── services/            # Frontier/reporting/research services     [READ-ONLY]
│   ├── policies/            # Stage/branch/selector/acquisition policy [READ-ONLY]
│   ├── profiles/            # Search-space/objective profiles          [READ-ONLY]
│   └── templates/           # Domain templates                         [READ-ONLY]
├── domains/
│   └── <<<domain>>>/        # ═══ domain execution ═══
│       ├── program.md       # Domain-specific agent instructions
│       ├── game.py          # Game engine, renderer, batch self-play    [READ-ONLY]
│       ├── prepare.py       # Opponents, evaluation harness             [READ-ONLY]
│       ├── train.py         # NN, self-play, training loop              [AGENT-EDITABLE]
│       ├── play.py          # Human vs AI / AI vs AI                    [READ-ONLY]
│       └── replay.py        # Replay recorded games                     [READ-ONLY]
├── docs/
│   └── program.md           # THIS FILE — 模板
├── output/
│   └── tracker.db           # SQLite database — experiment history
└── pyproject.toml
```

## The single rule

**`domains/<<<domain>>>/train.py` is the ONLY file you modify.** All other source files are read-only.

## Setup

1. **Read the in-scope files**:
   - `domains/<<<domain>>>/game.py` — game engine, board representation, batch self-play
   - `domains/<<<domain>>>/prepare.py` — opponents, `evaluate_win_rate()` harness
   - `domains/<<<domain>>>/train.py` — the file you edit: architecture, self-play, training loop
2. **Install dependencies**: `uv sync`
3. **Read the current experiment report**: `uv run python framework/index.py analyze --report --format json`
4. **Begin the experiment loop.**

## Available tools

| Command | Purpose | When to use |
|---------|---------|-------------|
| `index.py analyze --report` | Full experiment report (markdown) | **START HERE** each iteration |
| `index.py analyze --report --format json` | Structured report for parsing | When you need precise numbers |
| `index.py analyze --compare RUN_A RUN_B` | Side-by-side run comparison | After an experiment |
| `index.py analyze --stability RUN_ID` | Detailed stability metrics | To diagnose training issues |
| `index.py analyze --frontier` | Win-rate progression frontier | To see historical progress |
| `index.py analyze --runs` | List all runs | Overview of experiment history |
| `index.py sweep --dry-run ...` | Preview sweep configurations | Before systematic search |
| `index.py sweep ...` | Run batch hyperparameter search | Explore a parameter range |
| `index.py analyze --matrix TAG` | View sweep results | After a sweep completes |

## Experimentation

### What you CAN do
- Modify `domains/<<<domain>>>/train.py` — everything inside is fair game: model architecture, optimizer, hyperparameters, self-play strategy, batch size, temperature schedule, replay buffer, MCTS, loss function, etc.

### What you CANNOT do
- Modify any file other than `domains/<<<domain>>>/train.py`
- Install new packages or add dependencies
- Modify the evaluation harness (`evaluate_win_rate` in prepare.py)

### Simplicity criterion
Simpler is better. A 1% win_rate improvement that adds ugly complexity? Probably not worth it. A 1% improvement from deleting code? Keep.

## Evaluation metric

The single metric is `win_rate` — the fraction of games won against a fixed opponent.

```
win_rate = wins / n_games   (higher is better, range [0.0, 1.0])
```

Evaluation happens automatically:
- **Probe eval**: Every N cycles during training (lightweight)
- **Full eval**: At each checkpoint milestone
- **Final eval**: At run end

## Stage promotion

As the model improves, promote to harder opponents:

| Stage | Opponent | Promotion threshold | Action |
|---|---|---|---|
| 0 | L0 (<<<weakest>>>) | win_rate > 0.95 | Use `--eval-level 1` |
| 1 | L1 (<<<medium>>>) | win_rate > 0.80 | Use `--eval-level 2` |
| 2 | L2 (<<<strong>>>) | win_rate > 0.60 | Use `--eval-level 3` |
| 3 | L3 (<<<strongest>>>) | Keep optimizing | No limit |

<<<DOMAIN>>> Fill in your opponent descriptions for each level.

## Experiment tracking

All experiment data is stored in `output/tracker.db` (SQLite). Key tables:

- **runs**: One row per training run. UUID, hyperparams, status, final metrics.
- **cycle_metrics**: Per-cycle loss, win_rate, buffer size, game counts.
- **checkpoints**: Model snapshots at WR milestones, with full eval results.
- **recordings**: Game records linked to checkpoints.
- **opponents**: Registered NN opponents with alias, source run, and WR.

## The experiment loop

```
LOOP:
  1. Read report: index.py analyze --report --format json
  2. Form hypothesis from signals + data
  3. Edit domains/<<<domain>>>/train.py
  4. Commit: git add domains/<<<domain>>>/train.py && git commit -m "experiment: <description>"
  5. Run: uv run python domains/<<<domain>>>/train.py --time-budget 300
  6. Check results: index.py analyze --report
  7. If improved: commit results, continue
  8. If same/worse: git reset, try different hypothesis
  9. Check for stage promotion (threshold exceeded → harder opponent)
  10. GOTO 1
```

**NEVER STOP**: Once the loop begins, do NOT pause to ask. Run indefinitely until manually stopped.

## Hints for the agent

**Architecture ideas to try:**
- More/fewer residual blocks (4–12)
- Wider/narrower filters (32–128)
- Different activations (ReLU → SiLU, GELU)
- Squeeze-and-Excitation blocks
- Attention mechanisms
- Separate learning rates for policy/value heads

**Training strategy ideas:**
- Temperature annealing during self-play
- Prioritized replay (prefer recent or surprising outcomes)
- Data augmentation with board symmetries
- Learning rate warmup/decay schedules
- Adjust policy vs value loss weighting
- MCTS during self-play (improves data quality, costs speed)

**What to watch for:**
- Win_rate flatline → more radical architecture change
- Very short games → both sides blundering
- Low loss but low win_rate → overfitting to replay buffer
- Few games completed → self-play too slow (reduce model size)
