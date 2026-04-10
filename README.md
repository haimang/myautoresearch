# MAG Gomoku

Train a Gomoku (五子棋) AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent autonomously modifies the training code, runs experiments, and evolves the neural network — all while you sleep.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run game.py          # play a two-player game (verify the game works)
uv run train.py         # run one 5-minute training experiment
uv run play.py --list   # list saved checkpoints
uv run play.py          # play against the trained AI
```

## Architecture

Three core files, same as autoresearch:

| File | Role | Editable? |
|---|---|---|
| `game.py` | Board engine, pygame renderer, batched self-play | No |
| `prepare.py` | Minimax opponents (L0-L3), evaluation, checkpoint archive | No |
| `train.py` | Neural network, self-play, training loop | **Yes — agent modifies this** |

Plus supporting files:

- `program.md` — autonomous experiment protocol (the agent reads this)
- `play.py` — human vs AI / AI vs AI gameplay
- `replay.py` — replay recorded games, export frames for video
- `results.tsv` — experiment log
- `action-plan.md` — detailed project plan and dev log

## How it works

```
autoresearch loop (agent modifies train.py → self-play + train → evaluate win_rate → keep/revert)
        ↓
  model.safetensors (a few MB of weights)
        ↓
  play.py (human vs AI game)
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
uv run train.py   # establish baseline
# Point Claude Code at program.md and let it run
```

Each experiment takes ~8 minutes (5 min training + eval). Expect ~7 experiments/hour, ~70 overnight.

## Playing against the AI

```bash
uv run play.py                              # vs latest model
uv run play.py --checkpoint best            # vs best archived model
uv run play.py --black stage0 --white best  # watch two AIs play
uv run play.py --level 2                    # vs minimax (no NN)
uv run play.py --list                       # list all checkpoints
```

## Recording and replay

Training automatically records games, metrics, and key frames to `recordings/`.

```bash
uv run replay.py recordings/games/exp001_game0.json          # replay a game
uv run replay.py recordings/games/exp001_game0.json --export  # export PNG frames
uv run replay.py --montage                                    # growth montage
```

## Hardware

Tested on M3 Max 128GB. The model is tiny (~1M params) — any Apple Silicon Mac will work. Default config: 64 parallel self-play games, 6 residual blocks, 64 filters.

## License

MIT
