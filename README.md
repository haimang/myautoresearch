# MAG Gomoku

Train a Gomoku (五子棋) AI from scratch using the [autoresearch](https://github.com/karpathy/autoresearch) paradigm on Apple Silicon (MLX). An AI agent autonomously modifies the training code, runs experiments, and evolves the neural network — all while you sleep.

Inspired by [Code Bullet](https://www.youtube.com/@CodeBullet) — the project includes a full game implementation, training recording system, and replay tools for video production.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python src/train.py         # run one 5-minute training experiment
uv run python src/play.py --list   # list saved checkpoints
uv run python src/play.py          # play against the trained AI
```

## Project structure

```
mag-gomoku/
├── src/                         # Python source code
│   ├── game.py                  #   board engine, pygame renderer, batched self-play (read-only)
│   ├── prepare.py               #   minimax opponents, evaluation, checkpoint archive (read-only)
│   ├── train.py                 #   neural network, self-play, training loop (agent-mutable)
│   ├── play.py                  #   human vs AI / AI vs AI gameplay
│   └── replay.py                #   replay recorded games, export frames for video
├── docs/                        # documentation
│   ├── program.md               #   autonomous experiment protocol (agent reads this)
│   └── action-plan.md           #   project plan, dev log & troubleshooting
├── data/                        # experiment tracking (version controlled)
│   └── results.tsv              #   experiment log
├── output/                      # generated artifacts (gitignored)
│   ├── model.safetensors        #   trained model weights
│   ├── run.log                  #   latest training run output
│   └── recordings/              #   training game recordings
│       ├── games/               #     full game records (JSON)
│       ├── metrics/             #     training metrics (CSV)
│       └── frames/              #     key frame captures (PNG)
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

## How it works

```
autoresearch loop (agent modifies src/train.py → self-play + train → evaluate win_rate → keep/revert)
        ↓
  output/model.safetensors (a few MB of weights)
        ↓
  src/play.py (human vs AI game)
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

Training automatically records games, metrics, and key frames to `output/recordings/`.

```bash
uv run python src/replay.py output/recordings/games/exp001_game0.json          # replay a game
uv run python src/replay.py output/recordings/games/exp001_game0.json --export  # export PNG frames
uv run python src/replay.py --montage                                           # growth montage
```

## Hardware

Tested on M3 Max 128GB. The model is tiny (~564K params) — any Apple Silicon Mac will work. Default config: 64 parallel self-play games, 6 residual blocks, 64 filters.

## License

MIT
