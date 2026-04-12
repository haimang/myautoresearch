"""
Human vs AI (or AI vs AI) gameplay interface.

Load any checkpoint from the archive and play against it, or watch two
different checkpoints play each other.

Usage (run from project root):
  uv run python src/play.py                              # play vs latest model
  uv run python src/play.py --checkpoint best            # play vs best archived model
  uv run python src/play.py --checkpoint stage1_beat_random
  uv run python src/play.py --list                       # list all checkpoints
  uv run python src/play.py --black stage0 --white best  # AI vs AI
  uv run python src/play.py --mcts 200                   # stronger AI with MCTS
  uv run python src/play.py --level 2                    # play vs minimax L2 (no NN)
"""

import argparse
import os
import sys
import time

# ── path setup for decoupled project structure ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if os.path.join(_PROJECT_ROOT, "framework") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "framework"))

import numpy as np

from game import (
    BOARD_SIZE, BLACK, WHITE, EMPTY,
    Board, Renderer, GameRecord,
)
from play_service import load_nn_player, load_registered_opponent, resolve_checkpoint
from prepare import OPPONENTS
import core.db as tracker


def print_checkpoints():
    """Print all available checkpoints from tracker DB."""
    try:
        conn = tracker.init_db()
        checkpoints = tracker.list_all_checkpoints(conn, limit=100)
        conn.close()
    except Exception as e:
        print(f"Error reading tracker.db: {e}")
        return

    if not checkpoints:
        print("No checkpoints found in tracker.db.")
        return

    print(f"{'#':>3}  {'Tag':<25}  {'Run':>8}  {'WR':>7}  {'L':>2}  {'Cycle':>5}  {'Model Path':<40}")
    print("-" * 100)
    for i, cp in enumerate(checkpoints):
        tag = cp.get("tag", "?")
        run_short = cp.get("run_id", "?")[:8]
        wr = cp.get("win_rate", 0)
        level = cp.get("eval_level", "?")
        cyc = cp.get("cycle", "?")
        path = cp.get("model_path", "?")
        print(f"{i:3d}  {tag:<25}  {run_short:>8}  {wr:>6.1%}  {level:>2}  {cyc:>5}  {path:<40}")


def main():
    parser = argparse.ArgumentParser(description="MAG Gomoku - Play")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--list-opponents", action="store_true",
                        help="List all registered NN opponents")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Checkpoint tag or path (human plays BLACK vs this AI)")
    parser.add_argument("--opponent", "-o", type=str, default=None,
                        help="Play against a registered NN opponent by alias (e.g., S0)")
    parser.add_argument("--black", type=str, default=None,
                        help="Black player checkpoint (for AI vs AI)")
    parser.add_argument("--white", type=str, default=None,
                        help="White player checkpoint (for AI vs AI)")
    parser.add_argument("--mcts", type=int, default=0,
                        help="MCTS simulations for NN player (0=pure policy)")
    parser.add_argument("--level", "-l", type=int, default=None,
                        help="Play vs minimax at this level (0-3, no NN)")
    parser.add_argument("--swap", action="store_true",
                        help="Human plays WHITE instead of BLACK")
    args = parser.parse_args()

    if args.list:
        print_checkpoints()
        return

    if args.list_opponents:
        conn = tracker.init_db()
        opponents = tracker.list_opponents(conn)
        conn.close()
        if not opponents:
            print("No registered opponents. Use train.py --register-opponent to register one.")
            return
        print(f"{'Alias':<12} {'WR':>6} {'Level':>6} {'Source':>10} {'Description'}")
        print("─" * 60)
        for o in opponents:
            alias = o["alias"]
            wr = f"{o['win_rate']:.0%}" if o.get("win_rate") else "—"
            lv = f"L{o['eval_level']}" if o.get("eval_level") is not None else "—"
            src = o["source_run"][:8] if o.get("source_run") else "—"
            desc = o.get("description") or ""
            print(f"{alias:<12} {wr:>6} {lv:>6} {src:>10} {desc}")
        return

    # Determine players
    ai_vs_ai = args.black is not None and args.white is not None

    if ai_vs_ai:
        # AI vs AI mode
        black_path = resolve_checkpoint(args.black)
        white_path = resolve_checkpoint(args.white)
        black_fn = load_nn_player(black_path, args.mcts)
        white_fn = load_nn_player(white_path, args.mcts)
        title = f"AI ({args.black}) vs AI ({args.white})"
        human_player = None
    elif args.opponent is not None:
        # Human vs registered NN opponent
        try:
            opp_fn, opp_info = load_registered_opponent(args.opponent, mcts_sims=args.mcts)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        opp_label = f"{args.opponent}"
        if args.mcts:
            opp_label += f" (MCTS-{args.mcts})"
        title = f"Human vs {opp_label}"
        if args.swap:
            black_fn = opp_fn
            white_fn = None
            human_player = WHITE
        else:
            black_fn = None
            white_fn = opp_fn
            human_player = BLACK
    elif args.level is not None:
        # Human vs minimax
        if args.level not in OPPONENTS:
            print(f"Error: level must be 0-3, got {args.level}")
            sys.exit(1)
        opponent_fn = OPPONENTS[args.level]
        title = f"Human vs Minimax L{args.level}"
        if args.swap:
            black_fn = opponent_fn
            white_fn = None
            human_player = WHITE
        else:
            black_fn = None
            white_fn = opponent_fn
            human_player = BLACK
    else:
        # Human vs NN
        checkpoint = args.checkpoint
        if checkpoint is None:
            if os.path.isfile("output/model.safetensors"):
                checkpoint = "latest"
            else:
                print("No model found. Use --checkpoint, --level, or train first.")
                sys.exit(1)
        cp_path = resolve_checkpoint(checkpoint)
        nn_fn = load_nn_player(cp_path, args.mcts)
        cp_label = args.checkpoint or "latest"
        title = f"Human vs NN ({cp_label})"
        if args.swap:
            black_fn = nn_fn
            white_fn = None
            human_player = WHITE
        else:
            black_fn = None
            white_fn = nn_fn
            human_player = BLACK

    # Initialize game
    board = Board()
    renderer = Renderer(title=f"MAG Gomoku - {title}")
    record = GameRecord()

    info = {
        "_title": "MAG Gomoku",
        "Mode": title,
        "Turn": "BLACK",
        "Move": 0,
    }

    print(f"\n  {title}")
    if human_player:
        color = "BLACK" if human_player == BLACK else "WHITE"
        print(f"  You play as {color}. Click to place stones.")
    elif ai_vs_ai:
        print("  Watching AI vs AI. Close window to exit.")
    print()

    # Game loop
    while not board.is_terminal():
        info["Turn"] = "BLACK" if board.current_player == BLACK else "WHITE"
        info["Move"] = board.move_count
        renderer.draw_board(board, info=info)

        if board.current_player == BLACK:
            player_fn = black_fn
        else:
            player_fn = white_fn

        if player_fn is None:
            # Human's turn
            move = renderer.get_human_move(board)
            if move is None:
                print("Game aborted.")
                renderer.close()
                return
            row, col = move
        else:
            # AI's turn
            if ai_vs_ai:
                time.sleep(0.3)  # brief pause for watchability
            row, col = player_fn(board)

        board.place(row, col)
        record.add_move(step=board.move_count - 1, row=row, col=col,
                        player=board.history[-1][2])

    # Show final state
    record.result = board.winner
    info["Turn"] = "GAME OVER"
    info["Move"] = board.move_count

    if board.winner == BLACK:
        info["Winner"] = "BLACK"
    elif board.winner == WHITE:
        info["Winner"] = "WHITE"
    else:
        info["Winner"] = "DRAW"

    renderer.draw_board(board, info=info)
    print(f"  Result: {info['Winner']} in {board.move_count} moves")

    # Wait for window close
    import pygame
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                waiting = False

    renderer.close()


if __name__ == "__main__":
    main()
