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

import numpy as np

from game import (
    BOARD_SIZE, BLACK, WHITE, EMPTY,
    Board, Renderer, GameRecord,
)
from prepare import OPPONENTS, CHECKPOINT_DIR, list_checkpoints


def resolve_checkpoint(tag: str) -> str:
    """Resolve a checkpoint tag to a file path."""
    # Direct file path
    if os.path.isfile(tag):
        return tag

    # Check archive
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.safetensors")
    if os.path.isfile(path):
        return path

    # Check local model.safetensors
    if tag in ("latest", "local") and os.path.isfile("output/model.safetensors"):
        return "output/model.safetensors"

    # Fuzzy match in manifest
    checkpoints = list_checkpoints()
    for cp in checkpoints:
        if tag in cp.get("tag", ""):
            return cp.get("archived_path", "")

    print(f"Error: checkpoint '{tag}' not found")
    print("Use --list to see available checkpoints")
    sys.exit(1)


def load_nn_player(checkpoint_path: str, mcts_sims: int = 0):
    """Load a NN model and return a player function (board) -> (row, col)."""
    import mlx.core as mx
    from train import load_model

    model = load_model(checkpoint_path)
    model.eval()

    def nn_move(board: Board) -> tuple[int, int]:
        encoded = board.encode()
        x = mx.array(encoded[np.newaxis, ...])
        policy_logits, value = model(x)
        mx.eval(policy_logits, value)

        policy = np.array(policy_logits[0])
        val = float(np.array(value[0, 0]))

        # Mask illegal moves
        legal_mask = board.get_legal_mask()
        policy[legal_mask == 0] = -1e9

        # Argmax (no MCTS for now)
        action = int(np.argmax(policy))
        row, col = divmod(action, BOARD_SIZE)
        return row, col

    return nn_move


def print_checkpoints():
    """Print all available checkpoints."""
    checkpoints = list_checkpoints()
    if not checkpoints:
        print("No archived checkpoints found.")
        if os.path.isfile("output/model.safetensors"):
            print("  Local output/model.safetensors is available (use --checkpoint latest)")
        return

    print(f"{'#':>3}  {'Tag':<35}  {'Win Rate':>8}  {'Level':>5}  {'Timestamp':<20}")
    print("-" * 80)
    for i, cp in enumerate(checkpoints):
        tag = cp.get("tag", "?")
        wr = cp.get("win_rate", "?")
        level = cp.get("eval_level", "?")
        ts = cp.get("timestamp", "?")[:19]
        if isinstance(wr, float):
            wr = f"{wr:.2%}"
        print(f"{i:3d}  {tag:<35}  {wr:>8}  {level:>5}  {ts:<20}")

    if os.path.isfile("output/model.safetensors"):
        print()
        print("  + Local output/model.safetensors (use --checkpoint latest)")


def main():
    parser = argparse.ArgumentParser(description="MAG Gomoku - Play")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Checkpoint tag or path (human plays BLACK vs this AI)")
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
