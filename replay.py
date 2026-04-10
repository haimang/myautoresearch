"""
Replay recorded games and export frames for video production.

Usage:
  uv run replay.py games/exp047_game003.json              # replay in pygame window
  uv run replay.py games/exp047_game003.json --export      # export PNG frame sequence
  uv run replay.py --montage                               # growth montage from all stages
  uv run replay.py --versus stage0_baseline best           # render two checkpoints playing
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from game import (
    BOARD_SIZE, BLACK, WHITE, EMPTY,
    Board, Renderer, GameRecord,
)


def replay_game(record: GameRecord, renderer: Renderer, speed: float = 1.0,
                export_dir: str = None, info_extra: dict = None):
    """
    Replay a recorded game move by move.

    Parameters
    ----------
    record : GameRecord
    renderer : Renderer
    speed : float         — seconds per move (default 1.0)
    export_dir : str      — if set, save each frame as PNG
    info_extra : dict     — additional info to display
    """
    import pygame

    board = Board()
    info = {
        "_title": "MAG Gomoku Replay",
        "Black": record.black_name,
        "White": record.white_name,
    }
    if info_extra:
        info.update(info_extra)

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    # Show empty board first
    info["Move"] = 0
    info["Turn"] = "BLACK"
    renderer.draw_board(board, info=info)
    if export_dir:
        renderer.save_frame(os.path.join(export_dir, "frame_000.png"))
    time.sleep(speed * 0.5)

    for i, move in enumerate(record.moves):
        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        board.place(move.row, move.col)

        info["Move"] = i + 1
        info["Turn"] = "BLACK" if board.current_player == BLACK else "WHITE"
        if move.value is not None:
            info["Value"] = f"{move.value:+.3f}"
        if move.policy_entropy is not None:
            info["Entropy"] = f"{move.policy_entropy:.2f}"

        renderer.draw_board(board, info=info)

        if export_dir:
            renderer.save_frame(os.path.join(export_dir, f"frame_{i+1:03d}.png"))

        time.sleep(speed)

    # Show result
    result_map = {BLACK: "BLACK wins", WHITE: "WHITE wins", -1: "DRAW", 0: "???"}
    info["Result"] = result_map.get(record.result, "???")
    renderer.draw_board(board, info=info)

    if export_dir:
        renderer.save_frame(os.path.join(export_dir, f"frame_{len(record.moves)+1:03d}_final.png"))

    # Wait a bit at the end
    time.sleep(speed * 3)


def find_stage_games() -> dict[str, str]:
    """Find representative games for each stage from recordings."""
    games_dir = "recordings/games"
    if not os.path.isdir(games_dir):
        return {}

    games = {}
    files = sorted(glob.glob(os.path.join(games_dir, "*.json")))
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Look for wins by NN
            if data.get("result") in ("black_win", "white_win"):
                nn_side = None
                if "nn" in data.get("black", ""):
                    nn_side = "black"
                elif "nn" in data.get("white", ""):
                    nn_side = "white"

                if nn_side:
                    result_player = "black" if data["result"] == "black_win" else "white"
                    if nn_side == result_player:
                        # NN won this game
                        exp = data.get("experiment", os.path.basename(f))
                        key = f"exp_{exp}"
                        if key not in games:
                            games[key] = f
        except (json.JSONDecodeError, KeyError):
            continue

    return games


def montage(renderer: Renderer, speed: float = 0.5, export_dir: str = None):
    """Play a montage of representative games showing AI growth."""
    games = find_stage_games()

    if not games:
        # Fallback: just play all recorded games
        games_dir = "recordings/games"
        if os.path.isdir(games_dir):
            files = sorted(glob.glob(os.path.join(games_dir, "*.json")))[:10]
            games = {os.path.basename(f): f for f in files}

    if not games:
        print("No recorded games found in recordings/games/")
        return

    print(f"Montage: {len(games)} games")
    for i, (label, path) in enumerate(sorted(games.items())):
        print(f"  [{i+1}/{len(games)}] {label}")
        record = GameRecord.load(path)

        sub_export = None
        if export_dir:
            sub_export = os.path.join(export_dir, f"montage_{i:03d}_{label}")

        replay_game(record, renderer, speed=speed, export_dir=sub_export,
                    info_extra={"Montage": f"{i+1}/{len(games)}", "Label": label})


def main():
    parser = argparse.ArgumentParser(description="MAG Gomoku - Replay")
    parser.add_argument("game_file", nargs="?", help="Path to game JSON file")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export frames as PNG sequence")
    parser.add_argument("--output", "-o", type=str, default="replay_frames",
                        help="Output directory for exported frames")
    parser.add_argument("--speed", "-s", type=float, default=0.8,
                        help="Seconds per move (default 0.8)")
    parser.add_argument("--montage", "-m", action="store_true",
                        help="Play growth montage from all stages")
    args = parser.parse_args()

    if args.montage:
        renderer = Renderer(title="MAG Gomoku - Growth Montage")
        export_dir = args.output if args.export else None
        montage(renderer, speed=args.speed, export_dir=export_dir)
        renderer.close()
        return

    if not args.game_file:
        parser.print_help()
        print("\nExamples:")
        print("  uv run replay.py recordings/games/exp001_game0.json")
        print("  uv run replay.py recordings/games/exp001_game0.json --export")
        print("  uv run replay.py --montage")
        return

    if not os.path.isfile(args.game_file):
        print(f"Error: file not found: {args.game_file}")
        sys.exit(1)

    record = GameRecord.load(args.game_file)
    print(f"Replaying: {args.game_file}")
    print(f"  {record.black_name} (BLACK) vs {record.white_name} (WHITE)")
    print(f"  {len(record.moves)} moves")

    renderer = Renderer(title=f"MAG Gomoku - Replay")
    export_dir = args.output if args.export else None
    replay_game(record, renderer, speed=args.speed, export_dir=export_dir)

    if export_dir and args.export:
        frame_count = len(os.listdir(export_dir))
        print(f"Exported {frame_count} frames to {export_dir}/")

    renderer.close()


if __name__ == "__main__":
    main()
