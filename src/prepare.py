"""
Evaluation infrastructure for MAG-Gomoku autoresearch.

Provides:
  - Minimax opponents (L0-L3) for benchmarking neural-network agents
  - evaluate_win_rate():  play N games against a given opponent level
  - Frame capture for notable game moments

This module is READ-ONLY evaluation code — it never modifies model weights.
"""

import json
import os
import random
import time
from typing import Optional

import numpy as np

from game import (
    BOARD_SIZE,
    WIN_LENGTH,
    BLACK,
    WHITE,
    EMPTY,
    Board,
    GameRecord,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # 5 minutes training wall clock
EVAL_GAMES = 200  # games per evaluation
RECORDING_DIR = "output/recordings"

# ---------------------------------------------------------------------------
# Minimax evaluation heuristic
# ---------------------------------------------------------------------------

# Pattern scores: (consecutive_count, open_ends) -> score
_PATTERN_SCORES = {
    (5, 0): 100000,
    (5, 1): 100000,
    (5, 2): 100000,
    (4, 2): 10000,   # open four
    (4, 1): 1000,    # half-open four
    (3, 2): 1000,    # open three
    (3, 1): 100,     # half-open three
    (2, 2): 100,     # open two
    (2, 1): 10,      # half-open two
    (1, 2): 10,      # open one
    (1, 1): 1,       # half-open one
}


def _score_segment(grid: np.ndarray, row: int, col: int,
                   dr: int, dc: int, player: int) -> int:
    """
    Score a single direction from (row, col) for `player`.
    Count consecutive stones of `player` extending in both directions,
    and count how many ends are open (not blocked by opponent or edge).
    """
    count = 1

    # Extend in positive direction
    r, c = row + dr, col + dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
        count += 1
        r += dr
        c += dc
    # Check if positive end is open
    open_pos = (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                and grid[r, c] == EMPTY)

    # Extend in negative direction
    r, c = row - dr, col - dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r, c] == player:
        count += 1
        r -= dr
        c -= dc
    # Check if negative end is open
    open_neg = (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                and grid[r, c] == EMPTY)

    open_ends = int(open_pos) + int(open_neg)

    if count >= 5:
        return _PATTERN_SCORES.get((5, open_ends), 100000)
    if open_ends == 0:
        return 0  # completely blocked — worthless
    return _PATTERN_SCORES.get((count, open_ends), 0)


def evaluate_position(grid: np.ndarray, player: int) -> int:
    """
    Static evaluation of the board from `player`'s perspective.
    Scans every occupied cell in all four directions and sums pattern scores.
    Returns score_for_player - score_for_opponent.
    """
    opponent = WHITE if player == BLACK else BLACK
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    player_score = 0
    opponent_score = 0

    # To avoid double-counting, we only start a segment scan from
    # the "first" cell in each direction (the one whose predecessor
    # in the negative direction is NOT the same player).
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stone = grid[r, c]
            if stone == EMPTY:
                continue
            for dr, dc in directions:
                # Only score from the start of a run
                pr, pc = r - dr, c - dc
                if (0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE
                        and grid[pr, pc] == stone):
                    continue  # not the start of this segment

                score = _score_segment(grid, r, c, dr, dc, stone)
                if stone == player:
                    player_score += score
                else:
                    opponent_score += score

    return player_score - opponent_score


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def _minimax(grid: np.ndarray, depth: int, alpha: float, beta: float,
             maximizing: bool, player: int, opponent: int,
             candidate_fn, move_order_fn=None) -> tuple[float, Optional[tuple[int, int]]]:
    """
    Alpha-beta minimax.

    Parameters
    ----------
    grid : np.ndarray  — mutable BOARD_SIZE x BOARD_SIZE int8 array
    depth : int         — remaining search depth
    alpha, beta : float — pruning bounds
    maximizing : bool   — True when it is `player`'s turn
    player : int        — the player we are maximizing for
    opponent : int      — the other player
    candidate_fn       — callable(grid) -> list of (row, col) candidates
    move_order_fn      — optional callable(moves, grid) -> sorted moves
    """
    # Terminal checks
    # (We skip full win detection per move for speed; instead, the caller
    #  places the stone and checks.  At leaf nodes we use the heuristic.)
    if depth == 0:
        return float(evaluate_position(grid, player)), None

    moves = candidate_fn(grid)
    if not moves:
        return float(evaluate_position(grid, player)), None

    if move_order_fn is not None:
        moves = move_order_fn(moves, grid)

    best_move = moves[0]

    if maximizing:
        max_eval = -float("inf")
        for (r, c) in moves:
            grid[r, c] = player
            # Check immediate win
            if _check_win_fast(grid, r, c, player):
                grid[r, c] = EMPTY
                return 100000.0 + depth, (r, c)
            val, _ = _minimax(grid, depth - 1, alpha, beta,
                              False, player, opponent, candidate_fn, move_order_fn)
            grid[r, c] = EMPTY
            if val > max_eval:
                max_eval = val
                best_move = (r, c)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for (r, c) in moves:
            grid[r, c] = opponent
            # Check immediate win for opponent
            if _check_win_fast(grid, r, c, opponent):
                grid[r, c] = EMPTY
                return -100000.0 - depth, (r, c)
            val, _ = _minimax(grid, depth - 1, alpha, beta,
                              True, player, opponent, candidate_fn, move_order_fn)
            grid[r, c] = EMPTY
            if val < min_eval:
                min_eval = val
                best_move = (r, c)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move


def _check_win_fast(grid: np.ndarray, row: int, col: int, player: int) -> bool:
    """Check if the last placed stone at (row, col) creates five-in-a-row."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                   and grid[r, c] == player):
                count += 1
                r += dr * sign
                c += dc * sign
        if count >= WIN_LENGTH:
            return True
    return False


def _make_candidate_fn(radius: int):
    """Return a candidate-move generator that looks near existing stones."""
    def fn(grid):
        occupied = np.argwhere(grid != EMPTY)
        if len(occupied) == 0:
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        candidates = set()
        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and grid[nr, nc] == EMPTY):
                        candidates.add((int(nr), int(nc)))
        return list(candidates)
    return fn


def _move_order_basic(moves, grid):
    """Basic move ordering: prefer center, then random shuffle."""
    center = BOARD_SIZE // 2
    return sorted(moves, key=lambda m: abs(m[0] - center) + abs(m[1] - center))


def _move_order_heuristic(moves, grid):
    """
    Better move ordering: score each candidate position quickly
    by counting nearby stones (both own and opponent).
    """
    center = BOARD_SIZE // 2
    scored = []
    for (r, c) in moves:
        # Quick heuristic: count adjacent occupied cells + center bonus
        adj = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and grid[nr, nc] != EMPTY):
                    adj += 1
        center_bonus = max(0, 7 - abs(r - center) - abs(c - center))
        scored.append((-adj * 10 - center_bonus, r, c))
    scored.sort()
    return [(r, c) for (_, r, c) in scored]


# Killer-move heuristic storage (module-level for reuse across calls)
_killer_moves: dict[int, list[tuple[int, int]]] = {}


def _move_order_killer(moves, grid):
    """
    Move ordering with killer-move heuristic: prioritize moves that
    caused cutoffs at the same depth in previous searches, then
    fall back to the adjacency heuristic.
    """
    # First pass: separate killers from non-killers
    killer_set = set()
    for depth_killers in _killer_moves.values():
        for m in depth_killers:
            killer_set.add(m)

    killers = []
    rest = []
    for m in moves:
        if m in killer_set:
            killers.append(m)
        else:
            rest.append(m)

    # Sort the non-killer moves by heuristic
    rest = _move_order_heuristic(rest, grid)
    return killers + rest


def minimax_move(board: Board, depth: int, player: int,
                 move_order_fn=None) -> tuple[int, int]:
    """
    Find the best move for `player` on `board` using minimax with alpha-beta.

    Parameters
    ----------
    board : Board
    depth : int       — search depth
    player : int      — BLACK or WHITE
    move_order_fn     — optional move ordering function

    Returns
    -------
    (row, col)
    """
    grid = board.grid.copy()
    opponent = WHITE if player == BLACK else BLACK

    candidate_fn = _make_candidate_fn(radius=2)

    _, best_move = _minimax(
        grid, depth,
        -float("inf"), float("inf"),
        True, player, opponent,
        candidate_fn, move_order_fn,
    )

    if best_move is None:
        # Fallback: center or first legal move
        legal = board.get_legal_moves()
        if legal:
            return legal[0]
        return (BOARD_SIZE // 2, BOARD_SIZE // 2)

    return best_move


# ---------------------------------------------------------------------------
# Opponent functions (L0 – L3)
# ---------------------------------------------------------------------------

def opponent_l0(board: Board) -> tuple[int, int]:
    """Level 0: random legal move."""
    legal = board.get_legal_moves()
    return random.choice(legal)


def opponent_l1(board: Board) -> tuple[int, int]:
    """Level 1: minimax depth 2."""
    return minimax_move(board, depth=2, player=board.current_player,
                        move_order_fn=_move_order_basic)


def opponent_l2(board: Board) -> tuple[int, int]:
    """Level 2: minimax depth 4 with heuristic move ordering."""
    return minimax_move(board, depth=4, player=board.current_player,
                        move_order_fn=_move_order_heuristic)


def opponent_l3(board: Board) -> tuple[int, int]:
    """Level 3: minimax depth 6 with killer-move heuristic."""
    global _killer_moves
    _killer_moves.clear()
    return minimax_move(board, depth=6, player=board.current_player,
                        move_order_fn=_move_order_killer)


OPPONENTS = {0: opponent_l0, 1: opponent_l1, 2: opponent_l2, 3: opponent_l3}


# ---------------------------------------------------------------------------
# Evaluation: NN vs minimax opponent
# ---------------------------------------------------------------------------

def evaluate_win_rate(
    model_path: str,
    level: int = 0,
    n_games: int = EVAL_GAMES,
    record_games: int = 0,
    tag: str = "",
    run_id: str = "",
) -> dict:
    """
    Play `n_games` between a trained NN model and a minimax opponent.

    The NN plays as BLACK for the first half and as WHITE for the second half.
    Returns a dict with win_rate, wins, losses, draws, avg_game_length, level,
    recorded_files (list of paths), and eval_elapsed_s.
    """
    import mlx.core as mx
    from train import GomokuNet, load_model

    eval_start = time.time()

    model = load_model(model_path)
    model.eval()  # Crucial: disable BatchNorm running stats updates during inference
    opponent_fn = OPPONENTS[level]

    wins = 0
    losses = 0
    draws = 0
    total_length = 0
    recorded_files: list[str] = []

    games_dir = os.path.join(RECORDING_DIR, "games")
    os.makedirs(games_dir, exist_ok=True)

    # Per-game detail for data analysis
    game_details: list[dict] = []

    for game_i in range(n_games):
        nn_is_black = game_i < n_games // 2
        nn_player = BLACK if nn_is_black else WHITE

        board = Board()
        black_name = "nn" if nn_is_black else f"minimax_L{level}"
        white_name = f"minimax_L{level}" if nn_is_black else "nn"
        record = GameRecord(black_name=black_name, white_name=white_name)

        step = 0
        while not board.is_terminal():
            if board.current_player == nn_player:
                encoded = board.encode()
                x = mx.array(encoded[np.newaxis, ...])
                policy_logits, value = model(x)
                policy = policy_logits[0]

                legal_mask = mx.array(board.get_legal_mask())
                masked = mx.where(legal_mask > 0, policy, mx.array(float("-inf")))
                action = int(mx.argmax(masked).item())
                row, col = divmod(action, BOARD_SIZE)
            else:
                row, col = opponent_fn(board)

            board.place(row, col)
            record.add_move(step=step, row=row, col=col, player=board.history[-1][2])
            step += 1

        if game_i % 20 == 0:
            mx.clear_cache()

        total_length += board.move_count
        nn_won = 0
        if board.winner == nn_player:
            wins += 1
            nn_won = 1
            record.result = board.winner
        elif board.winner == -1:
            draws += 1
            record.result = -1
        else:
            losses += 1
            record.result = board.winner

        # Determine result string
        result_map = {BLACK: "black_win", WHITE: "white_win", -1: "draw", 0: "ongoing"}
        result_str = result_map.get(board.winner, "unknown")

        nn_side = "black" if nn_is_black else "white"
        game_details.append({
            "game_index": game_i,
            "result": result_str,
            "total_moves": board.move_count,
            "black": black_name,
            "white": white_name,
            "nn_side": nn_side,
            "nn_won": nn_won,
        })

        # Save recordings
        if game_i < record_games:
            if tag:
                filename = f"{tag}_game{game_i:03d}.json"
            else:
                filename = f"game{game_i:03d}.json"
            save_path = os.path.join(games_dir, filename)
            record.save(save_path)
            recorded_files.append(save_path)
            game_details[-1]["game_file"] = save_path

    eval_elapsed = time.time() - eval_start
    win_rate = wins / n_games if n_games > 0 else 0.0
    avg_length = total_length / n_games if n_games > 0 else 0.0

    result = {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_game_length": avg_length,
        "level": level,
        "n_games": n_games,
        "recorded_files": recorded_files,
        "game_details": game_details,
        "eval_elapsed_s": eval_elapsed,
    }

    # Print standard output format
    print("---")
    print(f"win_rate:         {win_rate:.4f}")
    print(f"eval_level:       {level}")
    print(f"wins:             {wins}")
    print(f"losses:           {losses}")
    print(f"draws:            {draws}")
    print(f"avg_game_length:  {avg_length:.1f}")
    print(f"eval_time:        {eval_elapsed:.1f}s")
    if recorded_files:
        print(f"recorded_games:   {len(recorded_files)}")

    return result


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_key_frames(
    board: Board,
    game_record: GameRecord,
    experiment_id: int,
    game_idx: int,
    trigger: str,
):
    """
    Render and save the current board state as a PNG frame.

    Only called for notable moments (first_win, stage_promotion, etc.).
    Silently skips in headless environments where pygame is unavailable.

    Parameters
    ----------
    board : Board
    game_record : GameRecord
    experiment_id : int
    game_idx : int
    trigger : str  — why this frame is being captured (e.g. "first_win")
    """
    try:
        from game import Renderer
    except ImportError:
        return
    except Exception:
        return

    frames_dir = os.path.join(RECORDING_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # Set SDL to use a dummy video driver for headless rendering
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        renderer = Renderer(title=f"exp{experiment_id} - {trigger}")
        info = {
            "_title": f"Exp {experiment_id}",
            "Game": game_idx,
            "Trigger": trigger,
            "Moves": board.move_count,
        }
        if board.winner == BLACK:
            info["Result"] = "BLACK wins"
        elif board.winner == WHITE:
            info["Result"] = "WHITE wins"
        elif board.winner == -1:
            info["Result"] = "Draw"

        renderer.draw_board(board, info=info)
        filename = f"exp{experiment_id}_game{game_idx}_{trigger}_move{board.move_count}.png"
        renderer.save_frame(os.path.join(frames_dir, filename))
        renderer.close()
    except Exception:
        # Headless environment — cannot render; skip silently
        pass


# ---------------------------------------------------------------------------
# Main: quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== prepare.py smoke test ===")
    print(f"Board size: {BOARD_SIZE}x{BOARD_SIZE}, Win length: {WIN_LENGTH}")
    print()

    # Play one game: L1 (black) vs L0 (white)
    board = Board()
    record = GameRecord(black_name="minimax_L1", white_name="random_L0")
    step = 0

    t0 = time.time()
    while not board.is_terminal():
        if board.current_player == BLACK:
            row, col = opponent_l1(board)
        else:
            row, col = opponent_l0(board)
        board.place(row, col)
        record.add_move(step=step, row=row, col=col, player=board.history[-1][2])
        step += 1
    elapsed = time.time() - t0

    if board.winner == BLACK:
        result_str = "BLACK (L1) wins"
    elif board.winner == WHITE:
        result_str = "WHITE (L0) wins"
    else:
        result_str = "Draw"

    record.result = board.winner
    print(f"Result: {result_str}")
    print(f"Moves:  {board.move_count}")
    print(f"Time:   {elapsed:.2f}s")
    print()

    # Display final board
    symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
    print("   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row_str = " ".join(f" {symbols[board.grid[r, c]]}" for c in range(BOARD_SIZE))
        print(f"{r:2d} {row_str}")
    print()
    print("=== smoke test complete ===")
