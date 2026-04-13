"""Native C wrapper for Gomoku minimax.

Drop-in replacement for the hot path in prepare.py. Loads minimax_c.so/dylib
via ctypes and exposes `root_scores` — a single call that returns an array
of (row, col, score) for all root candidates. Python still owns the top-k
softmax sampling so that `minimax_move_sampled`'s behaviour is unchanged
(see v15-update.md §4.3 C3).

Performance targets (v15 acceptance):
- L1 (depth 2) single call:  ≤ 1 ms
- L2 (depth 4) single call:  ≤ 50 ms
- L3 (depth 6) single call:  ≤ 500 ms

Set GOMOKU_MINIMAX_BACKEND=python to force the pure-Python fallback.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Optional

import numpy as np

_LIB = None


def _find_lib() -> Optional[str]:
    d = os.path.dirname(os.path.abspath(__file__))
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    path = os.path.join(d, "minimax_c" + ext)
    return path if os.path.isfile(path) else None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _find_lib()
    if path is None:
        raise ImportError(
            "minimax_c not found. Run: cd domains/gomoku && bash build_native.sh"
        )
    _LIB = ctypes.CDLL(path)

    # int gomoku_root_scores(
    #     const int8_t *grid, int player, int depth, int move_order,
    #     int *out_rows, int *out_cols, float *out_scores, int max_out);
    _LIB.gomoku_root_scores.restype = ctypes.c_int
    _LIB.gomoku_root_scores.argtypes = [
        ctypes.c_void_p,  # grid
        ctypes.c_int,     # player
        ctypes.c_int,     # depth
        ctypes.c_int,     # move_order
        ctypes.c_void_p,  # out_rows
        ctypes.c_void_p,  # out_cols
        ctypes.c_void_p,  # out_scores
        ctypes.c_int,     # max_out
    ]

    # int gomoku_evaluate_grid(const int8_t *grid, int player);
    _LIB.gomoku_evaluate_grid.restype = ctypes.c_int
    _LIB.gomoku_evaluate_grid.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # int gomoku_check_win_at(const int8_t *grid, int row, int col, int player);
    _LIB.gomoku_check_win_at.restype = ctypes.c_int
    _LIB.gomoku_check_win_at.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return _LIB


def is_available() -> bool:
    try:
        _load_lib()
        return True
    except (ImportError, OSError):
        return False


# Move-ordering backend IDs that mirror prepare.py constants.
MOVE_ORDER_BASIC = 0      # center-distance (L1)
MOVE_ORDER_HEURISTIC = 1  # adjacency-based (L2)
MOVE_ORDER_KILLER = 2     # killer heuristic (L3 — falls back to heuristic in v15)

_MAX_CANDIDATES = 225


def root_scores(grid: np.ndarray, player: int, depth: int,
                move_order: int = MOVE_ORDER_BASIC
                ) -> list[tuple[tuple[int, int], float]]:
    """Return [(move, score), ...] for all root candidates.

    Parameters
    ----------
    grid : np.ndarray  — int8 shape (BOARD_SIZE, BOARD_SIZE) or flat (225,)
    player : int       — 1 (BLACK) or 2 (WHITE)
    depth : int        — search depth (2/4/6 for L1/L2/L3)
    move_order : int   — one of MOVE_ORDER_* constants
    """
    lib = _load_lib()
    g = np.ascontiguousarray(grid, dtype=np.int8).reshape(-1)
    if g.size != 225:
        raise ValueError(f"expected 15x15 grid, got shape {grid.shape}")

    out_rows = np.zeros(_MAX_CANDIDATES, dtype=np.int32)
    out_cols = np.zeros(_MAX_CANDIDATES, dtype=np.int32)
    out_scores = np.zeros(_MAX_CANDIDATES, dtype=np.float32)

    n = lib.gomoku_root_scores(
        g.ctypes.data,
        int(player),
        int(depth),
        int(move_order),
        out_rows.ctypes.data,
        out_cols.ctypes.data,
        out_scores.ctypes.data,
        _MAX_CANDIDATES,
    )
    results = []
    for i in range(n):
        results.append(((int(out_rows[i]), int(out_cols[i])), float(out_scores[i])))
    return results


def evaluate_position_c(grid: np.ndarray, player: int) -> int:
    """Pure static eval (no search). For parity testing."""
    lib = _load_lib()
    g = np.ascontiguousarray(grid, dtype=np.int8).reshape(-1)
    return int(lib.gomoku_evaluate_grid(g.ctypes.data, int(player)))


def check_win_c(grid: np.ndarray, row: int, col: int, player: int) -> bool:
    lib = _load_lib()
    g = np.ascontiguousarray(grid, dtype=np.int8).reshape(-1)
    return bool(lib.gomoku_check_win_at(g.ctypes.data, int(row), int(col), int(player)))
