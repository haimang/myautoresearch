"""
Gomoku training script — MLX on Apple Silicon.

This is the MUTABLE training script for the autoresearch loop.
The agent modifies hyperparameters and architecture between runs.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import random
import shutil
import subprocess
import sys
import time as _time
import uuid

# ── path setup for decoupled project structure ──
# Domain dir (_THIS_DIR) MUST come before framework/ in sys.path so that
# `from prepare import OPPONENTS` resolves to domains/gomoku/prepare.py
# (with C minimax backend), NOT framework/prepare.py (pure-Python fallback).
# Python auto-adds the script dir to sys.path[0], but a subsequent insert(0)
# for framework/ would push it to [1].  We add framework/ at index 1 to keep
# the domain dir at [0].
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_fw_path = os.path.join(_PROJECT_ROOT, "framework")
if _fw_path not in sys.path:
    # Insert AFTER _THIS_DIR so domain prepare.py takes priority
    _idx = sys.path.index(_THIS_DIR) + 1 if _THIS_DIR in sys.path else 0
    sys.path.insert(_idx, _fw_path)
os.chdir(_PROJECT_ROOT)  # ensure output/ paths resolve correctly

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from game import BatchBoards, BOARD_SIZE, BLACK, WHITE, EMPTY
from core.tui import sparkline as _sparkline_fn, sparkline2 as _sparkline2_fn, sparkline3 as _sparkline3_fn, sparkline4 as _sparkline4_fn, progress_bar as _progress_bar_fn
from core.mcts import MCTSNode, mcts_search as _mcts_search_generic, mcts_search_batched as _mcts_search_batched, mcts_search_multi_root as _mcts_search_multi_root

# Try native C MCTS — 10-20x faster tree operations
try:
    from core.mcts_native import mcts_search_multi_root_native as _mcts_native, is_available as _mcts_native_available
    _USE_NATIVE_MCTS = _mcts_native_available()
except ImportError:
    _USE_NATIVE_MCTS = False

# Try to import TIME_BUDGET and evaluate_win_rate from prepare.py.
try:
    from prepare import TIME_BUDGET
except ImportError:
    TIME_BUDGET = 300

try:
    from prepare import evaluate_win_rate
except ImportError:
    evaluate_win_rate = None

# === Hyperparameters (autoresearch agent modifies these) ===
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256          # training batch size
PARALLEL_GAMES = 64       # number of simultaneous self-play games
MCTS_SIMULATIONS = 0      # 0 = pure policy network (no MCTS)
C_PUCT = 1.5              # MCTS exploration constant
DIRICHLET_ALPHA = 0.03    # Dirichlet noise alpha for root node exploration
DIRICHLET_FRAC = 0.25     # fraction of Dirichlet noise mixed into root prior
TEMPERATURE = 1.0         # self-play exploration temperature
TEMP_THRESHOLD = 30       # moves over which temperature decays (see run_self_play)
REPLAY_BUFFER_SIZE = 50000
TRAIN_STEPS_PER_CYCLE = 30
CYCLES_PER_REPORT = 5     # print stats every N cycles
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.0
EVAL_LEVEL = 0            # opponent level for evaluation (0=random, 1=minimax2, 2=minimax4, 3=minimax6)

# Signal-density replay tuning. These keep the win/loss objective intact,
# but make the model learn more often from states that expose weaknesses.
OPPONENT_SAMPLE_BOOST = 4.0
LOSS_SAMPLE_BOOST = 2.5
WIN_OPPONENT_SAMPLE_BOOST = 1.5
FOCUSED_SAMPLE_RATIO = 0.25
FAST_WIN_LEN_1 = 85
FAST_WIN_LEN_2 = 75
FAST_WIN_LEN_3 = 60
SLOW_WIN_LEN_1 = 85
SLOW_WIN_LEN_2 = 90
FAST_WIN_BOOST_1 = 1.08
FAST_WIN_BOOST_2 = 1.15
FAST_WIN_BOOST_3 = 1.25
SLOW_WIN_DAMP_1 = 0.96
SLOW_WIN_DAMP_2 = 0.90
FAST_LOSS_LEN = 60
FAST_LOSS_BOOST = 1.15

# ---------------------------------------------------------------------------
# D4 symmetry augmentation (8-fold: 4 rotations x 2 reflections)
# ---------------------------------------------------------------------------

def _apply_symmetry(board_np: np.ndarray, policy_np: np.ndarray,
                    transform_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of 8 D4 symmetry transforms to a (board, policy) pair.

    board_np: [3, H, W]  policy_np: [H*W]
    transform_id: 0-7 (0=identity, 1-3=rotations, 4-7=flip+rotations)
    """
    H = board_np.shape[1]
    p2d = policy_np.reshape(H, H)
    k = transform_id % 4
    if transform_id >= 4:
        board_np = np.flip(board_np, axis=2).copy()  # horizontal flip
        p2d = np.flip(p2d, axis=1).copy()
    if k > 0:
        board_np = np.rot90(board_np, k=k, axes=(1, 2)).copy()
        p2d = np.rot90(p2d, k=k).copy()
    return board_np, p2d.ravel()

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = nn.relu(x + residual)
        return x


class GomokuNet(nn.Module):
    def __init__(self, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()
        # Input conv: 3 channels -> num_filters
        self.input_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm(num_filters)

        # Residual tower
        self.res_blocks = [ResBlock(num_filters) for _ in range(num_blocks)]

        # Policy head: conv 1x1 -> 2 filters -> flatten -> FC -> 225
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value head: conv 1x1 -> 1 filter -> flatten -> FC -> 64 -> FC -> 1
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def __call__(self, x):
        # x: [B, 3, 15, 15] (NCHW from numpy encoding)
        # MLX Conv2d expects NHWC format, so transpose
        x = mx.transpose(x, (0, 2, 3, 1))  # [B, 15, 15, 3]

        # Input block
        x = nn.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = nn.relu(self.policy_bn(self.policy_conv(x)))
        p = mx.reshape(p, (p.shape[0], -1))  # [B, 2*15*15]
        p = self.policy_fc(p)  # [B, 225]

        # Value head
        v = nn.relu(self.value_bn(self.value_conv(x)))
        v = mx.reshape(v, (v.shape[0], -1))  # [B, 1*15*15]
        v = nn.relu(self.value_fc1(v))
        v = mx.tanh(self.value_fc2(v))  # [B, 1], range [-1, 1]

        return p, v


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------

def save_model(model: GomokuNet, path: str):
    """Save model weights to safetensors format."""
    model.save_weights(path)


def load_model(path: str, num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS) -> GomokuNet:
    """Load model weights from safetensors."""
    model = GomokuNet(num_blocks=num_blocks, num_filters=num_filters)
    model.load_weights(path)
    return model


# v15 B1: weight snapshot for async eval. We need to freeze the model state
# at probe-submit time so the background eval thread can read weights without
# racing the training loop's optimizer.update(). MLX arrays are immutable, so
# parameters() returns a tree of immutable refs — copying them via tree_map
# is safe and cheap (~10MB / ~50ms for an 8x128 model).
def snapshot_model(model: GomokuNet,
                   num_blocks: int = NUM_RES_BLOCKS,
                   num_filters: int = NUM_FILTERS) -> GomokuNet:
    """Return a frozen copy of `model` for use by background eval threads.

    The returned model is in eval mode and detached from any training state.
    """
    snap = GomokuNet(num_blocks=num_blocks, num_filters=num_filters)
    # tree_unflatten + flatten to deep-copy the parameter tree
    src_params = nn.utils.tree_flatten(model.parameters())
    snap_params = {k: mx.array(v) for k, v in src_params}
    snap.update(nn.utils.tree_unflatten(list(snap_params.items())))
    snap.eval()
    mx.eval(snap.parameters())
    return snap


def _eval_worker(snapshot: "GomokuNet", level: int, n_games: int,
                 num_openings: int, opponent_model: "GomokuNet | None") -> dict:
    """v15 B2: thread-pool worker. Runs in background while training continues.

    Receives a frozen weight snapshot, runs `_in_process_eval`, returns the
    full result dict (including `wr_by_opening`). Pure function — no shared
    mutable state with the main thread.
    """
    return _in_process_eval(snapshot, level, n_games,
                            opponent_model=opponent_model,
                            num_openings=num_openings)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: GomokuNet) -> int:
    """Count total trainable parameters using tree_flatten."""
    params = model.parameters()
    leaves = nn.utils.tree_flatten(params)
    total = 0
    for _, v in leaves:
        total += v.size
    return total


@dataclass(slots=True)
class ReplaySample:
    board: np.ndarray
    policy: np.ndarray
    value: float
    source: str
    game_length: int
    priority: float


def _length_priority_multiplier(value_target: float, game_length: int) -> float:
    """Low-risk length shaping applied only to replay priority.

    We deliberately keep this mild and bounded. The objective is still win/loss;
    game length only changes how often a sample is revisited.
    """
    multiplier = 1.0
    if value_target > 0.0:
        if game_length < FAST_WIN_LEN_3:
            multiplier *= FAST_WIN_BOOST_3
        elif game_length < FAST_WIN_LEN_2:
            multiplier *= FAST_WIN_BOOST_2
        elif game_length < FAST_WIN_LEN_1:
            multiplier *= FAST_WIN_BOOST_1
        elif game_length > SLOW_WIN_LEN_2:
            multiplier *= SLOW_WIN_DAMP_2
        elif game_length > SLOW_WIN_LEN_1:
            multiplier *= SLOW_WIN_DAMP_1
    elif value_target < 0.0 and game_length < FAST_LOSS_LEN:
        multiplier *= FAST_LOSS_BOOST
    return multiplier


def _compute_sample_priority(value_target: float, source: str, game_length: int) -> float:
    """Higher priority means the sample is drawn more often from replay."""
    priority = 1.0
    if source == "opponent":
        priority *= OPPONENT_SAMPLE_BOOST
        if value_target > 0:
            priority *= WIN_OPPONENT_SAMPLE_BOOST
    if value_target < 0:
        priority *= LOSS_SAMPLE_BOOST
    priority *= _length_priority_multiplier(value_target, game_length)
    return priority


def _make_replay_sample(board: np.ndarray, policy: np.ndarray,
                        value_target: float, source: str,
                        game_length: int) -> ReplaySample:
    return ReplaySample(
        board=board,
        policy=policy,
        value=float(value_target),
        source=source,
        game_length=int(game_length),
        priority=_compute_sample_priority(float(value_target), source, int(game_length)),
    )


def _sample_replay_indices(replay_buffer: list[ReplaySample],
                           batch_size: int) -> tuple[np.ndarray, dict[str, float]]:
    """Sample a batch with a guaranteed quota of corrective examples.

    Corrective examples are samples from opponent-play or from trajectories
    that eventually lost. This increases signal density without inventing a
    shaped reward.
    """
    buf_len = len(replay_buffer)
    recency_w = np.linspace(1.0, 3.0, buf_len, dtype=np.float64)
    priority_w = np.array([sample.priority for sample in replay_buffer], dtype=np.float64)
    weights = recency_w * priority_w

    focus_mask = np.array(
        [sample.source == "opponent" or sample.value < 0.0 for sample in replay_buffer],
        dtype=bool,
    )
    focus_quota = 0
    focus_count = int(focus_mask.sum())
    if focus_count > 0:
        focus_quota = min(max(1, int(batch_size * FOCUSED_SAMPLE_RATIO)), focus_count)

    chosen = np.empty(0, dtype=np.int64)
    if focus_quota > 0:
        focus_idx = np.flatnonzero(focus_mask)
        focus_w = weights[focus_idx]
        focus_sum = focus_w.sum()
        focus_p = (focus_w / focus_sum) if focus_sum > 0 else None
        chosen = np.random.choice(focus_idx, size=focus_quota, replace=False, p=focus_p)

    remaining = batch_size - chosen.size
    if remaining > 0:
        if chosen.size > 0:
            pool_mask = np.ones(buf_len, dtype=bool)
            pool_mask[chosen] = False
            pool_idx = np.flatnonzero(pool_mask)
        else:
            pool_idx = np.arange(buf_len)
        pool_w = weights[pool_idx]
        pool_sum = pool_w.sum()
        pool_p = (pool_w / pool_sum) if pool_sum > 0 else None
        extra = np.random.choice(pool_idx, size=remaining, replace=False, p=pool_p)
        chosen = np.concatenate([chosen, extra])

    np.random.shuffle(chosen)

    chosen_samples = [replay_buffer[i] for i in chosen]
    chosen_count = len(chosen_samples)
    focus_hits = sum(
        1 for sample in chosen_samples
        if sample.source == "opponent" or sample.value < 0.0
    )
    len_multipliers = [
        _length_priority_multiplier(sample.value, sample.game_length)
        for sample in chosen_samples
    ]
    boost_hits = sum(1 for mult in len_multipliers if mult > 1.001)
    damp_hits = sum(1 for mult in len_multipliers if mult < 0.999)
    stats = {
        "corrective_ratio": focus_hits / chosen_count if chosen_count > 0 else 0.0,
        "len_boost_ratio": boost_hits / chosen_count if chosen_count > 0 else 0.0,
        "len_damp_ratio": damp_hits / chosen_count if chosen_count > 0 else 0.0,
        "len_priority_avg": sum(len_multipliers) / chosen_count if chosen_count > 0 else 1.0,
    }
    return chosen, stats


# ---------------------------------------------------------------------------
# MCTS — Gomoku adapter for framework/core/mcts.py
# ---------------------------------------------------------------------------

MCTS_BATCH_SIZE = 8       # leaf nodes per GPU batch call in batched MCTS
MCTS_VIRTUAL_LOSS = 3.0   # virtual loss for batched search path diversification


def mcts_search(board, model, num_simulations: int,
                c_puct: float = C_PUCT,
                dirichlet_alpha: float = DIRICHLET_ALPHA,
                dirichlet_frac: float = DIRICHLET_FRAC,
                _compiled_forward=None) -> np.ndarray:
    """Run batched MCTS from the given Gomoku board state.

    Uses framework/core/mcts.mcts_search_batched() with virtual loss for
    GPU-efficient leaf batching. Falls back to serial if batch_size=1.

    Returns a [225] visit count distribution (unnormalized).
    """
    fwd = _compiled_forward or model

    def _evaluate_batch(states):
        """Batch evaluate multiple board states in one MLX forward pass."""
        encodings = np.stack([s.encode() for s in states])  # [B, 3, 15, 15]
        enc_mx = mx.array(encodings)
        if _compiled_forward is not None:
            priors, values = fwd(enc_mx)
            mx.eval(priors, values)
            priors_all = np.array(priors)
        else:
            logits, values = fwd(enc_mx)
            mx.eval(logits, values)
            priors_all = np.array(mx.softmax(logits, axis=-1))  # [B, 225]
        values_np = np.array(values).flatten()               # [B]
        return [(priors_all[i], float(values_np[i])) for i in range(len(states))]

    def _apply(state, action):
        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        state.place(row, col)

    def _terminal_value(state):
        return 0.0 if state.winner == -1 else 1.0

    return _mcts_search_batched(
        root_state=board,
        evaluate_batch_fn=_evaluate_batch,
        copy_fn=lambda s: s.copy(),
        legal_mask_fn=lambda s: s.get_legal_mask(),
        apply_fn=_apply,
        terminal_fn=lambda s: s.is_terminal(),
        terminal_value_fn=_terminal_value,
        action_size=BOARD_SIZE * BOARD_SIZE,
        num_simulations=num_simulations,
        batch_size=MCTS_BATCH_SIZE,
        virtual_loss=MCTS_VIRTUAL_LOSS,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_frac=dirichlet_frac,
    )


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def _run_self_play_mcts(model, num_games: int, mcts_sims: int,
                        temperature: float = TEMPERATURE) -> tuple[list, int, float, dict]:
    """Run self-play games using MCTS for policy target generation.

    Uses mcts_search_multi_root() to run ALL boards' MCTS searches with a
    shared GPU batch — leaves from all N trees are merged into one batch call.
    This maximizes GPU utilization (batch~N vs batch~1 per board).

    Returns (training_data, games_completed, avg_game_length, mcts_stats).
    """
    from game import Board

    model.eval()

    # v15.4: compile forward pass for ~15-20% inference speedup.
    # Safe because model is in eval mode (no BatchNorm state mutation).
    def _raw_forward(x):
        logits, values = model(x)
        return mx.softmax(logits, axis=-1), values
    _compiled_forward = mx.compile(_raw_forward)

    # Intra-search MLX cache release counter. mx.clear_cache() is cheap
    # (~10μs) but calling it on every single forward is still wasteful for
    # small MCTS batches. Every 8 forwards is the sweet spot: keeps the
    # peak pool size bounded to ~8 × activation_size while paying <1% of
    # wall time on the release call itself.
    _ebc = [0]

    def _evaluate_batch(states):
        """Shared batch evaluate for all boards' MCTS trees."""
        if not states:
            return []
        encodings = np.stack([s.encode() for s in states])
        enc_mx = mx.array(encodings)
        priors, values = _compiled_forward(enc_mx)
        mx.eval(priors, values)
        priors_all = np.array(priors)
        values_np = np.array(values).flatten()
        _ebc[0] += 1
        if _ebc[0] % 8 == 0:
            mx.clear_cache()
        return [(priors_all[i], float(values_np[i])) for i in range(len(states))]

    def _apply(state, action):
        """Fast-path action application for MCTS simulation.

        Skips Board.place()'s is_legal() check (MCTS paths are legal by
        construction — the tree only stores legal children), and skips
        the history.append() because _fast_copy zeroed history and we
        never replay a sim state. These two elisions cut ~35% off the
        profiled cost of place() at pg=16 per v14.1 profile.
        """
        row = action // BOARD_SIZE
        col = action % BOARD_SIZE
        player = state.current_player
        state.grid[row, col] = player
        state.move_count += 1
        state.last_move = (row, col)
        if state._check_win(row, col, player):
            state.winner = player
        elif state.move_count >= BOARD_SIZE * BOARD_SIZE:
            state.winner = -1
        state.current_player = WHITE if player == BLACK else BLACK

    def _terminal_value(state):
        return 0.0 if state.winner == -1 else 1.0

    def _fast_copy(board):
        """Lightweight board copy — skip history (not needed for MCTS sim)."""
        b = Board.__new__(Board)
        b.grid = board.grid.copy()
        b.current_player = board.current_player
        b.move_count = board.move_count
        b.last_move = board.last_move
        b.winner = board.winner
        b.history = []
        return b

    # Select search function: native C or Python fallback
    _search_fn = _mcts_native if _USE_NATIVE_MCTS else _mcts_search_multi_root

    boards = [Board() for _ in range(num_games)]
    move_data: list[list[tuple[np.ndarray, int, np.ndarray]]] = [[] for _ in range(num_games)]
    finished = [False] * num_games

    total_search_time = 0.0
    total_moves = 0
    top1_shares: list[float] = []
    entropies: list[float] = []

    # Play all games concurrently, one move at a time
    while not all(finished):
        # Collect active boards for this round's multi-root MCTS
        active_indices = [i for i in range(num_games) if not finished[i] and not boards[i].is_terminal()]
        if not active_indices:
            for i in range(num_games):
                if boards[i].is_terminal():
                    finished[i] = True
            break

        active_boards = [boards[i] for i in active_indices]

        # Run MCTS on ALL active boards — native C or Python fallback
        t0 = _time.time()
        all_visits = _search_fn(
            root_states=active_boards,
            evaluate_batch_fn=_evaluate_batch,
            copy_fn=_fast_copy,
            legal_mask_fn=lambda s: s.get_legal_mask(),
            apply_fn=_apply,
            terminal_fn=lambda s: s.is_terminal(),
            terminal_value_fn=_terminal_value,
            action_size=BOARD_SIZE * BOARD_SIZE,
            num_simulations=mcts_sims,
            sims_per_round=MCTS_BATCH_SIZE,
            virtual_loss=MCTS_VIRTUAL_LOSS,
            c_puct=C_PUCT,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_frac=DIRICHLET_FRAC,
        )
        total_search_time += _time.time() - t0
        total_moves += len(active_indices)

        # Process each board's search result
        for board_pos, board_idx in enumerate(active_indices):
            board = boards[board_idx]
            visits = all_visits[board_pos]

            # Track visit concentration
            visit_sum = visits.sum()
            if visit_sum > 0:
                top1_shares.append(float(visits.max() / visit_sum))
                dist = visits / visit_sum
                dist_nz = dist[dist > 0]
                entropies.append(float(-np.sum(dist_nz * np.log(dist_nz))))

            # Temperature-scaled policy target
            move_num = board.move_count
            if TEMP_THRESHOLD > 0 and move_num < TEMP_THRESHOLD:
                frac = move_num / TEMP_THRESHOLD
                temp = temperature * (1.0 - frac) + 0.3 * frac
            else:
                temp = 0.3

            if temp < 0.01:
                policy_dist = np.zeros_like(visits)
                policy_dist[int(np.argmax(visits))] = 1.0
            else:
                powered = visits ** (1.0 / temp)
                total = powered.sum()
                if total > 0:
                    policy_dist = powered / total
                else:
                    policy_dist = board.get_legal_mask()
                    lsum = policy_dist.sum()
                    if lsum > 0:
                        policy_dist /= lsum

            action = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=policy_dist)
            move_data[board_idx].append((board.encode(), board.current_player, policy_dist.copy()))
            board.place(action // BOARD_SIZE, action % BOARD_SIZE)

            if board.is_terminal():
                finished[board_idx] = True

        # Metal allocator cache release.
        #
        # MLX on Apple Silicon retains forward-pass buffers in an internal
        # pool across calls — on unified-memory systems without periodic
        # clear_cache() the pool grows monotonically until it hits the
        # system RAM ceiling. v14 ran with cadence "every 80 moves" which
        # was far too sparse: at pg=16 / 800 sims, one MCTS search does
        # ~100 GPU forwards per move, so "every 80 moves" = ~8000 forwards
        # between clears. See v14-update §12.3 for the full post-mortem.
        #
        # New cadence: every round (= one complete batch of leaves across
        # all active boards). The mx.clear_cache() call is cheap — tens
        # of μs — so per-round is cost-effectively infinite headroom vs
        # the RAM cost of retaining ~1 GB/forward of activations.
        mx.clear_cache()

    # Collect training data
    all_training_data = []
    game_lengths = []
    for i in range(num_games):
        board = boards[i]
        game_length = board.move_count
        winner = board.winner
        for encoded, player, policy in move_data[i]:
            if winner == -1:
                value_target = 0.0
            elif winner == player:
                value_target = 1.0
            else:
                value_target = -1.0
            all_training_data.append(
                _make_replay_sample(encoded, policy, value_target,
                                    source="self", game_length=game_length)
            )
        game_lengths.append(game_length)

    avg_game_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
    mcts_stats = {
        "search_time_s": total_search_time,
        "sims_per_sec": (total_moves * mcts_sims / total_search_time) if total_search_time > 0 else 0.0,
        "moves_total": total_moves,
        "avg_top1_share": float(np.mean(top1_shares)) if top1_shares else 0.0,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    return all_training_data, num_games, avg_game_len, mcts_stats


def run_self_play(model, num_games=PARALLEL_GAMES, temperature=TEMPERATURE):
    """
    Run batched self-play games using the model.

    When MCTS_SIMULATIONS > 0: uses MCTS for high-quality policy targets.
    When MCTS_SIMULATIONS == 0: uses batched softmax sampling (original behavior).

    Returns (training_data, games_completed, avg_game_length, mcts_stats).
    mcts_stats is {} when MCTS is disabled.
    """
    if MCTS_SIMULATIONS > 0:
        return _run_self_play_mcts(model, num_games, MCTS_SIMULATIONS, temperature)

    # --- Original batched self-play (no MCTS) ---
    model.eval()

    batch = BatchBoards(num_games)
    move_policies: list[list[np.ndarray]] = [[] for _ in range(num_games)]

    # Play all games to completion
    while not batch.get_finished_mask().all():
        encoded = batch.encode_all()  # [N, 3, 15, 15]
        legal_masks = batch.get_legal_masks()  # [N, 225]

        # Forward pass through model
        encoded_mx = mx.array(encoded)
        logits, values = model(encoded_mx)
        mx.eval(logits, values)
        logits_np = np.array(logits)  # [N, 225]

        # Mask illegal moves: set to -inf before softmax
        logits_np[legal_masks == 0] = -1e9

        # Compute actions for each game
        actions = np.zeros(num_games, dtype=np.int32)
        for i in range(num_games):
            if batch.winners[i] != 0:
                actions[i] = 0  # dummy, won't be applied
                continue

            game_logits = logits_np[i]

            # Gradual temperature decay over TEMP_THRESHOLD moves
            move_num = batch.move_counts[i]
            if TEMP_THRESHOLD > 0 and move_num < TEMP_THRESHOLD:
                frac = move_num / TEMP_THRESHOLD
                temp = temperature * (1.0 - frac) + 0.3 * frac  # linear decay
            else:
                temp = 0.3

            if temp < 0.01:
                # Greedy
                action = int(np.argmax(game_logits))
                policy_dist = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                policy_dist[action] = 1.0
            else:
                # Apply temperature and softmax
                scaled = game_logits / temp
                scaled -= np.max(scaled)  # numerical stability
                probs = np.exp(scaled)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    # Fallback: uniform over legal moves
                    probs = legal_masks[i].copy()
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs /= probs_sum
                    else:
                        # No legal moves — game should be terminal
                        actions[i] = 0
                        continue

                # Sample from distribution
                action = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=probs)
                policy_dist = probs

            actions[i] = action
            move_policies[i].append(policy_dist.copy())

        # Apply actions to all games
        batch.step(actions)

    # Collect training data from all completed games
    all_training_data = []
    game_lengths = []
    for i in range(num_games):
        game_length = int(batch.move_counts[i])
        game_data = _collect_game_with_policies(
            batch, i, move_policies[i], source="self", game_length=game_length
        )
        all_training_data.extend(game_data)
        game_lengths.append(game_length)

    avg_game_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
    return all_training_data, num_games, avg_game_len, {}


def run_opponent_play(model, opponent_model, num_games: int,
                      temperature: float = TEMPERATURE) -> tuple[list, int, float]:
    """
    Play games where `model` faces `opponent_model`.
    Each game: model plays one colour, opponent the other (alternating).
    Training data is collected only from the model's perspective.

    Returns (training_data, games_completed, avg_game_length).
    """
    from game import Board

    model.eval()
    opponent_model.eval()

    all_data: list[ReplaySample] = []
    game_lengths: list[int] = []

    for game_i in range(num_games):
        nn_is_black = game_i < num_games // 2
        nn_player = BLACK if nn_is_black else WHITE

        board = Board()
        # Collect (encoded, policy_dist) for each model move
        model_moves: list[tuple[np.ndarray, np.ndarray, int]] = []

        while not board.is_terminal():
            if board.current_player == nn_player:
                # Model's turn — with temperature exploration
                encoded = board.encode()
                x = mx.array(encoded[np.newaxis, ...])
                logits, _ = model(x)
                mx.eval(logits)
                logits_np = np.array(logits[0])
                legal_mask = board.get_legal_mask()
                logits_np[legal_mask == 0] = -1e9

                move_num = board.move_count
                if TEMP_THRESHOLD > 0 and move_num < TEMP_THRESHOLD:
                    frac = move_num / TEMP_THRESHOLD
                    temp = temperature * (1.0 - frac) + 0.3 * frac
                else:
                    temp = 0.3

                if temp < 0.01:
                    action = int(np.argmax(logits_np))
                    policy_dist = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                    policy_dist[action] = 1.0
                else:
                    scaled = logits_np / temp
                    scaled -= np.max(scaled)
                    probs = np.exp(scaled)
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs /= probs_sum
                    else:
                        probs = legal_mask.copy()
                        probs /= probs.sum()
                    action = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=probs)
                    policy_dist = probs

                model_moves.append((encoded, policy_dist, nn_player))
                row, col = divmod(action, BOARD_SIZE)
            else:
                # Opponent's turn
                row, col = _nn_opponent_move(opponent_model, board)

            board.place(row, col)

        # Determine value targets
        winner = board.winner
        game_lengths.append(board.move_count)
        for encoded, policy_dist, player in model_moves:
            if winner == -1:
                value_target = 0.0
            elif winner == player:
                value_target = 1.0
            else:
                value_target = -1.0
            all_data.append(
                _make_replay_sample(
                    encoded, policy_dist, value_target,
                    source="opponent", game_length=board.move_count,
                )
            )

        if game_i % 20 == 0:
            mx.clear_cache()

    avg_game_len = sum(game_lengths) / len(game_lengths) if game_lengths else 0.0
    return all_data, num_games, avg_game_len


def _collect_game_with_policies(batch, game_idx, policies, source: str = "self",
                                game_length: int | None = None):
    """
    Collect training data for a finished game, pairing each move's
    board encoding with its policy distribution and value target.

    Returns list of replay samples.
    """
    winner = batch.winners[game_idx]
    history = batch.histories[game_idx]
    data = []
    resolved_game_length = game_length if game_length is not None else len(history)

    for move_idx, (encoded, action, player) in enumerate(history):
        # Value target from this player's perspective
        if winner == -1:
            value_target = 0.0
        elif winner == player:
            value_target = 1.0
        else:
            value_target = -1.0

        # Policy target: use the actual distribution that was sampled from
        if move_idx < len(policies):
            policy_target = policies[move_idx]
        else:
            # Fallback: one-hot on the action taken
            policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            policy_target[action] = 1.0

        data.append(
            _make_replay_sample(
                encoded, policy_target, value_target,
                source=source, game_length=resolved_game_length,
            )
        )

    return data


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(model, batch_boards, batch_policies, batch_values):
    """
    Combined policy + value loss — returns only the weighted total so this
    function can be differentiated via ``mlx.nn.value_and_grad``.

    batch_boards:   [B, 3, 15, 15]
    batch_policies: [B, 225] probability targets
    batch_values:   [B] float targets in [-1, 1]
    """
    pred_policies, pred_values = model(batch_boards)

    log_probs = mx.log(mx.softmax(pred_policies, axis=-1) + 1e-8)
    policy_loss = -mx.mean(mx.sum(batch_policies * log_probs, axis=-1))
    value_loss = mx.mean((pred_values.squeeze() - batch_values) ** 2)

    return POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss


def compute_loss_split(model, batch_boards, batch_policies, batch_values):
    """
    Same loss as ``compute_loss`` but also returns the (policy, value)
    components for diagnostics. Called outside the grad path.
    """
    pred_policies, pred_values = model(batch_boards)
    log_probs = mx.log(mx.softmax(pred_policies, axis=-1) + 1e-8)
    policy_loss = -mx.mean(mx.sum(batch_policies * log_probs, axis=-1))
    value_loss = mx.mean((pred_values.squeeze() - batch_values) ** 2)
    total = POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss
    return total, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    """Main training function with checkpoint tracking and text TUI."""
    import core.db as _tracker

    run_id = str(uuid.uuid4())
    run_id_short = run_id[:8]
    start_time = _time.time()

    # Seed control for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        mx.random.seed(args.seed)

    # Resolve effective parameters (CLI args override module constants)
    time_budget = args.time_budget
    eval_level = args.eval_level
    eval_interval = args.eval_interval
    probe_games = args.probe_games
    full_eval_games = args.full_eval_games
    target_win_rate = args.target_win_rate
    target_games = args.target_games
    parallel_games = args.parallel_games
    probe_window = args.probe_window
    num_blocks = args.num_blocks
    num_filters = args.num_filters
    buffer_size = args.buffer_size
    learning_rate = args.learning_rate
    steps_per_cycle = args.steps_per_cycle
    mcts_sims = args.mcts_sims
    c_puct_val = args.c_puct
    dirichlet_alpha_val = args.dirichlet_alpha
    mcts_batch = args.mcts_batch  # sims_per_round override
    auto_stop_stagnation = args.auto_stop_stagnation
    stagnation_window = args.stagnation_window

    # Apply MCTS settings to module-level constants so run_self_play sees them
    global MCTS_SIMULATIONS, C_PUCT, DIRICHLET_ALPHA, MCTS_BATCH_SIZE
    MCTS_SIMULATIONS = mcts_sims
    C_PUCT = c_puct_val
    if mcts_batch is not None:
        MCTS_BATCH_SIZE = mcts_batch
    elif mcts_sims > 0:
        MCTS_BATCH_SIZE = min(8, mcts_sims)
    DIRICHLET_ALPHA = dirichlet_alpha_val

    # Handle --no-eval-opponent: override to disable NN opponent
    if args.no_eval_opponent:
        args.eval_opponent = None

    # Detect benchmark mode: fixed budget, minimax only, no resume
    is_benchmark = (
        time_budget is not None
        and args.eval_opponent is None
        and args.resume is None
    )

    # Ensure at least one stop condition
    if time_budget is None and target_win_rate is None and target_games is None:
        print("Warning: no stop condition set, defaulting to --time-budget 300")
        time_budget = 300

    # --- Output paths (per-run UUID directory) ---
    output_dir = f"output/{run_id}"
    model_path = f"{output_dir}/model.safetensors"
    ckpt_dir = f"{output_dir}/checkpoints"
    recording_dir = f"{output_dir}/recordings"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(recording_dir, "games"), exist_ok=True)

    # --- Resume support ---
    resumed_from = None
    initial_cycle = 0
    initial_ckpt_wr = 0.0
    resume_model_path = None

    if args.resume:
        db_tmp = _tracker.init_db()
        old_run = _tracker.get_run(db_tmp, args.resume)
        if not old_run:
            print(f"Error: run '{args.resume}' not found in tracker.db")
            sys.exit(1)
        resolved_id = old_run["id"]
        latest_ckpt = _tracker.get_latest_checkpoint(db_tmp, resolved_id)
        if not latest_ckpt:
            print(f"Error: no checkpoint found for run '{resolved_id[:8]}'")
            sys.exit(1)
        resumed_from = resolved_id
        resume_model_path = latest_ckpt["model_path"]
        initial_cycle = latest_ckpt["cycle"]

        # v15 A1: cross-level resume must reset the checkpoint threshold chain.
        #
        # Prior behaviour inherited parent's final WR verbatim as
        # `initial_ckpt_wr`. When resuming into a different `--eval-level`
        # (e.g. mcts_10 vs L1 @100% → mcts_11 vs L2), the inherited 1.0 made
        # `crossed_threshold()` always return None and no checkpoint was ever
        # saved during the child run. See mcts_11 post-mortem in v15-update.md
        # §3.1. We now check parent's eval_level; if different, reset to 0 so
        # the child run starts a fresh threshold chain vs the new opponent.
        parent_eval_level = old_run.get("eval_level")
        if parent_eval_level is None or parent_eval_level != args.eval_level:
            initial_ckpt_wr = 0.0
            _ckpt_note = (
                f" [cross-level resume: parent eval_level="
                f"{parent_eval_level}, new={args.eval_level}, "
                f"threshold chain reset from 0]"
            )
        else:
            initial_ckpt_wr = latest_ckpt["win_rate"]
            _ckpt_note = ""
        print(f"Resuming from run {resolved_id[:8]}  "
              f"cycle={initial_cycle}  parent_wr={latest_ckpt['win_rate']:.1%}  "
              f"model={resume_model_path}{_ckpt_note}")
        db_tmp.close()

    # Initialize tracker DB
    db_conn = _tracker.init_db()
    hw_info = _tracker.collect_hardware_info()

    hyperparams = {
        "num_res_blocks": num_blocks,
        "num_filters": num_filters,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "parallel_games": parallel_games,
        "mcts_simulations": mcts_sims,
        "temperature": TEMPERATURE,
        "temp_threshold": TEMP_THRESHOLD,
        "replay_buffer_size": buffer_size,
        "train_steps_per_cycle": steps_per_cycle,
        "time_budget": time_budget,
        "target_win_rate": target_win_rate,
        "target_games": target_games,
        "eval_level": eval_level,
        "eval_opponent": args.eval_opponent,
        "train_opponent": args.train_opponent,
        "opponent_mix": args.opponent_mix if args.train_opponent else None,
        "seed": args.seed,
        "sweep_tag": args.sweep_tag,
    }
    _tracker.create_run(db_conn, run_id, hyperparams, hw_info,
                        resumed_from=resumed_from, output_dir=output_dir,
                        is_benchmark=is_benchmark,
                        eval_opponent=args.eval_opponent)

    # Initialize model
    # Priority: --resume > --initial-opponent > fresh init
    if resume_model_path and os.path.exists(resume_model_path):
        model = load_model(resume_model_path, num_blocks=num_blocks,
                           num_filters=num_filters)
    elif args.initial_opponent:
        # v15 F2: from-scratch run but starting weights come from a
        # registered opponent. This is the entry point for v16's
        # "S2 vs S2 from scratch" training mode.
        opp = _tracker.get_opponent(db_conn, args.initial_opponent)
        if not opp:
            print(f"Error: --initial-opponent '{args.initial_opponent}' not found")
            sys.exit(1)
        if not os.path.exists(opp["model_path"]):
            print(f"Error: opponent weights not found: {opp['model_path']}")
            sys.exit(1)
        opp_nb = opp.get("num_res_blocks") or num_blocks
        opp_nf = opp.get("num_filters") or num_filters
        if opp_nb != num_blocks or opp_nf != num_filters:
            print(f"Warning: --initial-opponent {args.initial_opponent} arch "
                  f"({opp_nb}x{opp_nf}) != --num-blocks/--num-filters "
                  f"({num_blocks}x{num_filters}). Using opponent's arch.")
            num_blocks = opp_nb
            num_filters = opp_nf
        model = load_model(opp["model_path"],
                           num_blocks=num_blocks, num_filters=num_filters)
        print(f"Initial weights loaded from opponent '{args.initial_opponent}' "
              f"({num_blocks}x{num_filters})")
    else:
        model = GomokuNet(num_blocks=num_blocks, num_filters=num_filters)
    dummy = mx.zeros((1, 3, BOARD_SIZE, BOARD_SIZE))
    _ = model(dummy)
    mx.eval(model.parameters())

    num_params = count_parameters(model)

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=learning_rate, weight_decay=WEIGHT_DECAY
    )

    # Replay buffer
    replay_buffer: list[ReplaySample] = []

    # Training state
    total_games = 0
    total_train_steps = 0
    cycle = initial_cycle
    last_loss = 0.0
    last_policy_loss: float | None = None
    last_value_loss: float | None = None
    last_probe_wr: float | None = None
    last_ckpt_wr: float = initial_ckpt_wr
    num_checkpoints = 0
    stop_reason = "time_budget"
    avg_game_length = 0.0
    last_mcts_stats: dict = {}
    mcts_sims_per_sec_history: list[float] = []
    mcts_entropy_history: list[float] = []
    policy_loss_history: list[float] = []
    value_loss_history: list[float] = []

    # v15 B2 → v15.1 hotfix: `eval_executor` and `pending_eval` are kept
    # on the books for a possible v16 multi-process redesign, but during
    # v15.1 normal training they are dormant. Eval runs synchronously inside
    # _run_probe_eval(); see the helper definitions below for the rationale
    # (MLX/Metal command-buffer thread-safety assertion). The C minimax
    # port made the synchronous path acceptable (~13s for 80 games vs L2).
    eval_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="eval-worker"
    )
    pending_eval: dict | None = None  # always None in v15.1
    eval_status_str = "idle"   # set briefly to "running" inside _run_probe_eval
    eval_submit_time: float = 0.0  # for TUI elapsed display while running

    # NN opponent (if --eval-opponent specified)
    eval_opponent_alias: str | None = args.eval_opponent
    opponent_model = None
    if eval_opponent_alias:
        opp = _tracker.get_opponent(db_conn, eval_opponent_alias)
        if not opp:
            print(f"Error: opponent '{eval_opponent_alias}' not found")
            sys.exit(1)
        if not os.path.exists(opp["model_path"]):
            print(f"Error: opponent model file not found: {opp['model_path']}")
            sys.exit(1)
        opp_nb = opp.get("num_res_blocks") or NUM_RES_BLOCKS
        opp_nf = opp.get("num_filters") or NUM_FILTERS
        opponent_model = load_model(opp["model_path"],
                                    num_blocks=opp_nb, num_filters=opp_nf)
        opponent_model.eval()
        mx.eval(opponent_model.parameters())
        print(f"Loaded NN opponent '{eval_opponent_alias}' ({opp_nb}x{opp_nf}) "
              f"from {opp['model_path']}")

    # NN training opponent (if --train-opponent specified)
    # v15 F2: when --initial-opponent is given but --train-opponent is not,
    # default the train opponent to the same alias. This is the v16 path:
    # "S2 vs S2 from scratch" boils down to one CLI flag.
    train_opponent_alias: str | None = args.train_opponent
    if train_opponent_alias is None and args.initial_opponent is not None:
        train_opponent_alias = args.initial_opponent
        if args.opponent_mix < 0.999:
            print(f"[v15 F2] --initial-opponent {args.initial_opponent} given "
                  f"without --train-opponent: using same alias as train-opponent. "
                  f"For pure S vs S, set --opponent-mix 1.0 "
                  f"(current: {args.opponent_mix})")
    train_opponent_model = None
    opponent_mix: float = args.opponent_mix if train_opponent_alias else 0.0
    if train_opponent_alias:
        opp = _tracker.get_opponent(db_conn, train_opponent_alias)
        if not opp:
            print(f"Error: train-opponent '{train_opponent_alias}' not found")
            sys.exit(1)
        if not os.path.exists(opp["model_path"]):
            print(f"Error: train-opponent model not found: {opp['model_path']}")
            sys.exit(1)
        opp_nb = opp.get("num_res_blocks") or NUM_RES_BLOCKS
        opp_nf = opp.get("num_filters") or NUM_FILTERS
        train_opponent_model = load_model(opp["model_path"],
                                          num_blocks=opp_nb, num_filters=opp_nf)
        train_opponent_model.eval()
        mx.eval(train_opponent_model.parameters())
        print(f"Loaded train-opponent '{train_opponent_alias}' ({opp_nb}x{opp_nf}) "
              f"(mix={opponent_mix:.0%}) from {opp['model_path']}")

    # History for sparklines
    loss_history: list[float] = []
    wr_history: list[float] = []
    corrective_ratio_history: list[float] = []
    len_boost_ratio_history: list[float] = []
    len_damp_ratio_history: list[float] = []
    len_priority_avg_history: list[float] = []

    last_corrective_ratio: float | None = None
    last_len_boost_ratio: float | None = None
    last_len_damp_ratio: float | None = None
    last_len_priority_avg: float | None = None

    # Loss and grad function
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # --- Plain text TUI ---
    use_tui = sys.stdout.isatty()
    events: list[str] = []

    def _sparkline(values: list[float], width: int = 30) -> str:
        return _sparkline_fn(values, width)

    def _sparkline2(values: list[float], width: int = 40) -> tuple[str, str]:
        return _sparkline2_fn(values, width)

    def _sparkline3(values: list[float], width: int = 40) -> tuple[str, str, str]:
        return _sparkline3_fn(values, width)

    def _sparkline4(values: list[float], width: int = 40) -> tuple[str, str, str, str]:
        return _sparkline4_fn(values, width)

    def _smoothed_wr(window: int = probe_window) -> float | None:
        """Sliding average of recent probe WRs for stable decisions."""
        if not wr_history:
            return None
        w = min(window, len(wr_history))
        return sum(wr_history[-w:]) / w

    # ─── v15.1 hotfix: synchronous eval helpers ──────────────────────────
    #
    # The original v15 B 系 design ran eval in a background ThreadPoolExecutor
    # to avoid blocking the main training loop. That design was killed by an
    # MLX threading bug:
    #
    #     [AGXG15XFamilyCommandBuffer ...]:1090: failed assertion
    #     `A command encoder is already encoding to this command buffer'
    #
    # Root cause: MLX/Metal command buffers are NOT thread-safe. Even with a
    # snapshot of the weights, the GPU command queue is shared across threads,
    # so two simultaneous `model(x)` calls (one in the eval thread, one in the
    # main self-play / training thread) race and fire the assertion.
    #
    # Fix: revert to synchronous eval. Because v15 C 系 already brought
    # probe eval down from ~30 min → ~13 sec via the C minimax port, the
    # synchronous block is now a tolerable cost (cf v15-update §11.2 numbers).
    # All other v15 work (C minimax, checkpoint policy, observability,
    # promotion gate, --auto-promote-to, --initial-opponent, README) is
    # preserved — only B 系 is rolled back.
    #
    # `pending_eval`, `eval_executor`, `snapshot_model` are kept on the
    # books for v16's potential multi-process redesign, but during normal
    # training pending_eval stays None.

    def _run_probe_eval(eval_games: int) -> dict:
        """v15.1 sync probe — replaces the old async submit/integrate flow.

        Called inline from the main loop. Returns the full eval result dict
        so the caller can update wr_history and trigger checkpoint logic.
        """
        nonlocal eval_status_str, eval_submit_time
        eval_status_str = "running"
        eval_submit_time = _time.time()
        # Use the live model directly — single-threaded, no race with itself.
        model.eval()
        result = _in_process_eval(model, eval_level, eval_games,
                                  opponent_model=opponent_model,
                                  num_openings=args.eval_openings)
        mx.clear_cache()
        eval_status_str = "idle"
        return result

    def _integrate_probe_result(result: dict, submit_cycle: int):
        """Apply a probe result to wr_history, cycle_metrics, threshold
        check, target WR, and stagnation logic. Triggers a synchronous full
        eval + checkpoint save if a threshold was crossed.
        """
        nonlocal last_probe_wr, last_ckpt_wr, num_checkpoints, stop_reason

        wr = result["win_rate"]
        uniq = result.get("unique_trajectories", 0)
        n_open = result.get("num_openings", 0)
        n_g = result.get("wins", 0) + result.get("losses", 0) + result.get("draws", 0)
        elapsed = result.get("eval_elapsed_s", 0.0)

        last_probe_wr = wr
        wr_history.append(wr)
        sm_wr = _smoothed_wr()
        sm_str = f" (avg:{sm_wr:.1%})" if sm_wr is not None and len(wr_history) > 1 else ""
        opp_label = eval_opponent_alias if eval_opponent_alias else f"L{eval_level}"
        _log_event(
            f"Probe c{submit_cycle}: {wr:.1%}{sm_str} "
            f"({n_g}g vs {opp_label}, {uniq}u/{n_open}o, {elapsed:.1f}s)"
        )

        _tracker.save_cycle_metric(db_conn, run_id, {
            "cycle": submit_cycle,
            "timestamp_s": _time.time() - start_time,
            "loss": last_loss,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "total_games": total_games,
            "total_steps": total_train_steps,
            "buffer_size": len(replay_buffer),
            "win_rate": wr,
            "eval_type": "probe",
            "eval_games": n_g,
            "eval_level": eval_level,
            "eval_submitted_cycle": submit_cycle,
        })

        # Threshold check: did smoothed WR cross a checkpoint gate?
        effective_wr = sm_wr if sm_wr is not None else wr
        crossed = _tracker.crossed_threshold(effective_wr, last_ckpt_wr)
        if crossed is not None:
            _log_event(f"→ Threshold {crossed:.0%} crossed, full eval ({full_eval_games}g)…")
            full_result = _run_probe_eval(full_eval_games)
            _save_full_eval_checkpoint(full_result, threshold=crossed)
            num_checkpoints += 1
            last_ckpt_wr = effective_wr

        # Target WR early stop
        if target_win_rate and effective_wr >= target_win_rate:
            _log_event(f"🎯 Target {target_win_rate:.0%} reached (avg:{effective_wr:.1%})")
            stop_reason = "target_win_rate"

        # Stagnation early stop
        if auto_stop_stagnation and len(wr_history) >= stagnation_window:
            _sw = wr_history[-stagnation_window:]
            _xs = list(range(len(_sw)))
            _n = len(_sw)
            _sx = sum(_xs); _sy = sum(_sw)
            _sxy = sum(x * y for x, y in zip(_xs, _sw))
            _sxx = sum(x * x for x in _xs)
            _denom = _n * _sxx - _sx * _sx
            if abs(_denom) > 1e-12:
                _slope = (_n * _sxy - _sx * _sy) / _denom
                _mean = _sy / _n
                _std = (sum((w - _mean) ** 2 for w in _sw) / _n) ** 0.5
                _expected = abs(_slope * _n)
                if _expected < _std and _std > 0.01:
                    _log_event(
                        f"⚠ Stagnation: WR plateau for {stagnation_window} "
                        f"evals (std={_std:.1%})"
                    )
                    stop_reason = "stagnation"

    def _save_full_eval_checkpoint(result: dict, threshold: float | None):
        """Save a checkpoint built from a synchronous full-eval result."""
        wr_pct = int((threshold if threshold is not None else result["win_rate"]) * 100)
        tag = f"wr{wr_pct:03d}_c{cycle:04d}"
        ckpt_path = f"{ckpt_dir}/{tag}.safetensors"
        save_model(model, ckpt_path)
        elapsed = _time.time() - start_time
        full_wr = result.get("win_rate", 0.0)
        uniq = result.get("unique_trajectories", 0)
        n_eval = result.get("wins", 0) + result.get("losses", 0) + result.get("draws", 0)

        ckpt_id = _tracker.save_checkpoint(db_conn, run_id, {
            "tag": tag,
            "cycle": cycle,
            "step": total_train_steps,
            "loss": last_loss,
            "win_rate": full_wr,
            "eval_level": eval_level,
            "eval_games": n_eval,
            "wins": result.get("wins"),
            "losses": result.get("losses"),
            "draws": result.get("draws"),
            "avg_game_length": result.get("avg_game_length"),
            "num_params": num_params,
            "model_path": ckpt_path,
            "model_size_bytes": os.path.getsize(ckpt_path),
            "train_elapsed_s": elapsed,
            "eval_elapsed_s": result.get("eval_elapsed_s"),
            "eval_unique_openings": uniq,
            "recent_smoothed_wr": list(wr_history[-5:]) if wr_history else [],
        })
        if result.get("wr_by_opening"):
            _tracker.save_eval_breakdown(db_conn, ckpt_id, result["wr_by_opening"])
        _log_event(
            f"✓ Checkpoint {tag} (full eval {full_wr:.1%}, "
            f"{result.get('wins',0)}W/{result.get('losses',0)}L, "
            f"{uniq}u/{result.get('num_openings',0)}o)"
        )

    def _recent_avg(values: list[float], window: int = 5) -> float | None:
        if not values:
            return None
        w = min(window, len(values))
        return sum(values[-w:]) / w

    def _progress_bar(elapsed: float, budget: float | None, width: int = 46) -> str:
        return _progress_bar_fn(elapsed, budget, width)

    def _draw_panel():
        """Render the full text TUI panel with fixed-width columns."""
        W = 76  # inner width (~20% wider than before)
        chip = hw_info.get("chip", "")
        elapsed = _time.time() - start_time
        gps = total_games / elapsed if elapsed > 0 else 0.0

        def row(text: str) -> str:
            """Pad or truncate text to exactly W chars inside box borders."""
            return "│" + text[:W].ljust(W) + "│"

        lines = []
        lines.append("╭" + "─" * W + "╮")
        prov = "[benchmark]" if is_benchmark else "[exploratory]"
        resumed = " resumed" if resumed_from else ""
        lines.append(row(f" Run: {run_id_short}   {chip}   {num_params/1000:.1f}K params   {prov}{resumed}"))
        if time_budget is not None:
            lines.append(row(" " + _progress_bar(elapsed, time_budget)))
        else:
            tgt = f"  Target: {target_win_rate:.0%} WR" if target_win_rate else ""
            em, es = divmod(int(elapsed), 60)
            lines.append(row(f" Elapsed: {em}:{es:02d}{tgt}"))

        # --- Stats block (3 rows, fixed column widths) ---
        # Col layout: "  {lbl} {val:>6}  │  {lbl} {val:>7}  │  {lbl} {val}"
        lines.append("├" + "─" * W + "┤")
        wr_str = f"{last_probe_wr:.0%}" if last_probe_wr is not None else "—"
        sm_wr = _smoothed_wr()
        sm_str = f" avg:{sm_wr:.0%}" if sm_wr is not None and len(wr_history) > 1 else ""
        opp_str = eval_opponent_alias if eval_opponent_alias else f"L{eval_level}"
        mix_str = f" mix:{train_opponent_alias}({opponent_mix:.0%})" if train_opponent_alias else ""
        lines.append(row(f"  Cycle   {cycle:>6d}   │   Loss     {last_loss:>8.4f}   │   Games   {total_games:>7d}"))
        lines.append(row(f"  Steps   {total_train_steps:>6d}   │   Buffer   {len(replay_buffer):>8d}   │   WR  {wr_str:>5}{sm_str}"))
        lines.append(row(f"  Gm/s    {gps:>6.1f}   │   AvgLen   {avg_game_length:>8.1f}   │   vs {opp_str}{mix_str}"))
        if last_policy_loss is not None and last_value_loss is not None:
            lines.append(row(
                f"  P-Loss  {last_policy_loss:>6.3f}   │   V-Loss   {last_value_loss:>8.4f}   │   "
                f"policy entropy gap: {max(0.0, last_policy_loss):.2f} nats"
            ))

        # --- v15.1 sync eval status row ---
        # eval_status_str is set briefly to "running" inside _run_probe_eval()
        # while the synchronous eval is in progress, then back to "idle".
        # Most cycles will see "idle" because eval is fast.
        if eval_status_str == "running":
            elapsed_e = _time.time() - eval_submit_time
            lines.append(row(
                f"  Eval    ⋯ running (sync, {elapsed_e:>4.0f}s)                              "
            ))
        else:
            lines.append(row(f"  Eval    ★ idle                                                       "))

        # --- v15 E4: training health row ---
        h_learning = "★"
        if len(policy_loss_history) >= 20:
            recent = policy_loss_history[-10:]
            older = policy_loss_history[-20:-10]
            if sum(recent)/10 >= sum(older)/10 - 0.005:
                h_learning = "⚠"
        h_diverse = "★"
        if last_probe_wr is not None and last_value_loss is not None:
            # Diversity proxy: probe was recently run and finished
            pass
        h_value = "★" if (last_value_loss is None or 0.05 <= last_value_loss <= 0.6) else "⚠"
        h_plateau = "⚠" if (len(wr_history) >= 5
                              and max(wr_history[-5:]) - min(wr_history[-5:]) < 0.02
                              and max(wr_history[-5:]) < 0.95) else "★"
        h_collapse = "★"  # filled in by full-eval; default ★
        lines.append(row(
            f"  Health  learn:{h_learning}  diverse:{h_diverse}  plateau:{h_plateau}  "
            f"value:{h_value}  collapse:{h_collapse}"
        ))

        # --- MCTS stats row (only when MCTS is active) ---
        if MCTS_SIMULATIONS > 0:
            sps = last_mcts_stats.get("sims_per_sec", 0) if last_mcts_stats else 0
            top1 = last_mcts_stats.get("avg_top1_share", 0) if last_mcts_stats else 0
            ent = last_mcts_stats.get("avg_entropy", 0) if last_mcts_stats else 0
            sp_t = last_mcts_stats.get("search_time_s", 0) if last_mcts_stats else 0
            lines.append("│" + "╌" * W + "│")
            lines.append(row(f"  MCTS    {MCTS_SIMULATIONS:>4}sims   │   Sim/s   {sps:>8.0f}   │   SP time  {sp_t:>5.1f}s"))
            lines.append(row(f"  Focus   {top1:>6.0%}   │   Entropy  {ent:>8.2f}   │   c_puct    {C_PUCT:>4.1f}"))

        # --- Charts block (WR + Loss, each 4-row height) ---
        CW = 52  # chart width
        if wr_history or loss_history:
            lines.append("├" + "─" * W + "┤")
            if wr_history:
                top, up, mid, lo = _sparkline4(wr_history, CW)
                wr_last = wr_history[-1]
                lines.append(row(f"  Win Rate   {' ' * (CW - len(top) + 2)}{top}   {wr_last:>4.0%}"))
                lines.append(row(f"             {' ' * (CW - len(up) + 2)}{up}        "))
                lines.append(row(f"             {' ' * (CW - len(mid) + 2)}{mid}        "))
                lines.append(row(f"             {' ' * (CW - len(lo) + 2)}{lo}        "))
            if wr_history and loss_history:
                lines.append("│" + "╌" * W + "│")
            if loss_history:
                top, up, mid, lo = _sparkline4(loss_history, CW)
                lines.append(row(f"  Loss       {' ' * (CW - len(top) + 2)}{top}   {last_loss:>5.2f}"))
                lines.append(row(f"             {' ' * (CW - len(up) + 2)}{up}        "))
                lines.append(row(f"             {' ' * (CW - len(mid) + 2)}{mid}        "))
                lines.append(row(f"             {' ' * (CW - len(lo) + 2)}{lo}        "))

        # --- MCTS quality block (entropy + sims/sec sparklines) ---
        if MCTS_SIMULATIONS > 0 and len(mcts_entropy_history) > 1:
            lines.append("├" + "─" * W + "┤")
            ent_up, ent_lo = _sparkline2(mcts_entropy_history, CW)
            sps_up, sps_lo = _sparkline2(mcts_sims_per_sec_history, CW)
            ent_last = mcts_entropy_history[-1] if mcts_entropy_history else 0
            sps_last = mcts_sims_per_sec_history[-1] if mcts_sims_per_sec_history else 0
            lines.append(row(f"  Entropy    {' ' * (CW - len(ent_up) + 2)}{ent_up}   {ent_last:>5.2f}"))
            lines.append(row(f"             {' ' * (CW - len(ent_lo) + 2)}{ent_lo}        "))
            lines.append(row(f"  Sim/s      {' ' * (CW - len(sps_up) + 2)}{sps_up}   {sps_last:>5.0f}"))
            lines.append(row(f"             {' ' * (CW - len(sps_lo) + 2)}{sps_lo}        "))

        # --- Signal-quality block ---
        if last_corrective_ratio is not None:
            lines.append("├" + "─" * W + "┤")
            corr_avg = _recent_avg(corrective_ratio_history)
            corr_avg_str = f"{corr_avg:.0%}" if corr_avg is not None and len(corrective_ratio_history) > 1 else "—"
            len_avg = _recent_avg(len_priority_avg_history)
            len_avg_val = f"x{len_avg:.2f}" if len_avg is not None and len(len_priority_avg_history) > 1 else "—"
            corr_str = f"{last_corrective_ratio:.0%}"
            boost_str = f"{last_len_boost_ratio:.0%}" if last_len_boost_ratio is not None else "—"
            damp_str = f"{last_len_damp_ratio:.0%}" if last_len_damp_ratio is not None else "—"
            lp_str = f"x{(last_len_priority_avg or 1.0):.2f}"
            sig_up, sig_lo = _sparkline2(corrective_ratio_history, CW)
            len_up, len_lo = _sparkline2(len_priority_avg_history, CW)
            lines.append(row(f"  Signal  {corr_str:>6}   │   Avg      {corr_avg_str:>8}   │"))
            lines.append(row(f"          {' ' * (CW - len(sig_up) + 2)}{sig_up}        "))
            lines.append(row(f"          {' ' * (CW - len(sig_lo) + 2)}{sig_lo}        "))
            lines.append(row(f"  Length  {lp_str:>6}   │   Boost    {boost_str:>8}   │   Damp    {damp_str:>6}"))
            lines.append(row(f"          {' ' * (CW - len(len_up) + 2)}{len_up}        "))
            lines.append(row(f"          {' ' * (CW - len(len_lo) + 2)}{len_lo}        "))

        # --- Events block ---
        if events:
            lines.append("├" + "─" * W + "┤")
            for e in events[-10:]:
                lines.append(row(f"  {e}"))
        lines.append("╰" + "─" * W + "╯")
        return "\n".join(lines)

    def _update_tui():
        if use_tui:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_draw_panel() + "\n")
            sys.stdout.flush()

    def _log_event(msg: str):
        ts = _time.strftime("%H:%M:%S")
        events.append(f"[{ts}] {msg}")
        if not use_tui:
            print(f"[{ts}] {msg}")

    mcts_backend = "C-native" if (_USE_NATIVE_MCTS and mcts_sims > 0) else ("Python" if mcts_sims > 0 else "off")
    # v15.2: also log the minimax backend so users can confirm the C minimax
    # is actually loaded on Mac (not silently falling back to Python).
    try:
        import prepare as _prepare_mod
        _mm_backend = getattr(_prepare_mod, "MINIMAX_BACKEND", "?")
        _prepare_file = _prepare_mod.__file__
    except Exception:
        _mm_backend = "?"
        _prepare_file = "?"
    _log_event(f"Started run {run_id_short} | {num_params/1000:.1f}K params"
               + (f" | budget {time_budget}s" if time_budget else "")
               + (f" | MCTS {mcts_sims}sims [{mcts_backend}]" if mcts_sims > 0 else "")
               + f" | minimax [{_mm_backend}]")
    _log_event(f"prepare.py → {_prepare_file}")

    # Show TUI immediately so user sees something before first cycle
    _update_tui()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    try:
        while True:
            elapsed = _time.time() - start_time
            if time_budget is not None and elapsed >= time_budget:
                stop_reason = "time_budget"
                break
            if target_games and total_games >= target_games:
                stop_reason = "target_games"
                break

            cycle += 1

            # ----- Self-play (+ optional opponent games) -----
            # v15 F1: opponent_mix=1.0 means PURE opponent play (no self-play).
            # This is the entry point for v16's "S2 vs S2 from scratch" path.
            # Old behaviour forced n_self ≥ 1 even at mix=1.0, which prevented
            # the pure-opponent training mode from running.
            model.eval()
            if train_opponent_model and opponent_mix > 0 and parallel_games >= 1:
                if opponent_mix >= 0.999:
                    n_opp = parallel_games
                    n_self = 0
                else:
                    n_opp = max(1, int(parallel_games * opponent_mix))
                    n_self = max(1, parallel_games - n_opp)
            else:
                n_opp = 0
                n_self = parallel_games

            sp_t0 = _time.time()
            if n_self > 0:
                data, games_done, avg_gl, mcts_st = run_self_play(
                    model, num_games=n_self, temperature=TEMPERATURE
                )
                last_mcts_stats = mcts_st
            else:
                data, games_done, avg_gl = [], 0, 0.0
            selfplay_time = _time.time() - sp_t0
            total_games += games_done
            avg_game_length = avg_gl

            if n_opp > 0 and train_opponent_model is not None:
                opp_data, opp_done, opp_gl = run_opponent_play(
                    model, train_opponent_model, n_opp, temperature=TEMPERATURE
                )
                data.extend(opp_data)
                total_games += opp_done
                # Weighted average of game lengths
                total_g = games_done + opp_done
                avg_game_length = (avg_gl * games_done + opp_gl * opp_done) / total_g if total_g > 0 else opp_gl

            for item in data:
                if len(replay_buffer) >= buffer_size:
                    idx = random.randint(0, len(replay_buffer) - 1)
                    replay_buffer[idx] = item
                else:
                    replay_buffer.append(item)

            # Track MCTS stats for TUI
            if last_mcts_stats:
                if last_mcts_stats.get("sims_per_sec"):
                    mcts_sims_per_sec_history.append(last_mcts_stats["sims_per_sec"])
                if last_mcts_stats.get("avg_entropy") is not None:
                    mcts_entropy_history.append(last_mcts_stats["avg_entropy"])

            # ----- Training -----
            steps_run_this_cycle = 0
            if len(replay_buffer) >= BATCH_SIZE:
                model.train()
                cycle_loss = 0.0
                cycle_corrective_ratios: list[float] = []
                cycle_len_boost_ratios: list[float] = []
                cycle_len_damp_ratios: list[float] = []
                cycle_len_priority_avgs: list[float] = []
                # Last-batch tensors for out-of-grad loss-split diagnostics
                last_batch_boards_mx = None
                last_batch_policies_mx = None
                last_batch_values_mx = None
                steps_this_cycle = min(
                    steps_per_cycle,
                    len(replay_buffer) // BATCH_SIZE
                )
                steps_this_cycle = max(steps_this_cycle, 1)

                for step in range(steps_this_cycle):
                    if time_budget is not None and _time.time() - start_time >= time_budget:
                        break

                    indices, sample_stats = _sample_replay_indices(replay_buffer, BATCH_SIZE)
                    cycle_corrective_ratios.append(sample_stats["corrective_ratio"])
                    cycle_len_boost_ratios.append(sample_stats["len_boost_ratio"])
                    cycle_len_damp_ratios.append(sample_stats["len_damp_ratio"])
                    cycle_len_priority_avgs.append(sample_stats["len_priority_avg"])
                    # Apply random D4 symmetry transform to each sample
                    aug_boards = []
                    aug_policies = []
                    aug_values = []
                    for i in indices:
                        sample = replay_buffer[i]
                        b, p, v = sample.board, sample.policy, sample.value
                        t = random.randint(0, 7)
                        if t > 0:
                            b, p = _apply_symmetry(b, p, t)
                        aug_boards.append(b)
                        aug_policies.append(p)
                        aug_values.append(v)
                    batch_boards_np = np.stack(aug_boards)
                    batch_policies_np = np.stack(aug_policies)
                    batch_values_np = np.array(aug_values, dtype=np.float32)

                    batch_boards_mx = mx.array(batch_boards_np)
                    batch_policies_mx = mx.array(batch_policies_np)
                    batch_values_mx = mx.array(batch_values_np)

                    loss, grads = loss_and_grad(
                        model, batch_boards_mx, batch_policies_mx, batch_values_mx
                    )
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)

                    cycle_loss += loss.item()
                    total_train_steps += 1
                    steps_run_this_cycle += 1
                    last_batch_boards_mx = batch_boards_mx
                    last_batch_policies_mx = batch_policies_mx
                    last_batch_values_mx = batch_values_mx

                    # Release MLX allocator pool during training. Without
                    # this, 50 grad steps × ~1 GB activations+gradients per
                    # step can retain tens of GB of Metal buffers that are
                    # never returned to the OS — the root cause of the
                    # mcts_11 "RAM locked at 115 GB" symptom. Every 4 steps
                    # is a reasonable amortisation: clear_cache is ~10μs
                    # but we don't want to pay it on every single step.
                    # See v14-update §12.3.
                    if total_train_steps % 4 == 0:
                        mx.clear_cache()

                if steps_run_this_cycle > 0:
                    last_loss = cycle_loss / steps_run_this_cycle
                    # Recompute on the last mini-batch to recover the
                    # policy / value breakdown (one extra forward pass per
                    # cycle — negligible vs steps_run_this_cycle=50).
                    if last_batch_boards_mx is not None:
                        _total, _pl, _vl = compute_loss_split(
                            model,
                            last_batch_boards_mx,
                            last_batch_policies_mx,
                            last_batch_values_mx,
                        )
                        mx.eval(_pl, _vl)
                        last_policy_loss = float(_pl.item())
                        last_value_loss = float(_vl.item())
                        policy_loss_history.append(last_policy_loss)
                        value_loss_history.append(last_value_loss)
                if cycle_corrective_ratios:
                    last_corrective_ratio = sum(cycle_corrective_ratios) / len(cycle_corrective_ratios)
                    last_len_boost_ratio = sum(cycle_len_boost_ratios) / len(cycle_len_boost_ratios)
                    last_len_damp_ratio = sum(cycle_len_damp_ratios) / len(cycle_len_damp_ratios)
                    last_len_priority_avg = sum(cycle_len_priority_avgs) / len(cycle_len_priority_avgs)
                    corrective_ratio_history.append(last_corrective_ratio)
                    len_boost_ratio_history.append(last_len_boost_ratio)
                    len_damp_ratio_history.append(last_len_damp_ratio)
                    len_priority_avg_history.append(last_len_priority_avg)

            # Record cycle metrics
            metric = {
                "cycle": cycle,
                "timestamp_s": _time.time() - start_time,
                "loss": last_loss if steps_run_this_cycle > 0 else None,
                "policy_loss": last_policy_loss if steps_run_this_cycle > 0 else None,
                "value_loss": last_value_loss if steps_run_this_cycle > 0 else None,
                "total_games": total_games,
                "total_steps": total_train_steps,
                "buffer_size": len(replay_buffer),
            }

            # ----- Probe evaluation (v15.1: synchronous call, see hotfix in
            #  _run_probe_eval). Async eval was killed by an MLX/Metal
            #  command-buffer thread-safety assertion; with C minimax the
            #  synchronous probe is ~13s for 80 games vs L2, acceptable. ----
            if cycle % eval_interval == 0 and evaluate_win_rate is not None:
                _probe_result = _run_probe_eval(probe_games)
                _integrate_probe_result(_probe_result, submit_cycle=cycle)

            if stop_reason in ("target_win_rate", "stagnation"):
                _update_tui()
                break

            if steps_run_this_cycle > 0 and last_loss > 0:
                loss_history.append(last_loss)

            # Only emit a cycle_metrics row if something meaningful happened
            # this cycle. Probe rows are written separately by
            # _integrate_probe_result() (v15.1 sync path).
            if steps_run_this_cycle > 0:
                _tracker.save_cycle_metric(db_conn, run_id, metric)

            # Update TUI or print
            if use_tui:
                _update_tui()
            elif cycle % CYCLES_PER_REPORT == 0:
                elapsed = _time.time() - start_time
                wr_str = f" | WR: {last_probe_wr:.1%}" if last_probe_wr is not None else ""
                mcts_str = ""
                if last_mcts_stats:
                    sps = last_mcts_stats.get("sims_per_sec", 0)
                    top1 = last_mcts_stats.get("avg_top1_share", 0)
                    mcts_str = f" | MCTS: {sps:.0f}sim/s focus:{top1:.0%}"
                print(
                    f"Cycle {cycle:4d} | "
                    f"Loss: {last_loss:.4f} | "
                    f"Games: {total_games:6d} | "
                    f"Steps: {total_train_steps:6d} | "
                    f"Buffer: {len(replay_buffer):6d}{wr_str}{mcts_str} | "
                    f"Time: {elapsed:.1f}s"
                )

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        _log_event("Training interrupted by user")
    finally:
        # v15.1: eval is synchronous now (see hotfix comment near
        # _run_probe_eval). No in-flight future to drain. The unused
        # eval_executor and pending_eval are kept on the books for v16's
        # potential multi-process redesign, but they're no-ops in v15.1.
        try:
            eval_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        if use_tui:
            # Print final state after clearing
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(_draw_panel() + "\n")
            sys.stdout.flush()

    # ----- Save final model -----
    total_elapsed = _time.time() - start_time
    save_model(model, model_path)
    _log_event(f"Model saved to {model_path}")

    # ----- Final evaluation -----
    # v15.2: wrap in try/except so a SECOND Ctrl+C during the final eval
    # doesn't skip finish_run() and --auto-promote-to. mcts_12 hit this:
    # user pressed Ctrl+C, then again during the slow final eval, and the
    # run ended up with status='running' and no auto-promote at all.
    final_wr = last_probe_wr
    opp_label = eval_opponent_alias if eval_opponent_alias else f"L{eval_level}"
    result = None
    if evaluate_win_rate is not None:
        print(f"\nRunning final evaluation vs {opp_label} ({full_eval_games} games)...")
        tag = f"final_c{cycle:04d}"

        try:
            model.eval()
            result = _in_process_eval(model, eval_level, full_eval_games,
                                      opponent_model=opponent_model,
                                      num_openings=args.eval_openings)
            mx.clear_cache()
        except KeyboardInterrupt:
            print("\n⚠ Final eval interrupted by second Ctrl+C — "
                  "skipping checkpoint save, proceeding to finalize.")
            stop_reason = "interrupted"
            result = None
        if result:
            final_wr = result.get("win_rate", final_wr)
            uniq = result.get("unique_trajectories", 0)
            n_open = result.get("num_openings", 0)
            print(
                f"Final win_rate: {final_wr:.1%}  "
                f"({result.get('wins', 0)}W/{result.get('losses', 0)}L/"
                f"{result.get('draws', 0)}D, "
                f"{uniq} unique games / {n_open} openings)"
            )

            # Save as checkpoint
            ckpt_path = f"{ckpt_dir}/{tag}.safetensors"
            shutil.copy2(model_path, ckpt_path)

            ckpt_id = _tracker.save_checkpoint(db_conn, run_id, {
                "tag": tag,
                "cycle": cycle,
                "step": total_train_steps,
                "loss": last_loss,
                "win_rate": final_wr,
                "eval_level": eval_level,
                "eval_games": full_eval_games,
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "draws": result.get("draws"),
                "avg_game_length": result.get("avg_game_length"),
                "num_params": num_params,
                "model_path": ckpt_path,
                "model_size_bytes": os.path.getsize(ckpt_path),
                "train_elapsed_s": total_elapsed,
                "eval_elapsed_s": result.get("eval_elapsed_s"),
                "eval_unique_openings": uniq,
                # v15 E2: include recent smoothed WR so can_promote() runs
                "recent_smoothed_wr": list(wr_history[-5:]) if wr_history else [],
            })
            # v15 E3: persist per-opening WR breakdown
            if result.get("wr_by_opening"):
                _tracker.save_eval_breakdown(db_conn, ckpt_id,
                                             result["wr_by_opening"])
            # Save recording metadata
            for gd in result.get("game_details", []):
                if "game_file" in gd:
                    _tracker.save_recording(db_conn, ckpt_id, run_id, gd)
            db_conn.commit()
            num_checkpoints += 1

    # ----- Finalize run -----
    # final_loss: write None rather than 0 if no training step ever ran,
    # so analyze.py doesn't display a misleading "loss=0.0000" summary.
    _final_loss = last_loss if total_train_steps > 0 else None
    _tracker.finish_run(db_conn, run_id, {
        "status": "completed" if stop_reason != "interrupted" else "interrupted",
        "total_cycles": cycle - initial_cycle,
        "total_games": total_games,
        "total_steps": total_train_steps,
        "final_loss": _final_loss,
        "final_win_rate": final_wr,
        "num_params": num_params,
        "num_checkpoints": num_checkpoints,
        "wall_time_s": total_elapsed,
        "peak_memory_mb": None,
    })

    # ----- v15 E5: --auto-promote-to <alias> -----
    # If the user asked for auto-promotion and at least one checkpoint in
    # this run is promotion_eligible, register the most-recent eligible
    # checkpoint as the target alias. If nothing is eligible, print the
    # reason so the user sees why promotion was declined.
    if args.auto_promote_to:
        import shutil as _shutil
        eligible_ckpts = db_conn.execute(
            "SELECT id, tag, cycle, win_rate, eval_level, model_path, "
            "promotion_eligible, promotion_reason "
            "FROM checkpoints WHERE run_id = ? "
            "ORDER BY cycle DESC", (run_id,)
        ).fetchall()

        chosen = None
        for row in eligible_ckpts:
            if row["promotion_eligible"] == 1:
                chosen = row
                break

        if chosen is None:
            reasons = [f"  {r['tag']}: {r['promotion_reason'] or 'unknown'}"
                       for r in eligible_ckpts[:5]]
            print(f"\n⚠ --auto-promote-to {args.auto_promote_to}: "
                  f"no checkpoint is promotion_eligible")
            for r in reasons:
                print(r)
        else:
            alias = args.auto_promote_to
            dst_dir = os.path.join("output", "opponents", alias)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, "model.safetensors")
            _shutil.copy2(chosen["model_path"], dst_path)

            # Detect prev_alias from resumed_from chain (if any)
            prev_alias = None
            if resumed_from:
                parent_opp = db_conn.execute(
                    "SELECT alias FROM opponents WHERE source_run = ?",
                    (resumed_from,),
                ).fetchone()
                if parent_opp:
                    prev_alias = parent_opp["alias"]

            _tracker.register_opponent(
                db_conn, alias, dst_path,
                source_run=run_id,
                source_tag=chosen["tag"],
                win_rate=chosen["win_rate"],
                eval_level=chosen["eval_level"],
                description=f"Auto-promoted from {run_id[:8]}/{chosen['tag']}",
                num_res_blocks=num_blocks,
                num_filters=num_filters,
                prev_alias=prev_alias,
            )
            print(f"\n✓ Auto-promoted to {alias}: "
                  f"{chosen['tag']} WR={chosen['win_rate']:.1%} "
                  f"(prev_alias={prev_alias or '—'})")
            print(f"  Model copied to {dst_path}")

    db_conn.close()

    # ----- Print summary -----
    print()
    print("=" * 60)
    prov_label = "benchmark" if is_benchmark else "exploratory"
    resumed_str = f"  (resumed from {resumed_from[:8]})" if resumed_from else ""
    opp_label = eval_opponent_alias if eval_opponent_alias else f"L{eval_level}"
    print(f"Run:        {run_id_short}{resumed_str} ({stop_reason}) [{prov_label}]")
    print(f"Opponent:   {opp_label}")
    if train_opponent_alias:
        print(f"Train-opp:  {train_opponent_alias} (mix={opponent_mix:.0%})")
    print(f"Model:      {num_blocks}x{num_filters} ({num_params/1000:.1f}K params)")
    print(f"Cycles:     {cycle - initial_cycle} (total cycle #{cycle})")
    print(f"Games:      {total_games}")
    print(f"Steps:      {total_train_steps}")
    if total_train_steps > 0:
        pv = ""
        if last_policy_loss is not None and last_value_loss is not None:
            pv = f"  (P:{last_policy_loss:.3f} V:{last_value_loss:.4f})"
        print(f"Final loss: {last_loss:.4f}{pv}")
    else:
        print("Final loss: — (no training steps executed)")
    if final_wr is not None:
        print(f"Win rate:   {final_wr:.1%} (vs {opp_label})")
    print(f"Checkpoints:{num_checkpoints}")
    print(f"Wall time:  {total_elapsed:.1f}s")
    print(f"Output:     {output_dir}/")
    print(f"Tracker:    output/tracker.db")

    # Promotion hints
    if final_wr is not None and final_wr >= 0.65:
        print()
        best_tag = None
        if num_checkpoints > 0:
            ckpts = _tracker.init_db()
            clist = _tracker.get_checkpoints(ckpts, run_id)
            ckpts.close()
            if clist:
                best = max(clist, key=lambda c: c.get("win_rate", 0))
                best_tag = best["tag"]
                best_wr = best.get("win_rate", 0)
                print(f"💡 Best checkpoint: {best_tag} (WR {best_wr:.1%})")
        print("💡 Consider registering this model as an opponent:")
        if best_tag:
            print(f"   uv run python src/train.py --register-opponent my_opponent "
                  f"--from-run {run_id_short} --from-tag {best_tag}")
        print(f"   Then train against it: uv run python src/train.py --eval-opponent my_opponent")
    print("=" * 60)


def _quick_eval(model, level: int, n_games: int,
                opponent_model=None, num_openings: int = 0) -> float:
    """In-process lightweight evaluation. Returns win_rate only.

    Thin wrapper over ``_in_process_eval`` kept for call-site ergonomics.
    """
    result = _in_process_eval(model, level, n_games, opponent_model,
                              num_openings=num_openings)
    return result["win_rate"]


# Openings bank — a small set of canonical / near-canonical 2-move seeds.
# Forces each eval game to start from a distinct position so deterministic
# argmax + (previously) deterministic minimax can no longer collapse the
# entire eval to a single replayed trajectory. Combined with the stochastic
# minimax opponents in prepare.py, this restores real statistical power
# to "N games" eval numbers. See v14 findings §7.3 / §7.8.
#
# Each opening is a list of (row, col) stones placed alternating BLACK,
# WHITE starting from the empty board. Positions are chosen to cover a
# wide range of classical gomoku openings without handing either side a
# decisive head-start.
_EVAL_OPENING_SEEDS: list[list[tuple[int, int]]] = [
    [],                              # empty — strongest baseline
    [(7, 7)],                        # center-only
    [(7, 7), (7, 8)],                # direct (center → right)
    [(7, 7), (8, 8)],                # indirect (center → diag)
    [(7, 7), (6, 8)],                # indirect (center → anti-diag)
    [(7, 7), (8, 7)],                # direct (center → down)
    [(7, 7), (6, 7)],                # direct (center → up)
    [(7, 7), (7, 6)],                # direct (center → left)
    [(7, 7), (5, 7)],                # long-range vertical
    [(7, 7), (7, 5)],                # long-range horizontal
    [(7, 7), (5, 5)],                # far diagonal
    [(7, 7), (9, 9)],                # far diagonal opposite
    [(7, 7), (5, 9)],                # far anti-diagonal
    [(7, 7), (9, 5)],                # far anti-diagonal opposite
    [(6, 7), (7, 8)],                # off-center
    [(7, 8), (8, 7)],                # off-center opposite
]


def _apply_opening(board, seed: list[tuple[int, int]]) -> None:
    """Apply an opening sequence to an empty board. Caller owns the board."""
    for row, col in seed:
        if board.is_terminal():
            return
        # Legal check is defensive — all seeds fit an empty 15x15 board
        if not board.place(row, col):
            return


def _in_process_eval(model, level: int, n_games: int,
                     opponent_model=None,
                     num_openings: int = 0) -> dict:
    """In-process evaluation with BATCHED NN calls across concurrent games.

    v15.2 hotfix — replaces the v14/v15 sequential implementation that did
    one batch=1 MLX forward per move per game. mcts_12 on Mac M3 Max showed
    that the sequential design takes ~10-20 minutes per probe (vs the ~13
    seconds my Linux fake-NN benchmark predicted). Root cause: at batch=1,
    MLX has 50-300 ms of GPU dispatch latency per call, and a 150-game
    probe makes ~1800 such calls — almost all the wall time was MLX
    dispatch overhead, not actual compute.

    The new design plays all `n_games` games concurrently, wave by wave.
    On each wave we batch ALL boards whose current player is the NN into
    a single forward pass, then sequentially apply the minimax opponent's
    moves. This pushes batch sizes up from 1 to ~75 (half of n_games),
    where MLX overhead is amortised properly.

    Expected speedup: 8-15× on Mac M3 Max for 80-200 game probes against
    L2/L3 minimax. The C minimax opponent calls remain sequential — they
    are already fast enough that batching them would not help.
    """
    from game import Board

    use_nn_opponent = opponent_model is not None
    opponent_fn = None
    if not use_nn_opponent:
        from prepare import OPPONENTS
        opponent_fn = OPPONENTS[level]

    if num_openings <= 0:
        num_openings = min(len(_EVAL_OPENING_SEEDS), max(1, n_games // 4))
    num_openings = min(num_openings, len(_EVAL_OPENING_SEEDS))
    openings = _EVAL_OPENING_SEEDS[:num_openings]

    # Per-game state arrays
    boards = [Board() for _ in range(n_games)]
    nn_player_of = [BLACK if i < n_games // 2 else WHITE for i in range(n_games)]
    trajectories: list[list[int]] = [[] for _ in range(n_games)]
    opening_index_of: list[int] = [0] * n_games
    finished = [False] * n_games

    # Apply opening seeds + record opening_index for per-opening breakdown
    for i in range(n_games):
        nn_is_black = nn_player_of[i] == BLACK
        local_i = i if nn_is_black else i - n_games // 2
        oi = local_i % len(openings)
        opening_index_of[i] = oi
        _apply_opening(boards[i], openings[oi])

    start = _time.time()
    nn_call_count = 0

    # v15.4: compiled forward for eval NN (model is in eval mode)
    def _fwd(x):
        return model(x)
    _compiled_fwd = mx.compile(_fwd)

    while True:
        # Mark newly-terminal games
        for i in range(n_games):
            if not finished[i] and boards[i].is_terminal():
                finished[i] = True
        if all(finished):
            break

        # Collect all boards where it's the NN's turn
        nn_turn_idx: list[int] = []
        opp_turn_idx: list[int] = []
        for i in range(n_games):
            if finished[i]:
                continue
            if boards[i].current_player == nn_player_of[i]:
                nn_turn_idx.append(i)
            else:
                opp_turn_idx.append(i)

        # ── Batched NN move ─────────────────────────────────────────────
        # ONE GPU forward pass for all NN-turn boards (batch ~ n_games/2).
        # This amortises MLX dispatch latency from ~250 ms × N → ~30 ms × 1.
        if nn_turn_idx:
            encodings = np.stack([boards[i].encode() for i in nn_turn_idx])
            x = mx.array(encodings)
            policy_logits, _ = _compiled_fwd(x)
            mx.eval(policy_logits)
            policy_np = np.array(policy_logits)  # [B, 225]
            for k, i in enumerate(nn_turn_idx):
                legal = boards[i].get_legal_mask()
                masked = policy_np[k].copy()
                masked[legal == 0] = -np.inf
                action = int(np.argmax(masked))
                row, col = divmod(action, BOARD_SIZE)
                boards[i].place(row, col)
                trajectories[i].append(row * BOARD_SIZE + col)
            nn_call_count += 1

        # ── Sequential opponent moves ──────────────────────────────────
        # Minimax opponent. Each call is ~10-65ms on C path; sequential
        # is fine because opp_turn_idx is short and the bottleneck is no
        # longer here.
        if opp_turn_idx:
            for i in opp_turn_idx:
                if use_nn_opponent:
                    # NN opponent: batch this group too for symmetry.
                    pass  # handled below via batched path
                else:
                    row, col = opponent_fn(boards[i])
                    boards[i].place(row, col)
                    trajectories[i].append(row * BOARD_SIZE + col)

            # Optionally batch NN opponent moves
            if use_nn_opponent:
                opp_encodings = np.stack([boards[i].encode() for i in opp_turn_idx])
                opp_x = mx.array(opp_encodings)
                opp_logits, _ = opponent_model(opp_x)
                mx.eval(opp_logits)
                opp_np = np.array(opp_logits)
                for k, i in enumerate(opp_turn_idx):
                    legal = boards[i].get_legal_mask()
                    masked = opp_np[k].copy()
                    masked[legal == 0] = -np.inf
                    action = int(np.argmax(masked))
                    row, col = divmod(action, BOARD_SIZE)
                    boards[i].place(row, col)
                    trajectories[i].append(row * BOARD_SIZE + col)

        # Periodic MLX cache release. Coarser than the sequential version
        # because each iteration here does ~75 forwards-worth of work.
        if nn_call_count % 8 == 0:
            mx.clear_cache()

    elapsed = _time.time() - start
    mx.clear_cache()

    # ── Tally results + per-opening breakdown ──────────────────────────
    wins, losses, draws = 0, 0, 0
    total_moves = 0
    trajectory_fingerprints: set[tuple] = set()
    per_opening: list[dict] = [
        {"opening_index": i, "opening_moves": str(openings[i]),
         "wins": 0, "losses": 0, "draws": 0,
         "_lengths": [], "_traj": set()}
        for i in range(len(openings))
    ]

    for i in range(n_games):
        b = boards[i]
        nn_p = nn_player_of[i]
        traj_tuple = tuple(trajectories[i])
        trajectory_fingerprints.add(traj_tuple)
        bucket = per_opening[opening_index_of[i]]
        bucket["_lengths"].append(b.move_count)
        bucket["_traj"].add(traj_tuple)
        total_moves += b.move_count
        if b.winner == nn_p:
            wins += 1
            bucket["wins"] += 1
        elif b.winner is not None and b.winner != 0:
            losses += 1
            bucket["losses"] += 1
        else:
            draws += 1
            bucket["draws"] += 1

    breakdown = []
    for b in per_opening:
        avg_len_b = sum(b["_lengths"]) / max(1, len(b["_lengths"]))
        breakdown.append({
            "opening_index": b["opening_index"],
            "opening_moves": b["opening_moves"],
            "wins": b["wins"],
            "losses": b["losses"],
            "draws": b["draws"],
            "avg_length": avg_len_b,
            "unique_games": len(b["_traj"]),
        })

    return {
        "win_rate": wins / n_games if n_games > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_game_length": total_moves / n_games if n_games > 0 else 0,
        "eval_elapsed_s": elapsed,
        "num_openings": len(openings),
        "unique_trajectories": len(trajectory_fingerprints),
        "wr_by_opening": breakdown,
        "nn_batch_calls": nn_call_count,  # diagnostic
    }


def _nn_opponent_move(opp_model, board, temperature: float = 0.0) -> tuple[int, int]:
    """Make a move using an NN opponent model.

    Default is deterministic argmax (temperature=0). Evaluation against a
    temperature>0 NN opponent is a v13-era artifact — a noisy opponent
    inflates win rates with "gift" losses from random opponent blunders
    rather than measuring actual strength. Callers that want stochastic
    NN play should pass temperature explicitly.
    """
    encoded = board.encode()
    x = mx.array(encoded[np.newaxis, ...])
    policy_logits, _ = opp_model(x)
    policy = policy_logits[0]
    legal_mask = mx.array(board.get_legal_mask())
    masked = mx.where(legal_mask > 0, policy, mx.array(float("-inf")))
    if temperature > 0:
        probs = mx.softmax(masked / temperature)
        action = int(mx.random.categorical(mx.log(probs + 1e-10)).item())
    else:
        action = int(mx.argmax(masked).item())
    row, col = divmod(action, BOARD_SIZE)
    return row, col


def _subprocess_eval(model_path: str, level: int, n_games: int,
                     tag: str, run_id: str,
                     recording_dir: str = "",
                     num_blocks: int = NUM_RES_BLOCKS,
                     num_filters: int = NUM_FILTERS) -> dict | None:
    """Run full evaluation in subprocess, return parsed result dict."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    fw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.pardir, os.pardir, "framework")
    fw_dir = os.path.abspath(fw_dir)
    # Domain dir MUST come before framework dir so gomoku/prepare.py shadows framework/prepare.py
    env = {**os.environ, "PYTHONPATH": f"{src_dir}:{fw_dir}", "PYTHONUNBUFFERED": "1"}
    rec_arg = f", recording_dir='{recording_dir}'" if recording_dir else ""
    # Monkey-patch load_model in BOTH train and prepare modules.
    # Circular import: `import train` triggers `import prepare` which binds
    # prepare.load_model to the ORIGINAL before our patch runs. So we must
    # also patch prepare.load_model after import.
    patch = (
        f"import train; import prepare; "
        f"_orig = train.load_model; "
        f"_patched = lambda p, **kw: _orig(p, num_blocks={num_blocks}, num_filters={num_filters}); "
        f"train.load_model = _patched; "
        f"prepare.load_model = _patched; "
    )
    code = (
        f"{patch}"
        f"import json; "
        f"r = prepare.evaluate_win_rate('{model_path}', level={level}, "
        f"n_games={n_games}, record_games={n_games}, "
        f"tag='{tag}', run_id='{run_id}'{rec_arg}); "
        f"print('JSON_RESULT:' + json.dumps(r, default=str))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"Evaluation error (exit {proc.returncode}):")
        if proc.stderr:
            print(proc.stderr.strip()[:500])
        return None

    # Parse JSON result from stdout
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])

    # Fallback: print raw output
    if proc.stdout:
        print(proc.stdout)
    return None


def _do_checkpoint(model, db_conn, run_id, run_id_short, cycle,
                   total_steps, loss, win_rate, eval_level,
                   full_eval_games, num_params, start_time,
                   events, log_event_fn, ckpt_dir, recording_dir,
                   threshold=None,
                   num_blocks=NUM_RES_BLOCKS, num_filters=NUM_FILTERS,
                   opponent_model=None, eval_opponent_alias=None,
                   num_openings: int = 0):
    """Save checkpoint, run full eval (NN or subprocess minimax), record to DB."""
    import core.db as _tracker

    # Tag uses the crossed threshold (not raw probe WR)
    wr_pct = int((threshold if threshold is not None else win_rate) * 100)
    tag = f"wr{wr_pct:03d}_c{cycle:04d}"
    ckpt_path = f"{ckpt_dir}/{tag}.safetensors"
    save_model(model, ckpt_path)

    opp_label = eval_opponent_alias if eval_opponent_alias else f"L{eval_level}"
    log_event_fn(f"✓ Checkpoint {tag}  wr={win_rate:.1%}")

    # Full eval: always in-process with opening diversification.
    elapsed = _time.time() - start_time
    model.eval()
    result = _in_process_eval(model, eval_level, full_eval_games,
                              opponent_model=opponent_model,
                              num_openings=num_openings)
    mx.clear_cache()

    if result:
        full_wr = result.get("win_rate", win_rate)
        uniq = result.get("unique_trajectories", 0)
        n_open = result.get("num_openings", 0)
        log_event_fn(
            f"  Full eval: {full_wr:.1%} "
            f"({result.get('wins', '?')}W/{result.get('losses', '?')}L "
            f"in {result.get('eval_elapsed_s', 0):.1f}s, "
            f"{uniq}uniq/{n_open}op)"
        )

        ckpt_id = _tracker.save_checkpoint(db_conn, run_id, {
            "tag": tag,
            "cycle": cycle,
            "step": total_steps,
            "loss": loss,
            "win_rate": full_wr,
            "eval_level": eval_level,
            "eval_games": full_eval_games,
            "wins": result.get("wins"),
            "losses": result.get("losses"),
            "draws": result.get("draws"),
            "avg_game_length": result.get("avg_game_length"),
            "num_params": num_params,
            "model_path": ckpt_path,
            "model_size_bytes": os.path.getsize(ckpt_path),
            "train_elapsed_s": elapsed,
            "eval_elapsed_s": result.get("eval_elapsed_s"),
            "eval_unique_openings": uniq,
        })
        # Save recording metadata
        for gd in result.get("game_details", []):
            if "game_file" in gd:
                _tracker.save_recording(db_conn, ckpt_id, run_id, gd)
        db_conn.commit()
    else:
        # Eval failed — save checkpoint without eval details
        _tracker.save_checkpoint(db_conn, run_id, {
            "tag": tag,
            "cycle": cycle,
            "step": total_steps,
            "loss": loss,
            "win_rate": win_rate,
            "eval_level": eval_level,
            "eval_games": 0,
            "num_params": num_params,
            "model_path": ckpt_path,
            "model_size_bytes": os.path.getsize(ckpt_path),
            "train_elapsed_s": elapsed,
        })
        db_conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAG-Gomoku Training")
    p.add_argument("--time-budget", type=int, default=None,
                   help="Training time budget in seconds (default: unlimited)")
    p.add_argument("--target-win-rate", type=float, default=None,
                   help="Stop when smoothed win rate reaches this value")
    p.add_argument("--target-games", type=int, default=None,
                   help="Stop after this many self-play games")
    p.add_argument("--eval-level", type=int, default=EVAL_LEVEL,
                   help=f"Evaluation opponent level 0-3 (default: {EVAL_LEVEL})")
    p.add_argument("--eval-opponent", type=str, default=None,
                   help="Evaluate against a registered NN opponent alias "
                        "(default: None = use minimax at --eval-level). "
                        "This used to default to 'L4' which broke any fresh "
                        "tracker.db that didn't have that legacy alias.")
    p.add_argument("--no-eval-opponent", action="store_true",
                   help="Deprecated no-op — kept for command compatibility. "
                        "The new default is already 'no NN opponent'.")
    p.add_argument("--eval-interval", type=int, default=15,
                   help="Probe evaluation every N cycles (default: 15)")
    p.add_argument("--probe-games", type=int, default=100,
                   help="Games per probe evaluation (default: 100)")
    p.add_argument("--probe-window", type=int, default=5,
                   help="Sliding window size for smoothed win rate (default: 5)")
    p.add_argument("--full-eval-games", type=int, default=200,
                   help="Games per full evaluation at checkpoint (default: 200)")
    p.add_argument("--eval-openings", type=int, default=0,
                   help="Number of distinct opening seeds to force in eval "
                        "(0 = auto from eval bank, typically ~16). Fixes the "
                        "'200 games = 2 unique games' collapse documented in "
                        "v14 findings §7.3.")
    p.add_argument("--parallel-games", type=int, default=PARALLEL_GAMES,
                   help=f"Number of simultaneous self-play games (default: {PARALLEL_GAMES})")
    # Model capacity
    p.add_argument("--num-blocks", type=int, default=NUM_RES_BLOCKS,
                   help=f"Number of residual blocks (default: {NUM_RES_BLOCKS})")
    p.add_argument("--num-filters", type=int, default=NUM_FILTERS,
                   help=f"Number of convolutional filters (default: {NUM_FILTERS})")
    # Training dynamics
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                   help=f"Learning rate (default: {LEARNING_RATE})")
    p.add_argument("--steps-per-cycle", type=int, default=TRAIN_STEPS_PER_CYCLE,
                   help=f"Training steps per self-play cycle (default: {TRAIN_STEPS_PER_CYCLE})")
    # Replay buffer
    p.add_argument("--buffer-size", type=int, default=REPLAY_BUFFER_SIZE,
                   help=f"Replay buffer capacity (default: {REPLAY_BUFFER_SIZE})")
    # MCTS
    p.add_argument("--mcts-sims", type=int, default=MCTS_SIMULATIONS,
                   help=f"MCTS simulations per move (0=disable, default: {MCTS_SIMULATIONS})")
    p.add_argument("--c-puct", type=float, default=C_PUCT,
                   help=f"MCTS exploration constant (default: {C_PUCT})")
    p.add_argument("--dirichlet-alpha", type=float, default=DIRICHLET_ALPHA,
                   help=f"Dirichlet noise alpha for MCTS root (default: {DIRICHLET_ALPHA})")
    p.add_argument("--mcts-batch", type=int, default=None,
                   help="Sims per tree per GPU batch round (default: auto = min(8, mcts-sims))")
    # Reproducibility
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility (default: None = non-deterministic)")
    # Sweep integration (set by sweep.py, not typically used directly)
    p.add_argument("--sweep-tag", type=str, default=None,
                   help=argparse.SUPPRESS)
    # Mixed opponent training
    p.add_argument("--train-opponent", type=str, default=None,
                   help="Train with mixed self-play + games vs a registered NN opponent")
    p.add_argument("--opponent-mix", type=float, default=0.2,
                   help="Fraction of games played vs train-opponent (default: 0.2)")
    p.add_argument("--resume", type=str, default=None,
                   help="UUID of a previous run to resume from its last checkpoint")
    # v15 F2: from-scratch training but start with an opponent's weights.
    # Used by v16's "S2 vs S2 from scratch" path — unlike --resume which
    # inherits cycle count and optimizer state, --initial-opponent only
    # copies the weights into a fresh run.
    p.add_argument("--initial-opponent", type=str, default=None, metavar="ALIAS",
                   help="Load weights from a registered opponent as the "
                        "starting point (fresh run, not a resume)")
    # Stagnation early stopping
    p.add_argument("--auto-stop-stagnation", action="store_true",
                   help="Auto-stop training when WR plateaus (stagnation detected)")
    p.add_argument("--stagnation-window", type=int, default=10,
                   help="Number of eval points for stagnation detection (default: 10)")
    # v15 E5: auto-promotion. On completion, if any checkpoint is eligible
    # per can_promote(), register the newest eligible one as this alias.
    p.add_argument("--auto-promote-to", type=str, default=None, metavar="ALIAS",
                   help="On success, auto-register the best eligible "
                        "checkpoint as an opponent alias (e.g. S2, S3)")
    # Opponent registration (non-training mode)
    p.add_argument("--register-opponent", type=str, default=None, metavar="ALIAS",
                   help="Register a checkpoint as a named NN opponent")
    p.add_argument("--from-run", type=str, default=None,
                   help="Source run UUID for --register-opponent")
    p.add_argument("--from-tag", type=str, default=None,
                   help="Source checkpoint tag for --register-opponent")
    p.add_argument("--description", type=str, default=None,
                   help="Description for --register-opponent")
    return p.parse_args()


def _handle_register_opponent(args: argparse.Namespace) -> None:
    """Register a checkpoint model as a named opponent, then exit."""
    import core.db as _tracker
    import shutil

    alias = args.register_opponent
    if not args.from_run or not args.from_tag:
        print("Error: --register-opponent requires --from-run and --from-tag")
        sys.exit(1)

    conn = _tracker.init_db()
    run = _tracker.get_run(conn, args.from_run)
    if not run:
        print(f"Error: run '{args.from_run}' not found")
        sys.exit(1)

    resolved_run_id = run["id"]
    ckpt = _tracker.find_checkpoint_by_tag(conn, args.from_tag)
    if not ckpt or ckpt["run_id"] != resolved_run_id:
        # Try run-scoped search
        ckpts = _tracker.get_checkpoints(conn, resolved_run_id)
        ckpt = next((c for c in ckpts if args.from_tag in c["tag"]), None)
        if not ckpt:
            print(f"Error: checkpoint '{args.from_tag}' not found in run {resolved_run_id[:8]}")
            sys.exit(1)

    src_path = ckpt["model_path"]
    if not os.path.exists(src_path):
        print(f"Error: model file not found: {src_path}")
        sys.exit(1)

    dst_dir = os.path.join("output", "opponents", alias)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, "model.safetensors")
    shutil.copy2(src_path, dst_path)

    _tracker.register_opponent(
        conn, alias, dst_path,
        source_run=resolved_run_id,
        source_tag=ckpt["tag"],
        win_rate=ckpt.get("win_rate"),
        eval_level=ckpt.get("eval_level"),
        description=args.description,
        num_res_blocks=run.get("num_res_blocks"),
        num_filters=run.get("num_filters"),
    )
    conn.close()

    nb = run.get("num_res_blocks") or NUM_RES_BLOCKS
    nf = run.get("num_filters") or NUM_FILTERS
    print(f"✓ Registered opponent '{alias}'")
    print(f"  Source: run {resolved_run_id[:8]} / {ckpt['tag']}")
    print(f"  WR: {ckpt.get('win_rate', 0):.1%}  Level: L{ckpt.get('eval_level', '?')}")
    print(f"  Model: {dst_path} ({nb}x{nf})")
    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    if args.register_opponent:
        _handle_register_opponent(args)
    else:
        train(args)
