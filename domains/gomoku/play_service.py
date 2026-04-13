"""Shared gameplay services for CLI and web frontends."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Callable, Optional

# ── path setup for decoupled project structure ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_fw_path = os.path.join(_PROJECT_ROOT, "framework")
if _fw_path not in sys.path:
    # Insert AFTER domain dir so domains/gomoku/prepare.py takes priority
    _idx = sys.path.index(_THIS_DIR) + 1 if _THIS_DIR in sys.path else 0
    sys.path.insert(_idx, _fw_path)

import numpy as np

from game import BOARD_SIZE, Board
from prepare import OPPONENTS
import core.db as tracker


PlayerFn = Callable[[Board], tuple[int, int]]



def resolve_checkpoint(tag: str) -> str:
    """Resolve a checkpoint tag or direct path to an existing weight file."""
    if os.path.isfile(tag):
        return tag

    try:
        conn = tracker.init_db()
        cp = tracker.find_checkpoint_by_tag(conn, tag)
        conn.close()
        if cp and os.path.isfile(cp["model_path"]):
            return cp["model_path"]
    except Exception:
        pass

    if tag in ("latest", "local", "best"):
        try:
            conn = tracker.init_db()
            all_cp = tracker.list_all_checkpoints(conn, limit=1)
            conn.close()
            if all_cp and os.path.isfile(all_cp[0]["model_path"]):
                return all_cp[0]["model_path"]
        except Exception:
            pass

    raise FileNotFoundError(f"checkpoint '{tag}' not found")


@lru_cache(maxsize=32)
def _load_cached_model(checkpoint_path: str,
                       num_blocks: Optional[int],
                       num_filters: Optional[int]):
    import mlx.core as mx
    from train import load_model

    model = load_model(
        checkpoint_path,
        num_blocks=num_blocks if num_blocks is not None else 6,
        num_filters=num_filters if num_filters is not None else 64,
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def load_nn_player(checkpoint_path: str, mcts_sims: int = 0,
                   num_blocks: Optional[int] = None,
                   num_filters: Optional[int] = None) -> PlayerFn:
    """Load a NN model and return a player function (board) -> (row, col).

    When mcts_sims > 0, each move runs MCTS search for stronger play.
    """
    import mlx.core as mx

    model = _load_cached_model(checkpoint_path, num_blocks, num_filters)

    if mcts_sims > 0:
        from core.mcts import mcts_search_batched

        def _evaluate_batch(states):
            encodings = np.stack([s.encode() for s in states])
            enc_mx = mx.array(encodings)
            logits, values = model(enc_mx)
            mx.eval(logits, values)
            priors_all = np.array(mx.softmax(logits, axis=-1))
            values_np = np.array(values).flatten()
            return [(priors_all[i], float(values_np[i])) for i in range(len(states))]

        def mcts_move(board: Board) -> tuple[int, int]:
            visits = mcts_search_batched(
                root_state=board,
                evaluate_batch_fn=_evaluate_batch,
                copy_fn=lambda s: s.copy(),
                legal_mask_fn=lambda s: s.get_legal_mask(),
                apply_fn=lambda s, a: s.place(a // BOARD_SIZE, a % BOARD_SIZE),
                terminal_fn=lambda s: s.is_terminal(),
                terminal_value_fn=lambda s: 0.0 if s.winner == -1 else 1.0,
                action_size=BOARD_SIZE * BOARD_SIZE,
                num_simulations=mcts_sims,
                batch_size=min(8, mcts_sims),
            )
            action = int(np.argmax(visits))
            return divmod(action, BOARD_SIZE)

        return mcts_move

    def nn_move(board: Board) -> tuple[int, int]:
        encoded = board.encode()
        x = mx.array(encoded[np.newaxis, ...])
        policy_logits, value = model(x)
        mx.eval(policy_logits, value)

        policy = np.array(policy_logits[0])
        legal_mask = board.get_legal_mask()
        policy[legal_mask == 0] = -1e9

        action = int(np.argmax(policy))
        row, col = divmod(action, BOARD_SIZE)
        return row, col

    return nn_move


def load_registered_opponent(alias: str, mcts_sims: int = 0) -> tuple[PlayerFn, dict]:
    conn = tracker.init_db()
    opponent = tracker.get_opponent(conn, alias)
    conn.close()
    if not opponent:
        raise ValueError(f"unknown opponent alias: {alias}")
    if not os.path.isfile(opponent["model_path"]):
        raise FileNotFoundError(f"opponent model not found: {opponent['model_path']}")

    player = load_nn_player(
        opponent["model_path"],
        mcts_sims=mcts_sims,
        num_blocks=opponent.get("num_res_blocks"),
        num_filters=opponent.get("num_filters"),
    )
    return player, opponent


def get_frontend_opponents() -> list[dict]:
    """Return all playable browser opponents."""
    items: list[dict] = []
    for level in sorted(OPPONENTS):
        items.append({
            "id": f"minimax-{level}",
            "label": f"Minimax L{level}",
            "type": "minimax",
            "level": level,
            "description": f"Built-in minimax opponent level {level}",
        })

    conn = tracker.init_db()
    opponents = tracker.list_opponents(conn)
    conn.close()
    for opponent in opponents:
        desc = opponent.get("description") or "Registered neural opponent"
        items.append({
            "id": opponent["alias"],
            "label": opponent["alias"],
            "type": "nn",
            "description": desc,
            "win_rate": opponent.get("win_rate"),
            "eval_level": opponent.get("eval_level"),
            "num_res_blocks": opponent.get("num_res_blocks"),
            "num_filters": opponent.get("num_filters"),
            "source_run": opponent.get("source_run"),
        })

    return items


def create_player(player_type: str, player_id: str, mcts_sims: int = 0) -> tuple[PlayerFn, dict]:
    if player_type == "minimax":
        if not player_id.startswith("minimax-"):
            raise ValueError(f"invalid minimax player id: {player_id}")
        level = int(player_id.split("-", 1)[1])
        if level not in OPPONENTS:
            raise ValueError(f"unknown minimax level: {level}")
        player = OPPONENTS[level]
        return player, {
            "id": player_id,
            "label": f"Minimax L{level}",
            "type": "minimax",
            "level": level,
        }

    if player_type == "nn":
        player, opponent = load_registered_opponent(player_id, mcts_sims=mcts_sims)
        return player, {
            "id": opponent["alias"],
            "label": opponent["alias"],
            "type": "nn",
            "description": opponent.get("description") or "Registered neural opponent",
            "num_res_blocks": opponent.get("num_res_blocks"),
            "num_filters": opponent.get("num_filters"),
            "win_rate": opponent.get("win_rate"),
        }

    raise ValueError(f"unsupported player type: {player_type}")