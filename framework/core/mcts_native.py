"""
Native C MCTS wrapper — drop-in replacement for mcts_search_multi_root.

Uses batch C calls: ONE C call to select K*N paths, Python does board ops + GPU,
ONE C call to expand+backup all paths. Minimizes Python↔C transitions.
"""

import ctypes
import os
import platform
from typing import Any, Callable

import numpy as np

_LIB = None


def _find_lib():
    d = os.path.dirname(os.path.abspath(__file__))
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    path = os.path.join(d, "mcts_c" + ext)
    return path if os.path.isfile(path) else None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _find_lib()
    if path is None:
        raise ImportError("mcts_c not found. Run: cd framework/core && bash build_native.sh")
    _LIB = ctypes.CDLL(path)
    c = _LIB

    # Pool
    c.mcts_pool_reset.restype = None
    c.mcts_pool_usage.restype = ctypes.c_int

    # Node ops
    c.mcts_create_root.restype = ctypes.c_int
    c.mcts_expand.restype = None
    c.mcts_expand.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    c.mcts_add_dirichlet.restype = None
    c.mcts_add_dirichlet.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    c.mcts_set_root_value.restype = None
    c.mcts_set_root_value.argtypes = [ctypes.c_int, ctypes.c_float]
    c.mcts_get_visits.restype = None
    c.mcts_get_visits.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    # Batch select
    c.mcts_batch_select.restype = ctypes.c_int
    c.mcts_batch_select.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_float, ctypes.c_float]
    c.mcts_batch_get_path_actions.restype = ctypes.POINTER(ctypes.c_int)
    c.mcts_batch_get_path_nodes.restype = ctypes.POINTER(ctypes.c_int)
    c.mcts_batch_get_path_lens.restype = ctypes.POINTER(ctypes.c_int)
    c.mcts_batch_get_leaf_nodes.restype = ctypes.POINTER(ctypes.c_int)

    # Batch expand+backup
    c.mcts_batch_expand_backup.restype = None
    c.mcts_batch_expand_backup.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_float,
    ]

    # Single path (legacy)
    c.mcts_node_action.restype = ctypes.c_int
    c.mcts_node_is_expanded.restype = ctypes.c_int
    c.mcts_backup.restype = None
    c.mcts_backup.argtypes = [ctypes.c_int, ctypes.c_float]
    c.mcts_revert_path_virtual_loss.restype = None
    c.mcts_revert_path_virtual_loss.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

    return _LIB


def is_available() -> bool:
    try:
        _load_lib()
        return True
    except (ImportError, OSError):
        return False


MAX_PATH_DEPTH = 128  # must match C


def mcts_search_multi_root_native(
    root_states: list[Any],
    evaluate_batch_fn: Callable[[list[Any]], list[tuple[np.ndarray, float]]],
    copy_fn: Callable[[Any], Any],
    legal_mask_fn: Callable[[Any], np.ndarray],
    apply_fn: Callable[[Any, int], None],
    terminal_fn: Callable[[Any], bool],
    terminal_value_fn: Callable[[Any], float],
    action_size: int,
    num_simulations: int,
    sims_per_round: int = 8,
    virtual_loss: float = 3.0,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_frac: float = 0.25,
) -> list[np.ndarray]:
    """Multi-root MCTS using C native tree ops with batch select/backup.

    Only 2 Python↔C transitions per sim round (batch_select + batch_expand_backup)
    instead of K*N transitions. Board ops + GPU eval stay in Python.
    """
    lib = _load_lib()
    n = len(root_states)
    if n == 0:
        return []

    lib.mcts_pool_reset()

    # Init roots — batch GPU eval
    lms = [legal_mask_fn(s) for s in root_states]
    init_results = evaluate_batch_fn(root_states)

    root_arr = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ri = lib.mcts_create_root()
        priors_c = np.ascontiguousarray(init_results[i][0], dtype=np.float32)
        lm_c = np.ascontiguousarray(lms[i], dtype=np.float32)
        lib.mcts_expand(ri, priors_c.ctypes.data, lm_c.ctypes.data, action_size)
        if dirichlet_frac > 0:
            nc = int(lm_c.sum())
            if nc > 0:
                noise = np.ascontiguousarray(
                    np.random.dirichlet([dirichlet_alpha] * nc), dtype=np.float32
                )
                lib.mcts_add_dirichlet(ri, noise.ctypes.data, nc, dirichlet_frac)
        lib.mcts_set_root_value(ri, float(init_results[i][1]))
        root_arr[i] = ri

    remaining = num_simulations
    while remaining > 0:
        k = min(sims_per_round, remaining)

        # ── ONE C call: select K*N paths ──
        total_paths = lib.mcts_batch_select(
            root_arr.ctypes.data, n, k, c_puct, virtual_loss,
        )

        # Read batch results from C static buffers
        c_actions_ptr = lib.mcts_batch_get_path_actions()
        c_lens_ptr = lib.mcts_batch_get_path_lens()
        c_leaf_ptr = lib.mcts_batch_get_leaf_nodes()

        # Read path lens
        path_lens = np.ctypeslib.as_array(c_lens_ptr, shape=(total_paths,)).copy()
        leaf_nodes = np.ctypeslib.as_array(c_leaf_ptr, shape=(total_paths,)).copy()

        # Read all actions (flat, MAX_PATH_DEPTH per path)
        all_actions = np.ctypeslib.as_array(c_actions_ptr,
                                             shape=(total_paths, MAX_PATH_DEPTH)).copy()

        # ── Python: board ops for each path ──
        expand_indices = []     # path indices that need GPU expand
        expand_states = []      # board states for evaluation
        term_indices = []       # path indices that hit terminal
        term_values = []        # terminal values

        for pi in range(total_paths):
            plen = int(path_lens[pi])
            sim_state = copy_fn(root_states[pi % n])  # path pi came from root pi%n
            is_term = False
            for step in range(1, plen):
                action = int(all_actions[pi, step])
                if action >= 0:
                    apply_fn(sim_state, action)
                if terminal_fn(sim_state):
                    is_term = True
                    break
            if is_term:
                term_indices.append(pi)
                term_values.append(terminal_value_fn(sim_state))
            else:
                expand_indices.append(pi)
                expand_states.append(sim_state)

        # ── ONE GPU call: evaluate all leaves ──
        if expand_states:
            evals = evaluate_batch_fn(expand_states)
            all_priors = np.zeros((len(expand_states), action_size), dtype=np.float32)
            all_masks = np.zeros((len(expand_states), action_size), dtype=np.float32)
            all_values = np.zeros(len(expand_states), dtype=np.float32)
            for i, (priors, value) in enumerate(evals):
                all_priors[i] = priors
                all_masks[i] = legal_mask_fn(expand_states[i])
                all_values[i] = -float(value)  # negate for parent perspective
        else:
            all_priors = np.empty(0, dtype=np.float32)
            all_masks = np.empty(0, dtype=np.float32)
            all_values = np.empty(0, dtype=np.float32)

        # ── ONE C call: expand+backup all paths ──
        expand_idx_arr = np.array(expand_indices, dtype=np.int32)
        term_idx_arr = np.array(term_indices, dtype=np.int32)
        term_val_arr = np.array(term_values, dtype=np.float32)

        lib.mcts_batch_expand_backup(
            total_paths,
            expand_idx_arr.ctypes.data if len(expand_idx_arr) else None,
            len(expand_idx_arr),
            all_priors.ctypes.data if all_priors.size else None,
            all_masks.ctypes.data if all_masks.size else None,
            all_values.ctypes.data if all_values.size else None,
            action_size,
            term_idx_arr.ctypes.data if len(term_idx_arr) else None,
            len(term_idx_arr),
            term_val_arr.ctypes.data if term_val_arr.size else None,
            virtual_loss,
        )

        remaining -= k

    # Extract visit distributions
    visits_buf = np.zeros(action_size, dtype=np.float32)
    results = []
    for i in range(n):
        visits_buf[:] = 0
        lib.mcts_get_visits(int(root_arr[i]), visits_buf.ctypes.data, action_size)
        results.append(visits_buf.copy())
    return results
