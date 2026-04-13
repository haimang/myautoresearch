"""
Native C MCTS wrapper — drop-in replacement for mcts_search_multi_root.

Uses batch C calls: ONE C call to select K*N paths, Python does board ops + GPU,
ONE C call to expand+backup all paths. Minimizes Python↔C transitions.

v14.1 (2026-04-13):
- Detects the C-side MAX_BATCH_PATHS cap and warns the caller if the request
  (sims_per_round × n_roots) would be silently truncated. Prior versions would
  just drop work on the floor — see v14-update §12.2 for the full post-mortem.
- Adds an (opt-in) ThreadPoolExecutor for the Python path-walk loop. This
  was intended as the main P1 CPU-side speedup, but benchmarks on the dev
  Linux box (pg=16..32, batch=16..64) actually showed 0.81–0.87× (i.e.
  slower) because Board.place + _check_win are pure-Python GIL-bound and
  the thread dispatch overhead isn't recovered. The infrastructure is kept
  in place behind env var MCTS_NATIVE_WORKERS (default=1 = serial). If
  M3 Max performance cores shift the measurement, `MCTS_NATIVE_WORKERS=4`
  enables chunked parallel execution. The real fix (vectorising Board ops
  or moving them to C) is P2 — see v14-update §12.6.
"""

import ctypes
import os
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

_LIB = None

# Persistent worker pool for parallel Python path walks.
# Default to 1 (serial) because parallel threading measured slower on the
# pure-Python Board implementation — see module docstring. Flip via env var
# MCTS_NATIVE_WORKERS=4 to enable chunked parallel dispatch.
_WORKER_COUNT = max(1, int(os.environ.get("MCTS_NATIVE_WORKERS", "1")))
_POOL: ThreadPoolExecutor | None = None

# Compile-time cap reported by the loaded C library.
_MAX_BATCH_PATHS: int | None = None
# Track whether we've already emitted a truncation warning this process.
_WARNED_TRUNCATION = False


def _get_pool() -> ThreadPoolExecutor | None:
    global _POOL
    if _WORKER_COUNT <= 1:
        return None
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=_WORKER_COUNT,
            thread_name_prefix="mcts-walk",
        )
    return _POOL


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

    # Compile-time batch cap (v14.1+). Older .so files don't export this
    # symbol — fall back to the historical 256 in that case so the caller
    # still gets a sensible warning.
    try:
        c.mcts_max_batch_paths.restype = ctypes.c_int
        global _MAX_BATCH_PATHS
        _MAX_BATCH_PATHS = int(c.mcts_max_batch_paths())
    except (AttributeError, OSError):
        _MAX_BATCH_PATHS = 256

    return _LIB


def max_batch_paths() -> int:
    """Return the C-side MAX_BATCH_PATHS cap (or 256 for legacy builds)."""
    if _MAX_BATCH_PATHS is None:
        _load_lib()
    return _MAX_BATCH_PATHS or 256


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

    # Detect silent truncation of the C-side path batch buffer and warn ONCE
    # per process. See v14-update §12.2 for the root-cause story.
    global _WARNED_TRUNCATION
    cap = _MAX_BATCH_PATHS or 256
    if sims_per_round * n > cap and not _WARNED_TRUNCATION:
        effective_sims = cap // max(n, 1)
        warnings.warn(
            f"mcts_batch_select: sims_per_round({sims_per_round}) × "
            f"n_roots({n}) = {sims_per_round * n} exceeds C cap "
            f"MAX_BATCH_PATHS={cap}. Effective sims/round={effective_sims}. "
            f"Reduce --parallel-games or --mcts-batch, or rebuild the C "
            f"extension with a larger MAX_BATCH_PATHS.",
            RuntimeWarning, stacklevel=2,
        )
        _WARNED_TRUNCATION = True

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
        #
        # This loop used to be the dominant self-play CPU hotspot: each path
        # does a grid.copy() + a handful of apply_fn calls, all in Python.
        #
        # First attempt at P1 (one task per path) was slower than serial
        # because ThreadPoolExecutor dispatch overhead (~50 μs per task)
        # dominated the ~200 μs of actual work per path. Replaced with
        # CHUNK parallelism: split total_paths into a few large chunks and
        # dispatch those — only K submits instead of total_paths submits.
        # numpy's grid.copy() releases the GIL, so even with Python being
        # the majority of place()/is_terminal(), a ~1.3-1.5× speedup is
        # still achievable on CPU-rich (16-core) boxes.

        def _walk_chunk(start: int, end: int):
            out = []
            for pi in range(start, end):
                plen = int(path_lens[pi])
                sim_state = copy_fn(root_states[pi % n])
                is_term = False
                for step in range(1, plen):
                    action = int(all_actions[pi, step])
                    if action >= 0:
                        apply_fn(sim_state, action)
                    if terminal_fn(sim_state):
                        is_term = True
                        break
                if is_term:
                    out.append((True, pi, None, terminal_value_fn(sim_state)))
                else:
                    out.append((False, pi, sim_state, 0.0))
            return out

        pool = _get_pool()
        # Only parallelise if: (a) pool is enabled, (b) total_paths is
        # large enough that chunked work dominates dispatch, and (c) we
        # have enough paths to actually split into ≥2 meaningful chunks.
        MIN_PARALLEL_PATHS = 256
        if pool is not None and total_paths >= MIN_PARALLEL_PATHS:
            chunks = max(2, min(_WORKER_COUNT, total_paths // 128))
            chunk_size = (total_paths + chunks - 1) // chunks
            futures = [
                pool.submit(_walk_chunk, i, min(i + chunk_size, total_paths))
                for i in range(0, total_paths, chunk_size)
            ]
            walk_results: list = []
            for f in futures:
                walk_results.extend(f.result())
        else:
            walk_results = _walk_chunk(0, total_paths)

        expand_indices: list[int] = []
        expand_states: list[Any] = []
        term_indices: list[int] = []
        term_values: list[float] = []
        for is_term, pi, sim_state, tval in walk_results:
            if is_term:
                term_indices.append(pi)
                term_values.append(tval)
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
