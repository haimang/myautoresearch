"""
autoresearch MCTS（蒙特卡洛树搜索）— 域无关实现。

v12: numpy-vectorized PUCT in select_child. Children stored as numpy
arrays (actions, priors, visits, values) for batch PUCT computation.
Child MCTSNode objects created lazily (only for visited children).
Each child knows its index in parent's arrays → O(1) sync.
"""

from typing import Any, Callable

import numpy as np


class MCTSNode:
    """MCTS tree node with numpy-vectorized children.

    Children data stored as parallel arrays for fast PUCT (no Python loop).
    Child MCTSNode objects are created lazily on first visit.
    """
    __slots__ = ("parent", "parent_idx", "action", "prior",
                 "visit_count", "value_sum", "is_expanded", "n_children",
                 "child_actions", "child_priors", "child_visits", "child_values",
                 "child_nodes")

    def __init__(self, parent: "MCTSNode | None" = None, parent_idx: int = -1,
                 action: int = -1, prior: float = 0.0):
        self.parent = parent
        self.parent_idx = parent_idx  # index in parent's child_* arrays
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.n_children = 0
        self.child_actions: np.ndarray = np.empty(0, dtype=np.int32)
        self.child_priors: np.ndarray = np.empty(0, dtype=np.float32)
        self.child_visits: np.ndarray = np.empty(0, dtype=np.int32)
        self.child_values: np.ndarray = np.empty(0, dtype=np.float32)
        self.child_nodes: list["MCTSNode | None"] = []

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Vectorized PUCT select — numpy over all children, no Python loop."""
        v = self.child_visits.astype(np.float32)
        q = np.divide(self.child_values, v,
                       out=np.zeros(self.n_children, dtype=np.float32),
                       where=v > 0)
        exploration = c_puct * self.child_priors * (self.visit_count ** 0.5) / (1.0 + v)
        best_idx = int(np.argmax(q + exploration))

        if self.child_nodes[best_idx] is None:
            self.child_nodes[best_idx] = MCTSNode(
                parent=self, parent_idx=best_idx,
                action=int(self.child_actions[best_idx]),
                prior=float(self.child_priors[best_idx]),
            )
        return self.child_nodes[best_idx]

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray):
        """Expand: store children as numpy arrays. No child MCTSNode created."""
        masked = priors * legal_mask
        total = masked.sum()
        if total > 0:
            masked *= (1.0 / total)
        else:
            masked = legal_mask.copy()
            total = masked.sum()
            if total > 0:
                masked *= (1.0 / total)

        legal_indices = np.flatnonzero(legal_mask)
        n = len(legal_indices)
        self.n_children = n
        self.child_actions = legal_indices.astype(np.int32)
        self.child_priors = masked[legal_indices].astype(np.float32)
        self.child_visits = np.zeros(n, dtype=np.int32)
        self.child_values = np.zeros(n, dtype=np.float32)
        self.child_nodes = [None] * n
        self.is_expanded = True

    def _sync_to_parent(self):
        """O(1) sync own stats to parent's arrays."""
        if self.parent is not None and self.parent_idx >= 0:
            self.parent.child_visits[self.parent_idx] = self.visit_count
            self.parent.child_values[self.parent_idx] = self.value_sum

    def backup(self, value: float):
        """Propagate value up, alternating sign. O(1) parent sync per level."""
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            node._sync_to_parent()
            v = -v
            node = node.parent

    def apply_virtual_loss(self, vl: float):
        self.visit_count += 1
        self.value_sum -= vl
        self._sync_to_parent()

    def revert_virtual_loss(self, vl: float):
        self.visit_count -= 1
        self.value_sum += vl
        self._sync_to_parent()


# ── Serial MCTS (backward compat) ────────────────────────────────────────

def mcts_search(
    root_state: Any,
    evaluate_fn: Callable[[Any], tuple[np.ndarray, float]],
    copy_fn: Callable[[Any], Any],
    legal_mask_fn: Callable[[Any], np.ndarray],
    apply_fn: Callable[[Any, int], None],
    terminal_fn: Callable[[Any], bool],
    terminal_value_fn: Callable[[Any], float],
    action_size: int,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_frac: float = 0.25,
) -> np.ndarray:
    """Serial MCTS (single-state evaluate). Kept for backward compat."""
    root = MCTSNode()
    legal_mask = legal_mask_fn(root_state)
    priors, root_value = evaluate_fn(root_state)
    root.expand(priors, legal_mask)

    if dirichlet_frac > 0 and root.n_children > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * root.n_children)
        root.child_priors = ((1 - dirichlet_frac) * root.child_priors
                             + dirichlet_frac * noise).astype(np.float32)

    root.visit_count = 1
    root.value_sum = root_value

    for _ in range(num_simulations):
        node = root
        sim_state = copy_fn(root_state)
        while node.is_expanded and not terminal_fn(sim_state):
            node = node.select_child(c_puct)
            apply_fn(sim_state, node.action)
        if terminal_fn(sim_state):
            leaf_value = terminal_value_fn(sim_state)
        else:
            lm = legal_mask_fn(sim_state)
            p, v = evaluate_fn(sim_state)
            leaf_value = -v
            node.expand(p, lm)
        node.backup(leaf_value)

    visits = np.zeros(action_size, dtype=np.float32)
    for i in range(root.n_children):
        visits[root.child_actions[i]] = root.child_visits[i]
    return visits


# ── Batched MCTS (single root) ───────────────────────────────────────────

def mcts_search_batched(
    root_state: Any,
    evaluate_batch_fn: Callable[[list[Any]], list[tuple[np.ndarray, float]]],
    copy_fn: Callable[[Any], Any],
    legal_mask_fn: Callable[[Any], np.ndarray],
    apply_fn: Callable[[Any, int], None],
    terminal_fn: Callable[[Any], bool],
    terminal_value_fn: Callable[[Any], float],
    action_size: int,
    num_simulations: int,
    batch_size: int = 8,
    virtual_loss: float = 3.0,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_frac: float = 0.25,
) -> np.ndarray:
    """Batched MCTS — collects batch_size leaf nodes per GPU call."""
    root = MCTSNode()
    legal_mask = legal_mask_fn(root_state)
    rp, rv = evaluate_batch_fn([root_state])[0]
    root.expand(rp, legal_mask)
    if dirichlet_frac > 0 and root.n_children > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * root.n_children)
        root.child_priors = ((1 - dirichlet_frac) * root.child_priors
                             + dirichlet_frac * noise).astype(np.float32)
    root.visit_count = 1
    root.value_sum = rv

    remaining = num_simulations
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        paths: list[list[MCTSNode]] = []
        leaf_states: list[Any] = []
        leaf_path_idx: list[int] = []
        term_pairs: list[tuple[int, float]] = []

        for pi in range(batch_n):
            node = root
            ss = copy_fn(root_state)
            path = [node]
            while node.is_expanded and not terminal_fn(ss):
                node.apply_virtual_loss(virtual_loss)
                node = node.select_child(c_puct)
                apply_fn(ss, node.action)
                path.append(node)
            node.apply_virtual_loss(virtual_loss)
            paths.append(path)
            if terminal_fn(ss):
                term_pairs.append((pi, terminal_value_fn(ss)))
            else:
                leaf_states.append(ss)
                leaf_path_idx.append(pi)

        evals = evaluate_batch_fn(leaf_states) if leaf_states else []
        for i, (p, v) in enumerate(evals):
            path = paths[leaf_path_idx[i]]
            leaf = path[-1]
            leaf.expand(p, legal_mask_fn(leaf_states[i]))
            for n in path:
                n.revert_virtual_loss(virtual_loss)
            leaf.backup(-v)
        for pi, tv in term_pairs:
            path = paths[pi]
            for n in path:
                n.revert_virtual_loss(virtual_loss)
            path[-1].backup(tv)
        remaining -= batch_n

    visits = np.zeros(action_size, dtype=np.float32)
    for i in range(root.n_children):
        visits[root.child_actions[i]] = root.child_visits[i]
    return visits


# ── Multi-root MCTS (N trees, shared GPU) ────────────────────────────────

def mcts_search_multi_root(
    root_states: list[Any],
    evaluate_batch_fn: Callable[[list[Any]], list[tuple[np.ndarray, float]]],
    copy_fn: Callable[[Any], Any],
    legal_mask_fn: Callable[[Any], np.ndarray],
    apply_fn: Callable[[Any, int], None],
    terminal_fn: Callable[[Any], bool],
    terminal_value_fn: Callable[[Any], float],
    action_size: int,
    num_simulations: int,
    sims_per_round: int = 4,
    virtual_loss: float = 3.0,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.03,
    dirichlet_frac: float = 0.25,
) -> list[np.ndarray]:
    """Multi-root MCTS — N trees, K sims/tree/round, shared GPU batch.

    Each round: K sims per tree × N trees → up to K*N leaves in one GPU batch.
    This reduces GPU round-trips from num_sims to num_sims/K while
    keeping the same total search quality.

    Args:
        sims_per_round: K — number of sims per tree per GPU batch round.
            Higher K = fewer GPU calls but more virtual loss distortion.
            Default 4 gives good balance: 50 sims / 4 = ~13 GPU calls
            with batch size ~4*8=32 (vs 50 calls of batch 8).
    """
    n = len(root_states)
    if n == 0:
        return []

    lms = [legal_mask_fn(s) for s in root_states]
    init_results = evaluate_batch_fn(root_states)
    roots: list[MCTSNode] = []
    for i in range(n):
        r = MCTSNode()
        p, v = init_results[i]
        r.expand(p, lms[i])
        if dirichlet_frac > 0 and r.n_children > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * r.n_children)
            r.child_priors = ((1 - dirichlet_frac) * r.child_priors
                              + dirichlet_frac * noise).astype(np.float32)
        r.visit_count = 1
        r.value_sum = v
        roots.append(r)

    remaining = num_simulations
    while remaining > 0:
        k = min(sims_per_round, remaining)
        leaf_items: list[tuple[int, int, Any]] = []  # (tree_idx, path_list_idx, state)
        term_items: list[tuple[int, float]] = []      # (path_list_idx, value)
        all_paths: list[list[MCTSNode]] = []

        # K sims per tree × N trees — collect all leaves
        for _sim in range(k):
            for ti in range(n):
                node = roots[ti]
                ss = copy_fn(root_states[ti])
                path = [node]
                while node.is_expanded and not terminal_fn(ss):
                    node.apply_virtual_loss(virtual_loss)
                    node = node.select_child(c_puct)
                    apply_fn(ss, node.action)
                    path.append(node)
                node.apply_virtual_loss(virtual_loss)
                pi = len(all_paths)
                all_paths.append(path)
                if terminal_fn(ss):
                    term_items.append((pi, terminal_value_fn(ss)))
                else:
                    leaf_items.append((ti, pi, ss))

        # ONE GPU batch for all K*N leaves
        if leaf_items:
            leaf_states = [item[2] for item in leaf_items]
            evals = evaluate_batch_fn(leaf_states)
            for (ti, pi, ss), (p, v) in zip(leaf_items, evals):
                path = all_paths[pi]
                leaf = path[-1]
                leaf.expand(p, legal_mask_fn(ss))
                for nd in path:
                    nd.revert_virtual_loss(virtual_loss)
                leaf.backup(-v)

        for pi, tv in term_items:
            path = all_paths[pi]
            for nd in path:
                nd.revert_virtual_loss(virtual_loss)
            path[-1].backup(tv)

        remaining -= k

    out = []
    for r in roots:
        v = np.zeros(action_size, dtype=np.float32)
        for i in range(r.n_children):
            v[r.child_actions[i]] = r.child_visits[i]
        out.append(v)
    return out
