"""
autoresearch MCTS（蒙特卡洛树搜索）— 域无关实现。

提供 MCTSNode 和 mcts_search()，通过回调函数与具体游戏/模型解耦。
任何 domain 只需提供 evaluate_fn / copy_fn / legal_mask_fn / apply_fn /
terminal_fn / terminal_value_fn 即可复用 MCTS 搜索。
"""

from typing import Any, Callable

import numpy as np


class MCTSNode:
    """A node in the MCTS search tree."""
    __slots__ = ("parent", "action", "prior", "visit_count", "value_sum",
                 "children", "is_expanded")

    def __init__(self, parent: "MCTSNode | None" = None, action: int = -1,
                 prior: float = 0.0):
        self.parent = parent
        self.action = action          # action that led to this node
        self.prior = prior            # P(s, a) from the network
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: list["MCTSNode"] = []
        self.is_expanded = False

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """PUCT score: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))"""
        exploration = c_puct * self.prior * (parent_visits ** 0.5) / (1 + self.visit_count)
        return self.q_value() + exploration

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Select the child with highest PUCT score."""
        best_score = -float("inf")
        best_child = None
        pv = self.visit_count
        for child in self.children:
            score = child.ucb_score(pv, c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, priors: np.ndarray, legal_mask: np.ndarray):
        """Expand this node, creating children for each legal action.

        priors: [action_size] softmax output from the policy head
        legal_mask: [action_size] float mask (1=legal, 0=illegal)
        """
        masked = priors * legal_mask
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            # fallback: uniform over legal
            masked = legal_mask.copy()
            total = masked.sum()
            if total > 0:
                masked /= total

        for action in range(len(masked)):
            if legal_mask[action] > 0:
                self.children.append(MCTSNode(parent=self, action=action,
                                              prior=float(masked[action])))
        self.is_expanded = True

    def backup(self, value: float):
        """Propagate value up the tree, alternating sign for opponent."""
        node = self
        v = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            v = -v  # flip for opponent's perspective
            node = node.parent

    def apply_virtual_loss(self, vl: float):
        """Apply virtual loss during batched select to discourage path overlap."""
        self.visit_count += 1
        self.value_sum -= vl

    def revert_virtual_loss(self, vl: float):
        """Revert virtual loss before real backup."""
        self.visit_count -= 1
        self.value_sum += vl


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
    """Run MCTS from the given state. Domain-agnostic.

    Args:
        root_state: opaque game state object
        evaluate_fn: (state) -> (priors[action_size], value_for_current_player)
        copy_fn: (state) -> deep copy of state
        legal_mask_fn: (state) -> float mask[action_size] (1=legal)
        apply_fn: (state, action_int) -> None (mutates state in-place)
        terminal_fn: (state) -> bool
        terminal_value_fn: (state) -> float value from last-mover's perspective
            (+1.0 for win, -1.0 for loss, 0.0 for draw)
        action_size: total number of possible actions
        num_simulations: MCTS rollout count
        c_puct: exploration constant
        dirichlet_alpha: root noise alpha
        dirichlet_frac: root noise mixing fraction

    Returns:
        visits: np.ndarray[action_size] — visit count distribution

    Value convention:
        node.value_sum stores value from the PARENT's player perspective.
        select_child picks the child with highest Q (best for the selecting player).
    """
    root = MCTSNode()

    # Evaluate root
    legal_mask = legal_mask_fn(root_state)
    priors, root_value = evaluate_fn(root_state)
    root.expand(priors, legal_mask)

    # Add Dirichlet noise to root for exploration diversity
    if dirichlet_frac > 0 and root.children:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children):
            child.prior = (1 - dirichlet_frac) * child.prior + dirichlet_frac * noise[i]

    root.visit_count = 1
    root.value_sum = root_value

    for _ in range(num_simulations):
        node = root
        sim_state = copy_fn(root_state)

        # 1. Select — walk down the tree using PUCT
        while node.is_expanded and not terminal_fn(sim_state):
            node = node.select_child(c_puct)
            apply_fn(sim_state, node.action)

        # 2. Expand + Evaluate
        if terminal_fn(sim_state):
            # Terminal: value from last-mover's perspective (= parent's player).
            leaf_value = terminal_value_fn(sim_state)
        else:
            lm = legal_mask_fn(sim_state)
            p, v = evaluate_fn(sim_state)
            # NN value is from current player's perspective at the expanded node.
            # Current player = opponent of the parent's player → negate.
            leaf_value = -v
            node.expand(p, lm)

        # 3. Backup — propagate leaf value up the tree
        node.backup(leaf_value)

    # Build visit count distribution
    visits = np.zeros(action_size, dtype=np.float32)
    for child in root.children:
        visits[child.action] = child.visit_count
    return visits


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
    """Batched MCTS — collects multiple leaf nodes per GPU call.

    Uses virtual loss to diversify concurrent search paths, then evaluates
    all leaves in a single batch call to the neural network. This increases
    GPU utilization from ~5% (serial) to ~60-80% (batched).

    Args:
        evaluate_batch_fn: (list[state]) -> list[(priors, value)]
            Batch version of evaluate_fn. Single call handles N states.
        batch_size: number of concurrent search paths per GPU call.
        virtual_loss: penalty applied during select to discourage path overlap.
        (all other args identical to mcts_search)

    Returns:
        visits: np.ndarray[action_size] — visit count distribution
    """
    root = MCTSNode()

    # Evaluate root (single state, wrap in list)
    legal_mask = legal_mask_fn(root_state)
    results = evaluate_batch_fn([root_state])
    root_priors, root_value = results[0]
    root.expand(root_priors, legal_mask)

    # Dirichlet noise on root
    if dirichlet_frac > 0 and root.children:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children):
            child.prior = (1 - dirichlet_frac) * child.prior + dirichlet_frac * noise[i]

    root.visit_count = 1
    root.value_sum = root_value

    remaining = num_simulations
    while remaining > 0:
        batch_n = min(batch_size, remaining)

        # Collect search paths, applying virtual loss along each path
        search_paths: list[list[MCTSNode]] = []   # nodes visited per path
        leaf_states: list[Any] = []                # states to evaluate
        leaf_indices: list[int] = []               # which path index needs eval
        terminal_pairs: list[tuple[int, float]] = []  # (path_idx, value)

        for path_idx in range(batch_n):
            node = root
            sim_state = copy_fn(root_state)
            path = [node]

            # Select with virtual loss
            while node.is_expanded and not terminal_fn(sim_state):
                node.apply_virtual_loss(virtual_loss)
                node = node.select_child(c_puct)
                apply_fn(sim_state, node.action)
                path.append(node)
            node.apply_virtual_loss(virtual_loss)

            search_paths.append(path)

            if terminal_fn(sim_state):
                terminal_pairs.append((path_idx, terminal_value_fn(sim_state)))
            else:
                leaf_states.append(sim_state)
                leaf_indices.append(path_idx)

        # Batch evaluate all non-terminal leaves in one GPU call
        if leaf_states:
            eval_results = evaluate_batch_fn(leaf_states)
        else:
            eval_results = []

        # Expand + backup non-terminal paths
        for i, (state, (priors, value)) in enumerate(zip(leaf_states, eval_results)):
            path_idx = leaf_indices[i]
            path = search_paths[path_idx]
            leaf_node = path[-1]

            lm = legal_mask_fn(state)
            leaf_node.expand(priors, lm)
            leaf_value = -value  # negate: NN value is current player, we want parent's

            # Revert virtual loss then backup
            for n in path:
                n.revert_virtual_loss(virtual_loss)
            leaf_node.backup(leaf_value)

        # Backup terminal paths
        for path_idx, tv in terminal_pairs:
            path = search_paths[path_idx]
            for n in path:
                n.revert_virtual_loss(virtual_loss)
            path[-1].backup(tv)

        remaining -= batch_n

    # Build visit count distribution
    visits = np.zeros(action_size, dtype=np.float32)
    for child in root.children:
        visits[child.action] = child.visit_count
    return visits
