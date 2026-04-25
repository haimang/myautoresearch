"""Generic loop helpers for Bayesian-style refinement benchmarks."""

from __future__ import annotations

import random

from .acquisition import select_bayesian


def run_strategy(
    *,
    name: str,
    universe: list[dict],
    results_cache: dict[str, dict],
    budget: int,
    seed_observations: int,
    batch_size: int,
    seed: int,
    point_from_result,
    encode_candidates,
    utility_from_result,
    feasibility_from_result,
    prior_for_candidate,
) -> tuple[list[str], list[dict]]:
    rng = random.Random(seed)
    observed: dict[str, dict] = {}
    selected: list[str] = []
    seed_batch = rng.sample(universe, min(seed_observations, budget, len(universe)))
    for candidate in seed_batch:
        key = candidate["_key"]
        observed[key] = {"candidate": candidate, "result": results_cache[key]}
        selected.append(key)

    while len(selected) < min(budget, len(universe)):
        if name == "random":
            unseen = [candidate for candidate in universe if candidate["_key"] not in observed]
            batch = rng.sample(unseen, min(batch_size, len(unseen), budget - len(selected)))
        else:
            batch = select_bayesian(
                universe=universe,
                observed_keys=set(observed),
                observations=observed,
                batch_size=min(batch_size, budget - len(selected)),
                seed=seed + len(selected),
                encode_candidates=encode_candidates,
                utility_from_result=utility_from_result,
                feasibility_from_result=feasibility_from_result,
                prior_for_candidate=prior_for_candidate,
            )
        if not batch:
            break
        for candidate in batch:
            key = candidate["_key"]
            observed[key] = {"candidate": candidate, "result": results_cache[key]}
            selected.append(key)

    points = [point_from_result(observed[key]["candidate"], observed[key]["result"], key) for key in selected]
    return selected, points
