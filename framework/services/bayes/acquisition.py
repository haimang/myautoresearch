"""Acquisition logic for generic Bayesian-style refinement."""

from __future__ import annotations

import numpy as np

from .surrogate import fit_ensemble, predict_ensemble


def select_bayesian(
    *,
    universe: list[dict],
    observed_keys: set[str],
    observations: dict[str, dict],
    batch_size: int,
    seed: int,
    encode_candidates,
    utility_from_result,
    feasibility_from_result,
    prior_for_candidate,
) -> list[dict]:
    observed_candidates = [payload["candidate"] for payload in observations.values()]
    feature_schema = encode_candidates(universe)[1]
    X_obs, _ = encode_candidates(observed_candidates, feature_schema)
    y_util = np.array(
        [utility_from_result(payload["result"]) for payload in observations.values()],
        dtype=float,
    )
    y_feas = np.array(
        [feasibility_from_result(payload["result"]) for payload in observations.values()],
        dtype=float,
    )
    util_models = fit_ensemble(X_obs, y_util, seed=seed)
    feas_models = fit_ensemble(X_obs, y_feas, seed=seed + 1)

    unseen = [candidate for candidate in universe if candidate["_key"] not in observed_keys]
    if not unseen:
        return []
    X_unseen, _ = encode_candidates(unseen, feature_schema)
    mu, sigma = predict_ensemble(util_models, X_unseen)
    p_feas, _ = predict_ensemble(feas_models, X_unseen)
    p_feas = np.clip(p_feas, 0.0, 1.0)
    priors = np.array([prior_for_candidate(candidate) for candidate in unseen], dtype=float)
    acq = p_feas * (mu + 2.75 * sigma + priors) - (1.0 - p_feas) * 24.0
    order = np.argsort(-acq)
    return [unseen[int(i)] for i in order[:batch_size]]
