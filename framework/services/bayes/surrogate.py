"""Lightweight surrogate helpers for Bayesian-style refinement."""

from __future__ import annotations

import numpy as np


def fit_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_models: int = 24,
    ridge: float = 1e-3,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    models: list[np.ndarray] = []
    n = X.shape[0]
    X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
    for _ in range(n_models):
        idx = rng.integers(0, n, size=max(n, 2))
        xb = X_aug[idx]
        yb = y[idx]
        reg = np.eye(xb.shape[1]) * ridge
        reg[0, 0] = 0.0
        beta = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ yb
        models.append(beta)
    return models


def predict_ensemble(models: list[np.ndarray], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    preds = np.stack([X_aug @ beta for beta in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)
