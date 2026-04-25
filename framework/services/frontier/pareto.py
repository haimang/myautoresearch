"""Generic Pareto and knee-point helpers."""

from __future__ import annotations

import math


def pareto_front(
    points: list[dict],
    *,
    maximize: list[str],
    minimize: list[str],
) -> tuple[list[dict], list[dict]]:
    front: list[dict] = []
    dominated: list[dict] = []
    for i, point in enumerate(points):
        is_dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            better_or_equal = True
            strict = False
            for key in maximize:
                if other[key] < point[key]:
                    better_or_equal = False
                    break
                if other[key] > point[key]:
                    strict = True
            if not better_or_equal:
                continue
            for key in minimize:
                if other[key] > point[key]:
                    better_or_equal = False
                    break
                if other[key] < point[key]:
                    strict = True
            if better_or_equal and strict:
                is_dominated = True
                break
        (dominated if is_dominated else front).append(point)
    return front, dominated


def compute_knee_point(
    front: list[dict],
    *,
    maximize: list[str],
    minimize: list[str],
) -> dict | None:
    if not front:
        return None
    axes = maximize + minimize
    ranges = {axis: (min(p[axis] for p in front), max(p[axis] for p in front)) for axis in axes}
    best = None
    best_dist = None
    for point in front:
        total = 0.0
        for axis in axes:
            lo, hi = ranges[axis]
            if hi == lo:
                norm = 1.0
            elif axis in maximize:
                norm = (point[axis] - lo) / (hi - lo)
            else:
                norm = (hi - point[axis]) / (hi - lo)
            total += (1.0 - norm) ** 2
        dist = math.sqrt(total / len(axes))
        if best_dist is None or dist < best_dist:
            best = point
            best_dist = dist
    return best
