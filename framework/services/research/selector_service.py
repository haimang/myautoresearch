#!/usr/bin/env python3
"""autoresearch selector engine — compatibility coordinator for selector services."""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))
if _FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _FRAMEWORK_DIR)

from .selector_candidates import (  # noqa: E402,F401
    generate_branch_candidates,
    generate_generic_point_candidates,
    generate_point_candidates,
    generic_utility as _generic_utility,
    mean as _mean,
    objective_profile_for_campaign as _objective_profile_for_campaign,
    stable_json as _stable_json,
)
from .selector_recommendations import (  # noqa: E402,F401
    build_recommendation_id,
    load_branch_policy as _load_branch_policy,
    recommend_for_campaign,
)
from .selector_scoring import is_dominated as _is_dominated, score_candidate as _score_candidate  # noqa: E402,F401

__all__ = [
    "_generic_utility",
    "_is_dominated",
    "_load_branch_policy",
    "_mean",
    "_objective_profile_for_campaign",
    "_score_candidate",
    "_stable_json",
    "build_recommendation_id",
    "generate_branch_candidates",
    "generate_generic_point_candidates",
    "generate_point_candidates",
    "recommend_for_campaign",
]
