"""Build and validate Pareto fronts for the Figure 5 workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .figure5_scalarization import FIGURE5_OBJECTIVE_SENSES, FIGURE5_OBJECTIVES


def is_dominated(candidate: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Return True when ``other`` Pareto-dominates ``candidate``.

    Figure 5 uses mixed objective directions: kappa/E are maximized and rho is
    minimized. Domination requires ``other`` to be no worse on every objective
    and strictly better on at least one objective.
    """
    candidate_values = _objective_values(candidate)
    other_values = _objective_values(other)
    no_worse = True
    strictly_better = False

    for objective in FIGURE5_OBJECTIVES:
        candidate_value = candidate_values[objective]
        other_value = other_values[objective]
        sense = FIGURE5_OBJECTIVE_SENSES[objective]
        if sense == "maximize":
            no_worse = no_worse and other_value >= candidate_value
            strictly_better = strictly_better or other_value > candidate_value
        elif sense == "minimize":
            no_worse = no_worse and other_value <= candidate_value
            strictly_better = strictly_better or other_value < candidate_value
        else:
            raise ValueError(f"Unsupported objective sense {sense!r} for {objective!r}")

    return bool(no_worse and strictly_better)


def pareto_front(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return unique non-dominated points while preserving first-seen order.

    Figure 5 solution records are deduplicated by composition. Generic
    objective-only points are deduplicated by their exact objective tuple.
    """
    validated_points: list[Mapping[str, Any]] = []
    seen_identities: set[tuple[str, tuple[float, ...]]] = set()
    for point in points:
        objective_values = _objective_values(point)
        identity = _point_identity(point, objective_values)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        validated_points.append(point)

    front: list[Mapping[str, Any]] = []
    for candidate_index, candidate in enumerate(validated_points):
        dominated = False
        for other_index, other in enumerate(validated_points):
            if candidate_index == other_index:
                continue
            if is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _point_identity(
    point: Mapping[str, Any],
    objective_values: Mapping[str, float],
) -> tuple[str, tuple[float, ...]]:
    if "composition" not in point:
        return "objectives", tuple(objective_values[name] for name in FIGURE5_OBJECTIVES)

    composition = point["composition"]
    if not isinstance(composition, (list, tuple)) or len(composition) != 4:
        raise ValueError("point composition must contain four finite fractions")
    try:
        fractions = tuple(float(value) for value in composition)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("point composition must contain four finite fractions") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in fractions):
        raise ValueError("point composition must contain four finite non-negative fractions")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("point composition must sum to 1")
    return "composition", tuple(round(value, 10) for value in fractions)


def _objective_values(point: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for objective in FIGURE5_OBJECTIVES:
        if objective not in point:
            raise ValueError(f"point missing objective {objective!r}")
        value = float(point[objective])
        if not math.isfinite(value):
            raise ValueError(f"objective {objective!r} must be finite")
        values[objective] = value
    return values


__all__ = ["is_dominated", "pareto_front"]
