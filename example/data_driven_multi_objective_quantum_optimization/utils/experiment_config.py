"""Configuration shared by the Figure 4 and Figure 5 experiment runners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from .experiment_runtime import require_integer

QUBO_NORMALIZATION_SCHEME = "max_abs_polynomial_coefficient_v1"
QuboNormalizationScheme = Literal["max_abs_polynomial_coefficient_v1"]


def _require_non_negative_finite_real(field_name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite non-negative real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative real number") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative real number")
    return normalized


@dataclass(frozen=True)
class EncodingConfig:
    """Discrete encoding configuration shared by both figures."""

    num_levels: int = 50

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        num_levels = require_integer("num_levels", self.num_levels, minimum=2)
        object.__setattr__(self, "num_levels", num_levels)


@dataclass(frozen=True)
class FMConfig:
    """Factorization-machine tuning and device configuration."""

    optuna_trials: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        object.__setattr__(
            self,
            "optuna_trials",
            require_integer("optuna_trials", self.optuna_trials, minimum=0),
        )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class SAConfig:
    """Simulated-annealing reads and sweeps configuration."""

    reads: int = 1000
    sweeps: int = 3000

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        object.__setattr__(self, "reads", require_integer("sa_reads", self.reads, minimum=1))
        object.__setattr__(self, "sweeps", require_integer("sa_sweeps", self.sweeps, minimum=1))


@dataclass(frozen=True)
class QuboConfig:
    """Weights and normalization semantics for the complete production QUBO."""

    fm_objective_weight: float = 1.0
    system_penalty_weight: float = 650.0
    one_hot_penalty_weight: float = 1.0
    normalization_scheme: QuboNormalizationScheme = QUBO_NORMALIZATION_SCHEME

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        for field_name in (
            "fm_objective_weight",
            "system_penalty_weight",
            "one_hot_penalty_weight",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_non_negative_finite_real(field_name, getattr(self, field_name)),
            )
        if type(self.normalization_scheme) is not str or self.normalization_scheme != QUBO_NORMALIZATION_SCHEME:
            raise ValueError(f"normalization_scheme must be {QUBO_NORMALIZATION_SCHEME!r}")


__all__ = [
    "EncodingConfig",
    "FMConfig",
    "QUBO_NORMALIZATION_SCHEME",
    "QuboConfig",
    "QuboNormalizationScheme",
    "SAConfig",
]
