"""Strict post-run validation for the project's Figure 4/5 workflows.

The validator checks only evidence already stored in a summary. It never
re-trains an FM, rebuilds a QUBO, or runs simulated annealing. Paper-facing
trend and Pareto-quality measurements are diagnostics and do not decide pass
or fail.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .alloy_dataset_generator import build_dataset_row, generate_initial_dataset_multi_objective
from .experiment_config import QUBO_NORMALIZATION_SCHEME
from .experiment_runtime import (
    FIGURE4_SEED_INDEX,
    FIGURE5_SEED_INDEX,
    FM_FACTORIZATION_RANK,
    FM_TRAINING_BACKEND,
    MAX_RANDOM_REPLACEMENT_ATTEMPTS,
    NEAL_SEED_MAX,
    SEED_DERIVATION_SCHEME,
    derive_fm_seed_root,
    fm_seed_block_size,
    fm_seed_plan,
    load_json_object,
    require_integer,
    same_json_value,
    write_json_atomic,
)
from .figure4_experiment_config import (
    FIGURE4_OBJECTIVE_NAMES,
    FIGURE4_OBJECTIVES,
    FIGURE4_PRESET_NUM_SEEDS,
    FIGURE4_SETTINGS,
)
from .figure4_experiment_config import (
    preset_config as figure4_preset_config,
)
from .figure4_outputs import (
    SUMMARY_SCHEMA_VERSION as FIGURE4_SCHEMA_VERSION,
)
from .figure4_outputs import (
    figure4_output_layout,
)
from .figure5_experiment_config import FIGURE5_PRESET_NUM_SEEDS
from .figure5_experiment_config import preset_config as figure5_preset_config
from .figure5_outputs import (
    SUMMARY_SCHEMA_VERSION as FIGURE5_SCHEMA_VERSION,
)
from .figure5_outputs import (
    figure5_output_layout,
)
from .figure5_pareto import pareto_front
from .figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
    compute_ddts_targets,
    compute_individual_objective_targets,
    preference_weights_for_iteration,
)
from .qubo_math import prepare_discrete_composition

REPORT_SCHEMA_VERSION = 3
_NUMBER_TYPES = (int, float)
_CONFIG_FIELDS = frozenset({"num_samples", "iterations", "encoding", "fm", "sa", "qubo"})
_ENCODING_FIELDS = frozenset({"num_levels"})
_FM_CONFIG_FIELDS = frozenset({"optuna_trials", "device"})
_SA_CONFIG_FIELDS = frozenset({"reads", "sweeps"})
_QUBO_CONFIG_FIELDS = frozenset(
    {
        "fm_objective_weight",
        "system_penalty_weight",
        "one_hot_penalty_weight",
        "normalization_scheme",
    }
)
_FM_METADATA_FIELDS = frozenset(
    {
        "tuner",
        "rank",
        "seed_plan",
        "init_std",
        "l2_reg_w",
        "l2_reg_v",
        "train_loss",
        "validation_loss",
        "test_loss",
    }
)
_QUBO_STATS_FIELDS = frozenset(
    {
        "normalization_scheme",
        "fm_objective_weight",
        "system_penalty_weight",
        "one_hot_penalty_weight",
        "fm_scale",
        "system_scale",
        "one_hot_scale",
        "num_variables",
        "max_abs",
    }
)
_F4_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "seed_derivation",
        "training_backend",
        "config",
        "seed_list",
        "objectives",
        "settings",
        "trajectories",
        "aggregated",
    }
)
_F4_TRAJECTORY_FIELDS = frozenset(
    {
        "objective",
        "setting",
        "seed",
        "best_so_far",
        "final_dataset_size",
        "training_backend",
        "fm_metadata",
        "qubo_stats",
        "duplicate_replacements",
        "random_replacements",
        "random_replacement_draws",
        "accepted_sa_candidates",
        "infeasible_sa_samples_skipped",
        "max_feasible_candidate_rank",
        "completed_iterations",
    }
)
_F4_AGGREGATE_FIELDS = frozenset(
    {
        "objective",
        "setting",
        "best_so_far_mean",
        "best_so_far_std",
        "best_so_far_min",
        "best_so_far_max",
        "num_trajectories",
    }
)
_F5_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "seed_derivation",
        "training_backend",
        "config",
        "seed_list",
        "objectives",
        "settings",
        "trajectories",
        "solutions",
        "pareto_fronts",
    }
)
_F5_TRAJECTORY_FIELDS = frozenset(
    {
        "setting",
        "seed",
        "iteration_records",
        "final_dataset_size",
        "training_backend",
        "latest_weights",
        "latest_fm_metadata",
        "latest_scalarization_metadata",
        "qubo_stats",
        "duplicate_replacements",
        "random_replacements",
        "random_replacement_draws",
        "accepted_sa_candidates",
        "infeasible_sa_samples_skipped",
        "max_feasible_candidate_rank",
        "completed_iterations",
    }
)
_F5_RECORD_FIELDS = frozenset(
    {
        "setting",
        "seed",
        "iteration",
        "weights",
        "scalarization_method",
        "decision_status",
        "proposed_solution",
        "added_solution",
        "replacement_draws",
        "sa_energy",
        "feasible_candidate_rank",
        "infeasible_sa_samples_skipped",
    }
)
_SOLUTION_POINT_FIELDS = frozenset({"sample_id", "composition", "kappa", "E", "rho"})
_TOP_SOLUTION_FIELDS = frozenset(
    {
        "setting",
        "seed",
        "iteration",
        "status",
        "weights",
        "sample_id",
        "composition",
        "kappa",
        "E",
        "rho",
    }
)
_PARETO_FRONT_FIELDS = frozenset({"setting", "seed", "solutions"})


def _json_safe(value: Any) -> Any:
    """Return a representation that can always be dumped with allow_nan=False."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _error(path: str, rule: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "path": path,
        "rule": rule,
        "expected": _json_safe(expected),
        "actual": _json_safe(actual),
    }


def _store_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    errors: Sequence[dict[str, Any]],
    *,
    evidence: Any | None = None,
) -> None:
    detail: dict[str, Any] = {"errors": list(errors)}
    if evidence is not None:
        detail["evidence"] = _json_safe(evidence)
    checks[name] = {"passed": not errors, "detail": detail}


def _expect_fields(
    value: Any,
    path: str,
    expected_fields: frozenset[str],
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if type(value) is not dict:
        errors.append(_error(path, "json_object", "object", type(value).__name__))
        return None
    actual_fields = frozenset(value)
    if actual_fields != expected_fields:
        errors.append(
            _error(path, "exact_fields", sorted(expected_fields), sorted(str(field) for field in actual_fields))
        )
    return value


def _strict_integer(
    value: Any,
    path: str,
    errors: list[dict[str, Any]],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if type(value) is not int:
        errors.append(_error(path, "native_json_integer", "non-bool integer", value))
        return None
    if minimum is not None and value < minimum:
        errors.append(_error(path, "minimum", minimum, value))
        return None
    if maximum is not None and value > maximum:
        errors.append(_error(path, "maximum", maximum, value))
        return None
    return value


def _strict_number(
    value: Any,
    path: str,
    errors: list[dict[str, Any]],
    *,
    minimum: float | None = None,
    positive: bool = False,
) -> float | None:
    if type(value) not in _NUMBER_TYPES or not math.isfinite(float(value)):
        errors.append(_error(path, "finite_json_number", "finite non-bool number", value))
        return None
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        errors.append(_error(path, "minimum", minimum, value))
        return None
    if positive and normalized <= 0.0:
        errors.append(_error(path, "positive", "> 0", value))
        return None
    return normalized


def _strict_string_list(value: Any, path: str, errors: list[dict[str, Any]]) -> list[str] | None:
    if type(value) is not list:
        errors.append(_error(path, "json_array", "array of strings", value))
        return None
    if any(type(item) is not str for item in value):
        errors.append(_error(path, "string_items", "all items are strings", value))
        return None
    return list(value)


def _strict_seed_list(value: Any, path: str, errors: list[dict[str, Any]]) -> list[int] | None:
    if type(value) is not list or not value:
        errors.append(_error(path, "nonempty_seed_array", "nonempty integer array", value))
        return None
    seeds: list[int] = []
    for index, seed in enumerate(value):
        parsed = _strict_integer(seed, f"{path}[{index}]", errors, minimum=0, maximum=NEAL_SEED_MAX)
        if parsed is not None:
            seeds.append(parsed)
    if len(seeds) != len(value):
        return None
    if len(set(seeds)) != len(seeds):
        errors.append(_error(path, "unique_seeds", "no duplicate seeds", seeds))
        return None
    return seeds


def _strict_numeric_list(
    value: Any,
    path: str,
    errors: list[dict[str, Any]],
    *,
    length: int,
) -> list[float] | None:
    if type(value) is not list or len(value) != length:
        errors.append(_error(path, "array_length", length, value))
        return None
    parsed: list[float] = []
    for index, item in enumerate(value):
        number = _strict_number(item, f"{path}[{index}]", errors)
        if number is not None:
            parsed.append(number)
    return parsed if len(parsed) == length else None


def _parse_config(value: Any, path: str, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    config = _expect_fields(value, path, _CONFIG_FIELDS, errors)
    if config is None:
        return None
    encoding = _expect_fields(config.get("encoding"), f"{path}.encoding", _ENCODING_FIELDS, errors)
    fm = _expect_fields(config.get("fm"), f"{path}.fm", _FM_CONFIG_FIELDS, errors)
    sa = _expect_fields(config.get("sa"), f"{path}.sa", _SA_CONFIG_FIELDS, errors)
    qubo = _expect_fields(config.get("qubo"), f"{path}.qubo", _QUBO_CONFIG_FIELDS, errors)
    num_samples = _strict_integer(config.get("num_samples"), f"{path}.num_samples", errors, minimum=1)
    iterations = _strict_integer(config.get("iterations"), f"{path}.iterations", errors, minimum=1)
    num_levels = (
        _strict_integer(encoding.get("num_levels"), f"{path}.encoding.num_levels", errors, minimum=2)
        if encoding is not None
        else None
    )
    optuna_trials = (
        _strict_integer(fm.get("optuna_trials"), f"{path}.fm.optuna_trials", errors, minimum=0)
        if fm is not None
        else None
    )
    device = fm.get("device") if fm is not None else None
    if type(device) is not str or device not in {"cpu", "cuda"}:
        errors.append(_error(f"{path}.fm.device", "supported_device", ["cpu", "cuda"], device))
        device = None
    reads = _strict_integer(sa.get("reads"), f"{path}.sa.reads", errors, minimum=1) if sa is not None else None
    sweeps = _strict_integer(sa.get("sweeps"), f"{path}.sa.sweeps", errors, minimum=1) if sa is not None else None
    qubo_weights = (
        {
            field_name: _strict_number(
                qubo.get(field_name),
                f"{path}.qubo.{field_name}",
                errors,
                minimum=0.0,
            )
            for field_name in (
                "fm_objective_weight",
                "system_penalty_weight",
                "one_hot_penalty_weight",
            )
        }
        if qubo is not None
        else {}
    )
    normalization_scheme = qubo.get("normalization_scheme") if qubo is not None else None
    if type(normalization_scheme) is not str or normalization_scheme != QUBO_NORMALIZATION_SCHEME:
        errors.append(
            _error(
                f"{path}.qubo.normalization_scheme",
                "supported_normalization_scheme",
                QUBO_NORMALIZATION_SCHEME,
                normalization_scheme,
            )
        )
        normalization_scheme = None
    if (
        any(
            item is None
            for item in (
                num_samples,
                iterations,
                num_levels,
                optuna_trials,
                device,
                reads,
                sweeps,
                normalization_scheme,
                *qubo_weights.values(),
            )
        )
        or len(qubo_weights) != 3
    ):
        return None
    return {
        "num_samples": num_samples,
        "iterations": iterations,
        "num_levels": num_levels,
        "optuna_trials": optuna_trials,
        "device": device,
        "reads": reads,
        "sweeps": sweeps,
        **qubo_weights,
        "normalization_scheme": normalization_scheme,
        "raw": config,
    }


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            type(actual) is dict
            and actual.keys() == expected.keys()
            and all(_metadata_matches(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(_metadata_matches(left, right) for left, right in zip(actual, expected))
        )
    if type(expected) in _NUMBER_TYPES:
        return (
            type(actual) in _NUMBER_TYPES
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_fm_metadata(
    value: Any,
    path: str,
    expected_seed_plan: Mapping[str, Any],
    optuna_trials: int,
    errors: list[dict[str, Any]],
) -> None:
    metadata = _expect_fields(value, path, _FM_METADATA_FIELDS, errors)
    if metadata is None:
        return
    expected_tuner = "optuna" if optuna_trials > 0 else "fixed"
    if metadata.get("tuner") != expected_tuner or type(metadata.get("tuner")) is not str:
        errors.append(_error(f"{path}.tuner", "tuning_method", expected_tuner, metadata.get("tuner")))
    rank = _strict_integer(metadata.get("rank"), f"{path}.rank", errors, minimum=1)
    if rank is not None and rank != FM_FACTORIZATION_RANK:
        errors.append(_error(f"{path}.rank", "fm_rank", FM_FACTORIZATION_RANK, rank))
    if not same_json_value(metadata.get("seed_plan"), dict(expected_seed_plan)):
        errors.append(
            _error(f"{path}.seed_plan", "derived_fm_seed_plan", expected_seed_plan, metadata.get("seed_plan"))
        )
    _strict_number(metadata.get("init_std"), f"{path}.init_std", errors, positive=True)
    for field in ("l2_reg_w", "l2_reg_v", "train_loss", "validation_loss", "test_loss"):
        _strict_number(metadata.get(field), f"{path}.{field}", errors, minimum=0.0)


def _validate_qubo_stats(
    value: Any,
    path: str,
    *,
    figure: int,
    setting: str,
    config: Mapping[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    stats = _expect_fields(value, path, _QUBO_STATS_FIELDS, errors)
    if stats is None:
        return
    parsed: dict[str, float] = {}
    for field in (
        "fm_objective_weight",
        "system_penalty_weight",
        "one_hot_penalty_weight",
        "fm_scale",
        "system_scale",
        "one_hot_scale",
        "max_abs",
    ):
        number = _strict_number(stats.get(field), f"{path}.{field}", errors, minimum=0.0)
        if number is not None:
            parsed[field] = number
    num_variables = _strict_integer(stats.get("num_variables"), f"{path}.num_variables", errors, minimum=1)
    scheme = stats.get("normalization_scheme")
    if type(scheme) is not str or scheme != config["normalization_scheme"]:
        errors.append(
            _error(
                f"{path}.normalization_scheme",
                "configured_normalization_scheme",
                config["normalization_scheme"],
                scheme,
            )
        )
    expected_variables = (4 if figure == 5 or setting == "wo_cgfm" else 3) * config["num_levels"]
    if num_variables is not None and num_variables != expected_variables:
        errors.append(_error(f"{path}.num_variables", "encoding_size", expected_variables, num_variables))
    expected_weights = {
        "fm": config["fm_objective_weight"],
        "system": (config["system_penalty_weight"] if figure == 5 or setting == "wo_cgfm" else 0.0),
        "one_hot": config["one_hot_penalty_weight"],
    }
    weight_fields = {
        "fm": "fm_objective_weight",
        "system": "system_penalty_weight",
        "one_hot": "one_hot_penalty_weight",
    }
    for term_name, expected_weight in expected_weights.items():
        weight_field = weight_fields[term_name]
        if weight_field in parsed and not math.isclose(
            parsed[weight_field], expected_weight, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(
                _error(
                    f"{path}.{weight_field}",
                    "configured_applied_weight",
                    expected_weight,
                    parsed[weight_field],
                )
            )
        scale_field = f"{term_name}_scale"
        if scale_field in parsed:
            valid_scale = (
                parsed[scale_field] > 0.0
                if expected_weight > 0.0
                else math.isclose(parsed[scale_field], 0.0, rel_tol=0.0, abs_tol=1e-12)
            )
            if not valid_scale:
                errors.append(
                    _error(
                        f"{path}.{scale_field}",
                        "applied_weight_normalization_scale",
                        "> 0" if expected_weight > 0.0 else 0.0,
                        parsed[scale_field],
                    )
                )
    if "max_abs" in parsed:
        expected_nonzero_qubo = any(weight > 0.0 for weight in expected_weights.values())
        valid_max_abs = (
            parsed["max_abs"] > 0.0
            if expected_nonzero_qubo
            else math.isclose(parsed["max_abs"], 0.0, rel_tol=0.0, abs_tol=1e-12)
        )
        if not valid_max_abs:
            errors.append(
                _error(
                    f"{path}.max_abs",
                    "applied_weight_final_qubo_scale",
                    "> 0" if expected_nonzero_qubo else 0.0,
                    parsed["max_abs"],
                )
            )


def _validate_common_counters(
    trajectory: Mapping[str, Any],
    path: str,
    *,
    completed: int,
    reads: int,
    errors: list[dict[str, Any]],
) -> dict[str, int] | None:
    counters: dict[str, int] = {}
    for field in (
        "duplicate_replacements",
        "random_replacements",
        "random_replacement_draws",
        "accepted_sa_candidates",
        "infeasible_sa_samples_skipped",
        "max_feasible_candidate_rank",
    ):
        parsed = _strict_integer(trajectory.get(field), f"{path}.{field}", errors, minimum=0)
        if parsed is not None:
            counters[field] = parsed
    if len(counters) != 6:
        return None
    if counters["accepted_sa_candidates"] + counters["duplicate_replacements"] != completed:
        errors.append(
            _error(
                path,
                "candidate_partition",
                completed,
                counters["accepted_sa_candidates"] + counters["duplicate_replacements"],
            )
        )
    if counters["random_replacements"] != counters["duplicate_replacements"]:
        errors.append(
            _error(
                f"{path}.random_replacements",
                "replacement_count",
                counters["duplicate_replacements"],
                counters["random_replacements"],
            )
        )
    minimum_draws = counters["random_replacements"]
    maximum_draws = counters["duplicate_replacements"] * MAX_RANDOM_REPLACEMENT_ATTEMPTS
    if not minimum_draws <= counters["random_replacement_draws"] <= maximum_draws:
        errors.append(
            _error(
                f"{path}.random_replacement_draws",
                "replacement_draw_range",
                [minimum_draws, maximum_draws],
                counters["random_replacement_draws"],
            )
        )
    maximum_skipped = completed * (reads - 1)
    rank = counters["max_feasible_candidate_rank"]
    skipped = counters["infeasible_sa_samples_skipped"]
    if not 1 <= rank <= reads:
        errors.append(_error(f"{path}.max_feasible_candidate_rank", "sa_rank_range", [1, reads], rank))
    if not max(0, rank - 1) <= skipped <= maximum_skipped:
        errors.append(
            _error(
                f"{path}.infeasible_sa_samples_skipped",
                "sa_skipped_range",
                [max(0, rank - 1), maximum_skipped],
                skipped,
            )
        )
    return counters


def _quick_config_error(config: dict[str, Any] | None, *, figure: int) -> list[dict[str, Any]]:
    if config is None:
        return [_error("$.config", "canonical_quick_config", "valid config", None)]
    expected = (
        figure4_preset_config("quick", device=config["device"]).to_dict()
        if figure == 4
        else figure5_preset_config("quick", device=config["device"]).to_dict()
    )
    if same_json_value(config["raw"], expected):
        return []
    return [_error("$.config", "canonical_quick_config", expected, config["raw"])]


def _f4_diagnostics(final_statistics: Mapping[str, tuple[float, float]]) -> dict[str, Any]:
    main_trends: dict[str, Any] = {}
    for objective, maximize in (("kappa", True), ("E", True), ("rho", False)):
        with_cgfm = final_statistics[f"{objective}:w_cgfm"]
        without_cgfm = final_statistics[f"{objective}:wo_cgfm"]
        observed = with_cgfm[0] > without_cgfm[0] if maximize else with_cgfm[0] < without_cgfm[0]
        main_trends[objective] = {
            "w_cgfm": list(with_cgfm),
            "wo_cgfm": list(without_cgfm),
            "paper_direction_observed": observed,
        }
    similarity: dict[str, Any] = {}
    for objective in ("delta_alpha", "delta_T"):
        with_cgfm = final_statistics[f"{objective}:w_cgfm"]
        without_cgfm = final_statistics[f"{objective}:wo_cgfm"]
        spread = max(with_cgfm[1], without_cgfm[1], 1e-12)
        similarity[objective] = {
            "w_cgfm": list(with_cgfm),
            "wo_cgfm": list(without_cgfm),
            "standardized_difference": abs(with_cgfm[0] - without_cgfm[0]) / spread,
        }
    return {"cgfm_paper_direction": main_trends, "delta_metric_similarity": similarity}


def validate_figure4_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a Figure 4 summary and recompute every stored aggregate."""
    checks: dict[str, dict[str, Any]] = {}
    contract_errors: list[dict[str, Any]] = []
    root = _expect_fields(summary, "$", _F4_TOP_FIELDS, contract_errors)
    config = _parse_config(root.get("config") if root else None, "$.config", contract_errors)
    seeds = _strict_seed_list(root.get("seed_list") if root else None, "$.seed_list", contract_errors)
    if root is not None:
        if root.get("schema_version") != FIGURE4_SCHEMA_VERSION or type(root.get("schema_version")) is not int:
            contract_errors.append(
                _error("$.schema_version", "summary_schema", FIGURE4_SCHEMA_VERSION, root.get("schema_version"))
            )
        if root.get("seed_derivation") != SEED_DERIVATION_SCHEME:
            contract_errors.append(
                _error("$.seed_derivation", "seed_derivation", SEED_DERIVATION_SCHEME, root.get("seed_derivation"))
            )
        if root.get("training_backend") != FM_TRAINING_BACKEND:
            contract_errors.append(
                _error("$.training_backend", "training_backend", FM_TRAINING_BACKEND, root.get("training_backend"))
            )
    _store_check(checks, "summary_contract", contract_errors)

    axes_errors: list[dict[str, Any]] = []
    objectives = _strict_string_list(root.get("objectives") if root else None, "$.objectives", axes_errors)
    settings = _strict_string_list(root.get("settings") if root else None, "$.settings", axes_errors)
    if objectives is not None and objectives != list(FIGURE4_OBJECTIVE_NAMES):
        axes_errors.append(_error("$.objectives", "canonical_order", list(FIGURE4_OBJECTIVE_NAMES), objectives))
    if settings is not None and settings != list(FIGURE4_SETTINGS):
        axes_errors.append(_error("$.settings", "canonical_order", list(FIGURE4_SETTINGS), settings))
    _store_check(checks, "canonical_axes", axes_errors)

    quick_errors = _quick_config_error(config, figure=4)
    expected_seeds = list(range(FIGURE4_PRESET_NUM_SEEDS["quick"]))
    if seeds is not None and seeds != expected_seeds:
        quick_errors.append(_error("$.seed_list", "canonical_quick_seeds", expected_seeds, seeds))
    _store_check(checks, "quick_configuration", quick_errors)

    trajectory_errors: list[dict[str, Any]] = []
    curves: dict[str, list[list[float]]] = {
        f"{objective}:{setting}": [] for objective in FIGURE4_OBJECTIVE_NAMES for setting in FIGURE4_SETTINGS
    }
    raw_trajectories = root.get("trajectories") if root else None
    expected_identities = (
        [
            (seed, objective, setting)
            for seed in seeds
            for objective in FIGURE4_OBJECTIVE_NAMES
            for setting in FIGURE4_SETTINGS
        ]
        if seeds is not None
        else []
    )
    if type(raw_trajectories) is not list:
        trajectory_errors.append(_error("$.trajectories", "json_array", "trajectory array", raw_trajectories))
        raw_trajectories = []
    if len(raw_trajectories) != len(expected_identities):
        trajectory_errors.append(
            _error("$.trajectories", "canonical_trajectory_count", len(expected_identities), len(raw_trajectories))
        )
    objective_specs = {objective.name: objective for objective in FIGURE4_OBJECTIVES}
    if config is not None:
        block_size = fm_seed_block_size(config["optuna_trials"])
        for index, raw_trajectory in enumerate(raw_trajectories):
            path = f"$.trajectories[{index}]"
            trajectory = _expect_fields(raw_trajectory, path, _F4_TRAJECTORY_FIELDS, trajectory_errors)
            if trajectory is None:
                continue
            expected_identity = expected_identities[index] if index < len(expected_identities) else None
            identity = (trajectory.get("seed"), trajectory.get("objective"), trajectory.get("setting"))
            identity_types = tuple(type(item) for item in identity)
            if expected_identity is None or identity != expected_identity or identity_types != (int, str, str):
                trajectory_errors.append(_error(path, "canonical_trajectory_order", expected_identity, identity))
                continue
            seed, objective_name, setting = expected_identity
            completed = _strict_integer(
                trajectory.get("completed_iterations"), f"{path}.completed_iterations", trajectory_errors, minimum=0
            )
            if completed is not None and completed != config["iterations"]:
                trajectory_errors.append(
                    _error(f"{path}.completed_iterations", "configured_iterations", config["iterations"], completed)
                )
            final_size = _strict_integer(
                trajectory.get("final_dataset_size"), f"{path}.final_dataset_size", trajectory_errors, minimum=1
            )
            expected_size = config["num_samples"] + config["iterations"]
            if final_size is not None and final_size != expected_size:
                trajectory_errors.append(
                    _error(f"{path}.final_dataset_size", "dataset_growth", expected_size, final_size)
                )
            if trajectory.get("training_backend") != FM_TRAINING_BACKEND:
                trajectory_errors.append(
                    _error(
                        f"{path}.training_backend",
                        "training_backend",
                        FM_TRAINING_BACKEND,
                        trajectory.get("training_backend"),
                    )
                )
            series = _strict_numeric_list(
                trajectory.get("best_so_far"),
                f"{path}.best_so_far",
                trajectory_errors,
                length=config["iterations"],
            )
            if series is not None:
                maximize = objective_specs[objective_name].maximize
                for point_index, (previous, current) in enumerate(zip(series, series[1:]), start=1):
                    monotonic = current + 1e-12 >= previous if maximize else current <= previous + 1e-12
                    if not monotonic:
                        trajectory_errors.append(
                            _error(
                                f"{path}.best_so_far[{point_index}]",
                                "best_so_far_monotonicity",
                                "nondecreasing" if maximize else "nonincreasing",
                                [previous, current],
                            )
                        )
                        break
                curves[f"{objective_name}:{setting}"].append(series)
            _validate_common_counters(
                trajectory,
                path,
                completed=config["iterations"],
                reads=config["reads"],
                errors=trajectory_errors,
            )
            trajectory_index = FIGURE4_OBJECTIVE_NAMES.index(objective_name) * len(
                FIGURE4_SETTINGS
            ) + FIGURE4_SETTINGS.index(setting)
            root_seed = derive_fm_seed_root(
                seed,
                trajectory_count=len(FIGURE4_OBJECTIVE_NAMES) * len(FIGURE4_SETTINGS),
                trajectory_index=trajectory_index,
                iterations=config["iterations"],
                iteration=config["iterations"] - 1,
                model_count=1,
                model_index=0,
                figure_index=FIGURE4_SEED_INDEX,
                block_size=block_size,
            )
            _validate_fm_metadata(
                trajectory.get("fm_metadata"),
                f"{path}.fm_metadata",
                fm_seed_plan(root_seed, config["optuna_trials"]),
                config["optuna_trials"],
                trajectory_errors,
            )
            _validate_qubo_stats(
                trajectory.get("qubo_stats"),
                f"{path}.qubo_stats",
                figure=4,
                setting=setting,
                config=config,
                errors=trajectory_errors,
            )
    _store_check(
        checks,
        "trajectories",
        trajectory_errors,
        evidence={"actual": len(raw_trajectories), "expected": len(expected_identities)},
    )

    aggregate_errors: list[dict[str, Any]] = []
    final_statistics: dict[str, tuple[float, float]] = {}
    raw_aggregated = root.get("aggregated") if root else None
    expected_keys = set(curves)
    if type(raw_aggregated) is not dict:
        aggregate_errors.append(_error("$.aggregated", "json_object", "aggregate object", raw_aggregated))
        raw_aggregated = {}
    if set(raw_aggregated) != expected_keys:
        aggregate_errors.append(
            _error("$.aggregated", "exact_aggregate_keys", sorted(expected_keys), sorted(raw_aggregated))
        )
    if config is not None and seeds is not None:
        for key in sorted(expected_keys):
            path = f"$.aggregated.{key}"
            entry = _expect_fields(raw_aggregated.get(key), path, _F4_AGGREGATE_FIELDS, aggregate_errors)
            objective_name, setting = key.split(":", 1)
            source = curves[key]
            if entry is None:
                continue
            if entry.get("objective") != objective_name or type(entry.get("objective")) is not str:
                aggregate_errors.append(
                    _error(f"{path}.objective", "aggregate_identity", objective_name, entry.get("objective"))
                )
            if entry.get("setting") != setting or type(entry.get("setting")) is not str:
                aggregate_errors.append(_error(f"{path}.setting", "aggregate_identity", setting, entry.get("setting")))
            count = _strict_integer(
                entry.get("num_trajectories"), f"{path}.num_trajectories", aggregate_errors, minimum=1
            )
            if count is not None and count != len(seeds):
                aggregate_errors.append(_error(f"{path}.num_trajectories", "seed_count", len(seeds), count))
            stored: dict[str, list[float] | None] = {}
            for field in ("best_so_far_mean", "best_so_far_std", "best_so_far_min", "best_so_far_max"):
                stored[field] = _strict_numeric_list(
                    entry.get(field), f"{path}.{field}", aggregate_errors, length=config["iterations"]
                )
            if len(source) != len(seeds) or any(value is None for value in stored.values()):
                continue
            array = np.asarray(source, dtype=np.float64)
            expected_curves = {
                "best_so_far_mean": np.mean(array, axis=0),
                "best_so_far_std": np.std(array, axis=0),
                "best_so_far_min": np.min(array, axis=0),
                "best_so_far_max": np.max(array, axis=0),
            }
            for field, expected_curve in expected_curves.items():
                actual_curve = np.asarray(stored[field], dtype=np.float64)
                if not np.allclose(actual_curve, expected_curve, rtol=1e-12, atol=1e-12):
                    mismatch = int(np.flatnonzero(~np.isclose(actual_curve, expected_curve, rtol=1e-12, atol=1e-12))[0])
                    aggregate_errors.append(
                        _error(
                            f"{path}.{field}[{mismatch}]",
                            "recomputed_aggregate",
                            float(expected_curve[mismatch]),
                            float(actual_curve[mismatch]),
                        )
                    )
            means = np.asarray(stored["best_so_far_mean"], dtype=np.float64)
            stds = np.asarray(stored["best_so_far_std"], dtype=np.float64)
            minima = np.asarray(stored["best_so_far_min"], dtype=np.float64)
            maxima = np.asarray(stored["best_so_far_max"], dtype=np.float64)
            if np.any(stds < 0.0) or np.any(minima > means + 1e-12) or np.any(means > maxima + 1e-12):
                aggregate_errors.append(_error(path, "aggregate_bounds", "min <= mean <= max and std >= 0", "violated"))
            final_statistics[key] = (
                float(expected_curves["best_so_far_mean"][-1]),
                float(expected_curves["best_so_far_std"][-1]),
            )
    _store_check(checks, "aggregate_curves", aggregate_errors)

    integrity_checks = ("summary_contract", "canonical_axes", "trajectories", "aggregate_curves")
    diagnostics: dict[str, Any]
    if all(checks[name]["passed"] for name in integrity_checks) and len(final_statistics) == len(expected_keys):
        diagnostics = _f4_diagnostics(final_statistics)
    else:
        diagnostics = {"skipped_reason": "Figure 4 structural or numerical integrity checks failed"}
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "figure": 4,
        "scope": "quick_workflow",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "metrics": None,
        "diagnostics": diagnostics,
    }


def _composition_key(values: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(round(value, 10) for value in values)  # type: ignore[return-value]


def _strict_composition(
    value: Any,
    path: str,
    errors: list[dict[str, Any]],
    *,
    num_levels: int,
) -> tuple[float, float, float, float] | None:
    fractions = _strict_numeric_list(value, path, errors, length=4)
    if fractions is None:
        return None
    if any(fraction < 0.0 for fraction in fractions):
        errors.append(_error(path, "nonnegative_composition", "all fractions >= 0", fractions))
        return None
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-10):
        errors.append(_error(path, "composition_sum", 1.0, sum(fractions)))
        return None
    off_grid = [
        fraction
        for fraction in fractions
        if not math.isclose(fraction * num_levels, round(fraction * num_levels), rel_tol=0.0, abs_tol=1e-10)
    ]
    if off_grid:
        errors.append(_error(path, "composition_grid", f"multiples of 1/{num_levels}", fractions))
        return None
    return fractions[0], fractions[1], fractions[2], fractions[3]


def _validate_solution_point(
    value: Any,
    path: str,
    *,
    seed: int,
    sample_id: int,
    num_levels: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    point = _expect_fields(value, path, _SOLUTION_POINT_FIELDS, errors)
    if point is None:
        return None
    parsed_id = _strict_integer(point.get("sample_id"), f"{path}.sample_id", errors, minimum=0)
    if parsed_id is not None and parsed_id != sample_id:
        errors.append(_error(f"{path}.sample_id", "iteration_sample_id", sample_id, parsed_id))
    composition = _strict_composition(point.get("composition"), f"{path}.composition", errors, num_levels=num_levels)
    objectives: dict[str, float] = {}
    for objective in FIGURE5_OBJECTIVES:
        number = _strict_number(point.get(objective), f"{path}.{objective}", errors)
        if number is not None:
            objectives[objective] = number
    if parsed_id is None or composition is None or len(objectives) != len(FIGURE5_OBJECTIVES):
        return None
    expected_row = build_dataset_row(sample_id, seed, composition)
    for objective, actual in objectives.items():
        expected = float(expected_row[objective])
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(_error(f"{path}.{objective}", "material_property", expected, actual))
    return {
        "sample_id": parsed_id,
        "composition": list(point["composition"]),
        **{objective: point[objective] for objective in FIGURE5_OBJECTIVES},
    }


def _solution_matches_row(solution: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    row_composition = tuple(float(row[f"f{index}_norm"]) for index in range(1, 5))
    return _composition_key(solution["composition"]) == _composition_key(row_composition) and all(
        math.isclose(float(solution[objective]), float(row[objective]), rel_tol=1e-12, abs_tol=1e-12)
        for objective in FIGURE5_OBJECTIVES
    )


def _solutions_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["sample_id"] == second["sample_id"]
        and _composition_key(first["composition"]) == _composition_key(second["composition"])
        and all(
            math.isclose(float(first[objective]), float(second[objective]), rel_tol=1e-12, abs_tol=1e-12)
            for objective in FIGURE5_OBJECTIVES
        )
    )


def _initial_figure5_rows(seed: int, num_samples: int, num_levels: int) -> list[dict[str, Any]]:
    continuous_rows, _ = generate_initial_dataset_multi_objective(num_samples=num_samples, seed=seed)
    rows: list[dict[str, Any]] = []
    for row in continuous_rows:
        composition = prepare_discrete_composition(
            [row["f1_norm"], row["f2_norm"], row["f3_norm"], row["f4_norm"]], num_levels
        )
        rows.append(dict(build_dataset_row(row["sample_id"], seed, composition)))
    return rows


def _expected_scalarization_metadata(
    setting: str,
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    if setting == "w_ddts":
        return dict(compute_ddts_targets(rows, weights).metadata)
    individual = compute_individual_objective_targets(rows)
    return {
        **individual.metadata,
        "merge_weights": {objective: float(weight) for objective, weight in zip(FIGURE5_OBJECTIVES, weights)},
        "merge_stage": "qubo",
    }


def _validate_figure5_trajectory(
    trajectory: Mapping[str, Any],
    path: str,
    *,
    expected_seed: int,
    expected_setting: str,
    config: Mapping[str, Any],
    initial_rows: Sequence[Mapping[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    identity = (trajectory.get("setting"), trajectory.get("seed"))
    if identity != (expected_setting, expected_seed) or tuple(type(item) for item in identity) != (str, int):
        errors.append(_error(path, "canonical_trajectory_order", (expected_setting, expected_seed), identity))
        return None
    iterations = config["iterations"]
    completed = _strict_integer(
        trajectory.get("completed_iterations"), f"{path}.completed_iterations", errors, minimum=0
    )
    if completed is not None and completed != iterations:
        errors.append(_error(f"{path}.completed_iterations", "configured_iterations", iterations, completed))
    final_size = _strict_integer(trajectory.get("final_dataset_size"), f"{path}.final_dataset_size", errors, minimum=1)
    expected_size = config["num_samples"] + iterations
    if final_size is not None and final_size != expected_size:
        errors.append(_error(f"{path}.final_dataset_size", "dataset_growth", expected_size, final_size))
    if trajectory.get("training_backend") != FM_TRAINING_BACKEND:
        errors.append(
            _error(
                f"{path}.training_backend", "training_backend", FM_TRAINING_BACKEND, trajectory.get("training_backend")
            )
        )
    raw_records = trajectory.get("iteration_records")
    if type(raw_records) is not list or len(raw_records) != iterations:
        errors.append(_error(f"{path}.iteration_records", "iteration_record_count", iterations, raw_records))
        return None

    rows = [dict(row) for row in initial_rows]
    seen_rows: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
    for row in rows:
        composition = tuple(float(row[f"f{index}_norm"]) for index in range(1, 5))
        seen_rows.setdefault(_composition_key(composition), row)
    proposed_solutions: list[dict[str, Any]] = []
    derived = {
        "accepted_sa_candidates": 0,
        "duplicate_replacements": 0,
        "random_replacements": 0,
        "random_replacement_draws": 0,
        "infeasible_sa_samples_skipped": 0,
        "max_feasible_candidate_rank": 0,
    }
    last_weights: list[float] | None = None
    expected_method = "ddts" if expected_setting == "w_ddts" else "weighted_sum"
    records_valid = True
    for iteration, raw_record in enumerate(raw_records):
        record_path = f"{path}.iteration_records[{iteration}]"
        record = _expect_fields(raw_record, record_path, _F5_RECORD_FIELDS, errors)
        if record is None:
            records_valid = False
            continue
        record_identity = (record.get("setting"), record.get("seed"), record.get("iteration"))
        expected_identity = (expected_setting, expected_seed, iteration)
        if record_identity != expected_identity or tuple(type(item) for item in record_identity) != (str, int, int):
            errors.append(_error(record_path, "record_identity", expected_identity, record_identity))
            records_valid = False
        weights = _strict_numeric_list(
            record.get("weights"), f"{record_path}.weights", errors, length=len(FIGURE5_OBJECTIVES)
        )
        if weights is not None:
            if any(weight < 0.0 for weight in weights) or not math.isclose(
                sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-10
            ):
                errors.append(_error(f"{record_path}.weights", "preference_simplex", "nonnegative sum=1", weights))
                records_valid = False
            expected_weights = preference_weights_for_iteration(expected_seed, iteration)
            if not np.allclose(weights, expected_weights, rtol=0.0, atol=1e-12):
                errors.append(
                    _error(f"{record_path}.weights", "deterministic_preference_weights", expected_weights, weights)
                )
                records_valid = False
            last_weights = weights
        else:
            records_valid = False
        if record.get("scalarization_method") != expected_method or type(record.get("scalarization_method")) is not str:
            errors.append(
                _error(
                    f"{record_path}.scalarization_method",
                    "setting_scalarization",
                    expected_method,
                    record.get("scalarization_method"),
                )
            )
            records_valid = False
        status = record.get("decision_status")
        if type(status) is not str or status not in {"accepted", "duplicate_replacement"}:
            errors.append(
                _error(
                    f"{record_path}.decision_status",
                    "candidate_status",
                    ["accepted", "duplicate_replacement"],
                    status,
                )
            )
            records_valid = False
        draws = _strict_integer(record.get("replacement_draws"), f"{record_path}.replacement_draws", errors, minimum=0)
        _strict_number(record.get("sa_energy"), f"{record_path}.sa_energy", errors)
        rank = _strict_integer(
            record.get("feasible_candidate_rank"),
            f"{record_path}.feasible_candidate_rank",
            errors,
            minimum=1,
            maximum=config["reads"],
        )
        skipped = _strict_integer(
            record.get("infeasible_sa_samples_skipped"),
            f"{record_path}.infeasible_sa_samples_skipped",
            errors,
            minimum=0,
        )
        if rank is not None and skipped is not None and rank != skipped + 1:
            errors.append(_error(record_path, "sa_rank_equals_skipped_plus_one", skipped + 1, rank))
            records_valid = False
        sample_id = config["num_samples"] + iteration
        proposed = _validate_solution_point(
            record.get("proposed_solution"),
            f"{record_path}.proposed_solution",
            seed=expected_seed,
            sample_id=sample_id,
            num_levels=config["num_levels"],
            errors=errors,
        )
        added = _validate_solution_point(
            record.get("added_solution"),
            f"{record_path}.added_solution",
            seed=expected_seed,
            sample_id=sample_id,
            num_levels=config["num_levels"],
            errors=errors,
        )
        if (
            proposed is None
            or added is None
            or weights is None
            or status not in {"accepted", "duplicate_replacement"}
            or draws is None
        ):
            records_valid = False
            continue
        proposed_key = _composition_key(proposed["composition"])
        added_key = _composition_key(added["composition"])
        if status == "accepted":
            derived["accepted_sa_candidates"] += 1
            if draws != 0:
                errors.append(_error(f"{record_path}.replacement_draws", "accepted_draws", 0, draws))
                records_valid = False
            if not _solutions_equal(proposed, added):
                errors.append(_error(record_path, "accepted_proposed_equals_added", proposed, added))
                records_valid = False
            if proposed_key in seen_rows:
                errors.append(
                    _error(
                        f"{record_path}.proposed_solution.composition",
                        "accepted_is_novel",
                        "novel",
                        proposed["composition"],
                    )
                )
                records_valid = False
        else:
            derived["duplicate_replacements"] += 1
            derived["random_replacements"] += 1
            if not 1 <= draws <= MAX_RANDOM_REPLACEMENT_ATTEMPTS:
                errors.append(
                    _error(
                        f"{record_path}.replacement_draws",
                        "replacement_draw_range",
                        [1, MAX_RANDOM_REPLACEMENT_ATTEMPTS],
                        draws,
                    )
                )
                records_valid = False
            if proposed_key not in seen_rows:
                errors.append(
                    _error(
                        f"{record_path}.proposed_solution.composition",
                        "duplicate_was_seen",
                        "seen",
                        proposed["composition"],
                    )
                )
                records_valid = False
            elif not _solution_matches_row(proposed, seen_rows[proposed_key]):
                errors.append(
                    _error(
                        f"{record_path}.proposed_solution",
                        "duplicate_matches_prior_row",
                        "matching prior row",
                        proposed,
                    )
                )
                records_valid = False
            if added_key in seen_rows or added_key == proposed_key:
                errors.append(
                    _error(
                        f"{record_path}.added_solution.composition",
                        "replacement_is_novel",
                        "novel",
                        added["composition"],
                    )
                )
                records_valid = False
        derived["random_replacement_draws"] += draws
        if skipped is not None:
            derived["infeasible_sa_samples_skipped"] += skipped
        if rank is not None:
            derived["max_feasible_candidate_rank"] = max(derived["max_feasible_candidate_rank"], rank)
        added_row = dict(build_dataset_row(sample_id, expected_seed, added["composition"]))
        rows.append(added_row)
        seen_rows[added_key] = added_row
        proposed_solutions.append(
            {
                "setting": expected_setting,
                "seed": expected_seed,
                "iteration": iteration,
                "status": status,
                "weights": list(weights),
                **proposed,
            }
        )

    for field, expected in derived.items():
        actual = _strict_integer(trajectory.get(field), f"{path}.{field}", errors, minimum=0)
        if actual is not None and actual != expected:
            errors.append(_error(f"{path}.{field}", "derived_from_iteration_records", expected, actual))
            records_valid = False
    if not records_valid or len(rows) != expected_size or len(proposed_solutions) != iterations or last_weights is None:
        return None
    latest_weights = _strict_numeric_list(
        trajectory.get("latest_weights"),
        f"{path}.latest_weights",
        errors,
        length=len(FIGURE5_OBJECTIVES),
    )
    if latest_weights is None or not np.allclose(latest_weights, last_weights, rtol=0.0, atol=1e-12):
        errors.append(_error(f"{path}.latest_weights", "last_record_weights", last_weights, latest_weights))

    fm_metadata = trajectory.get("latest_fm_metadata")
    expected_model_fields = {"artificial_target"} if expected_setting == "w_ddts" else set(FIGURE5_OBJECTIVES)
    if type(fm_metadata) is not dict or set(fm_metadata) != expected_model_fields:
        errors.append(
            _error(f"{path}.latest_fm_metadata", "setting_model_fields", sorted(expected_model_fields), fm_metadata)
        )
    else:
        trajectory_index = FIGURE5_SETTINGS.index(expected_setting)
        block_size = fm_seed_block_size(config["optuna_trials"])
        models = [(0, "artificial_target")] if expected_setting == "w_ddts" else list(enumerate(FIGURE5_OBJECTIVES))
        for model_index, model_name in models:
            root_seed = derive_fm_seed_root(
                expected_seed,
                trajectory_count=len(FIGURE5_SETTINGS),
                trajectory_index=trajectory_index,
                iterations=iterations,
                iteration=iterations - 1,
                model_count=len(FIGURE5_OBJECTIVES),
                model_index=model_index,
                figure_index=FIGURE5_SEED_INDEX,
                block_size=block_size,
            )
            _validate_fm_metadata(
                fm_metadata[model_name],
                f"{path}.latest_fm_metadata.{model_name}",
                fm_seed_plan(root_seed, config["optuna_trials"]),
                config["optuna_trials"],
                errors,
            )
    expected_metadata = _expected_scalarization_metadata(expected_setting, rows[:-1], last_weights)
    if not _metadata_matches(trajectory.get("latest_scalarization_metadata"), expected_metadata):
        errors.append(
            _error(
                f"{path}.latest_scalarization_metadata",
                "recomputed_scalarization_metadata",
                expected_metadata,
                trajectory.get("latest_scalarization_metadata"),
            )
        )
    _validate_qubo_stats(
        trajectory.get("qubo_stats"),
        f"{path}.qubo_stats",
        figure=5,
        setting=expected_setting,
        config=config,
        errors=errors,
    )
    return proposed_solutions


@lru_cache(maxsize=8)
def exact_figure5_front(
    num_levels: int,
) -> dict[tuple[float, float, float, float], tuple[float, float, float]]:
    """Enumerate the direct composition grid and return its exact mixed-sense front."""
    levels = require_integer("num_levels", num_levels, minimum=2)
    rows: list[dict[str, Any]] = []
    keys: list[tuple[float, float, float, float]] = []
    for first in range(levels + 1):
        for second in range(levels - first + 1):
            for third in range(levels - first - second + 1):
                fourth = levels - first - second - third
                composition = (first / levels, second / levels, third / levels, fourth / levels)
                rows.append(dict(build_dataset_row(len(rows), 0, composition)))
                keys.append(_composition_key(composition))
    oriented = np.asarray([[row["kappa"], row["E"], -row["rho"]] for row in rows], dtype=np.float64)
    non_dominated = np.ones(len(rows), dtype=bool)
    for index, candidate in enumerate(oriented):
        dominates = np.all(oriented >= candidate, axis=1) & np.any(oriented > candidate, axis=1)
        non_dominated[index] = not bool(np.any(dominates))
    return {keys[index]: tuple(float(value) for value in oriented[index]) for index in np.flatnonzero(non_dominated)}


def _spacing_cv(oriented_points: Sequence[Sequence[float]], exact_values: np.ndarray) -> float | None:
    if len(oriented_points) < 2:
        return None
    points = np.asarray(oriented_points, dtype=np.float64)
    minimum = np.min(exact_values, axis=0)
    span = np.ptp(exact_values, axis=0)
    span = np.where(span > 0.0, span, 1.0)
    normalized = (points - minimum) / span
    distances = np.sqrt(np.sum((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    mean = float(np.mean(nearest))
    return 0.0 if mean <= 0.0 else float(np.std(nearest) / mean)


def _mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    finite = [value for value in values if value is not None]
    if not finite:
        return None, None
    array = np.asarray(finite, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _front_metrics_from_validated(
    *,
    num_levels: int,
    seeds: Sequence[int],
    settings: Sequence[str],
    solutions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    exact = exact_figure5_front(num_levels)
    exact_values = np.asarray(list(exact.values()), dtype=np.float64)
    by_setting_seed: list[dict[str, Any]] = []
    for seed in seeds:
        for setting in settings:
            unique: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
            for point in solutions:
                if point["seed"] == seed and point["setting"] == setting:
                    unique.setdefault(_composition_key(point["composition"]), point)
            exact_hits = [key for key in unique if key in exact]
            by_setting_seed.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "unique_proposals": len(unique),
                    "exact_front_hits": len(exact_hits),
                    "exact_front_coverage": len(exact_hits) / len(exact) if exact else 0.0,
                    "exact_front_precision": len(exact_hits) / len(unique) if unique else 0.0,
                    "spacing_cv": _spacing_cv([exact[key] for key in exact_hits], exact_values),
                }
            )
    aggregates: dict[str, dict[str, Any]] = {}
    for setting in settings:
        records = [record for record in by_setting_seed if record["setting"] == setting]
        coverage_mean, coverage_std = _mean_std([record["exact_front_coverage"] for record in records])
        precision_mean, precision_std = _mean_std([record["exact_front_precision"] for record in records])
        spacing_mean, spacing_std = _mean_std([record["spacing_cv"] for record in records])
        aggregates[setting] = {
            "num_seeds": len(records),
            "exact_front_coverage_mean": coverage_mean,
            "exact_front_coverage_std": coverage_std,
            "exact_front_precision_mean": precision_mean,
            "exact_front_precision_std": precision_std,
            "spacing_cv_mean": spacing_mean,
            "spacing_cv_std": spacing_std,
        }
    return {"exact_front_size": len(exact), "by_setting_seed": by_setting_seed, "setting_aggregates": aggregates}


def figure5_front_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compute per-setting/per-seed metrics only from strictly validated proposals."""
    errors: list[dict[str, Any]] = []
    if type(summary) is not dict:
        raise ValueError("Invalid Figure 5 metrics input at $: root must be a JSON object")
    config = _parse_config(summary.get("config"), "$.config", errors)
    seeds = _strict_seed_list(summary.get("seed_list"), "$.seed_list", errors)
    settings = _strict_string_list(summary.get("settings"), "$.settings", errors)
    if settings is not None:
        expected_subset = [setting for setting in FIGURE5_SETTINGS if setting in settings]
        if settings != expected_subset or len(settings) != len(set(settings)):
            errors.append(_error("$.settings", "canonical_supported_order", list(FIGURE5_SETTINGS), settings))
    raw_solutions = summary.get("solutions")
    if type(raw_solutions) is not list:
        errors.append(_error("$.solutions", "json_array", "solution array", raw_solutions))
        raw_solutions = []
    validated: list[dict[str, Any]] = []
    if config is not None and seeds is not None and settings is not None:
        for index, raw_solution in enumerate(raw_solutions):
            path = f"$.solutions[{index}]"
            solution = _expect_fields(raw_solution, path, _TOP_SOLUTION_FIELDS, errors)
            if solution is None:
                continue
            seed = _strict_integer(solution.get("seed"), f"{path}.seed", errors, minimum=0, maximum=NEAL_SEED_MAX)
            iteration = _strict_integer(solution.get("iteration"), f"{path}.iteration", errors, minimum=0)
            setting = solution.get("setting")
            status = solution.get("status")
            if type(setting) is not str or setting not in settings:
                errors.append(_error(f"{path}.setting", "selected_setting", settings, setting))
            if type(status) is not str or status not in {"accepted", "duplicate_replacement"}:
                errors.append(
                    _error(f"{path}.status", "candidate_status", ["accepted", "duplicate_replacement"], status)
                )
            weights = _strict_numeric_list(
                solution.get("weights"), f"{path}.weights", errors, length=len(FIGURE5_OBJECTIVES)
            )
            if seed is None or iteration is None or setting not in settings or weights is None:
                continue
            if seed not in seeds:
                errors.append(_error(f"{path}.seed", "selected_seed", seeds, seed))
                continue
            expected_weights = preference_weights_for_iteration(seed, iteration)
            if not np.allclose(weights, expected_weights, rtol=0.0, atol=1e-12):
                errors.append(_error(f"{path}.weights", "deterministic_preference_weights", expected_weights, weights))
            point = _validate_solution_point(
                {field: solution.get(field) for field in _SOLUTION_POINT_FIELDS},
                path,
                seed=seed,
                sample_id=config["num_samples"] + iteration,
                num_levels=config["num_levels"],
                errors=errors,
            )
            if point is not None:
                validated.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "iteration": iteration,
                        "status": status,
                        "weights": list(weights),
                        **point,
                    }
                )
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid Figure 5 metrics input at {first['path']}: {first['rule']}")
    return _front_metrics_from_validated(
        num_levels=config["num_levels"],  # type: ignore[index]
        seeds=seeds,  # type: ignore[arg-type]
        settings=settings,  # type: ignore[arg-type]
        solutions=validated,
    )


def _ddts_comparison(metrics: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    records = {(record["seed"], record["setting"]): record for record in metrics["by_setting_seed"]}
    by_seed: list[dict[str, Any]] = []
    for seed in seeds:
        ddts = records[(seed, "w_ddts")]
        weighted = records[(seed, "wo_ddts")]
        coverage_denominator = weighted["exact_front_coverage"]
        coverage_ratio = ddts["exact_front_coverage"] / coverage_denominator if coverage_denominator != 0.0 else None
        weighted_spacing = weighted["spacing_cv"]
        spacing_ratio = (
            ddts["spacing_cv"] / weighted_spacing
            if ddts["spacing_cv"] is not None and weighted_spacing not in {None, 0.0}
            else None
        )
        by_seed.append(
            {
                "seed": seed,
                "coverage_ratio_w_ddts_over_wo_ddts": coverage_ratio,
                "spacing_cv_ratio_w_ddts_over_wo_ddts": spacing_ratio,
            }
        )
    coverage_mean, coverage_std = _mean_std([record["coverage_ratio_w_ddts_over_wo_ddts"] for record in by_seed])
    spacing_mean, spacing_std = _mean_std([record["spacing_cv_ratio_w_ddts_over_wo_ddts"] for record in by_seed])
    return {
        "by_seed": by_seed,
        "aggregate": {
            "num_seeds": len(by_seed),
            "coverage_ratio_mean": coverage_mean,
            "coverage_ratio_std": coverage_std,
            "spacing_cv_ratio_mean": spacing_mean,
            "spacing_cv_ratio_std": spacing_std,
        },
    }


def validate_figure5_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly rebuild Figure 5 rows, records, proposals, fronts, and diagnostics."""
    checks: dict[str, dict[str, Any]] = {}
    contract_errors: list[dict[str, Any]] = []
    root = _expect_fields(summary, "$", _F5_TOP_FIELDS, contract_errors)
    config = _parse_config(root.get("config") if root else None, "$.config", contract_errors)
    seeds = _strict_seed_list(root.get("seed_list") if root else None, "$.seed_list", contract_errors)
    if root is not None:
        if root.get("schema_version") != FIGURE5_SCHEMA_VERSION or type(root.get("schema_version")) is not int:
            contract_errors.append(
                _error("$.schema_version", "summary_schema", FIGURE5_SCHEMA_VERSION, root.get("schema_version"))
            )
        if root.get("seed_derivation") != SEED_DERIVATION_SCHEME:
            contract_errors.append(
                _error("$.seed_derivation", "seed_derivation", SEED_DERIVATION_SCHEME, root.get("seed_derivation"))
            )
        if root.get("training_backend") != FM_TRAINING_BACKEND:
            contract_errors.append(
                _error("$.training_backend", "training_backend", FM_TRAINING_BACKEND, root.get("training_backend"))
            )
    _store_check(checks, "summary_contract", contract_errors)

    axes_errors: list[dict[str, Any]] = []
    objectives = _strict_string_list(root.get("objectives") if root else None, "$.objectives", axes_errors)
    settings = _strict_string_list(root.get("settings") if root else None, "$.settings", axes_errors)
    if objectives is not None and objectives != list(FIGURE5_OBJECTIVES):
        axes_errors.append(_error("$.objectives", "canonical_order", list(FIGURE5_OBJECTIVES), objectives))
    if settings is not None and settings != list(FIGURE5_SETTINGS):
        axes_errors.append(_error("$.settings", "canonical_order", list(FIGURE5_SETTINGS), settings))
    _store_check(checks, "canonical_axes", axes_errors)

    quick_errors = _quick_config_error(config, figure=5)
    expected_seeds = list(range(FIGURE5_PRESET_NUM_SEEDS["quick"]))
    if seeds is not None and seeds != expected_seeds:
        quick_errors.append(_error("$.seed_list", "canonical_quick_seeds", expected_seeds, seeds))
    _store_check(checks, "quick_configuration", quick_errors)

    trajectory_errors: list[dict[str, Any]] = []
    expected_solutions: list[dict[str, Any]] = []
    raw_trajectories = root.get("trajectories") if root else None
    expected_identities = (
        [(seed, setting) for seed in seeds for setting in FIGURE5_SETTINGS] if seeds is not None else []
    )
    if type(raw_trajectories) is not list:
        trajectory_errors.append(_error("$.trajectories", "json_array", "trajectory array", raw_trajectories))
        raw_trajectories = []
    if len(raw_trajectories) != len(expected_identities):
        trajectory_errors.append(
            _error("$.trajectories", "canonical_trajectory_count", len(expected_identities), len(raw_trajectories))
        )
    if config is not None and seeds is not None:
        initial_by_seed = {
            seed: _initial_figure5_rows(seed, config["num_samples"], config["num_levels"]) for seed in seeds
        }
        for index, raw_trajectory in enumerate(raw_trajectories):
            path = f"$.trajectories[{index}]"
            trajectory = _expect_fields(raw_trajectory, path, _F5_TRAJECTORY_FIELDS, trajectory_errors)
            if trajectory is None or index >= len(expected_identities):
                continue
            expected_seed, expected_setting = expected_identities[index]
            proposals = _validate_figure5_trajectory(
                trajectory,
                path,
                expected_seed=expected_seed,
                expected_setting=expected_setting,
                config=config,
                initial_rows=initial_by_seed[expected_seed],
                errors=trajectory_errors,
            )
            if proposals is not None:
                expected_solutions.extend(proposals)
    _store_check(
        checks,
        "trajectories",
        trajectory_errors,
        evidence={"actual": len(raw_trajectories), "expected": len(expected_identities)},
    )

    solution_errors: list[dict[str, Any]] = []
    raw_solutions = root.get("solutions") if root else None
    if type(raw_solutions) is not list:
        solution_errors.append(_error("$.solutions", "json_array", "solution array", raw_solutions))
        raw_solutions = []
    for index, raw_solution in enumerate(raw_solutions):
        _expect_fields(raw_solution, f"$.solutions[{index}]", _TOP_SOLUTION_FIELDS, solution_errors)
    if not same_json_value(raw_solutions, expected_solutions):
        solution_errors.append(
            _error(
                "$.solutions",
                "records_proposed_flattening",
                {"count": len(expected_solutions), "content": "exact proposed record order"},
                {"count": len(raw_solutions), "content_matches": False},
            )
        )
    _store_check(checks, "solutions", solution_errors)

    front_errors: list[dict[str, Any]] = []
    raw_fronts = root.get("pareto_fronts") if root else None
    if type(raw_fronts) is not list:
        front_errors.append(_error("$.pareto_fronts", "json_array", "front array", raw_fronts))
        raw_fronts = []
    if len(raw_fronts) != len(expected_identities):
        front_errors.append(_error("$.pareto_fronts", "front_count", len(expected_identities), len(raw_fronts)))
    for index, raw_front in enumerate(raw_fronts):
        path = f"$.pareto_fronts[{index}]"
        front = _expect_fields(raw_front, path, _PARETO_FRONT_FIELDS, front_errors)
        if front is None or index >= len(expected_identities):
            continue
        seed, setting = expected_identities[index]
        identity = (front.get("seed"), front.get("setting"))
        if identity != (seed, setting) or tuple(type(item) for item in identity) != (int, str):
            front_errors.append(_error(path, "canonical_front_order", (seed, setting), identity))
            continue
        source = [point for point in expected_solutions if point["seed"] == seed and point["setting"] == setting]
        expected_front = list(pareto_front(source))
        if not same_json_value(front.get("solutions"), expected_front):
            front_errors.append(
                _error(
                    f"{path}.solutions",
                    "recomputed_mixed_direction_pareto_front",
                    {"count": len(expected_front), "content": "exact first-seen front"},
                    front.get("solutions"),
                )
            )
    _store_check(checks, "pareto_fronts", front_errors)

    integrity_checks = ("summary_contract", "canonical_axes", "trajectories", "solutions", "pareto_fronts")
    metrics: dict[str, Any] | None = None
    diagnostics: dict[str, Any]
    if config is not None and seeds is not None and all(checks[name]["passed"] for name in integrity_checks):
        metrics = _front_metrics_from_validated(
            num_levels=config["num_levels"],
            seeds=seeds,
            settings=FIGURE5_SETTINGS,
            solutions=expected_solutions,
        )
        diagnostics = {"ddts_comparison": _ddts_comparison(metrics, seeds)}
    else:
        diagnostics = {"skipped_reason": "Figure 5 structural or numerical integrity checks failed"}
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "figure": 5,
        "scope": "quick_workflow",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _input_failure_report(figure: str, path: Path, error: ValueError) -> dict[str, Any]:
    figure_number = 4 if figure == "figure4" else 5
    input_errors = [_error(str(path), "read_valid_json_object", "readable JSON object", str(error))]
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "figure": figure_number,
        "scope": "quick_workflow",
        "passed": False,
        "checks": {"input": {"passed": False, "detail": {"errors": input_errors}}},
        "metrics": None,
        "diagnostics": {"skipped_reason": "summary input could not be parsed"},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate a Figure 4 or Figure 5 workflow summary.")
    parser.add_argument("figure", choices=("figure4", "figure5"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report path; defaults to the Figure-specific validation file beside the summary.",
    )
    return parser.parse_args(argv)


def _default_report_path(figure: str, summary_path: Path) -> Path:
    output_dir = summary_path.parent
    if figure == "figure4":
        return figure4_output_layout(output_dir).validation_report_path
    return figure5_output_layout(output_dir).validation_report_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    args = parse_args(argv)
    try:
        summary = load_json_object(args.summary, document_name="reproduction summary")
    except ValueError as error:
        report = _input_failure_report(args.figure, args.summary, error)
    else:
        report = validate_figure4_summary(summary) if args.figure == "figure4" else validate_figure5_summary(summary)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    output_path = args.output if args.output is not None else _default_report_path(args.figure, args.summary)
    write_json_atomic(output_path, report)
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "exact_figure5_front",
    "figure5_front_metrics",
    "main",
    "parse_args",
    "validate_figure4_summary",
    "validate_figure5_summary",
]
