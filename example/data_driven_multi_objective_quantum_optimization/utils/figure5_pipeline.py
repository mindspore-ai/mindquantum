"""Figure 5 多目标 FM+QUBO active-learning 主流程。.

一条 trajectory 对应固定的 `setting + seed`。`w_ddts` 每轮训练一个
DDTS 人工目标 FM；`wo_ddts` 每轮分别训练三个 objective FM，然后按
preference weights 合并 QUBO。两条流程都使用直接四相 one-hot 编码和
system penalty，不使用 CGFM。
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .alloy_dataset_generator import (
    CSV_FIELDNAMES,
    build_dataset_row,
    generate_initial_dataset_multi_objective_batch,
    sample_multi_objective_design,
)
from .experiment_runtime import (
    FIGURE5_REPLACEMENT_NAMESPACE,
    FIGURE5_SEED_INDEX,
    FM_TRAINING_BACKEND,
    MAX_RANDOM_REPLACEMENT_ATTEMPTS,
    SEED_DERIVATION_SCHEME,
    derive_bounded_seed,
    derive_fm_seed_root,
    derive_python_seed,
    fm_seed_block_size,
    fm_seed_plan,
    require_integer,
    same_json_value,
    validate_seed_schedule,
    write_json_atomic,
)
from .figure5_experiment_config import Figure5ExperimentConfig
from .figure5_outputs import (
    SUMMARY_SCHEMA_VERSION,
    checkpoint_payload,
    figure5_output_layout,
    load_checkpoint,
    write_checkpoint,
)
from .figure5_pareto import pareto_front
from .figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
    Figure5Setting,
    compute_ddts_targets,
    compute_individual_objective_targets,
    validate_preference_weights,
    validate_settings,
)
from .figure5_scalarization import (
    preference_weights_for_iteration as _preference_weights_for_iteration,
)
from .fm_torch import fit_torch_fm, fm_to_qubo
from .qubo_math import (
    IterationEncoding,
    QuboBuildResult,
    QuboStats,
    build_single_objective_qubo,
    create_iteration_encoding,
    decode_candidate_bits_to_composition,
    diagnose_no_feasible_candidate,
    encode_single_objective_rows,
    prepare_discrete_composition,
    select_lowest_energy_feasible_sample,
    solve_qubo_with_sa,
    validate_candidate_composition,
)

TRAINING_BACKEND = FM_TRAINING_BACKEND
CandidateStatus = Literal["accepted", "duplicate_replacement"]
LOGGER = logging.getLogger(__name__)
_STATE_COUNTER_FIELDS = (
    "duplicate_replacements",
    "random_replacements",
    "random_replacement_draws",
    "accepted_sa_candidates",
    "infeasible_sa_samples_skipped",
    "max_feasible_candidate_rank",
)
_QUBO_FLOAT_FIELDS = (
    "fm_objective_weight",
    "system_penalty_weight",
    "one_hot_penalty_weight",
    "fm_scale",
    "system_scale",
    "one_hot_scale",
    "max_abs",
)
_DATASET_ROW_FIELDS = frozenset(CSV_FIELDNAMES)
_DATASET_NUMERIC_FIELDS = tuple(name for name in CSV_FIELDNAMES if name not in {"sample_id", "seed"})
FIGURE5_TRAJECTORY_COUNT = len(FIGURE5_SETTINGS)
FIGURE5_BOUNDED_STREAM_COUNT = 1
FIGURE5_SA_STREAM_INDEX = 0
FIGURE5_FM_MODEL_COUNT = len(FIGURE5_OBJECTIVES)


def _trajectory_index(setting: Figure5Setting) -> int:
    try:
        return FIGURE5_SETTINGS.index(setting)
    except ValueError as exc:
        raise ValueError(f"Cannot derive Figure 5 trajectory seed for setting={setting!r}") from exc


def _validate_seed_schedule(seed_list: Sequence[int], config: Figure5ExperimentConfig) -> list[int]:
    return validate_seed_schedule(
        seed_list,
        figure_index=FIGURE5_SEED_INDEX,
        trajectory_count=FIGURE5_TRAJECTORY_COUNT,
        iterations=config.iterations,
        bounded_stream_count=FIGURE5_BOUNDED_STREAM_COUNT,
        fm_model_count=FIGURE5_FM_MODEL_COUNT,
        fm_seed_block_size=fm_seed_block_size(config.optuna_trials),
    )


@dataclass(frozen=True)
class SolutionPoint:
    """A validated alloy design stored in trajectory and summary records."""

    sample_id: int
    composition: list[float]
    kappa: float
    E: float
    rho: float


@dataclass(frozen=True)
class IterationRecord:
    """One active-learning iteration, including proposed and actually added designs."""

    setting: str
    seed: int
    iteration: int
    weights: list[float]
    scalarization_method: str
    decision_status: CandidateStatus
    proposed_solution: SolutionPoint
    added_solution: SolutionPoint
    replacement_draws: int
    sa_energy: float
    feasible_candidate_rank: int = 1
    infeasible_sa_samples_skipped: int = 0


@dataclass
class Figure5TrajectoryState:
    """Serializable in-progress state for one setting/seed trajectory."""

    rows: list[dict[str, Any]]
    iteration_records: list[IterationRecord] = field(default_factory=list)
    latest_weights: list[float] = field(default_factory=list)
    latest_fm_metadata: dict[str, Any] = field(default_factory=dict)
    latest_scalarization_metadata: dict[str, Any] = field(default_factory=dict)
    qubo_stats: QuboStats | None = None
    duplicate_replacements: int = 0
    random_replacements: int = 0
    random_replacement_draws: int = 0
    accepted_sa_candidates: int = 0
    infeasible_sa_samples_skipped: int = 0
    max_feasible_candidate_rank: int = 0

    @property
    def completed_iterations(self) -> int:
        return len(self.iteration_records)


@dataclass(frozen=True)
class Figure5TrajectoryResult:
    """Store figure5 trajectory result data."""

    setting: str
    seed: int
    iteration_records: list[IterationRecord]
    final_dataset_size: int
    training_backend: str
    latest_weights: list[float]
    latest_fm_metadata: dict[str, Any]
    latest_scalarization_metadata: dict[str, Any]
    qubo_stats: QuboStats | None
    duplicate_replacements: int
    random_replacements: int
    random_replacement_draws: int
    accepted_sa_candidates: int
    infeasible_sa_samples_skipped: int = 0
    max_feasible_candidate_rank: int = 0
    completed_iterations: int = 0


@dataclass(frozen=True)
class CandidateDecision:
    """Store candidate decision data."""

    status: CandidateStatus
    proposed_row: dict[str, Any]
    added_row: dict[str, Any]
    replacement_draws: int = 0


@dataclass(frozen=True)
class TrainingIterationResult:
    """Store training iteration result data."""

    encoding: IterationEncoding
    candidate_bits: np.ndarray
    weights: tuple[float, ...]
    scalarization_method: str
    fm_metadata: dict[str, Any]
    scalarization_metadata: dict[str, Any]
    qubo_stats: QuboStats
    sa_energy: float
    feasible_candidate_rank: int
    infeasible_sa_samples_skipped: int


@dataclass(frozen=True)
class Figure5ParetoFront:
    """Store figure5 pareto front data."""

    setting: str
    seed: int
    solutions: list[Mapping[str, Any]]


@dataclass(frozen=True)
class Figure5Summary:
    """Store figure5 summary data."""

    schema_version: int
    seed_derivation: str
    training_backend: str
    config: dict[str, Any]
    seed_list: list[int]
    objectives: list[str]
    settings: list[str]
    trajectories: list[Figure5TrajectoryResult]
    solutions: list[dict[str, Any]]
    pareto_fronts: list[Figure5ParetoFront]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_composition_from_row(row: Mapping[str, Any]) -> np.ndarray:
    """Run normalized composition from row."""
    return np.array(
        [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
        dtype=np.float64,
    )


def discretize_row_for_figure5(row: Mapping[str, Any], num_levels: int) -> dict[str, Any]:
    """Put a continuous initial design onto the direct Figure 5 fraction grid."""
    composition = prepare_discrete_composition(normalized_composition_from_row(row), num_levels)
    return dict(build_dataset_row(row["sample_id"], row["seed"], composition))


def discretize_rows_for_figure5(rows: Sequence[Mapping[str, Any]], num_levels: int) -> list[dict[str, Any]]:
    """Discretize dataset rows for the Figure 5 encoding."""
    return [discretize_row_for_figure5(row, num_levels) for row in rows]


def _sample_random_replacement_row(
    sample_id: int,
    seed: int,
    rng: random.Random,
    num_levels: int,
) -> dict[str, Any]:
    continuous_design = sample_multi_objective_design(rng)
    composition = prepare_discrete_composition(continuous_design, num_levels)
    return dict(build_dataset_row(sample_id, seed, composition))


def _composition_key(composition: Sequence[float]) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 10) for value in composition)


def _row_composition_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return _composition_key(normalized_composition_from_row(row))


def _solution_point_from_row(row: Mapping[str, Any]) -> SolutionPoint:
    return SolutionPoint(
        sample_id=require_integer("sample_id", row["sample_id"], minimum=0),
        composition=[float(value) for value in normalized_composition_from_row(row)],
        kappa=float(row["kappa"]),
        E=float(row["E"]),
        rho=float(row["rho"]),
    )


def _checkpoint_state_error(checkpoint: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid checkpoint state {checkpoint}: {detail}")


def _is_json_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected_fields: set[str],
    *,
    checkpoint: Path,
    context: str,
) -> None:
    missing_fields = sorted(expected_fields - payload.keys())
    unexpected_fields = sorted(payload.keys() - expected_fields)
    if missing_fields:
        raise _checkpoint_state_error(checkpoint, f"{context} is missing fields: {', '.join(missing_fields)}")
    if unexpected_fields:
        raise _checkpoint_state_error(
            checkpoint,
            f"{context} has unexpected fields: {', '.join(unexpected_fields)}",
        )


def _solution_point_from_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: Path,
    context: str,
) -> SolutionPoint:
    _require_exact_fields(
        payload,
        {item.name for item in dataclass_fields(SolutionPoint)},
        checkpoint=checkpoint,
        context=context,
    )
    if type(payload["sample_id"]) is not int:
        raise _checkpoint_state_error(checkpoint, f"{context}.sample_id must be an integer")
    composition = payload["composition"]
    if (
        not isinstance(composition, list)
        or len(composition) != 4
        or any(not _is_json_number(value) for value in composition)
    ):
        raise _checkpoint_state_error(checkpoint, f"{context}.composition must contain four numbers")
    for objective in FIGURE5_OBJECTIVES:
        if not _is_json_number(payload[objective]):
            raise _checkpoint_state_error(checkpoint, f"{context}.{objective} must be a number")
    return SolutionPoint(
        sample_id=payload["sample_id"],
        composition=[float(value) for value in composition],
        kappa=float(payload["kappa"]),
        E=float(payload["E"]),
        rho=float(payload["rho"]),
    )


def _iteration_record_from_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: Path,
    record_index: int,
) -> IterationRecord:
    context = f"iteration_records[{record_index}]"
    _require_exact_fields(
        payload,
        {item.name for item in dataclass_fields(IterationRecord)},
        checkpoint=checkpoint,
        context=context,
    )
    for field_name in ("setting", "scalarization_method", "decision_status"):
        if type(payload[field_name]) is not str:
            raise _checkpoint_state_error(checkpoint, f"{context}.{field_name} must be a string")
    for field_name in (
        "seed",
        "iteration",
        "replacement_draws",
        "feasible_candidate_rank",
        "infeasible_sa_samples_skipped",
    ):
        if type(payload[field_name]) is not int:
            raise _checkpoint_state_error(checkpoint, f"{context}.{field_name} must be an integer")
    weights = payload["weights"]
    if not isinstance(weights, list) or any(not _is_json_number(value) for value in weights):
        raise _checkpoint_state_error(checkpoint, f"{context}.weights must be a list of numbers")
    if not _is_json_number(payload["sa_energy"]):
        raise _checkpoint_state_error(checkpoint, f"{context}.sa_energy must be a number")
    raw_proposed_solution = payload["proposed_solution"]
    raw_added_solution = payload["added_solution"]
    if not isinstance(raw_proposed_solution, dict):
        raise _checkpoint_state_error(checkpoint, f"{context}.proposed_solution must be a JSON object")
    if not isinstance(raw_added_solution, dict):
        raise _checkpoint_state_error(checkpoint, f"{context}.added_solution must be a JSON object")
    return IterationRecord(
        setting=payload["setting"],
        seed=payload["seed"],
        iteration=payload["iteration"],
        weights=[float(value) for value in weights],
        scalarization_method=payload["scalarization_method"],
        decision_status=payload["decision_status"],
        proposed_solution=_solution_point_from_payload(
            raw_proposed_solution,
            checkpoint=checkpoint,
            context=f"{context}.proposed_solution",
        ),
        added_solution=_solution_point_from_payload(
            raw_added_solution,
            checkpoint=checkpoint,
            context=f"{context}.added_solution",
        ),
        replacement_draws=payload["replacement_draws"],
        sa_energy=float(payload["sa_energy"]),
        feasible_candidate_rank=payload["feasible_candidate_rank"],
        infeasible_sa_samples_skipped=payload["infeasible_sa_samples_skipped"],
    )


def _state_from_checkpoint(payload: Mapping[str, Any], checkpoint: Path) -> Figure5TrajectoryState:
    raw_state = payload["state"]
    _require_exact_fields(
        raw_state,
        {item.name for item in dataclass_fields(Figure5TrajectoryState)},
        checkpoint=checkpoint,
        context="state",
    )

    if not isinstance(raw_state["rows"], list) or any(not isinstance(row, dict) for row in raw_state["rows"]):
        raise _checkpoint_state_error(checkpoint, "state.rows must be a list of JSON objects")
    raw_records = raw_state["iteration_records"]
    if not isinstance(raw_records, list) or any(not isinstance(record, dict) for record in raw_records):
        raise _checkpoint_state_error(checkpoint, "state.iteration_records must be a list of JSON objects")
    latest_weights = raw_state["latest_weights"]
    if not isinstance(latest_weights, list) or any(not _is_json_number(value) for value in latest_weights):
        raise _checkpoint_state_error(checkpoint, "state.latest_weights must be a list of numbers")
    for field_name in ("latest_fm_metadata", "latest_scalarization_metadata"):
        if not isinstance(raw_state[field_name], dict):
            raise _checkpoint_state_error(checkpoint, f"state.{field_name} must be a JSON object")
    for counter_name in _STATE_COUNTER_FIELDS:
        if type(raw_state[counter_name]) is not int:
            raise _checkpoint_state_error(checkpoint, f"state.{counter_name} must be an integer")

    raw_qubo_stats = raw_state["qubo_stats"]
    if raw_qubo_stats is None:
        qubo_stats = None
    else:
        if not isinstance(raw_qubo_stats, dict):
            raise _checkpoint_state_error(checkpoint, "state.qubo_stats must be null or a JSON object")
        _require_exact_fields(
            raw_qubo_stats,
            {item.name for item in dataclass_fields(QuboStats)},
            checkpoint=checkpoint,
            context="state.qubo_stats",
        )
        for field_name in _QUBO_FLOAT_FIELDS:
            if not _is_json_number(raw_qubo_stats[field_name]):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"state.qubo_stats.{field_name} must be a finite number",
                )
        if type(raw_qubo_stats["normalization_scheme"]) is not str:
            raise _checkpoint_state_error(
                checkpoint,
                "state.qubo_stats.normalization_scheme must be a string",
            )
        if type(raw_qubo_stats["num_variables"]) is not int:
            raise _checkpoint_state_error(checkpoint, "state.qubo_stats.num_variables must be an integer")
        qubo_stats = QuboStats(
            normalization_scheme=raw_qubo_stats["normalization_scheme"],
            fm_objective_weight=float(raw_qubo_stats["fm_objective_weight"]),
            system_penalty_weight=float(raw_qubo_stats["system_penalty_weight"]),
            one_hot_penalty_weight=float(raw_qubo_stats["one_hot_penalty_weight"]),
            fm_scale=float(raw_qubo_stats["fm_scale"]),
            system_scale=float(raw_qubo_stats["system_scale"]),
            one_hot_scale=float(raw_qubo_stats["one_hot_scale"]),
            num_variables=raw_qubo_stats["num_variables"],
            max_abs=float(raw_qubo_stats["max_abs"]),
        )

    return Figure5TrajectoryState(
        rows=[dict(row) for row in raw_state["rows"]],
        iteration_records=[
            _iteration_record_from_payload(record, checkpoint=checkpoint, record_index=index)
            for index, record in enumerate(raw_records)
        ],
        latest_weights=[float(value) for value in latest_weights],
        latest_fm_metadata=dict(raw_state["latest_fm_metadata"]),
        latest_scalarization_metadata=dict(raw_state["latest_scalarization_metadata"]),
        qubo_stats=qubo_stats,
        duplicate_replacements=raw_state["duplicate_replacements"],
        random_replacements=raw_state["random_replacements"],
        random_replacement_draws=raw_state["random_replacement_draws"],
        accepted_sa_candidates=raw_state["accepted_sa_candidates"],
        infeasible_sa_samples_skipped=raw_state["infeasible_sa_samples_skipped"],
        max_feasible_candidate_rank=raw_state["max_feasible_candidate_rank"],
    )


def _validated_composition(
    composition: Sequence[float],
    *,
    checkpoint: Path,
    context: str,
) -> tuple[float, float, float, float]:
    if len(composition) != 4:
        raise _checkpoint_state_error(checkpoint, f"{context} must contain four fractions")
    fractions = tuple(float(value) for value in composition)
    if any(not math.isfinite(value) or value < 0.0 for value in fractions):
        raise _checkpoint_state_error(checkpoint, f"{context} must contain finite non-negative fractions")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise _checkpoint_state_error(checkpoint, f"{context} must sum to 1")
    return fractions[0], fractions[1], fractions[2], fractions[3]


def _solution_values_match(first: SolutionPoint, second: SolutionPoint) -> bool:
    return (
        first.sample_id == second.sample_id
        and _composition_key(first.composition) == _composition_key(second.composition)
        and all(
            math.isclose(getattr(first, objective), getattr(second, objective), rel_tol=1e-12, abs_tol=1e-12)
            for objective in FIGURE5_OBJECTIVES
        )
    )


def _solution_matches_row(
    solution: SolutionPoint,
    row: Mapping[str, Any],
    *,
    compare_sample_id: bool,
) -> bool:
    if compare_sample_id and solution.sample_id != row["sample_id"]:
        return False
    if _composition_key(solution.composition) != _row_composition_key(row):
        return False
    return all(
        math.isclose(getattr(solution, objective), float(row[objective]), rel_tol=1e-12, abs_tol=1e-12)
        for objective in FIGURE5_OBJECTIVES
    )


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and actual.keys() == expected.keys()
            and all(_metadata_matches(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(
                _metadata_matches(actual_item, expected_item) for actual_item, expected_item in zip(actual, expected)
            )
        )
    if type(expected) in {int, float}:
        return (
            _is_json_number(actual)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def _expected_latest_scalarization_metadata(
    state: Figure5TrajectoryState,
    *,
    setting: Figure5Setting,
    config: Figure5ExperimentConfig,
) -> dict[str, Any]:
    last_record = state.iteration_records[-1]
    training_row_count = config.num_samples + state.completed_iterations - 1
    training_rows = state.rows[:training_row_count]
    if setting == "w_ddts":
        return dict(compute_ddts_targets(training_rows, last_record.weights).metadata)

    individual = compute_individual_objective_targets(training_rows)
    return {
        **individual.metadata,
        "merge_weights": {
            objective: float(weight) for objective, weight in zip(FIGURE5_OBJECTIVES, last_record.weights)
        },
        "merge_stage": "qubo",
    }


def _validate_trajectory_state(
    state: Figure5TrajectoryState,
    *,
    setting: Figure5Setting,
    seed: int,
    config: Figure5ExperimentConfig,
    checkpoint: Path,
) -> None:
    """Validate all state that affects resume, replay, and Figure 5 outputs."""
    completed = state.completed_iterations
    if completed > config.iterations:
        raise _checkpoint_state_error(
            checkpoint,
            f"completed_iterations={completed} exceeds configured iterations={config.iterations}",
        )
    expected_row_count = config.num_samples + completed
    if len(state.rows) != expected_row_count:
        raise _checkpoint_state_error(
            checkpoint,
            f"rows has length {len(state.rows)}; expected num_samples + completed_iterations = {expected_row_count}",
        )

    for row_index, row in enumerate(state.rows):
        missing_fields = sorted(_DATASET_ROW_FIELDS - row.keys())
        if missing_fields:
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}] is missing fields: {', '.join(missing_fields)}",
            )
        if type(row["sample_id"]) is not int or row["sample_id"] != row_index:
            raise _checkpoint_state_error(checkpoint, f"rows[{row_index}].sample_id must equal {row_index}")
        if type(row["seed"]) is not int or row["seed"] != int(seed):
            raise _checkpoint_state_error(
                checkpoint,
                f"rows[{row_index}].seed must equal trajectory seed {int(seed)}",
            )
        for field_name in _DATASET_NUMERIC_FIELDS:
            value = row[field_name]
            if not _is_json_number(value) or not math.isfinite(float(value)):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"rows[{row_index}].{field_name} must be a finite number",
                )
        _validated_composition(
            normalized_composition_from_row(row),
            checkpoint=checkpoint,
            context=f"rows[{row_index}] composition",
        )

    accepted_count = 0
    duplicate_count = 0
    replacement_draw_sum = 0
    skipped_sum = 0
    maximum_rank = 0
    seen_rows: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
    for row in state.rows[: config.num_samples]:
        seen_rows.setdefault(_row_composition_key(row), row)

    expected_method = "ddts" if setting == "w_ddts" else "weighted_sum"
    for record_index, record in enumerate(state.iteration_records):
        context = f"iteration_records[{record_index}]"
        if record.setting != setting or record.seed != int(seed) or record.iteration != record_index:
            raise _checkpoint_state_error(
                checkpoint,
                f"{context} setting/seed/iteration does not match its trajectory position",
            )
        try:
            validate_preference_weights(
                record.weights,
                num_objectives=len(FIGURE5_OBJECTIVES),
            )
        except (TypeError, ValueError) as exc:
            raise _checkpoint_state_error(checkpoint, f"{context}.weights are invalid ({exc})") from exc
        expected_weights = _preference_weights_for_iteration(seed, record_index)
        if not np.allclose(record.weights, expected_weights, rtol=0.0, atol=1e-12):
            raise _checkpoint_state_error(
                checkpoint,
                f"{context}.weights do not match the deterministic preference weights",
            )
        if record.scalarization_method != expected_method:
            raise _checkpoint_state_error(
                checkpoint,
                f"{context}.scalarization_method must be {expected_method!r} for {setting}",
            )
        if not math.isfinite(record.sa_energy):
            raise _checkpoint_state_error(checkpoint, f"{context}.sa_energy must be finite")
        if not 1 <= record.feasible_candidate_rank <= config.sa_reads:
            raise _checkpoint_state_error(
                checkpoint,
                f"{context}.feasible_candidate_rank must be between 1 and sa_reads={config.sa_reads}",
            )
        if record.infeasible_sa_samples_skipped != record.feasible_candidate_rank - 1:
            raise _checkpoint_state_error(
                checkpoint,
                f"{context} feasible rank must equal skipped infeasible samples + 1",
            )

        expected_sample_id = config.num_samples + record_index
        for point_name, point in (
            ("proposed_solution", record.proposed_solution),
            ("added_solution", record.added_solution),
        ):
            if point.sample_id != expected_sample_id:
                raise _checkpoint_state_error(
                    checkpoint,
                    f"{context}.{point_name}.sample_id must equal {expected_sample_id}",
                )
            _validated_composition(
                point.composition,
                checkpoint=checkpoint,
                context=f"{context}.{point_name}.composition",
            )
            for objective in FIGURE5_OBJECTIVES:
                if not math.isfinite(getattr(point, objective)):
                    raise _checkpoint_state_error(
                        checkpoint,
                        f"{context}.{point_name}.{objective} must be finite",
                    )

        added_row = state.rows[expected_sample_id]
        if not _solution_matches_row(record.added_solution, added_row, compare_sample_id=True):
            raise _checkpoint_state_error(
                checkpoint,
                f"{context}.added_solution does not match rows[{expected_sample_id}]",
            )
        proposed_key = _composition_key(record.proposed_solution.composition)
        added_key = _composition_key(record.added_solution.composition)
        if record.decision_status == "accepted":
            accepted_count += 1
            if record.replacement_draws != 0:
                raise _checkpoint_state_error(checkpoint, f"{context} accepted decision must have zero draws")
            if not _solution_values_match(record.proposed_solution, record.added_solution):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"{context} accepted decision must have identical proposed and added solutions",
                )
            if proposed_key in seen_rows:
                raise _checkpoint_state_error(checkpoint, f"{context} accepted composition was already present")
        elif record.decision_status == "duplicate_replacement":
            duplicate_count += 1
            if not 1 <= record.replacement_draws <= MAX_RANDOM_REPLACEMENT_ATTEMPTS:
                raise _checkpoint_state_error(
                    checkpoint,
                    f"{context}.replacement_draws must be in [1, {MAX_RANDOM_REPLACEMENT_ATTEMPTS}]",
                )
            if proposed_key not in seen_rows:
                raise _checkpoint_state_error(checkpoint, f"{context} duplicate proposal was not previously present")
            if added_key in seen_rows or proposed_key == added_key:
                raise _checkpoint_state_error(checkpoint, f"{context} replacement composition must be novel")
            if not _solution_matches_row(
                record.proposed_solution,
                seen_rows[proposed_key],
                compare_sample_id=False,
            ):
                raise _checkpoint_state_error(
                    checkpoint,
                    f"{context}.proposed_solution does not match the prior row with that composition",
                )
        else:
            raise _checkpoint_state_error(
                checkpoint,
                f"{context}.decision_status must be 'accepted' or 'duplicate_replacement'",
            )

        replacement_draw_sum += record.replacement_draws
        skipped_sum += record.infeasible_sa_samples_skipped
        maximum_rank = max(maximum_rank, record.feasible_candidate_rank)
        seen_rows[added_key] = added_row

    derived_counters = {
        "accepted_sa_candidates": accepted_count,
        "duplicate_replacements": duplicate_count,
        "random_replacements": duplicate_count,
        "random_replacement_draws": replacement_draw_sum,
        "infeasible_sa_samples_skipped": skipped_sum,
        "max_feasible_candidate_rank": maximum_rank,
    }
    for counter_name, expected_value in derived_counters.items():
        actual_value = getattr(state, counter_name)
        if actual_value < 0 or actual_value != expected_value:
            raise _checkpoint_state_error(
                checkpoint,
                f"counter {counter_name!r}={actual_value}; expected {expected_value} from iteration records",
            )

    if completed == 0:
        if state.latest_weights or state.latest_fm_metadata or state.latest_scalarization_metadata:
            raise _checkpoint_state_error(checkpoint, "zero-iteration state must not contain training metadata")
        if state.qubo_stats is not None:
            raise _checkpoint_state_error(checkpoint, "zero-iteration state must not contain qubo_stats")
        return

    if len(state.latest_weights) != len(FIGURE5_OBJECTIVES) or not np.allclose(
        state.latest_weights,
        state.iteration_records[-1].weights,
        rtol=0.0,
        atol=1e-12,
    ):
        raise _checkpoint_state_error(checkpoint, "latest_weights must match the final iteration record")
    if setting == "w_ddts":
        valid_fm_metadata = state.latest_fm_metadata.keys() == {"artificial_target"} and isinstance(
            state.latest_fm_metadata["artificial_target"], dict
        )
    else:
        valid_fm_metadata = state.latest_fm_metadata.keys() == set(FIGURE5_OBJECTIVES) and all(
            isinstance(state.latest_fm_metadata[objective], dict) for objective in FIGURE5_OBJECTIVES
        )
    if not valid_fm_metadata:
        raise _checkpoint_state_error(checkpoint, "latest_fm_metadata is inconsistent with setting")

    trajectory_index = _trajectory_index(setting)
    block_size = fm_seed_block_size(config.optuna_trials)
    metadata_by_model = (
        [(0, state.latest_fm_metadata["artificial_target"])]
        if setting == "w_ddts"
        else [
            (model_index, state.latest_fm_metadata[objective])
            for model_index, objective in enumerate(FIGURE5_OBJECTIVES)
        ]
    )
    for model_index, model_metadata in metadata_by_model:
        expected_root = derive_fm_seed_root(
            seed,
            trajectory_count=FIGURE5_TRAJECTORY_COUNT,
            trajectory_index=trajectory_index,
            iterations=config.iterations,
            iteration=completed - 1,
            model_count=FIGURE5_FM_MODEL_COUNT,
            model_index=model_index,
            figure_index=FIGURE5_SEED_INDEX,
            block_size=block_size,
        )
        expected_seed_plan = fm_seed_plan(expected_root, config.optuna_trials)
        if "seed_plan" not in model_metadata:
            raise _checkpoint_state_error(
                checkpoint,
                f"latest_fm_metadata model {model_index} is missing field 'seed_plan'",
            )
        if not same_json_value(model_metadata["seed_plan"], expected_seed_plan):
            raise _checkpoint_state_error(
                checkpoint,
                f"latest_fm_metadata model {model_index} seed_plan does not match the derived schedule",
            )

    try:
        expected_scalarization_metadata = _expected_latest_scalarization_metadata(
            state,
            setting=setting,
            config=config,
        )
    except (TypeError, ValueError) as exc:
        raise _checkpoint_state_error(
            checkpoint,
            f"cannot derive latest scalarization metadata ({exc})",
        ) from exc
    if not _metadata_matches(state.latest_scalarization_metadata, expected_scalarization_metadata):
        raise _checkpoint_state_error(
            checkpoint,
            "latest_scalarization_metadata does not match rows, weights, and setting",
        )

    stats = state.qubo_stats
    if stats is None:
        raise _checkpoint_state_error(checkpoint, "completed state must contain qubo_stats")
    for field_name in _QUBO_FLOAT_FIELDS:
        value = float(getattr(stats, field_name))
        if not math.isfinite(value) or value < 0.0:
            raise _checkpoint_state_error(
                checkpoint,
                f"qubo_stats.{field_name} must be finite and non-negative",
            )
    if stats.normalization_scheme != config.qubo.normalization_scheme:
        raise _checkpoint_state_error(checkpoint, "qubo_stats.normalization_scheme is inconsistent")
    expected_num_variables = 4 * config.num_levels
    if stats.num_variables != expected_num_variables:
        raise _checkpoint_state_error(
            checkpoint,
            f"qubo_stats.num_variables={stats.num_variables}; expected {expected_num_variables}",
        )
    expected_weights = {
        "fm": config.qubo.fm_objective_weight,
        "system": config.qubo.system_penalty_weight,
        "one_hot": config.qubo.one_hot_penalty_weight,
    }
    weight_fields = {
        "fm": "fm_objective_weight",
        "system": "system_penalty_weight",
        "one_hot": "one_hot_penalty_weight",
    }
    for term_name, expected_weight in expected_weights.items():
        actual_weight = getattr(stats, weight_fields[term_name])
        if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise _checkpoint_state_error(
                checkpoint,
                f"qubo_stats.{term_name} weight is inconsistent with config",
            )
        scale = getattr(stats, f"{term_name}_scale")
        valid_scale = scale > 0.0 if expected_weight > 0.0 else math.isclose(scale, 0.0, rel_tol=0.0, abs_tol=1e-12)
        if not valid_scale:
            raise _checkpoint_state_error(
                checkpoint,
                f"qubo_stats.{term_name}_scale is inconsistent with its applied weight",
            )
    expected_nonzero_qubo = any(weight > 0.0 for weight in expected_weights.values())
    valid_max_abs = (
        stats.max_abs > 0.0 if expected_nonzero_qubo else math.isclose(stats.max_abs, 0.0, rel_tol=0.0, abs_tol=1e-12)
    )
    if not valid_max_abs:
        raise _checkpoint_state_error(
            checkpoint,
            "qubo_stats.max_abs is inconsistent with the applied QUBO weights",
        )


def _sample_unique_replacement_row(
    *,
    sample_id: int,
    seed: int,
    rng: random.Random,
    num_levels: int,
    seen_compositions: set[tuple[float, float, float, float]],
    max_attempts: int = MAX_RANDOM_REPLACEMENT_ATTEMPTS,
) -> tuple[dict[str, Any], int]:
    for attempt in range(1, max_attempts + 1):
        row = _sample_random_replacement_row(sample_id, seed, rng, num_levels)
        if _row_composition_key(row) not in seen_compositions:
            return row, attempt
    raise RuntimeError(f"Unable to generate a novel random replacement after {max_attempts} attempts")


def _fit_and_solve_iteration(
    *,
    rows: Sequence[dict[str, Any]],
    setting: Figure5Setting,
    config: Figure5ExperimentConfig,
    seed: int,
    iteration: int,
) -> TrainingIterationResult:
    """Run one Figure 5 preprocessing, FM, QUBO and SA iteration."""
    encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
    features = encode_single_objective_rows(rows, encoding)
    weights = _preference_weights_for_iteration(seed, iteration)
    trajectory_index = _trajectory_index(setting)
    block_size = fm_seed_block_size(config.optuna_trials)

    def fit_seed_root(model_index: int) -> int:
        return derive_fm_seed_root(
            seed,
            trajectory_count=FIGURE5_TRAJECTORY_COUNT,
            trajectory_index=trajectory_index,
            iterations=config.iterations,
            iteration=iteration,
            model_count=FIGURE5_FM_MODEL_COUNT,
            model_index=model_index,
            figure_index=FIGURE5_SEED_INDEX,
            block_size=block_size,
        )

    if setting == "w_ddts":
        scalarization = compute_ddts_targets(rows, weights)
        model, metadata = fit_torch_fm(
            features,
            scalarization.targets,
            optuna_trials=config.optuna_trials,
            device=config.device,
            seed=fit_seed_root(0),
        )
        merged_q, merged_bias = fm_to_qubo(model)
        fm_metadata: dict[str, Any] = {"artificial_target": dict(metadata)}
        scalarization_method = scalarization.method
        scalarization_metadata = dict(scalarization.metadata)
    elif setting == "wo_ddts":
        individual = compute_individual_objective_targets(rows)
        merged_q = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
        merged_bias = 0.0
        fm_metadata = {}
        for objective_index, objective in enumerate(FIGURE5_OBJECTIVES):
            model, metadata = fit_torch_fm(
                features,
                individual.targets[objective],
                optuna_trials=config.optuna_trials,
                device=config.device,
                seed=fit_seed_root(objective_index),
            )
            objective_q, objective_bias = fm_to_qubo(model)
            merged_q += float(weights[objective_index]) * objective_q
            merged_bias += float(weights[objective_index]) * objective_bias
            fm_metadata[objective] = dict(metadata)
        scalarization_method = "weighted_sum"
        scalarization_metadata = {
            **individual.metadata,
            "merge_weights": {name: float(weight) for name, weight in zip(FIGURE5_OBJECTIVES, weights)},
            "merge_stage": "qubo",
        }
    else:
        raise ValueError(f"Unsupported Figure 5 setting: {setting}")

    qubo_result: QuboBuildResult = build_single_objective_qubo(
        merged_q,
        merged_bias,
        encoding,
        config.qubo,
        include_system_penalty=True,
    )
    sampling = solve_qubo_with_sa(
        qubo_result.q,
        qubo_result.bias,
        reads=config.sa_reads,
        sweeps=config.sa_sweeps,
        seed=derive_bounded_seed(
            seed,
            trajectory_count=FIGURE5_TRAJECTORY_COUNT,
            trajectory_index=trajectory_index,
            iterations=config.iterations,
            iteration=iteration,
            stream_count=FIGURE5_BOUNDED_STREAM_COUNT,
            stream_index=FIGURE5_SA_STREAM_INDEX,
            figure_index=FIGURE5_SEED_INDEX,
        ),
    )

    def is_feasible_candidate(state: np.ndarray) -> bool:
        composition = decode_candidate_bits_to_composition(state, encoding)
        return composition is not None and validate_candidate_composition(composition)

    try:
        selected = select_lowest_energy_feasible_sample(sampling, is_feasible_candidate)
    except RuntimeError as exc:
        one_hot_valid = 0
        fully_feasible = 0
        for sample in sampling.samples:
            composition = decode_candidate_bits_to_composition(sample.state, encoding)
            if composition is None:
                continue
            one_hot_valid += 1
            if validate_candidate_composition(composition):
                fully_feasible += 1
        diagnostic = diagnose_no_feasible_candidate(
            qubo_result.q,
            qubo_result.bias,
            sampling,
            encoding,
            require_unit_sum=True,
        )
        detail = (
            "Figure 5 SA found no feasible candidate for "
            f"setting={setting!r}, seed={seed}, iteration={iteration + 1}/{config.iterations}: "
            f"one_hot_valid={one_hot_valid}/{sampling.num_samples}, "
            f"fully_feasible={fully_feasible}/{sampling.num_samples}, "
            f"num_levels={config.num_levels}, sa_reads={config.sa_reads}, "
            f"sa_sweeps={config.sa_sweeps}. {diagnostic.format_message()} "
            "The seed schedule is deterministic; unchanged --resume repeats the same SA batch."
        )
        LOGGER.error(detail)
        raise RuntimeError(detail) from exc
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=selected.state,
        weights=weights,
        scalarization_method=scalarization_method,
        fm_metadata=fm_metadata,
        scalarization_metadata=scalarization_metadata,
        qubo_stats=qubo_result.stats,
        sa_energy=float(selected.energy),
        feasible_candidate_rank=selected.rank,
        infeasible_sa_samples_skipped=selected.infeasible_samples_skipped,
    )


def _decide_candidate(
    *,
    candidate_bits: np.ndarray,
    encoding: IterationEncoding,
    sample_id: int,
    seed: int,
    replacement_rng: random.Random,
    num_levels: int,
    seen_compositions: set[tuple[float, float, float, float]],
) -> CandidateDecision:
    composition = decode_candidate_bits_to_composition(candidate_bits, encoding)
    if composition is None or not validate_candidate_composition(composition):
        raise RuntimeError("Internal error: selected SA candidate is not feasible")

    proposed_row = dict(build_dataset_row(sample_id, seed, composition))
    if _composition_key(composition) in seen_compositions:
        replacement, draws = _sample_unique_replacement_row(
            sample_id=sample_id,
            seed=seed,
            rng=replacement_rng,
            num_levels=num_levels,
            seen_compositions=seen_compositions,
        )
        return CandidateDecision("duplicate_replacement", proposed_row, replacement, draws)
    return CandidateDecision("accepted", proposed_row, proposed_row)


def _apply_candidate_decision(state: Figure5TrajectoryState, decision: CandidateDecision) -> None:
    if decision.status == "accepted":
        state.accepted_sa_candidates += 1
    elif decision.status == "duplicate_replacement":
        state.duplicate_replacements += 1
        state.random_replacements += 1
        state.random_replacement_draws += decision.replacement_draws
    else:
        raise ValueError(f"Unsupported candidate status: {decision.status}")
    state.rows.append(decision.added_row)


def _result_from_state(setting: Figure5Setting, seed: int, state: Figure5TrajectoryState) -> Figure5TrajectoryResult:
    return Figure5TrajectoryResult(
        setting=setting,
        seed=require_integer("seed", seed, minimum=0),
        iteration_records=list(state.iteration_records),
        final_dataset_size=len(state.rows),
        training_backend=TRAINING_BACKEND,
        latest_weights=list(state.latest_weights),
        latest_fm_metadata=dict(state.latest_fm_metadata),
        latest_scalarization_metadata=dict(state.latest_scalarization_metadata),
        qubo_stats=state.qubo_stats,
        duplicate_replacements=state.duplicate_replacements,
        random_replacements=state.random_replacements,
        random_replacement_draws=state.random_replacement_draws,
        accepted_sa_candidates=state.accepted_sa_candidates,
        infeasible_sa_samples_skipped=state.infeasible_sa_samples_skipped,
        max_feasible_candidate_rank=state.max_feasible_candidate_rank,
        completed_iterations=state.completed_iterations,
    )


def run_single_trajectory(
    initial_rows: Sequence[dict[str, Any]],
    setting: Figure5Setting,
    seed: int,
    config: Figure5ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> Figure5TrajectoryResult:
    """Internal runner/test orchestration for one fixed Figure 5 trajectory."""
    normalized_seed = _validate_seed_schedule([seed], config)[0]
    selected_setting = validate_settings([setting])[0]
    trajectory_index = _trajectory_index(selected_setting)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        loaded = load_checkpoint(checkpoint, setting=selected_setting, seed=normalized_seed, config=config)
        state = _state_from_checkpoint(loaded, checkpoint)
        _validate_trajectory_state(
            state,
            setting=selected_setting,
            seed=normalized_seed,
            config=config,
            checkpoint=checkpoint,
        )
        if state.completed_iterations == config.iterations:
            LOGGER.info(
                "Skipping completed Figure 5 trajectory setting=%s seed=%s completed=%s",
                selected_setting,
                seed,
                state.completed_iterations,
            )
            return _result_from_state(selected_setting, normalized_seed, state)
    else:
        state = Figure5TrajectoryState(rows=discretize_rows_for_figure5(initial_rows, config.num_levels))

    seen_compositions = {_row_composition_key(row) for row in state.rows}
    LOGGER.info(
        "Starting Figure 5 trajectory setting=%s seed=%s start=%s target=%s rows=%s",
        selected_setting,
        seed,
        state.completed_iterations,
        config.iterations,
        len(state.rows),
    )

    for iteration in range(state.completed_iterations, config.iterations):
        replacement_rng = random.Random(
            derive_python_seed(
                FIGURE5_REPLACEMENT_NAMESPACE,
                normalized_seed,
                trajectory_index=trajectory_index,
                iteration=iteration,
            )
        )
        training = _fit_and_solve_iteration(
            rows=state.rows,
            setting=selected_setting,
            config=config,
            seed=normalized_seed,
            iteration=iteration,
        )
        state.latest_weights = [float(value) for value in training.weights]
        state.latest_fm_metadata = dict(training.fm_metadata)
        state.latest_scalarization_metadata = dict(training.scalarization_metadata)
        state.qubo_stats = training.qubo_stats
        state.infeasible_sa_samples_skipped += training.infeasible_sa_samples_skipped
        state.max_feasible_candidate_rank = max(
            state.max_feasible_candidate_rank,
            training.feasible_candidate_rank,
        )

        decision = _decide_candidate(
            candidate_bits=training.candidate_bits,
            encoding=training.encoding,
            sample_id=len(state.rows),
            seed=normalized_seed,
            replacement_rng=replacement_rng,
            num_levels=config.num_levels,
            seen_compositions=seen_compositions,
        )
        _apply_candidate_decision(state, decision)
        seen_compositions.add(_row_composition_key(decision.added_row))
        state.iteration_records.append(
            IterationRecord(
                setting=selected_setting,
                seed=normalized_seed,
                iteration=iteration,
                weights=list(state.latest_weights),
                scalarization_method=training.scalarization_method,
                decision_status=decision.status,
                proposed_solution=_solution_point_from_row(decision.proposed_row),
                added_solution=_solution_point_from_row(decision.added_row),
                replacement_draws=decision.replacement_draws,
                sa_energy=training.sa_energy,
                feasible_candidate_rank=training.feasible_candidate_rank,
                infeasible_sa_samples_skipped=training.infeasible_sa_samples_skipped,
            )
        )

        LOGGER.info(
            "Completed Figure 5 iteration setting=%s seed=%s iteration=%s/%s decision=%s rows=%s",
            selected_setting,
            seed,
            iteration + 1,
            config.iterations,
            decision.status,
            len(state.rows),
        )
        if checkpoint is not None:
            write_checkpoint(
                checkpoint,
                checkpoint_payload(
                    setting=selected_setting,
                    seed=normalized_seed,
                    state=state,
                    config=config,
                ),
            )

    return _result_from_state(selected_setting, normalized_seed, state)


def _flatten_proposed_solutions(
    trajectories: Sequence[Figure5TrajectoryResult],
) -> list[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for record in trajectory.iteration_records:
            solutions.append(
                {
                    "setting": record.setting,
                    "seed": record.seed,
                    "iteration": record.iteration,
                    "status": record.decision_status,
                    "weights": list(record.weights),
                    **asdict(record.proposed_solution),
                }
            )
    return solutions


def run_figure5_experiment(
    seed_list: Sequence[int],
    config: Figure5ExperimentConfig,
    output_dir: str | Path,
    resume: bool = False,
    settings: Sequence[Figure5Setting] = FIGURE5_SETTINGS,
) -> Figure5Summary:
    """Internal runner/test orchestration that writes the current Figure 5 summary."""
    selected_settings = validate_settings(settings)
    selected_seeds = _validate_seed_schedule(seed_list, config)

    output_layout = figure5_output_layout(output_dir)
    output_layout.root.mkdir(parents=True, exist_ok=True)
    dataset_batch = generate_initial_dataset_multi_objective_batch(
        seed_list=selected_seeds,
        num_samples=config.num_samples,
    )
    trajectories: list[Figure5TrajectoryResult] = []
    for seed in selected_seeds:
        initial_rows, _ = dataset_batch[int(seed)]
        for setting in selected_settings:
            trajectories.append(
                run_single_trajectory(
                    initial_rows=initial_rows,
                    setting=setting,
                    seed=int(seed),
                    config=config,
                    checkpoint_path=output_layout.checkpoint_path(setting, int(seed)),
                    resume=resume,
                )
            )

    solutions = _flatten_proposed_solutions(trajectories)
    fronts = [
        Figure5ParetoFront(
            setting=trajectory.setting,
            seed=trajectory.seed,
            solutions=pareto_front(
                [
                    solution
                    for solution in solutions
                    if solution["setting"] == trajectory.setting and solution["seed"] == trajectory.seed
                ]
            ),
        )
        for trajectory in trajectories
    ]

    summary = Figure5Summary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        seed_derivation=SEED_DERIVATION_SCHEME,
        training_backend=TRAINING_BACKEND,
        config=config.to_dict(),
        seed_list=selected_seeds,
        objectives=list(FIGURE5_OBJECTIVES),
        settings=list(selected_settings),
        trajectories=trajectories,
        solutions=solutions,
        pareto_fronts=fronts,
    )
    write_json_atomic(output_layout.summary_path, summary.to_dict())
    LOGGER.info("Wrote Figure 5 summary path=%s", output_layout.summary_path)
    return summary


__all__ = [
    "CandidateDecision",
    "Figure5ParetoFront",
    "Figure5Summary",
    "Figure5TrajectoryResult",
    "Figure5TrajectoryState",
    "IterationRecord",
    "SolutionPoint",
    "discretize_row_for_figure5",
    "discretize_rows_for_figure5",
]
