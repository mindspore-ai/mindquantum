from __future__ import annotations

import contextlib
import copy
import io
import json
import unittest
from pathlib import Path

import numpy as np
from utils.alloy_dataset_generator import build_dataset_row, generate_initial_dataset_multi_objective
from utils.experiment_config import QUBO_NORMALIZATION_SCHEME
from utils.experiment_runtime import (
    FIGURE4_SEED_INDEX,
    FIGURE5_SEED_INDEX,
    FM_FACTORIZATION_RANK,
    FM_TRAINING_BACKEND,
    SEED_DERIVATION_SCHEME,
    derive_fm_seed_root,
    fm_seed_block_size,
    fm_seed_plan,
)
from utils.figure4_experiment_config import (
    FIGURE4_OBJECTIVE_NAMES,
    FIGURE4_OBJECTIVES,
    FIGURE4_SETTINGS,
    Figure4ExperimentConfig,
)
from utils.figure4_experiment_config import (
    preset_config as figure4_preset_config,
)
from utils.figure4_outputs import figure4_output_layout
from utils.figure5_experiment_config import Figure5ExperimentConfig
from utils.figure5_experiment_config import preset_config as figure5_preset_config
from utils.figure5_outputs import figure5_output_layout
from utils.figure5_pareto import pareto_front
from utils.figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
    compute_ddts_targets,
    compute_individual_objective_targets,
    preference_weights_for_iteration,
)
from utils.qubo_math import prepare_discrete_composition
from utils.validate_reproduction import (
    exact_figure5_front,
    figure5_front_metrics,
    main,
    parse_args,
    validate_figure4_summary,
    validate_figure5_summary,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test" / "validation_report_v3"
WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _fm_metadata(seed_plan: dict[str, object], trials: int) -> dict[str, object]:
    return {
        "tuner": "optuna" if trials else "fixed",
        "rank": FM_FACTORIZATION_RANK,
        "seed_plan": seed_plan,
        "init_std": 0.05,
        "l2_reg_w": 1e-4,
        "l2_reg_v": 1e-4,
        "train_loss": 0.1,
        "validation_loss": 0.2,
        "test_loss": 0.3,
    }


def _qubo_stats(
    *,
    setting: str,
    config: Figure4ExperimentConfig | Figure5ExperimentConfig,
    figure: int,
) -> dict[str, object]:
    include_system = figure == 5 or setting == "wo_cgfm"
    applied_system_weight = config.qubo.system_penalty_weight if include_system else 0.0
    return {
        "normalization_scheme": config.qubo.normalization_scheme,
        "fm_objective_weight": config.qubo.fm_objective_weight,
        "system_penalty_weight": applied_system_weight,
        "one_hot_penalty_weight": config.qubo.one_hot_penalty_weight,
        "fm_scale": 2.0 if config.qubo.fm_objective_weight > 0.0 else 0.0,
        "system_scale": 3.0 if applied_system_weight > 0.0 else 0.0,
        "one_hot_scale": 1.0 if config.qubo.one_hot_penalty_weight > 0.0 else 0.0,
        "num_variables": (4 if include_system else 3) * config.num_levels,
        "max_abs": (
            1.0
            if any(
                (
                    config.qubo.fm_objective_weight,
                    applied_system_weight,
                    config.qubo.one_hot_penalty_weight,
                )
            )
            else 0.0
        ),
    }


def _figure4_summary(
    config: Figure4ExperimentConfig | None = None,
    seeds: list[int] | None = None,
) -> dict[str, object]:
    config = figure4_preset_config("quick") if config is None else config
    seeds = [0, 1, 2] if seeds is None else seeds
    trajectories: list[dict[str, object]] = []
    grouped: dict[str, list[list[float]]] = {
        f"{objective}:{setting}": [] for objective in FIGURE4_OBJECTIVE_NAMES for setting in FIGURE4_SETTINGS
    }
    block_size = fm_seed_block_size(config.optuna_trials)
    for seed in seeds:
        for objective_index, objective in enumerate(FIGURE4_OBJECTIVES):
            for setting_index, setting in enumerate(FIGURE4_SETTINGS):
                offset = float(seed) * 0.1 + float(setting_index) * 0.5
                start = 10.0 + float(objective_index) + offset
                direction = 0.01 if objective.maximize else -0.01
                curve = [start + (direction * iteration) for iteration in range(config.iterations)]
                grouped[f"{objective.name}:{setting}"].append(curve)
                trajectory_index = objective_index * len(FIGURE4_SETTINGS) + setting_index
                root = derive_fm_seed_root(
                    seed,
                    trajectory_count=len(FIGURE4_OBJECTIVES) * len(FIGURE4_SETTINGS),
                    trajectory_index=trajectory_index,
                    iterations=config.iterations,
                    iteration=config.iterations - 1,
                    model_count=1,
                    model_index=0,
                    figure_index=FIGURE4_SEED_INDEX,
                    block_size=block_size,
                )
                trajectories.append(
                    {
                        "objective": objective.name,
                        "setting": setting,
                        "seed": seed,
                        "best_so_far": curve,
                        "final_dataset_size": config.num_samples + config.iterations,
                        "training_backend": FM_TRAINING_BACKEND,
                        "fm_metadata": _fm_metadata(fm_seed_plan(root, config.optuna_trials), config.optuna_trials),
                        "qubo_stats": _qubo_stats(setting=setting, config=config, figure=4),
                        "duplicate_replacements": 0,
                        "random_replacements": 0,
                        "random_replacement_draws": 0,
                        "accepted_sa_candidates": config.iterations,
                        "infeasible_sa_samples_skipped": 0,
                        "max_feasible_candidate_rank": 1,
                        "completed_iterations": config.iterations,
                    }
                )
    aggregated: dict[str, dict[str, object]] = {}
    for key, curves in grouped.items():
        objective, setting = key.split(":", 1)
        array = np.asarray(curves, dtype=np.float64)
        aggregated[key] = {
            "objective": objective,
            "setting": setting,
            "best_so_far_mean": np.mean(array, axis=0).tolist(),
            "best_so_far_std": np.std(array, axis=0).tolist(),
            "best_so_far_min": np.min(array, axis=0).tolist(),
            "best_so_far_max": np.max(array, axis=0).tolist(),
            "num_trajectories": len(seeds),
        }
    return {
        "schema_version": 5,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "training_backend": FM_TRAINING_BACKEND,
        "config": config.to_dict(),
        "seed_list": list(seeds),
        "objectives": list(FIGURE4_OBJECTIVE_NAMES),
        "settings": list(FIGURE4_SETTINGS),
        "trajectories": trajectories,
        "aggregated": aggregated,
    }


def _initial_rows(config: Figure5ExperimentConfig, seed: int) -> list[dict[str, object]]:
    raw_rows, _ = generate_initial_dataset_multi_objective(config.num_samples, seed)
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        composition = prepare_discrete_composition(
            [raw["f1_norm"], raw["f2_norm"], raw["f3_norm"], raw["f4_norm"]],
            config.num_levels,
        )
        rows.append(dict(build_dataset_row(raw["sample_id"], seed, composition)))
    return rows


def _grid_compositions(levels: int) -> list[tuple[float, float, float, float]]:
    values: list[tuple[float, float, float, float]] = []
    for first in range(levels + 1):
        for second in range(levels - first + 1):
            for third in range(levels - first - second + 1):
                fourth = levels - first - second - third
                values.append((first / levels, second / levels, third / levels, fourth / levels))
    return values


def _solution_point(sample_id: int, seed: int, composition: tuple[float, ...]) -> dict[str, object]:
    row = build_dataset_row(sample_id, seed, composition)
    return {
        "sample_id": sample_id,
        "composition": list(composition),
        "kappa": row["kappa"],
        "E": row["E"],
        "rho": row["rho"],
    }


def _figure5_summary(config: Figure5ExperimentConfig | None = None) -> dict[str, object]:
    config = figure5_preset_config("quick") if config is None else config
    seed = 0
    initial = _initial_rows(config, seed)
    seen = {tuple(round(float(row[f"f{index}_norm"]), 10) for index in range(1, 5)) for row in initial}
    candidates = [
        composition
        for composition in _grid_compositions(config.num_levels)
        if tuple(round(value, 10) for value in composition) not in seen
    ][: config.iterations]
    if len(candidates) != config.iterations:
        raise AssertionError("test fixture grid does not contain enough novel designs")

    trajectories: list[dict[str, object]] = []
    solutions: list[dict[str, object]] = []
    block_size = fm_seed_block_size(config.optuna_trials)
    for setting_index, setting in enumerate(FIGURE5_SETTINGS):
        rows = [dict(row) for row in initial]
        records: list[dict[str, object]] = []
        for iteration, composition in enumerate(candidates):
            sample_id = config.num_samples + iteration
            point = _solution_point(sample_id, seed, composition)
            weights = list(preference_weights_for_iteration(seed, iteration))
            record = {
                "setting": setting,
                "seed": seed,
                "iteration": iteration,
                "weights": weights,
                "scalarization_method": "ddts" if setting == "w_ddts" else "weighted_sum",
                "decision_status": "accepted",
                "proposed_solution": dict(point),
                "added_solution": dict(point),
                "replacement_draws": 0,
                "sa_energy": float(iteration),
                "feasible_candidate_rank": 1,
                "infeasible_sa_samples_skipped": 0,
            }
            records.append(record)
            rows.append(dict(build_dataset_row(sample_id, seed, composition)))
            solutions.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "iteration": iteration,
                    "status": "accepted",
                    "weights": weights,
                    **point,
                }
            )
        last_weights = records[-1]["weights"]
        if setting == "w_ddts":
            scalarization_metadata = dict(compute_ddts_targets(rows[:-1], last_weights).metadata)
            model_names = [(0, "artificial_target")]
        else:
            individual = compute_individual_objective_targets(rows[:-1])
            scalarization_metadata = {
                **individual.metadata,
                "merge_weights": {
                    objective: float(weight) for objective, weight in zip(FIGURE5_OBJECTIVES, last_weights)
                },
                "merge_stage": "qubo",
            }
            model_names = list(enumerate(FIGURE5_OBJECTIVES))
        latest_fm_metadata: dict[str, object] = {}
        for model_index, model_name in model_names:
            root = derive_fm_seed_root(
                seed,
                trajectory_count=len(FIGURE5_SETTINGS),
                trajectory_index=setting_index,
                iterations=config.iterations,
                iteration=config.iterations - 1,
                model_count=len(FIGURE5_OBJECTIVES),
                model_index=model_index,
                figure_index=FIGURE5_SEED_INDEX,
                block_size=block_size,
            )
            latest_fm_metadata[model_name] = _fm_metadata(
                fm_seed_plan(root, config.optuna_trials), config.optuna_trials
            )
        trajectories.append(
            {
                "setting": setting,
                "seed": seed,
                "iteration_records": records,
                "final_dataset_size": config.num_samples + config.iterations,
                "training_backend": FM_TRAINING_BACKEND,
                "latest_weights": list(last_weights),
                "latest_fm_metadata": latest_fm_metadata,
                "latest_scalarization_metadata": scalarization_metadata,
                "qubo_stats": _qubo_stats(setting=setting, config=config, figure=5),
                "duplicate_replacements": 0,
                "random_replacements": 0,
                "random_replacement_draws": 0,
                "accepted_sa_candidates": config.iterations,
                "infeasible_sa_samples_skipped": 0,
                "max_feasible_candidate_rank": 1,
                "completed_iterations": config.iterations,
            }
        )
    fronts = [
        {
            "setting": setting,
            "seed": seed,
            "solutions": pareto_front(
                [point for point in solutions if point["setting"] == setting and point["seed"] == seed]
            ),
        }
        for setting in FIGURE5_SETTINGS
    ]
    return {
        "schema_version": 5,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "training_backend": FM_TRAINING_BACKEND,
        "config": config.to_dict(),
        "seed_list": [seed],
        "objectives": list(FIGURE5_OBJECTIVES),
        "settings": list(FIGURE5_SETTINGS),
        "trajectories": trajectories,
        "solutions": solutions,
        "pareto_fronts": fronts,
    }


def _set_first_duplicate_replacement(summary: dict[str, object]) -> None:
    config = figure5_preset_config("quick")
    prior = _initial_rows(config, 0)[0]
    composition = tuple(float(prior[f"f{index}_norm"]) for index in range(1, 5))
    duplicate = _solution_point(config.num_samples, 0, composition)
    trajectory = summary["trajectories"][0]  # type: ignore[index]
    record = trajectory["iteration_records"][0]
    record["decision_status"] = "duplicate_replacement"
    record["replacement_draws"] = 1
    record["proposed_solution"] = duplicate
    trajectory["accepted_sa_candidates"] -= 1
    trajectory["duplicate_replacements"] = 1
    trajectory["random_replacements"] = 1
    trajectory["random_replacement_draws"] = 1
    summary["solutions"][0] = {  # type: ignore[index]
        "setting": "w_ddts",
        "seed": 0,
        "iteration": 0,
        "status": "duplicate_replacement",
        "weights": record["weights"],
        **duplicate,
    }
    setting_solutions = [
        point
        for point in summary["solutions"]  # type: ignore[index]
        if point["setting"] == "w_ddts" and point["seed"] == 0
    ]
    summary["pareto_fronts"][0]["solutions"] = pareto_front(setting_solutions)  # type: ignore[index]


class ReproductionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.figure4 = _figure4_summary()
        cls.figure5 = _figure5_summary()

    def test_complete_production_fixtures_pass_and_report_is_json_safe(self) -> None:
        figure4_report = validate_figure4_summary(copy.deepcopy(self.figure4))
        figure5_report = validate_figure5_summary(copy.deepcopy(self.figure5))
        self.assertEqual(figure4_report["report_schema_version"], 3)
        self.assertEqual(figure5_report["report_schema_version"], 3)
        self.assertTrue(figure4_report["passed"])
        self.assertTrue(figure5_report["passed"])
        self.assertEqual(
            self.figure4["config"]["qubo"]["normalization_scheme"],
            QUBO_NORMALIZATION_SCHEME,
        )
        self.assertIsNone(figure4_report["metrics"])
        self.assertIsNotNone(figure5_report["metrics"])
        json.dumps(figure4_report, allow_nan=False)
        json.dumps(figure5_report, allow_nan=False)

    def test_strict_integer_contract_rejects_bool_float_and_string(self) -> None:
        for invalid in (True, 100.0, "100"):
            with self.subTest(invalid=invalid):
                summary = copy.deepcopy(self.figure4)
                summary["config"]["iterations"] = invalid
                report = validate_figure4_summary(summary)
                self.assertFalse(report["passed"])
                self.assertFalse(report["checks"]["summary_contract"]["passed"])
                json.dumps(report, allow_nan=False)

    def test_exact_fields_and_canonical_order_are_enforced(self) -> None:
        for fixture, validator in (
            (self.figure4, validate_figure4_summary),
            (self.figure5, validate_figure5_summary),
        ):
            legacy = copy.deepcopy(fixture)
            legacy["schema_version"] = 4
            self.assertFalse(validator(legacy)["checks"]["summary_contract"]["passed"])

        extra = copy.deepcopy(self.figure4)
        extra["unexpected"] = 1
        self.assertFalse(validate_figure4_summary(extra)["checks"]["summary_contract"]["passed"])

        reversed_axes = copy.deepcopy(self.figure4)
        reversed_axes["settings"] = list(reversed(reversed_axes["settings"]))
        self.assertFalse(validate_figure4_summary(reversed_axes)["checks"]["canonical_axes"]["passed"])

        reversed_trajectories = copy.deepcopy(self.figure4)
        reversed_trajectories["trajectories"][0:2] = reversed(reversed_trajectories["trajectories"][0:2])
        self.assertFalse(validate_figure4_summary(reversed_trajectories)["checks"]["trajectories"]["passed"])

    def test_development_scale_keeps_diagnostics_but_fails_quick_contract(self) -> None:
        summary = _figure4_summary(config=figure4_preset_config("test"), seeds=[0, 1, 2])
        report = validate_figure4_summary(summary)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["quick_configuration"]["passed"])
        self.assertNotIn("skipped_reason", report["diagnostics"])

    def test_figure4_rejects_nonmonotonic_curves_and_forged_aggregate(self) -> None:
        nonmonotonic = copy.deepcopy(self.figure4)
        nonmonotonic["trajectories"][0]["best_so_far"][1] = -999.0
        self.assertFalse(validate_figure4_summary(nonmonotonic)["checks"]["trajectories"]["passed"])

        forged = copy.deepcopy(self.figure4)
        forged["aggregated"]["kappa:wo_cgfm"]["best_so_far_mean"][0] += 1.0
        self.assertFalse(validate_figure4_summary(forged)["checks"]["aggregate_curves"]["passed"])

    def test_figure4_rejects_fm_qubo_counter_and_rank_damage(self) -> None:
        mutations = (
            ("fm", lambda summary: summary["trajectories"][0]["fm_metadata"]["seed_plan"].__setitem__("root", 1)),
            ("qubo", lambda summary: summary["trajectories"][0]["qubo_stats"].__setitem__("num_variables", 1)),
            (
                "qubo_weight",
                lambda summary: summary["trajectories"][0]["qubo_stats"].__setitem__("fm_objective_weight", 2.0),
            ),
            (
                "qubo_scheme",
                lambda summary: summary["trajectories"][0]["qubo_stats"].__setitem__("normalization_scheme", "unknown"),
            ),
            ("counter", lambda summary: summary["trajectories"][0].__setitem__("accepted_sa_candidates", 99)),
            ("rank", lambda summary: summary["trajectories"][0].__setitem__("max_feasible_candidate_rank", 101)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                summary = copy.deepcopy(self.figure4)
                mutate(summary)
                self.assertFalse(validate_figure4_summary(summary)["checks"]["trajectories"]["passed"])

    def test_figure5_rejects_record_semantic_damage(self) -> None:
        mutations = (
            ("weight", lambda record: record["weights"].__setitem__(0, record["weights"][0] + 0.1)),
            ("method", lambda record: record.__setitem__("scalarization_method", "weighted_sum")),
            ("status", lambda record: record.__setitem__("decision_status", "unknown")),
            ("sample_id", lambda record: record["proposed_solution"].__setitem__("sample_id", 999)),
            ("property", lambda record: record["proposed_solution"].__setitem__("kappa", 999.0)),
            ("grid", lambda record: record["proposed_solution"].__setitem__("composition", [0.1, 0.2, 0.3, 0.4])),
            ("rank", lambda record: record.__setitem__("feasible_candidate_rank", 2)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                summary = copy.deepcopy(self.figure5)
                mutate(summary["trajectories"][0]["iteration_records"][0])
                report = validate_figure5_summary(summary)
                self.assertFalse(report["checks"]["trajectories"]["passed"])
                self.assertIn("skipped_reason", report["diagnostics"])

    def test_figure5_rejects_state_metadata_and_qubo_damage(self) -> None:
        mutations = (
            ("counter", lambda trajectory: trajectory.__setitem__("accepted_sa_candidates", 149)),
            ("latest_weights", lambda trajectory: trajectory["latest_weights"].__setitem__(0, 0.0)),
            (
                "scalar_metadata",
                lambda trajectory: trajectory["latest_scalarization_metadata"].__setitem__("utopian_space", "raw"),
            ),
            (
                "fm",
                lambda trajectory: trajectory["latest_fm_metadata"]["artificial_target"]["seed_plan"].__setitem__(
                    "root", 1
                ),
            ),
            ("qubo", lambda trajectory: trajectory["qubo_stats"].__setitem__("system_scale", 0.0)),
            (
                "qubo_weight",
                lambda trajectory: trajectory["qubo_stats"].__setitem__("one_hot_penalty_weight", 3.0),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                summary = copy.deepcopy(self.figure5)
                mutate(summary["trajectories"][0])
                self.assertFalse(validate_figure5_summary(summary)["checks"]["trajectories"]["passed"])

    def test_duplicate_replacement_keeps_added_design_out_of_proposal_front(self) -> None:
        summary = copy.deepcopy(self.figure5)
        _set_first_duplicate_replacement(summary)
        report = validate_figure5_summary(summary)
        self.assertTrue(report["passed"])
        proposed = summary["trajectories"][0]["iteration_records"][0]["proposed_solution"]
        added = summary["trajectories"][0]["iteration_records"][0]["added_solution"]
        self.assertEqual(summary["solutions"][0]["composition"], proposed["composition"])
        self.assertNotEqual(summary["solutions"][0]["composition"], added["composition"])

    def test_top_solutions_and_stored_front_must_match_reconstruction(self) -> None:
        stale_solution = copy.deepcopy(self.figure5)
        stale_solution["solutions"][0]["rho"] += 1.0
        self.assertFalse(validate_figure5_summary(stale_solution)["checks"]["solutions"]["passed"])

        stale_front = copy.deepcopy(self.figure5)
        stale_front["pareto_fronts"][0]["solutions"] = []
        self.assertFalse(validate_figure5_summary(stale_front)["checks"]["pareto_fronts"]["passed"])

    def test_front_metrics_are_seed_isolated_and_strict(self) -> None:
        config = figure5_preset_config("test").to_dict()
        config["num_samples"] = 1
        config["iterations"] = 2
        config["encoding"]["num_levels"] = 2

        def point(setting: str, seed: int, iteration: int, composition: tuple[float, ...]) -> dict[str, object]:
            solution = _solution_point(1 + iteration, seed, composition)
            return {
                "setting": setting,
                "seed": seed,
                "iteration": iteration,
                "status": "accepted",
                "weights": list(preference_weights_for_iteration(seed, iteration)),
                **solution,
            }

        first = (0.0, 0.0, 0.0, 1.0)
        second = (0.0, 0.0, 0.5, 0.5)
        summary = {
            "config": config,
            "seed_list": [0, 1],
            "settings": list(FIGURE5_SETTINGS),
            "solutions": [
                point("w_ddts", 0, 0, first),
                point("w_ddts", 0, 1, first),
                point("w_ddts", 1, 0, second),
                point("wo_ddts", 1, 0, first),
            ],
        }
        metrics = figure5_front_metrics(summary)
        records = metrics["by_setting_seed"]
        self.assertEqual(
            [(record["seed"], record["setting"]) for record in records],
            [(0, "w_ddts"), (0, "wo_ddts"), (1, "w_ddts"), (1, "wo_ddts")],
        )
        self.assertEqual(records[0]["unique_proposals"], 1)
        self.assertEqual(records[1]["unique_proposals"], 0)
        self.assertIsNone(metrics["setting_aggregates"]["w_ddts"]["spacing_cv_mean"])

        summary["solutions"][0]["kappa"] = "invalid"
        with self.assertRaisesRegex(ValueError, "finite_json_number"):
            figure5_front_metrics(summary)

    def test_exact_front_rejects_lossy_integer_inputs(self) -> None:
        for invalid in (True, 2.0, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    exact_figure5_front(invalid)  # type: ignore[arg-type]

    def test_cli_malformed_json_writes_report_v3_and_returns_one(self) -> None:
        summary_path = WORKSPACE_TMP_ROOT / "malformed.json"
        report_path = WORKSPACE_TMP_ROOT / "malformed_report.json"
        summary_path.write_text("{not-json", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            return_code = main(["figure5", "--summary", str(summary_path), "--output", str(report_path)])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(return_code, 1)
        self.assertEqual(report["report_schema_version"], 3)
        self.assertFalse(report["passed"])
        self.assertEqual(set(report["checks"]["input"]["detail"]["errors"][0]), {"path", "rule", "expected", "actual"})
        json.dumps(report, allow_nan=False)

    def test_cli_defaults_report_to_each_summary_directory(self) -> None:
        cases = (
            ("figure4", self.figure4, figure4_output_layout),
            ("figure5", self.figure5, figure5_output_layout),
        )
        for figure, summary, layout_factory in cases:
            with self.subTest(figure=figure):
                output_dir = WORKSPACE_TMP_ROOT / f"{figure}_default_output"
                summary_path = output_dir / f"{figure}_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    return_code = main([figure, "--summary", str(summary_path)])
                report_path = layout_factory(output_dir).validation_report_path
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(return_code, 0)
                self.assertTrue(report["passed"])

    def test_cli_validation_failure_uses_default_report_path(self) -> None:
        output_dir = WORKSPACE_TMP_ROOT / "figure4_failed_validation"
        summary_path = output_dir / "figure4_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("{}", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            return_code = main(["figure4", "--summary", str(summary_path)])
        report_path = figure4_output_layout(output_dir).validation_report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(return_code, 1)
        self.assertFalse(report["passed"])

    def test_cli_public_surface_and_no_paper_profile(self) -> None:
        args = parse_args(["figure4", "--summary", "summary.json"])
        self.assertFalse(hasattr(args, "profile"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["figure4", "--summary", "summary.json", "--profile", "paper"])


if __name__ == "__main__":
    unittest.main()
