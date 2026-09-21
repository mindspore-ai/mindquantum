from __future__ import annotations

import json
import random
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from utils import figure5_pipeline as pipeline
from utils import figure5_runner, figure5_scalarization
from utils.alloy_dataset_generator import build_dataset_row, generate_initial_dataset_multi_objective
from utils.experiment_config import (
    EncodingConfig,
    FMConfig,
    QuboConfig,
    SAConfig,
)
from utils.experiment_runtime import (
    FIGURE5_REPLACEMENT_NAMESPACE,
    FIGURE5_SEED_INDEX,
    derive_fm_seed_root,
    derive_python_seed,
    fm_seed_block_size,
    fm_seed_plan,
    resolve_contiguous_seeds,
)
from utils.figure5_experiment_config import (
    FIGURE5_PRESET_NUM_SEEDS,
    Figure5ExperimentConfig,
    preset_config,
    resolve_experiment_config,
)
from utils.figure5_outputs import (
    CHECKPOINT_SCHEMA_VERSION,
    VALIDATION_REPORT_FILENAME,
    checkpoint_payload,
    figure5_output_layout,
    load_checkpoint,
    write_checkpoint,
)
from utils.figure5_pipeline import (
    Figure5TrajectoryResult,
    IterationRecord,
    SolutionPoint,
    TrainingIterationResult,
    run_figure5_experiment,
    run_single_trajectory,
)
from utils.figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
    preference_weights_for_iteration,
    validate_settings,
)
from utils.qubo_math import (
    QuboStats,
    SASample,
    SASamplingResult,
    build_single_objective_qubo,
    create_iteration_encoding,
    encode_discrete_composition,
)

WORKSPACE_TMP_ROOT = (
    Path(__file__).resolve().parents[1] / ".tmp_test" / "figure5_schema_v5_qubo_config_v1_mixed_radix_v1"
)
WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _test_config(iterations: int = 2) -> Figure5ExperimentConfig:
    return Figure5ExperimentConfig(
        num_samples=10,
        iterations=iterations,
        encoding=EncodingConfig(num_levels=8),
        fm=FMConfig(optuna_trials=0, device="cpu"),
        sa=SAConfig(reads=32, sweeps=12),
    )


def _qubo_stats(
    num_variables: int = 32,
    qubo_config: QuboConfig | None = None,
) -> QuboStats:
    qubo_config = QuboConfig() if qubo_config is None else qubo_config
    return QuboStats(
        normalization_scheme=qubo_config.normalization_scheme,
        fm_objective_weight=qubo_config.fm_objective_weight,
        system_penalty_weight=qubo_config.system_penalty_weight,
        one_hot_penalty_weight=qubo_config.one_hot_penalty_weight,
        fm_scale=1.0 if qubo_config.fm_objective_weight > 0.0 else 0.0,
        system_scale=1.0 if qubo_config.system_penalty_weight > 0.0 else 0.0,
        one_hot_scale=1.0 if qubo_config.one_hot_penalty_weight > 0.0 else 0.0,
        num_variables=num_variables,
        max_abs=(
            1.0
            if any(
                (
                    qubo_config.fm_objective_weight,
                    qubo_config.system_penalty_weight,
                    qubo_config.one_hot_penalty_weight,
                )
            )
            else 0.0
        ),
    )


def _fm_metadata(
    config: Figure5ExperimentConfig,
    seed: int,
    iteration: int,
    setting: str,
) -> dict[str, object]:
    trajectory_index = 0 if setting == "w_ddts" else 1
    block_size = fm_seed_block_size(config.optuna_trials)

    def model_metadata(model_index: int) -> dict[str, object]:
        root = derive_fm_seed_root(
            seed,
            trajectory_count=2,
            trajectory_index=trajectory_index,
            iterations=config.iterations,
            iteration=iteration,
            model_count=3,
            model_index=model_index,
            figure_index=FIGURE5_SEED_INDEX,
            block_size=block_size,
        )
        return {"mock": True, "seed_plan": fm_seed_plan(root, config.optuna_trials)}

    if setting == "w_ddts":
        return {"artificial_target": model_metadata(0)}
    return {objective: model_metadata(model_index) for model_index, objective in enumerate(FIGURE5_OBJECTIVES)}


def _training_result(
    config: Figure5ExperimentConfig,
    seed: int,
    iteration: int,
    rows: list[dict[str, object]],
) -> TrainingIterationResult:
    encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
    composition = [1.0, 0.0, 0.0, 0.0] if iteration % 2 == 0 else [0.0, 1.0, 0.0, 0.0]
    scalarization = pipeline.compute_ddts_targets(rows, preference_weights_for_iteration(seed, iteration))
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=encode_discrete_composition(composition, encoding),
        weights=preference_weights_for_iteration(seed, iteration),
        scalarization_method="ddts",
        fm_metadata=_fm_metadata(config, seed, iteration, "w_ddts"),
        scalarization_metadata=dict(scalarization.metadata),
        qubo_stats=_qubo_stats(4 * config.num_levels, config.qubo),
        sa_energy=0.0,
        feasible_candidate_rank=1,
        infeasible_sa_samples_skipped=0,
    )


def _sampling_result(*states: np.ndarray) -> SASamplingResult:
    return SASamplingResult(
        samples=tuple(SASample(np.asarray(state, dtype=np.float64), float(index)) for index, state in enumerate(states))
    )


def _novel_grid_composition(rows: list[dict[str, object]], num_levels: int) -> list[float]:
    seen = {pipeline._row_composition_key(row) for row in rows}
    for first in range(num_levels + 1):
        for second in range(num_levels - first + 1):
            for third in range(num_levels - first - second + 1):
                fourth = num_levels - first - second - third
                composition = [value / num_levels for value in (first, second, third, fourth)]
                if pipeline._composition_key(composition) not in seen:
                    return composition
    raise AssertionError("test grid unexpectedly contains every composition")


class Figure5PipelineTests(unittest.TestCase):
    def test_config_presets_overrides_and_runner_default(self) -> None:
        paper = preset_config("paper", device="cpu")
        quick = preset_config("quick", device="cuda")
        test = preset_config("test", device="cpu")

        self.assertEqual((paper.num_samples, paper.iterations, paper.num_levels), (500, 1000, 25))
        self.assertEqual((quick.num_samples, quick.iterations, quick.num_levels), (500, 150, 25))
        self.assertEqual((quick.optuna_trials, quick.sa_reads, quick.sa_sweeps), (3, 100, 500))
        self.assertEqual((test.num_samples, test.iterations, test.num_levels), (10, 2, 8))
        self.assertEqual((test.sa_reads, test.sa_sweeps), (32, 12))
        self.assertEqual(paper.qubo, QuboConfig(one_hot_penalty_weight=1.0))
        self.assertEqual(quick.qubo, QuboConfig(one_hot_penalty_weight=2.0))
        self.assertEqual(test.qubo, QuboConfig(one_hot_penalty_weight=2.0))

        overridden = resolve_experiment_config(
            preset="test",
            device="cpu",
            iterations=4,
            num_levels=9,
            sa_reads=7,
            fm_objective_weight=2.0,
            system_penalty_weight=3.0,
            one_hot_penalty_weight=4.0,
        )
        self.assertEqual((overridden.iterations, overridden.num_levels, overridden.sa_reads), (4, 9, 7))
        self.assertEqual(
            overridden.qubo,
            QuboConfig(
                fm_objective_weight=2.0,
                system_penalty_weight=3.0,
                one_hot_penalty_weight=4.0,
            ),
        )

        args = figure5_runner.parse_args(["--output-dir", "figure5_default", "--settings", "w_ddts"])
        self.assertEqual(args.preset, "quick")
        self.assertEqual(FIGURE5_PRESET_NUM_SEEDS, {"paper": 1, "quick": 1, "test": 1})
        self.assertEqual(
            resolve_contiguous_seeds(
                FIGURE5_PRESET_NUM_SEEDS[args.preset],
                args.num_seeds,
                args.seed_start,
            ),
            [0],
        )

        qubo_args = figure5_runner.parse_args(
            [
                "--output-dir",
                "figure5_qubo_override",
                "--settings",
                "w_ddts",
                "--fm-objective-weight",
                "2.5",
                "--system-penalty-weight",
                "0",
                "--one-hot-penalty-weight",
                "3.5",
            ]
        )
        self.assertEqual(
            (
                qubo_args.fm_objective_weight,
                qubo_args.system_penalty_weight,
                qubo_args.one_hot_penalty_weight,
            ),
            (2.5, 0.0, 3.5),
        )

        alias_args = figure5_runner.parse_args(
            ["--output-dir", "figure5_alias", "--settings", "w_ddts", "--sa-runs", "9"]
        )
        self.assertEqual(alias_args.sa_reads, 9)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_settings(["w_ddts", "w_ddts"])

        multi_seed_args = figure5_runner.parse_args(
            [
                "--output-dir",
                "figure5_multi_seed",
                "--settings",
                "w_ddts",
                "wo_ddts",
                "--num-seeds",
                "3",
                "--seed-start",
                "4",
            ]
        )
        self.assertEqual(
            resolve_contiguous_seeds(
                FIGURE5_PRESET_NUM_SEEDS[multi_seed_args.preset],
                multi_seed_args.num_seeds,
                multi_seed_args.seed_start,
            ),
            [4, 5, 6],
        )

    def test_config_integer_contract_and_seed_schedule_match_figure4(self) -> None:
        config = Figure5ExperimentConfig(
            num_samples=np.int64(5),
            iterations=np.int64(2),
            encoding=EncodingConfig(np.int64(4)),
            fm=FMConfig(np.int64(0)),
            sa=SAConfig(np.int64(3), np.int64(4)),
        )
        for value in (
            config.num_samples,
            config.iterations,
            config.num_levels,
            config.optuna_trials,
            config.sa_reads,
            config.sa_sweeps,
        ):
            self.assertIs(type(value), int)

        for invalid in (True, 2.0, "2"):
            with self.subTest(num_samples=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    Figure5ExperimentConfig(num_samples=invalid)  # type: ignore[arg-type]
            with self.subTest(iterations=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    Figure5ExperimentConfig(iterations=invalid)  # type: ignore[arg-type]
            with self.subTest(override=invalid):
                with self.assertRaisesRegex(ValueError, "iterations"):
                    resolve_experiment_config(
                        preset="test",
                        device="cpu",
                        iterations=invalid,  # type: ignore[arg-type]
                    )

        from utils.experiment_runtime import NEAL_SEED_MAX

        manifest_mock = mock.Mock()
        run_mock = mock.Mock()
        with (
            mock.patch("utils.figure5_runner.configure_file_logging_path", return_value=Path("runner.log")),
            mock.patch("utils.figure5_runner.ensure_training_dependencies"),
            mock.patch("utils.figure5_runner.ensure_compute_device_available"),
            mock.patch("utils.figure5_runner._write_manifest", manifest_mock),
            mock.patch("utils.figure5_runner.run_figure5_experiment", run_mock),
        ):
            with self.assertRaisesRegex(ValueError, "derived bounded seed"):
                figure5_runner.main(
                    [
                        "--output-dir",
                        "invalid_figure5_seed_schedule",
                        "--settings",
                        "w_ddts",
                        "--iterations",
                        "2",
                        "--seed-start",
                        str(NEAL_SEED_MAX),
                    ]
                )
        manifest_mock.assert_not_called()
        run_mock.assert_not_called()

        with mock.patch("utils.figure5_pipeline.generate_initial_dataset_multi_objective_batch") as dataset_mock:
            with self.assertRaisesRegex(ValueError, "derived bounded seed"):
                run_figure5_experiment(
                    seed_list=[NEAL_SEED_MAX],
                    config=config,
                    output_dir=WORKSPACE_TMP_ROOT / "invalid_seed_schedule",
                    settings=("w_ddts",),
                )
        dataset_mock.assert_not_called()

    def test_runner_is_the_only_supported_experiment_entrypoint(self) -> None:
        self.assertEqual(FIGURE5_SETTINGS, ("w_ddts", "wo_ddts"))
        self.assertFalse(hasattr(figure5_scalarization, "SUPPORTED_SETTINGS"))
        self.assertEqual(figure5_runner.__all__, ["main", "parse_args"])
        self.assertFalse(hasattr(figure5_runner, "resolve_seed_list"))
        wildcard_namespace: dict[str, object] = {}
        exec("from utils.figure5_runner import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            {"main", "parse_args"},
        )
        self.assertNotIn("run_figure5_experiment", figure5_runner.__all__)
        self.assertNotIn("run_figure5_experiment", pipeline.__all__)
        self.assertNotIn("run_single_trajectory", pipeline.__all__)
        self.assertEqual(tuple(FIGURE5_OBJECTIVES), ("kappa", "E", "rho"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_settings(["unknown"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_settings(["w_ddts", "w_ddts"])

    def test_explicit_multi_seed_runs_setting_seed_cartesian_product(self) -> None:
        config = _test_config(iterations=1)

        def fake_run(*_: object, **kwargs: object) -> Figure5TrajectoryResult:
            return Figure5TrajectoryResult(
                setting=str(kwargs["setting"]),
                seed=int(kwargs["seed"]),
                iteration_records=[],
                final_dataset_size=config.num_samples,
                training_backend=pipeline.TRAINING_BACKEND,
                latest_weights=[],
                latest_fm_metadata={},
                latest_scalarization_metadata={},
                qubo_stats=None,
                duplicate_replacements=0,
                random_replacements=0,
                random_replacement_draws=0,
                accepted_sa_candidates=0,
                completed_iterations=0,
            )

        output_dir = WORKSPACE_TMP_ROOT / "figure5_multi_seed"
        with mock.patch("utils.figure5_pipeline.run_single_trajectory", side_effect=fake_run) as run_mock:
            summary = run_figure5_experiment(
                seed_list=[4, 5, 6],
                config=config,
                output_dir=output_dir,
                settings=("w_ddts", "wo_ddts"),
            )

        self.assertEqual(len(summary.trajectories), 6)
        self.assertEqual(
            [(result.setting, result.seed) for result in summary.trajectories],
            [
                ("w_ddts", 4),
                ("wo_ddts", 4),
                ("w_ddts", 5),
                ("wo_ddts", 5),
                ("w_ddts", 6),
                ("wo_ddts", 6),
            ],
        )
        for call_index in range(0, run_mock.call_count, 2):
            self.assertIs(
                run_mock.call_args_list[call_index].kwargs["initial_rows"],
                run_mock.call_args_list[call_index + 1].kwargs["initial_rows"],
            )

    def test_summary_fronts_are_per_seed_and_deduplicate_compositions(self) -> None:
        config = _test_config(iterations=2)

        def fake_run(*_: object, **kwargs: object) -> Figure5TrajectoryResult:
            setting = str(kwargs["setting"])
            seed = int(kwargs["seed"])
            if seed == 0:
                first = SolutionPoint(10, [1.0, 0.0, 0.0, 0.0], 100.0, 80.0, 3.0)
                second = SolutionPoint(11, [1.0, 0.0, 0.0, 0.0], 100.0, 80.0, 3.0)
            else:
                first = SolutionPoint(10, [0.0, 1.0, 0.0, 0.0], 120.0, 100.0, 2.5)
                second = SolutionPoint(11, [0.0, 0.0, 1.0, 0.0], 110.0, 90.0, 2.8)
            records = [
                IterationRecord(setting, seed, 0, [0.2, 0.3, 0.5], "ddts", "accepted", first, first, 0, -1.0),
                IterationRecord(
                    setting,
                    seed,
                    1,
                    [0.2, 0.3, 0.5],
                    "ddts",
                    "accepted",
                    second,
                    second,
                    0,
                    -2.0,
                ),
            ]
            return Figure5TrajectoryResult(
                setting=setting,
                seed=seed,
                iteration_records=records,
                final_dataset_size=config.num_samples + config.iterations,
                training_backend=pipeline.TRAINING_BACKEND,
                latest_weights=[0.2, 0.3, 0.5],
                latest_fm_metadata={"artificial_target": {}},
                latest_scalarization_metadata={},
                qubo_stats=_qubo_stats(),
                duplicate_replacements=0,
                random_replacements=0,
                random_replacement_draws=0,
                accepted_sa_candidates=2,
                completed_iterations=2,
            )

        output_dir = WORKSPACE_TMP_ROOT / "figure5_per_seed_fronts"
        with mock.patch("utils.figure5_pipeline.run_single_trajectory", side_effect=fake_run):
            summary = run_figure5_experiment(
                seed_list=[0, 1],
                config=config,
                output_dir=output_dir,
                settings=("w_ddts",),
            )

        self.assertEqual(
            [(front.setting, front.seed) for front in summary.pareto_fronts],
            [("w_ddts", 0), ("w_ddts", 1)],
        )
        self.assertEqual(len(summary.pareto_fronts[0].solutions), 1)
        self.assertEqual(summary.pareto_fronts[0].solutions[0]["seed"], 0)
        self.assertEqual(summary.pareto_fronts[0].solutions[0]["iteration"], 0)
        self.assertEqual(len(summary.pareto_fronts[1].solutions), 1)
        self.assertEqual(summary.pareto_fronts[1].solutions[0]["seed"], 1)

    def test_preference_weights_are_shared_and_deterministic(self) -> None:
        self.assertFalse(hasattr(pipeline, "preference_weights_for_iteration"))
        weights_a = preference_weights_for_iteration(seed=4, iteration=7)
        weights_b = preference_weights_for_iteration(seed=4, iteration=7)
        weights_c = preference_weights_for_iteration(seed=4, iteration=8)

        self.assertEqual(weights_a, weights_b)
        self.assertNotEqual(weights_a, weights_c)
        self.assertAlmostEqual(sum(weights_a), 1.0, places=12)
        for invalid in (True, 1.5, "1"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    preference_weights_for_iteration(invalid, 0)  # type: ignore[arg-type]
                with self.assertRaisesRegex(ValueError, "integer"):
                    preference_weights_for_iteration(0, invalid)  # type: ignore[arg-type]

    def test_training_uses_one_fm_for_ddts_and_three_for_weighted_sum(self) -> None:
        config = replace(
            _test_config(iterations=1),
            qubo=QuboConfig(
                fm_objective_weight=2.0,
                system_penalty_weight=3.0,
                one_hot_penalty_weight=4.0,
            ),
        )
        rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=2)
        rows = pipeline.discretize_rows_for_figure5(rows, config.num_levels)
        size = 4 * config.num_levels

        def fake_sa(q: np.ndarray, bias: float, **_: object) -> SASamplingResult:
            encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
            state = encode_discrete_composition([1.0, 0.0, 0.0, 0.0], encoding)
            return _sampling_result(state)

        with (
            mock.patch("utils.figure5_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})) as fit_mock,
            mock.patch("utils.figure5_pipeline.fm_to_qubo", return_value=(np.eye(size), 1.0)),
            mock.patch("utils.figure5_pipeline.solve_qubo_with_sa", side_effect=fake_sa),
            mock.patch("utils.figure5_pipeline.diagnose_no_feasible_candidate") as diagnostic_mock,
        ):
            ddts = pipeline._fit_and_solve_iteration(
                rows=rows,
                setting="w_ddts",
                config=config,
                seed=2,
                iteration=0,
            )
        diagnostic_mock.assert_not_called()
        self.assertEqual(fit_mock.call_count, 1)
        ddts_fit_seed = fit_mock.call_args.kwargs["seed"]
        self.assertEqual(ddts.qubo_stats.num_variables, size)
        self.assertEqual(ddts.qubo_stats.fm_objective_weight, 2.0)
        self.assertEqual(ddts.qubo_stats.system_penalty_weight, 3.0)
        self.assertEqual(ddts.qubo_stats.one_hot_penalty_weight, 4.0)

        captured: dict[str, object] = {}

        def capture_build(
            fm_q: np.ndarray,
            fm_bias: float,
            encoding: object,
            qubo_config: QuboConfig,
            include_system_penalty: bool,
        ) -> object:
            captured["q"] = fm_q.copy()
            captured["bias"] = fm_bias
            captured["qubo_config"] = qubo_config
            captured["include_system_penalty"] = include_system_penalty
            return build_single_objective_qubo(
                fm_q,
                fm_bias,
                encoding,
                qubo_config,
                include_system_penalty,
            )

        objective_qubos = [
            (np.eye(size) * 1.0, 1.0),
            (np.eye(size) * 2.0, 2.0),
            (np.eye(size) * 3.0, 3.0),
        ]
        with (
            mock.patch("utils.figure5_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})) as fit_mock,
            mock.patch("utils.figure5_pipeline.fm_to_qubo", side_effect=objective_qubos),
            mock.patch("utils.figure5_pipeline.build_single_objective_qubo", side_effect=capture_build),
            mock.patch("utils.figure5_pipeline.solve_qubo_with_sa", side_effect=fake_sa),
            mock.patch("utils.figure5_pipeline.diagnose_no_feasible_candidate") as diagnostic_mock,
        ):
            weighted = pipeline._fit_and_solve_iteration(
                rows=rows,
                setting="wo_ddts",
                config=config,
                seed=2,
                iteration=0,
            )
        diagnostic_mock.assert_not_called()

        weights = preference_weights_for_iteration(2, 0)
        expected_scale = sum(weight * (index + 1) for index, weight in enumerate(weights))
        self.assertEqual(fit_mock.call_count, 3)
        weighted_fit_seeds = [call.kwargs["seed"] for call in fit_mock.call_args_list]
        self.assertEqual(len(weighted_fit_seeds), len(set(weighted_fit_seeds)))
        self.assertNotIn(ddts_fit_seed, weighted_fit_seeds)
        self.assertTrue(np.allclose(captured["q"], np.eye(size) * expected_scale))
        self.assertAlmostEqual(float(captured["bias"]), expected_scale)
        self.assertEqual(captured["qubo_config"], config.qubo)
        self.assertTrue(captured["include_system_penalty"])
        self.assertEqual(weighted.scalarization_method, "weighted_sum")

    def test_empty_feasible_sa_batch_is_exactly_classified_with_context(self) -> None:
        config = Figure5ExperimentConfig(
            num_samples=10,
            iterations=1,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0, device="cpu"),
            sa=SAConfig(reads=1, sweeps=7),
        )
        rows, _ = generate_initial_dataset_multi_objective(
            num_samples=config.num_samples,
            seed=2,
        )
        rows = pipeline.discretize_rows_for_figure5(rows, config.num_levels)
        size = 4 * config.num_levels
        all_zero = np.zeros(size, dtype=np.float64)

        with (
            mock.patch("utils.figure5_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch(
                "utils.figure5_pipeline.fm_to_qubo",
                return_value=(np.zeros((size, size), dtype=np.float64), 0.0),
            ),
            mock.patch(
                "utils.figure5_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(all_zero),
            ),
            mock.patch(
                "utils.figure5_pipeline.diagnose_no_feasible_candidate",
                wraps=pipeline.diagnose_no_feasible_candidate,
            ) as diagnostic_mock,
        ):
            with self.assertRaises(RuntimeError) as raised:
                pipeline._fit_and_solve_iteration(
                    rows=rows,
                    setting="w_ddts",
                    config=config,
                    seed=2,
                    iteration=0,
                )

        diagnostic_mock.assert_called_once()
        self.assertTrue(diagnostic_mock.call_args.kwargs["require_unit_sum"])
        detail = str(raised.exception)
        self.assertRegex(
            detail,
            (
                "Figure 5.*setting='w_ddts'.*seed=2.*iteration=1/1.*"
                "one_hot_valid=1/1.*fully_feasible=0/1.*num_levels=8.*sa_reads=1.*"
                "sa_sweeps=7.*classification=heuristic_sampling_failure.*"
                "lowest_sampled_infeasible_energy=.*exact_best_feasible_energy=.*"
                "exact_feasible_state_count=165.*unchanged --resume"
            ),
        )

    def test_candidate_decision_separates_proposed_and_replacement(self) -> None:
        config = _test_config(iterations=1)
        encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
        composition = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        bits = encode_discrete_composition(composition, encoding)

        accepted = pipeline._decide_candidate(
            candidate_bits=bits,
            encoding=encoding,
            sample_id=10,
            seed=3,
            replacement_rng=random.Random(20),
            num_levels=config.num_levels,
            seen_compositions=set(),
        )
        self.assertEqual(accepted.status, "accepted")
        self.assertIs(accepted.proposed_row, accepted.added_row)

        duplicate = pipeline._decide_candidate(
            candidate_bits=bits,
            encoding=encoding,
            sample_id=10,
            seed=3,
            replacement_rng=random.Random(20),
            num_levels=config.num_levels,
            seen_compositions={pipeline._composition_key(composition)},
        )
        self.assertEqual(duplicate.status, "duplicate_replacement")
        self.assertIsNotNone(duplicate.proposed_row)
        self.assertNotEqual(
            pipeline._row_composition_key(duplicate.proposed_row),
            pipeline._row_composition_key(duplicate.added_row),
        )

        with self.assertRaisesRegex(RuntimeError, "not feasible"):
            pipeline._decide_candidate(
                candidate_bits=np.zeros(4 * config.num_levels),
                encoding=encoding,
                sample_id=10,
                seed=3,
                replacement_rng=random.Random(21),
                num_levels=config.num_levels,
                seen_compositions=set(),
            )

    def test_resume_rejects_inconsistent_completed_state_before_skip(self) -> None:
        seed = 21
        config = _test_config(iterations=1)
        initial_rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=seed)
        prepared_rows = pipeline.discretize_rows_for_figure5(initial_rows, config.num_levels)
        composition = _novel_grid_composition(prepared_rows, config.num_levels)
        added_row = dict(build_dataset_row(config.num_samples, seed, composition))
        weights = preference_weights_for_iteration(seed, 0)
        scalarization = pipeline.compute_ddts_targets(prepared_rows, weights)
        solution = pipeline._solution_point_from_row(added_row)
        state = pipeline.Figure5TrajectoryState(
            rows=[*prepared_rows, added_row],
            iteration_records=[
                IterationRecord(
                    setting="w_ddts",
                    seed=seed,
                    iteration=0,
                    weights=list(weights),
                    scalarization_method="ddts",
                    decision_status="accepted",
                    proposed_solution=solution,
                    added_solution=solution,
                    replacement_draws=0,
                    sa_energy=-1.0,
                    feasible_candidate_rank=1,
                    infeasible_sa_samples_skipped=0,
                )
            ],
            latest_weights=list(weights),
            latest_fm_metadata=_fm_metadata(config, seed, 0, "w_ddts"),
            latest_scalarization_metadata=dict(scalarization.metadata),
            qubo_stats=_qubo_stats(4 * config.num_levels),
            accepted_sa_candidates=1,
            max_feasible_candidate_rank=1,
        )
        checkpoint = figure5_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_internal_validation").checkpoint_path(
            "w_ddts", seed
        )
        valid_payload = checkpoint_payload(
            setting="w_ddts",
            seed=seed,
            state=state,
            config=config,
        )

        write_checkpoint(checkpoint, valid_payload)
        with mock.patch("utils.figure5_pipeline._fit_and_solve_iteration") as fit_mock:
            result = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=seed,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        fit_mock.assert_not_called()
        self.assertEqual(result.completed_iterations, 1)

        corruptions = (
            (
                "missing state field",
                lambda payload: payload["state"].pop("random_replacement_draws"),
                "missing fields.*random_replacement_draws",
            ),
            (
                "unexpected state field",
                lambda payload: payload["state"].__setitem__("unknown", 1),
                "unexpected fields.*unknown",
            ),
            (
                "missing record field",
                lambda payload: payload["state"]["iteration_records"][0].pop("weights"),
                "missing fields.*weights",
            ),
            (
                "unexpected solution field",
                lambda payload: payload["state"]["iteration_records"][0]["added_solution"].__setitem__("unknown", 1),
                "unexpected fields.*unknown",
            ),
            ("row count", lambda payload: payload["state"]["rows"].pop(), "rows has length"),
            (
                "row seed",
                lambda payload: payload["state"]["rows"][0].__setitem__("seed", seed + 1),
                r"rows\[0\]\.seed",
            ),
            (
                "composition",
                lambda payload: payload["state"]["rows"][0].__setitem__("f1_norm", 1.2),
                "composition",
            ),
            (
                "record identity",
                lambda payload: payload["state"]["iteration_records"][0].__setitem__("seed", seed + 1),
                "setting/seed/iteration",
            ),
            (
                "weights",
                lambda payload: payload["state"]["iteration_records"][0]["weights"].__setitem__(0, 0.0),
                "weights",
            ),
            (
                "method",
                lambda payload: payload["state"]["iteration_records"][0].__setitem__(
                    "scalarization_method", "weighted_sum"
                ),
                "scalarization_method",
            ),
            (
                "solution id",
                lambda payload: payload["state"]["iteration_records"][0]["added_solution"].__setitem__("sample_id", 99),
                "sample_id",
            ),
            (
                "accepted draws",
                lambda payload: payload["state"]["iteration_records"][0].__setitem__("replacement_draws", 1),
                "zero draws",
            ),
            (
                "candidate accounting",
                lambda payload: payload["state"].__setitem__("accepted_sa_candidates", 0),
                "accepted_sa_candidates",
            ),
            (
                "SA rank",
                lambda payload: payload["state"]["iteration_records"][0].__setitem__("feasible_candidate_rank", 2),
                "rank",
            ),
            (
                "latest weights",
                lambda payload: payload["state"]["latest_weights"].__setitem__(0, 0.0),
                "latest_weights",
            ),
            (
                "FM metadata",
                lambda payload: payload["state"].__setitem__("latest_fm_metadata", {"mock": True}),
                "latest_fm_metadata",
            ),
            (
                "FM seed plan",
                lambda payload: payload["state"]["latest_fm_metadata"]["artificial_target"]["seed_plan"].__setitem__(
                    "root",
                    payload["state"]["latest_fm_metadata"]["artificial_target"]["seed_plan"]["root"] + 1,
                ),
                "seed_plan",
            ),
            (
                "scalarization metadata",
                lambda payload: payload["state"]["latest_scalarization_metadata"]["utopian_point"].__setitem__(
                    "kappa", 999.0
                ),
                "latest_scalarization_metadata",
            ),
            (
                "QUBO variable count",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("num_variables", 31),
                "num_variables",
            ),
            (
                "QUBO normalization scheme",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("normalization_scheme", "unknown"),
                "normalization_scheme",
            ),
            (
                "unexpected QUBO field",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("legacy_penalty", 1.0),
                "unexpected fields.*legacy_penalty",
            ),
            (
                "QUBO objective weight",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("fm_objective_weight", 2.0),
                "fm.*weight",
            ),
            (
                "QUBO zero maximum coefficient",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("max_abs", 0.0),
                "max_abs",
            ),
            (
                "missing QUBO stats",
                lambda payload: payload["state"].__setitem__("qubo_stats", None),
                "contain qubo_stats",
            ),
        )
        for label, corrupt, error_pattern in corruptions:
            with self.subTest(label=label):
                payload = json.loads(json.dumps(valid_payload))
                corrupt(payload)
                write_checkpoint(checkpoint, payload)
                with mock.patch("utils.figure5_pipeline._fit_and_solve_iteration") as fit_mock:
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        run_single_trajectory(
                            initial_rows,
                            "w_ddts",
                            seed=seed,
                            config=config,
                            checkpoint_path=checkpoint,
                            resume=True,
                        )
                    fit_mock.assert_not_called()

    def test_completed_weighted_sum_checkpoint_validates_setting_metadata(self) -> None:
        seed = 22
        config = _test_config(iterations=1)
        initial_rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=seed)
        prepared_rows = pipeline.discretize_rows_for_figure5(initial_rows, config.num_levels)
        composition = _novel_grid_composition(prepared_rows, config.num_levels)
        added_row = dict(build_dataset_row(config.num_samples, seed, composition))
        weights = preference_weights_for_iteration(seed, 0)
        individual = pipeline.compute_individual_objective_targets(prepared_rows)
        scalarization_metadata = {
            **individual.metadata,
            "merge_weights": dict(zip(FIGURE5_OBJECTIVES, weights)),
            "merge_stage": "qubo",
        }
        solution = pipeline._solution_point_from_row(added_row)
        state = pipeline.Figure5TrajectoryState(
            rows=[*prepared_rows, added_row],
            iteration_records=[
                IterationRecord(
                    "wo_ddts",
                    seed,
                    0,
                    list(weights),
                    "weighted_sum",
                    "accepted",
                    solution,
                    solution,
                    0,
                    -1.0,
                )
            ],
            latest_weights=list(weights),
            latest_fm_metadata=_fm_metadata(config, seed, 0, "wo_ddts"),
            latest_scalarization_metadata=scalarization_metadata,
            qubo_stats=_qubo_stats(4 * config.num_levels),
            accepted_sa_candidates=1,
            max_feasible_candidate_rank=1,
        )
        checkpoint = figure5_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_weighted_sum_validation").checkpoint_path(
            "wo_ddts", seed
        )
        write_checkpoint(
            checkpoint,
            checkpoint_payload(setting="wo_ddts", seed=seed, state=state, config=config),
        )

        with mock.patch("utils.figure5_pipeline._fit_and_solve_iteration") as fit_mock:
            result = run_single_trajectory(
                initial_rows,
                "wo_ddts",
                seed=seed,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        fit_mock.assert_not_called()
        self.assertEqual(result.completed_iterations, 1)

    def test_partial_resume_and_completed_skip(self) -> None:
        config = _test_config(iterations=2)
        initial_rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=6)
        discretized_initial_rows = pipeline.discretize_rows_for_figure5(initial_rows, config.num_levels)
        output_dir = WORKSPACE_TMP_ROOT / "figure5_resume"
        checkpoint = figure5_output_layout(output_dir).checkpoint_path("w_ddts", 6)

        with mock.patch(
            "utils.figure5_pipeline._fit_and_solve_iteration",
            side_effect=[_training_result(config, 6, 0, discretized_initial_rows), RuntimeError("interrupt")],
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                run_single_trajectory(
                    initial_rows,
                    "w_ddts",
                    seed=6,
                    config=config,
                    checkpoint_path=checkpoint,
                    resume=False,
                )

        payload = load_checkpoint(checkpoint, setting="w_ddts", seed=6, config=config)
        self.assertEqual(len(payload["state"]["iteration_records"]), 1)

        with mock.patch(
            "utils.figure5_pipeline._fit_and_solve_iteration",
            return_value=_training_result(config, 6, 1, payload["state"]["rows"]),
        ) as fit_mock:
            resumed = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(fit_mock.call_count, 1)
        self.assertEqual(resumed.completed_iterations, 2)
        self.assertEqual(resumed.final_dataset_size, config.num_samples + config.iterations)
        self.assertEqual(resumed.random_replacements, 0)
        self.assertEqual(resumed.infeasible_sa_samples_skipped, 0)

        with mock.patch("utils.figure5_pipeline._fit_and_solve_iteration") as fit_mock:
            skipped = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )
        fit_mock.assert_not_called()
        self.assertEqual(skipped.completed_iterations, 2)

        changed = _test_config(iterations=3)
        with self.assertRaisesRegex(ValueError, "config"):
            run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=6,
                config=changed,
                checkpoint_path=checkpoint,
                resume=True,
            )

    def test_partial_resume_rebuilds_iteration_replacement_rng_exactly(self) -> None:
        seed = 23
        config = _test_config(iterations=2)
        initial_rows, _ = generate_initial_dataset_multi_objective(num_samples=config.num_samples, seed=seed)
        prepared_rows = pipeline.discretize_rows_for_figure5(initial_rows, config.num_levels)
        seen = {pipeline._row_composition_key(row) for row in prepared_rows}
        first_rng = random.Random(
            derive_python_seed(
                FIGURE5_REPLACEMENT_NAMESPACE,
                seed,
                trajectory_index=0,
                iteration=0,
            )
        )
        first_added, first_draws = pipeline._sample_unique_replacement_row(
            sample_id=config.num_samples,
            seed=seed,
            rng=first_rng,
            num_levels=config.num_levels,
            seen_compositions=seen,
        )
        proposed_composition = pipeline.normalized_composition_from_row(prepared_rows[0])
        first_proposed = dict(build_dataset_row(config.num_samples, seed, proposed_composition))
        first_weights = preference_weights_for_iteration(seed, 0)
        first_scalarization = pipeline.compute_ddts_targets(prepared_rows, first_weights)
        state = pipeline.Figure5TrajectoryState(
            rows=[*prepared_rows, first_added],
            iteration_records=[
                IterationRecord(
                    "w_ddts",
                    seed,
                    0,
                    list(first_weights),
                    "ddts",
                    "duplicate_replacement",
                    pipeline._solution_point_from_row(first_proposed),
                    pipeline._solution_point_from_row(first_added),
                    first_draws,
                    -1.0,
                )
            ],
            latest_weights=list(first_weights),
            latest_fm_metadata=_fm_metadata(config, seed, 0, "w_ddts"),
            latest_scalarization_metadata=dict(first_scalarization.metadata),
            qubo_stats=_qubo_stats(4 * config.num_levels),
            duplicate_replacements=1,
            random_replacements=1,
            random_replacement_draws=first_draws,
            max_feasible_candidate_rank=1,
        )
        checkpoint = figure5_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_rng_replay").checkpoint_path("w_ddts", seed)
        write_checkpoint(
            checkpoint,
            checkpoint_payload(setting="w_ddts", seed=seed, state=state, config=config),
        )

        expected_rng = random.Random(
            derive_python_seed(
                FIGURE5_REPLACEMENT_NAMESPACE,
                seed,
                trajectory_index=0,
                iteration=0,
            )
        )
        replayed_first, replayed_first_draws = pipeline._sample_unique_replacement_row(
            sample_id=config.num_samples,
            seed=seed,
            rng=expected_rng,
            num_levels=config.num_levels,
            seen_compositions=seen,
        )
        self.assertEqual(replayed_first_draws, first_draws)
        self.assertEqual(pipeline._row_composition_key(replayed_first), pipeline._row_composition_key(first_added))
        expected_seen = {*seen, pipeline._row_composition_key(first_added)}
        expected_second_rng = random.Random(
            derive_python_seed(
                FIGURE5_REPLACEMENT_NAMESPACE,
                seed,
                trajectory_index=0,
                iteration=1,
            )
        )
        expected_second, expected_second_draws = pipeline._sample_unique_replacement_row(
            sample_id=config.num_samples + 1,
            seed=seed,
            rng=expected_second_rng,
            num_levels=config.num_levels,
            seen_compositions=expected_seen,
        )

        encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
        second_weights = preference_weights_for_iteration(seed, 1)
        second_scalarization = pipeline.compute_ddts_targets(state.rows, second_weights)
        training = TrainingIterationResult(
            encoding=encoding,
            candidate_bits=encode_discrete_composition(proposed_composition, encoding),
            weights=second_weights,
            scalarization_method="ddts",
            fm_metadata=_fm_metadata(config, seed, 1, "w_ddts"),
            scalarization_metadata=dict(second_scalarization.metadata),
            qubo_stats=_qubo_stats(4 * config.num_levels),
            sa_energy=-2.0,
            feasible_candidate_rank=1,
            infeasible_sa_samples_skipped=0,
        )
        with mock.patch("utils.figure5_pipeline._fit_and_solve_iteration", return_value=training):
            result = run_single_trajectory(
                initial_rows,
                "w_ddts",
                seed=seed,
                config=config,
                checkpoint_path=checkpoint,
                resume=True,
            )

        final_record = result.iteration_records[-1]
        self.assertEqual(final_record.decision_status, "duplicate_replacement")
        self.assertEqual(final_record.replacement_draws, expected_second_draws)
        self.assertEqual(
            pipeline._composition_key(final_record.added_solution.composition),
            pipeline._row_composition_key(expected_second),
        )
        self.assertEqual(result.random_replacement_draws, first_draws + expected_second_draws)

    def test_summary_uses_proposed_solutions_and_excludes_replacements(self) -> None:
        config = _test_config(iterations=3)
        weak = SolutionPoint(10, [1.0, 0.0, 0.0, 0.0], 100.0, 70.0, 3.0)
        strong = SolutionPoint(11, [0.0, 1.0, 0.0, 0.0], 110.0, 80.0, 2.8)
        replacement = SolutionPoint(12, [0.0, 0.0, 1.0, 0.0], 90.0, 60.0, 3.2)

        def trajectory(setting: str, seed: int) -> Figure5TrajectoryResult:
            records = [
                IterationRecord(setting, seed, 0, [0.2, 0.3, 0.5], "ddts", "accepted", weak, weak, 0, -1.0),
                IterationRecord(
                    setting,
                    seed,
                    1,
                    [0.2, 0.3, 0.5],
                    "ddts",
                    "duplicate_replacement",
                    strong,
                    replacement,
                    1,
                    -2.0,
                ),
                IterationRecord(
                    setting,
                    seed,
                    2,
                    [0.2, 0.3, 0.5],
                    "ddts",
                    "duplicate_replacement",
                    weak,
                    replacement,
                    1,
                    -3.0,
                ),
            ]
            return Figure5TrajectoryResult(
                setting=setting,
                seed=seed,
                iteration_records=records,
                final_dataset_size=13,
                training_backend=pipeline.TRAINING_BACKEND,
                latest_weights=[0.2, 0.3, 0.5],
                latest_fm_metadata={"mock": True},
                latest_scalarization_metadata={"mock": True},
                qubo_stats=_qubo_stats(),
                duplicate_replacements=2,
                random_replacements=2,
                random_replacement_draws=2,
                accepted_sa_candidates=1,
                completed_iterations=3,
            )

        def fake_run(*_: object, **kwargs: object) -> Figure5TrajectoryResult:
            return trajectory(str(kwargs["setting"]), int(kwargs["seed"]))

        output_dir = WORKSPACE_TMP_ROOT / "figure5_summary"
        with mock.patch("utils.figure5_pipeline.run_single_trajectory", side_effect=fake_run):
            summary = run_figure5_experiment(
                seed_list=[0],
                config=config,
                output_dir=output_dir,
                settings=("w_ddts", "wo_ddts"),
            )

        self.assertEqual(len(summary.solutions), 6)
        self.assertTrue(all(solution["composition"] != replacement.composition for solution in summary.solutions))
        self.assertEqual(
            [(front.setting, front.seed) for front in summary.pareto_fronts],
            [("w_ddts", 0), ("wo_ddts", 0)],
        )
        self.assertEqual(len(summary.pareto_fronts[0].solutions), 1)
        self.assertEqual(summary.pareto_fronts[0].solutions[0]["composition"], strong.composition)
        payload = json.loads(figure5_output_layout(output_dir).summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["settings"], ["w_ddts", "wo_ddts"])
        self.assertNotIn("pareto_front", payload)

    def test_output_checkpoint_and_manifest_schema(self) -> None:
        config = _test_config(iterations=1)
        output_dir = WORKSPACE_TMP_ROOT / "figure5_outputs"
        layout = figure5_output_layout(output_dir)
        self.assertEqual(
            layout.validation_report_path,
            output_dir / VALIDATION_REPORT_FILENAME,
        )
        state = pipeline.Figure5TrajectoryState(rows=[])
        checkpoint = layout.checkpoint_path("wo_ddts", 4)
        write_checkpoint(
            checkpoint,
            checkpoint_payload(setting="wo_ddts", seed=4, state=state, config=config),
        )
        loaded = load_checkpoint(checkpoint, setting="wo_ddts", seed=4, config=config)
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(loaded["seed_derivation"], "mixed_radix_v1")
        self.assertEqual(checkpoint.name, "wo_ddts_seed_4.json")
        self.assertNotIn("\n  ", checkpoint.read_text(encoding="utf-8"))
        for invalid_seed in (True, 1.5, "1"):
            with self.subTest(seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "integer"):
                    layout.checkpoint_path("wo_ddts", invalid_seed)  # type: ignore[arg-type]

        args = figure5_runner.parse_args(
            ["--output-dir", str(output_dir), "--preset", "test", "--settings", "w_ddts", "wo_ddts"]
        )
        manifest_path = figure5_runner._write_manifest(
            output_layout=layout,
            args=args,
            resolved_config=config,
            seed_list=[0],
            settings=["w_ddts", "wo_ddts"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_path.name, "figure5_manifest.json")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["figure"], 5)
        self.assertEqual(manifest["seed_derivation"], "mixed_radix_v1")
        self.assertEqual(manifest["resolved_config"], config.to_dict())
        self.assertEqual(manifest["objectives"], ["kappa", "E", "rho"])
        self.assertEqual(manifest["settings"], ["w_ddts", "wo_ddts"])
        paper_metadata = manifest["runtime"]["reference_paper"]
        self.assertTrue(paper_metadata["path"].endswith("paper_2512_11479.pdf"))
        self.assertEqual(paper_metadata["sha256"] is not None, paper_metadata["exists"])

    def test_checkpoint_outer_schema_rejects_malformed_payloads(self) -> None:
        config = _test_config(iterations=1)
        layout = figure5_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_outer_validation")
        checkpoint = layout.checkpoint_path("w_ddts", 4)
        valid_payload = checkpoint_payload(
            setting="w_ddts",
            seed=4,
            state=pipeline.Figure5TrajectoryState(rows=[]),
            config=config,
        )
        malformed_payloads = (
            ("root", [], "root"),
            (
                "missing state",
                {key: value for key, value in valid_payload.items() if key != "state"},
                "missing top-level fields.*state",
            ),
            ("boolean schema", {**valid_payload, "schema_version": True}, "schema_version"),
            ("old QUBO schema", {**valid_payload, "schema_version": 4}, "schema_version"),
            ("numeric setting", {**valid_payload, "setting": 1}, "setting"),
            ("string seed", {**valid_payload, "seed": "4"}, "seed"),
            ("list config", {**valid_payload, "config": []}, "config"),
            ("list state", {**valid_payload, "state": []}, "state"),
            ("wrong schema", {**valid_payload, "schema_version": 1}, "schema_version"),
            (
                "wrong seed derivation",
                {**valid_payload, "seed_derivation": "legacy"},
                "seed_derivation",
            ),
            ("wrong setting", {**valid_payload, "setting": "wo_ddts"}, "setting metadata"),
            ("wrong seed", {**valid_payload, "seed": 5}, "seed metadata"),
            (
                "wrong config",
                {**valid_payload, "config": {**valid_payload["config"], "iterations": 2}},
                "different configuration",
            ),
            (
                "wrong QUBO config",
                {
                    **valid_payload,
                    "config": {
                        **valid_payload["config"],
                        "qubo": {
                            **valid_payload["config"]["qubo"],
                            "system_penalty_weight": 3.0,
                        },
                    },
                },
                "different configuration",
            ),
            (
                "type-coerced config",
                {**valid_payload, "config": {**valid_payload["config"], "iterations": True}},
                "config",
            ),
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for label, malformed_payload, error_pattern in malformed_payloads:
            with self.subTest(label=label):
                checkpoint.write_text(json.dumps(malformed_payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error_pattern) as caught:
                    load_checkpoint(checkpoint, setting="w_ddts", seed=4, config=config)
                self.assertIn(str(checkpoint), str(caught.exception))

        checkpoint.write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid JSON") as caught:
            load_checkpoint(checkpoint, setting="w_ddts", seed=4, config=config)
        self.assertIn(str(checkpoint), str(caught.exception))

    def test_runner_rejects_unavailable_cuda(self) -> None:
        from utils import experiment_runtime

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(EnvironmentError, "cuda"):
                experiment_runtime.ensure_compute_device_available("cuda")


if __name__ == "__main__":
    unittest.main()
