import inspect
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from utils import figure4_pipeline, figure4_runner, fm_torch, plot_figure4, qubo_math
from utils.alloy_dataset_generator import build_dataset_row, generate_initial_dataset_single_objective
from utils.experiment_config import (
    QUBO_NORMALIZATION_SCHEME,
    EncodingConfig,
    FMConfig,
    QuboConfig,
    SAConfig,
)
from utils.experiment_runtime import (
    FIGURE4_SEED_INDEX,
    SEED_DERIVATION_SCHEME,
    TRAINING_REQUIRED_MODULES,
    configure_file_logging_path,
    derive_fm_seed_root,
    ensure_training_dependencies,
    fm_seed_block_size,
    fm_seed_plan,
)
from utils.figure4_experiment_config import (
    FIGURE4_OBJECTIVES,
    FIGURE4_PRESET_NUM_SEEDS,
    FIGURE4_SETTINGS,
    Figure4ExperimentConfig,
    resolve_experiment_config,
)
from utils.figure4_experiment_config import (
    preset_config as figure4_preset_config,
)
from utils.figure4_outputs import (
    CHECKPOINT_DIR_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FIGURE_FILENAME,
    LOG_DIR_NAME,
    MANIFEST_FILENAME,
    PLOT_LOG_FILENAME,
    RUNNER_LOG_FILENAME,
    SUMMARY_FILENAME,
    VALIDATION_REPORT_FILENAME,
    checkpoint_payload,
    figure4_output_layout,
    load_checkpoint,
    load_summary,
    write_checkpoint,
)
from utils.figure4_pipeline import (
    TrainingIterationResult,
    TrajectoryResult,
    TrajectoryState,
    aggregate_trajectory_results,
    discretize_rows_for_figure4,
    run_figure4_experiment,
    run_single_trajectory,
)
from utils.figure4_setting_strategies import get_setting_strategy, validate_settings
from utils.figure5_experiment_config import preset_config as figure5_preset_config
from utils.fm_torch import TorchFMRegressor, fm_to_qubo
from utils.qubo_math import (
    MAX_EXACT_DIAGNOSTIC_STATES,
    QuboStats,
    SASample,
    SASamplingResult,
    build_one_hot_penalty_matrix,
    build_single_objective_qubo,
    build_system_penalty_matrix,
    cgfm_angles_to_composition,
    cgfm_composition_to_angles,
    create_cgfm_iteration_encoding,
    create_iteration_encoding,
    decode_candidate_bits_to_cgfm_composition,
    decode_candidate_bits_to_composition,
    diagnose_no_feasible_candidate,
    encode_cgfm_rows,
    encode_single_objective_rows,
    evaluate_qubo_energy,
    normalize_qubo_term,
    prepare_discrete_composition,
    select_lowest_energy_feasible_sample,
    solve_qubo_with_sa,
    validate_candidate_composition,
)

WORKSPACE_TMP_ROOT = (
    Path(__file__).resolve().parents[1] / ".tmp_test" / "figure4_schema_v6_qubo_config_v1_mixed_radix_v1"
)
WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _sampling_result(*states: np.ndarray) -> SASamplingResult:
    return SASamplingResult(
        samples=tuple(SASample(np.asarray(state, dtype=np.float64), float(index)) for index, state in enumerate(states))
    )


def _fm_metadata(
    config: Figure4ExperimentConfig,
    seed: int,
    iteration: int,
    *,
    trajectory_index: int = 0,
) -> dict[str, object]:
    block_size = fm_seed_block_size(config.optuna_trials)
    root = derive_fm_seed_root(
        seed,
        trajectory_count=10,
        trajectory_index=trajectory_index,
        iterations=config.iterations,
        iteration=iteration,
        model_count=1,
        model_index=0,
        figure_index=FIGURE4_SEED_INDEX,
        block_size=block_size,
    )
    return {"mock": True, "seed_plan": fm_seed_plan(root, config.optuna_trials)}


def _training_result(config: Figure4ExperimentConfig, seed: int, iteration: int) -> TrainingIterationResult:
    encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
    candidate_bits = encode_single_objective_rows(
        [build_dataset_row(0, seed, [1.0, 0.0, 0.0, 0.0])],
        encoding,
    )[0].astype(np.float64)
    return TrainingIterationResult(
        encoding=encoding,
        candidate_bits=candidate_bits,
        feasible_candidate_rank=1,
        infeasible_sa_samples_skipped=0,
        fm_metadata=_fm_metadata(config, seed, iteration),
        qubo_stats=_qubo_stats(config, 4 * config.num_levels),
    )


def _qubo_stats(
    config: Figure4ExperimentConfig,
    num_variables: int,
    *,
    include_system: bool = True,
) -> QuboStats:
    fm_weight = config.qubo.fm_objective_weight
    system_weight = config.qubo.system_penalty_weight if include_system else 0.0
    one_hot_weight = config.qubo.one_hot_penalty_weight
    return QuboStats(
        normalization_scheme=config.qubo.normalization_scheme,
        fm_objective_weight=fm_weight,
        system_penalty_weight=system_weight,
        one_hot_penalty_weight=one_hot_weight,
        fm_scale=1.0 if fm_weight > 0.0 else 0.0,
        system_scale=1.0 if system_weight > 0.0 else 0.0,
        one_hot_scale=1.0 if one_hot_weight > 0.0 else 0.0,
        num_variables=num_variables,
        max_abs=1.0 if any((fm_weight, system_weight, one_hot_weight)) else 0.0,
    )


class Figure4PipelineTests(unittest.TestCase):
    def test_required_runtime_dependencies_are_importable(self) -> None:
        ensure_training_dependencies(TRAINING_REQUIRED_MODULES)
        self.assertIsInstance(torch.__version__, str)
        self.assertTrue(torch.__version__)

    def test_config_validation_and_presets(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_levels"):
            EncodingConfig(num_levels=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            Figure4ExperimentConfig(iterations=0)
        with self.assertRaisesRegex(ValueError, "sa_reads"):
            SAConfig(reads=0, sweeps=1)
        with self.assertRaisesRegex(ValueError, "optuna_trials"):
            FMConfig(optuna_trials=-1)

        paper = figure4_preset_config("paper", device="cpu")
        quick = resolve_experiment_config(preset="quick", device="cuda")
        test = figure4_preset_config("test", device="cpu")
        self.assertEqual(quick.num_samples, 50)
        self.assertEqual(quick.iterations, 100)
        self.assertEqual(quick.num_levels, 25)
        self.assertEqual(quick.optuna_trials, 3)
        self.assertEqual(quick.sa_reads, 250)
        self.assertEqual(quick.sa_sweeps, 1000)
        self.assertEqual(quick.device, "cuda")
        self.assertEqual(paper.qubo, QuboConfig(one_hot_penalty_weight=1.0))
        self.assertEqual(quick.qubo, QuboConfig(one_hot_penalty_weight=2.0))
        self.assertEqual(test.qubo, QuboConfig(one_hot_penalty_weight=2.0))

        overridden = resolve_experiment_config(preset="test", device="cpu", iterations=3, sa_reads=4)
        self.assertEqual(overridden.iterations, 3)
        self.assertEqual(overridden.sa_reads, 4)
        self.assertEqual(overridden.sa_sweeps, 12)

        args = figure4_runner.parse_args(["--output-dir", "figure4_default", "--settings", "w_cgfm"])
        self.assertEqual(args.preset, "quick")
        self.assertEqual(FIGURE4_PRESET_NUM_SEEDS, {"paper": 20, "quick": 3, "test": 1})
        alias_args = figure4_runner.parse_args(
            ["--output-dir", "figure4_alias", "--settings", "w_cgfm", "--sa-runs", "7"]
        )
        self.assertEqual(alias_args.sa_reads, 7)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_settings(["w_cgfm", "w_cgfm"])

        from utils.experiment_runtime import validate_seed_list

        self.assertEqual(validate_seed_list([0, np.int64(2)]), [0, 2])
        for invalid_seeds in ([], [-1], [1, 1], [True]):
            with self.subTest(invalid_seeds=invalid_seeds):
                with self.assertRaises(ValueError):
                    validate_seed_list(invalid_seeds)

    def test_qubo_config_contract_and_runner_overrides(self) -> None:
        config = QuboConfig(
            fm_objective_weight=np.float64(0.0),
            system_penalty_weight=np.float32(2.5),
            one_hot_penalty_weight=np.int64(0),
        )
        self.assertEqual(config.normalization_scheme, QUBO_NORMALIZATION_SCHEME)
        for value in (
            config.fm_objective_weight,
            config.system_penalty_weight,
            config.one_hot_penalty_weight,
        ):
            self.assertIs(type(value), float)

        overridden = resolve_experiment_config(
            preset="test",
            device="cpu",
            fm_objective_weight=0.0,
            system_penalty_weight=3.0,
            one_hot_penalty_weight=4.0,
        )
        self.assertEqual(
            overridden.qubo,
            QuboConfig(0.0, 3.0, 4.0),
        )
        args = figure4_runner.parse_args(
            [
                "--output-dir",
                "figure4_qubo_override",
                "--settings",
                "wo_cgfm",
                "--fm-objective-weight",
                "0",
                "--system-penalty-weight",
                "3",
                "--one-hot-penalty-weight",
                "4",
            ]
        )
        self.assertEqual(
            (args.fm_objective_weight, args.system_penalty_weight, args.one_hot_penalty_weight),
            (0.0, 3.0, 4.0),
        )

        for field_name in (
            "fm_objective_weight",
            "system_penalty_weight",
            "one_hot_penalty_weight",
        ):
            for invalid in (True, "1", -1.0, np.nan, np.inf, 10**400):
                with self.subTest(field=field_name, value=invalid):
                    with self.assertRaisesRegex(ValueError, field_name):
                        QuboConfig(**{field_name: invalid})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "normalization_scheme"):
            QuboConfig(normalization_scheme="unknown")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "normalization_scheme"):
            QuboConfig(normalization_scheme=np.str_(QUBO_NORMALIZATION_SCHEME))  # type: ignore[arg-type]

    def test_config_integer_contract_rejects_coercion_and_normalizes_numpy_values(self) -> None:
        config = Figure4ExperimentConfig(
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

        constructors = (
            lambda value: Figure4ExperimentConfig(num_samples=value),
            lambda value: Figure4ExperimentConfig(iterations=value),
            lambda value: EncodingConfig(num_levels=value),
            lambda value: FMConfig(optuna_trials=value),
            lambda value: SAConfig(reads=value, sweeps=1),
            lambda value: SAConfig(reads=1, sweeps=value),
        )
        for invalid in (True, 2.0, "2"):
            for constructor in constructors:
                with self.subTest(value=invalid, constructor=constructor):
                    with self.assertRaisesRegex(ValueError, "integer"):
                        constructor(invalid)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "iterations"):
            resolve_experiment_config(preset="test", device="cpu", iterations=1.5)  # type: ignore[arg-type]

    def test_seed_schedule_is_validated_before_runner_or_pipeline_work(self) -> None:
        from utils.experiment_runtime import (
            FIGURE4_SEED_INDEX,
            NEAL_SEED_MAX,
            resolve_contiguous_seeds,
            validate_seed_list,
            validate_seed_schedule,
        )

        self.assertEqual(validate_seed_list([NEAL_SEED_MAX], iterations=2), [NEAL_SEED_MAX])
        with self.assertRaisesRegex(ValueError, "derived bounded seed"):
            validate_seed_schedule(
                [NEAL_SEED_MAX],
                figure_index=FIGURE4_SEED_INDEX,
                trajectory_count=10,
                iterations=2,
                bounded_stream_count=2,
                fm_model_count=1,
                fm_seed_block_size=4,
            )
        with self.assertRaisesRegex(ValueError, "base-seed maximum"):
            validate_seed_list([NEAL_SEED_MAX + 1])
        with self.assertRaisesRegex(ValueError, "contiguous seed range"):
            resolve_contiguous_seeds(2, None, NEAL_SEED_MAX)
        for invalid in (True, 1.0, "1"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    resolve_contiguous_seeds(1, invalid, 0)  # type: ignore[arg-type]

        manifest_mock = mock.Mock()
        run_mock = mock.Mock()
        with (
            mock.patch("utils.figure4_runner.configure_file_logging_path", return_value=Path("runner.log")),
            mock.patch("utils.figure4_runner.ensure_training_dependencies"),
            mock.patch("utils.figure4_runner.ensure_compute_device_available"),
            mock.patch("utils.figure4_runner._write_manifest", manifest_mock),
            mock.patch("utils.figure4_runner.run_figure4_experiment", run_mock),
        ):
            with self.assertRaisesRegex(ValueError, "derived bounded seed"):
                figure4_runner.main(
                    [
                        "--output-dir",
                        "invalid_seed_schedule",
                        "--settings",
                        "wo_cgfm",
                        "--iterations",
                        "2",
                        "--num-seeds",
                        "1",
                        "--seed-start",
                        str(NEAL_SEED_MAX),
                    ]
                )
        manifest_mock.assert_not_called()
        run_mock.assert_not_called()

        config = Figure4ExperimentConfig(
            num_samples=1,
            iterations=2,
            encoding=EncodingConfig(2),
            fm=FMConfig(0),
            sa=SAConfig(1, 1),
        )
        with mock.patch("utils.figure4_pipeline.generate_initial_dataset_batch") as dataset_mock:
            with self.assertRaisesRegex(ValueError, "derived bounded seed"):
                run_figure4_experiment(
                    seed_list=[NEAL_SEED_MAX],
                    config=config,
                    output_dir=WORKSPACE_TMP_ROOT / "invalid_seed_schedule",
                )
        dataset_mock.assert_not_called()

    def test_sa_batch_is_energy_sorted_and_feasible_selection_is_auditable(self) -> None:
        q = np.asarray([[-1.0]], dtype=np.float64)
        sampling = solve_qubo_with_sa(q, 0.0, reads=5, sweeps=10, seed=3)

        self.assertEqual(sampling.num_samples, 5)
        self.assertEqual(
            [sample.energy for sample in sampling.samples],
            sorted(sample.energy for sample in sampling.samples),
        )
        self.assertTrue(all(sample.state.shape == (1,) for sample in sampling.samples))

        controlled = _sampling_result(np.asarray([0.0]), np.asarray([1.0]), np.asarray([0.0]))
        selected = select_lowest_energy_feasible_sample(controlled, lambda state: bool(state[0] == 1.0))
        self.assertEqual(selected.rank, 2)
        self.assertEqual(selected.infeasible_samples_skipped, 1)
        np.testing.assert_array_equal(selected.state, [1.0])

        with self.assertRaisesRegex(RuntimeError, "No feasible"):
            select_lowest_energy_feasible_sample(controlled, lambda _: False)
        with self.assertRaisesRegex(ValueError, "reads"):
            solve_qubo_with_sa(q, 0.0, reads=0, sweeps=1, seed=0)
        for parameter, kwargs in (
            ("reads", {"reads": 1.5, "sweeps": 1, "seed": 0}),
            ("sweeps", {"reads": 1, "sweeps": "1", "seed": 0}),
            ("seed", {"reads": 1, "sweeps": 1, "seed": True}),
        ):
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    solve_qubo_with_sa(q, 0.0, **kwargs)  # type: ignore[arg-type]

        from utils.experiment_runtime import NEAL_SEED_MAX

        with self.assertRaisesRegex(ValueError, "seed"):
            solve_qubo_with_sa(q, 0.0, reads=1, sweeps=1, seed=NEAL_SEED_MAX + 1)

    def test_exact_no_feasible_diagnostic_classifies_energy_relationships(self) -> None:
        encoding = create_iteration_encoding(2, num_blocks=4)
        size = encoding.num_blocks * encoding.num_levels
        all_zero = np.zeros(size, dtype=np.float64)

        # An asymmetric stored coefficient must still contribute exactly once to x^T Q x.
        penalty_q = np.zeros((size, size), dtype=np.float64)
        penalty_q[0, 1] = -2.0
        multi_hot = all_zero.copy()
        multi_hot[[0, 1]] = 1.0
        penalty = diagnose_no_feasible_candidate(
            penalty_q,
            0.0,
            _sampling_result(multi_hot),
            encoding,
            require_unit_sum=True,
        )
        self.assertEqual(penalty.classification, "qubo_penalty_balance_failure")
        self.assertEqual(penalty.lowest_sampled_infeasible_energy, -2.0)
        self.assertEqual(penalty.exact_best_feasible_energy, 0.0)
        self.assertEqual(penalty.exact_feasible_state_count, math.comb(5, 3))
        self.assertIn("QUBO penalty-balance failure", penalty.format_message())

        heuristic_q = np.zeros((size, size), dtype=np.float64)
        heuristic_q[1, 1] = -1.0
        heuristic = diagnose_no_feasible_candidate(
            heuristic_q,
            0.0,
            _sampling_result(all_zero),
            encoding,
            require_unit_sum=True,
        )
        self.assertEqual(heuristic.classification, "heuristic_sampling_failure")
        self.assertEqual(heuristic.lowest_sampled_infeasible_energy, 0.0)
        self.assertEqual(heuristic.exact_best_feasible_energy, -1.0)
        self.assertIn("heuristic-sampling failure", heuristic.format_message())

        tied = diagnose_no_feasible_candidate(
            np.zeros((size, size), dtype=np.float64),
            0.0,
            _sampling_result(all_zero),
            encoding,
            require_unit_sum=True,
        )
        self.assertEqual(tied.classification, "unclassified_no_feasible_candidate")
        self.assertEqual(tied.exact_best_feasible_energy, 0.0)
        self.assertIn("no feasible SA candidate; cause not classified", tied.format_message())
        self.assertNotIn("heuristic-sampling failure", tied.format_message())

    def test_exact_no_feasible_diagnostic_counts_cgfm_and_skips_over_limit(self) -> None:
        encoding = create_cgfm_iteration_encoding(2, seed=3)
        size = encoding.num_blocks * encoding.num_levels
        q = np.zeros((size, size), dtype=np.float64)
        q[0, 0] = -1.0
        q[2, 2] = -2.0
        multi_hot = np.zeros(size, dtype=np.float64)
        multi_hot[[0, 1]] = 1.0

        diagnostic = diagnose_no_feasible_candidate(
            q,
            0.0,
            _sampling_result(multi_hot),
            encoding,
            require_unit_sum=False,
        )
        self.assertEqual(diagnostic.classification, "heuristic_sampling_failure")
        self.assertEqual(diagnostic.exact_feasible_state_count, 27)
        self.assertEqual(diagnostic.exact_best_feasible_energy, -3.0)

        with mock.patch(
            "utils.qubo_math._active_qubo_energy",
            wraps=qubo_math._active_qubo_energy,
        ) as energy_mock:
            over_limit = diagnose_no_feasible_candidate(
                q,
                0.0,
                _sampling_result(multi_hot),
                encoding,
                require_unit_sum=False,
                max_exact_diagnostic_states=10,
            )
        self.assertEqual(over_limit.classification, "unclassified_no_feasible_candidate")
        self.assertIsNone(over_limit.exact_best_feasible_energy)
        self.assertEqual(over_limit.exact_feasible_state_count, 27)
        # One call recomputes the sampled state; no exact state was enumerated.
        self.assertEqual(energy_mock.call_count, 1)
        self.assertIn("exact_best_feasible_energy=not_computed", over_limit.format_message())
        self.assertIn("exceeds the limit", over_limit.format_message())
        self.assertNotIn("heuristic-sampling failure", over_limit.format_message())

    def test_exact_no_feasible_diagnostic_rejects_malformed_inputs(self) -> None:
        encoding = create_iteration_encoding(2, num_blocks=4)
        size = encoding.num_blocks * encoding.num_levels
        q = np.zeros((size, size), dtype=np.float64)
        all_zero = np.zeros(size, dtype=np.float64)
        sampling = _sampling_result(all_zero)

        with self.assertRaisesRegex(ValueError, "shape"):
            diagnose_no_feasible_candidate(q[:-1], 0.0, sampling, encoding, require_unit_sum=True)
        non_finite_q = q.copy()
        non_finite_q[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnose_no_feasible_candidate(non_finite_q, 0.0, sampling, encoding, require_unit_sum=True)
        with self.assertRaisesRegex(ValueError, "bias"):
            diagnose_no_feasible_candidate(q, np.inf, sampling, encoding, require_unit_sum=True)
        with self.assertRaisesRegex(ValueError, "shape"):
            diagnose_no_feasible_candidate(
                q,
                0.0,
                _sampling_result(all_zero[:-1]),
                encoding,
                require_unit_sum=True,
            )
        string_sampling = SASamplingResult(samples=(SASample(np.asarray(["0"] * size), 123.0),))
        with self.assertRaisesRegex(ValueError, "numeric binary"):
            diagnose_no_feasible_candidate(q, 0.0, string_sampling, encoding, require_unit_sum=True)
        for invalid_value in (np.nan, 0.5):
            invalid_state = all_zero.copy()
            invalid_state[0] = invalid_value
            with self.subTest(state_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "finite binary"):
                    diagnose_no_feasible_candidate(
                        q,
                        0.0,
                        _sampling_result(invalid_state),
                        encoding,
                        require_unit_sum=True,
                    )
        with self.assertRaisesRegex(ValueError, "at least one"):
            diagnose_no_feasible_candidate(
                q,
                0.0,
                SASamplingResult(samples=()),
                encoding,
                require_unit_sum=True,
            )
        with self.assertRaisesRegex(ValueError, "require_unit_sum"):
            diagnose_no_feasible_candidate(
                q,
                0.0,
                sampling,
                encoding,
                require_unit_sum=1,  # type: ignore[arg-type]
            )
        feasible = all_zero.copy()
        feasible[1] = 1.0
        with self.assertRaisesRegex(ValueError, "no feasible"):
            diagnose_no_feasible_candidate(
                q,
                0.0,
                _sampling_result(feasible),
                encoding,
                require_unit_sum=True,
            )
        for invalid_limit in (True, 1.5, "10"):
            with self.subTest(max_exact_diagnostic_states=invalid_limit):
                with self.assertRaisesRegex(ValueError, "max_exact_diagnostic_states"):
                    diagnose_no_feasible_candidate(
                        q,
                        0.0,
                        sampling,
                        encoding,
                        require_unit_sum=True,
                        max_exact_diagnostic_states=invalid_limit,  # type: ignore[arg-type]
                    )

    def test_all_builtin_encoding_domains_fit_exact_diagnostic_limit(self) -> None:
        for preset in ("paper", "quick", "test"):
            with self.subTest(figure=4, preset=preset):
                levels = figure4_preset_config(preset).num_levels  # type: ignore[arg-type]
                self.assertLessEqual(math.comb(levels + 3, 3), MAX_EXACT_DIAGNOSTIC_STATES)
                self.assertLessEqual((levels + 1) ** 3, MAX_EXACT_DIAGNOSTIC_STATES)
            with self.subTest(figure=5, preset=preset):
                levels = figure5_preset_config(preset).num_levels  # type: ignore[arg-type]
                self.assertLessEqual(math.comb(levels + 3, 3), MAX_EXACT_DIAGNOSTIC_STATES)

    def test_fm_seed_and_integer_boundaries_fail_before_training(self) -> None:
        self.assertFalse(hasattr(fm_torch, "fm_seed_plan"))
        self.assertFalse(hasattr(fm_torch, "fm_seed_block_size"))
        x = np.zeros((2, 2), dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        split = fm_torch.split_train_validation_test(
            x,
            y,
            seed=fm_torch.NUMPY_SEED_MAX - 1,
        )
        self.assertEqual(len(split.train_y), 2)

        with self.assertRaisesRegex(ValueError, "seed"):
            fm_torch.split_train_validation_test(
                x,
                y,
                seed=fm_torch.NUMPY_SEED_MAX,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            fm_torch.tune_fm_hparams(
                split,
                optuna_trials=2,
                device="cpu",
                seed=fm_torch.NUMPY_SEED_MAX,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            fm_torch.fit_torch_fm(
                x,
                y,
                optuna_trials=0,
                device="cpu",
                seed=fm_torch.NUMPY_SEED_MAX,
            )
        for invalid_trials in (True, 1.0, "1"):
            with self.subTest(optuna_trials=invalid_trials):
                with self.assertRaisesRegex(ValueError, "optuna_trials"):
                    fm_torch.tune_fm_hparams(
                        split,
                        optuna_trials=invalid_trials,  # type: ignore[arg-type]
                        device="cpu",
                        seed=0,
                    )

        for invalid in (True, 2.0, "2"):
            with self.subTest(num_levels=invalid):
                with self.assertRaisesRegex(ValueError, "num_levels"):
                    create_iteration_encoding(invalid)  # type: ignore[arg-type]
            with self.subTest(num_blocks=invalid):
                with self.assertRaisesRegex(ValueError, "num_blocks"):
                    create_iteration_encoding(2, num_blocks=invalid)  # type: ignore[arg-type]

    def test_checkpoint_manager_rules(self) -> None:
        config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        output_dir = WORKSPACE_TMP_ROOT / "checkpoint_rules"
        layout = figure4_output_layout(output_dir)
        path = layout.checkpoint_path("wo_cgfm", "kappa", 4)
        self.assertEqual(path, layout.checkpoint_path("wo_cgfm", "kappa", 4))
        self.assertEqual(path.name, "wo_cgfm_kappa_seed_4.json")
        self.assertEqual(path.parent, layout.checkpoint_dir)

        rows = [build_dataset_row(0, 4, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))]
        payload = checkpoint_payload(
            objective=FIGURE4_OBJECTIVES[0],
            setting="wo_cgfm",
            seed=4,
            state=TrajectoryState(rows=rows),
            config=config,
        )
        write_checkpoint(path, payload)
        loaded = load_checkpoint(
            path,
            objective=FIGURE4_OBJECTIVES[0],
            seed=4,
            config=config,
            setting="wo_cgfm",
        )
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(loaded["seed_derivation"], "mixed_radix_v1")
        self.assertEqual(loaded["objective"], "kappa")
        self.assertNotIn("\n  ", path.read_text(encoding="utf-8"))

        changed_config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        with self.assertRaisesRegex(ValueError, "different configuration"):
            load_checkpoint(
                path,
                objective=FIGURE4_OBJECTIVES[0],
                seed=4,
                config=changed_config,
                setting="wo_cgfm",
            )
        changed_qubo_config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
            qubo=QuboConfig(system_penalty_weight=3.0),
        )
        with self.assertRaisesRegex(ValueError, "different configuration"):
            load_checkpoint(
                path,
                objective=FIGURE4_OBJECTIVES[0],
                seed=4,
                config=changed_qubo_config,
                setting="wo_cgfm",
            )

    def test_checkpoint_outer_schema_rejects_malformed_payloads(self) -> None:
        config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=2, sweeps=12),
        )
        layout = figure4_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_outer_validation")
        path = layout.checkpoint_path("wo_cgfm", "kappa", 4)
        row = build_dataset_row(0, 4, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        valid_payload = checkpoint_payload(
            objective=FIGURE4_OBJECTIVES[0],
            setting="wo_cgfm",
            seed=4,
            state=TrajectoryState(rows=[row]),
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
            ("old QUBO schema", {**valid_payload, "schema_version": 5}, "schema_version"),
            (
                "wrong seed derivation",
                {**valid_payload, "seed_derivation": "legacy"},
                "seed_derivation",
            ),
            ("numeric objective", {**valid_payload, "objective": 1}, "objective"),
            ("string seed", {**valid_payload, "seed": "4"}, "seed"),
            ("list config", {**valid_payload, "config": []}, "config"),
            ("list state", {**valid_payload, "state": []}, "state"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        for label, malformed_payload, error_pattern in malformed_payloads:
            with self.subTest(label=label):
                path.write_text(json.dumps(malformed_payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error_pattern):
                    load_checkpoint(
                        path,
                        objective=FIGURE4_OBJECTIVES[0],
                        seed=4,
                        config=config,
                        setting="wo_cgfm",
                    )

        path.write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            load_checkpoint(
                path,
                objective=FIGURE4_OBJECTIVES[0],
                seed=4,
                config=config,
                setting="wo_cgfm",
            )

    def test_output_layout_rules(self) -> None:
        output_dir = WORKSPACE_TMP_ROOT / "figure4_layout_rules"
        layout = figure4_output_layout(output_dir)
        self.assertEqual(layout.summary_path, output_dir / SUMMARY_FILENAME)
        self.assertEqual(layout.manifest_path, output_dir / MANIFEST_FILENAME)
        self.assertEqual(layout.figure_path, output_dir / DEFAULT_FIGURE_FILENAME)
        self.assertEqual(
            layout.validation_report_path,
            output_dir / VALIDATION_REPORT_FILENAME,
        )
        self.assertEqual(layout.runner_log_path, output_dir / LOG_DIR_NAME / RUNNER_LOG_FILENAME)
        self.assertEqual(layout.plot_log_path, output_dir / LOG_DIR_NAME / PLOT_LOG_FILENAME)
        self.assertEqual(
            layout.checkpoint_path("w_cgfm", "delta_T", 7),
            output_dir / CHECKPOINT_DIR_NAME / "w_cgfm_delta_T_seed_7.json",
        )
        for invalid_seed in (True, 1.5, "1"):
            with self.subTest(seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "integer"):
                    layout.checkpoint_path("w_cgfm", "delta_T", invalid_seed)  # type: ignore[arg-type]

        summary_payload = {
            "schema_version": 5,
            "seed_derivation": SEED_DERIVATION_SCHEME,
            "aggregated": {},
        }
        layout.summary_path.parent.mkdir(parents=True, exist_ok=True)
        layout.summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        self.assertEqual(load_summary(layout.summary_path), summary_payload)

        log_path = configure_file_logging_path(layout.runner_log_path)
        self.assertEqual(log_path, layout.runner_log_path)
        self.assertTrue(layout.runner_log_path.exists())

        config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(5),
            fm=FMConfig(0),
            sa=SAConfig(2, 12),
        )
        args = figure4_runner.parse_args(["--output-dir", str(output_dir), "--preset", "test", "--settings", "wo_cgfm"])
        manifest_path = figure4_runner._write_manifest(
            output_layout=layout,
            args=args,
            resolved_config=config,
            seed_list=[0],
            objective_names=["kappa"],
            settings=["wo_cgfm"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_path.name, "figure4_manifest.json")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["figure"], 4)
        self.assertEqual(manifest["seed_derivation"], "mixed_radix_v1")

    def test_runner_checks_requested_cuda_availability(self) -> None:
        from utils import experiment_runtime

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(EnvironmentError, "cuda"):
                experiment_runtime.ensure_compute_device_available("cuda")
            experiment_runtime.ensure_compute_device_available("cpu")
        with self.assertRaisesRegex(ValueError, "device"):
            experiment_runtime.ensure_compute_device_available("tpu")

    def test_runner_owns_objective_selection_contract(self) -> None:
        self.assertEqual(figure4_runner._selected_objectives(None), FIGURE4_OBJECTIVES)
        self.assertEqual(figure4_runner._selected_objectives([]), FIGURE4_OBJECTIVES)
        selected = figure4_runner._selected_objectives(["delta_T", "kappa", "delta_T"])
        self.assertEqual([objective.name for objective in selected], ["kappa", "delta_T"])
        with self.assertRaisesRegex(ValueError, "Unknown objectives"):
            figure4_runner._selected_objectives(["kappa", "not_an_objective"])

        self.assertEqual(FIGURE4_SETTINGS, ("wo_cgfm", "w_cgfm"))
        self.assertEqual(
            inspect.signature(run_figure4_experiment).parameters["settings"].default,
            FIGURE4_SETTINGS,
        )
        self.assertEqual(figure4_runner.__all__, ["main", "parse_args"])
        wildcard_namespace: dict[str, object] = {}
        exec("from utils.figure4_runner import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            {"main", "parse_args"},
        )
        self.assertNotIn("run_figure4_experiment", figure4_runner.__all__)
        self.assertNotIn("run_figure4_experiment", figure4_pipeline.__all__)
        self.assertNotIn("run_single_trajectory", figure4_pipeline.__all__)

        self.assertEqual(plot_figure4.__all__, ["main", "parse_args", "plot_values"])
        plot_namespace: dict[str, object] = {}
        exec("from utils.plot_figure4 import *", plot_namespace)
        self.assertEqual(
            {name for name in plot_namespace if not name.startswith("__")},
            {"main", "parse_args", "plot_values"},
        )

    def test_workflow_documentation_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        document_path = repo_root / "README.md"
        content = document_path.read_text(encoding="utf-8")
        for heading in (
            "## 第一部分：Figure 4 复现",
            "### Figure 4 当前状态",
            "### Figure 4 代码结构",
            "### Figure 4 算法流程",
            "### Figure 4 对象命名与调用规则",
            "### Figure 4 配置规则",
            "### Figure 4 运行方法",
            "### Figure 4 输出规则",
            "### Figure 4 验收规则",
            "## 第二部分：Figure 5 复现",
            "### Figure 5 当前状态",
            "### Figure 5 代码结构",
            "### Figure 5 算法流程",
            "### Figure 5 对象命名与调用规则",
            "### Figure 5 配置规则",
            "### Figure 5 运行方法",
            "### Figure 5 输出规则",
            "### Figure 5 验收规则",
        ):
            self.assertIn(heading, content)
        for obsolete_heading in (
            "### Figure 4 开发情况",
            "### Figure 5 开发情况",
            "### Figure 5 当前代码结构",
            "### Figure 5 实现顺序",
        ):
            self.assertNotIn(obsolete_heading, content)
        for figure4_keyword in ("figure4_experiment_config.py", "figure4_setting_strategies.py"):
            self.assertIn(figure4_keyword, content)
        for figure4_object in (
            "Figure4ExperimentConfig",
            "Figure4Summary",
            "figure4_runner.py",
        ):
            self.assertIn(figure4_object, content)
        for figure5_keyword in ("w_ddts", "wo_ddts", "weighted-sum", "Pareto front", "figure5_runner.py"):
            self.assertIn(figure5_keyword, content)
        for figure5_object in (
            "Figure5ExperimentConfig",
            "Figure5Summary",
            "Figure5ParetoFront",
            "compute_weighted_sum_reference_targets",
        ):
            self.assertIn(figure5_object, content)
        for qubo_contract_keyword in (
            "QuboConfig",
            "max_abs_polynomial_coefficient_v1",
            "--fm-objective-weight",
            "--system-penalty-weight",
            "--one-hot-penalty-weight",
            "不是 JSONL",
            "python -m json.tool",
        ):
            self.assertIn(qubo_contract_keyword, content)
        legacy_document_name = "FIGURE4" + "_WORKFLOW.md"
        self.assertFalse((repo_root / legacy_document_name).exists())
        figure4_section = content.split("## 第二部分：Figure 5 复现", 1)[0]
        self.assertNotIn("run_figure4_experiment", figure4_section)
        self.assertNotIn("程序化调用示例", figure4_section)
        figure5_section = content.split("## 第二部分：Figure 5 复现", 1)[1]
        self.assertNotIn("run_figure5_experiment", figure5_section)
        self.assertNotIn("程序化调用示例", figure5_section)

    def test_torch_fm_forward_shape(self) -> None:
        model = TorchFMRegressor(num_features=8, init_std=0.1)
        x = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1,))

    def test_five_samples_split_into_non_empty_integer_subsets(self) -> None:
        x = np.arange(20, dtype=np.float32).reshape(5, 4)
        y = np.arange(5, dtype=np.float32)

        split = fm_torch.split_train_validation_test(x, y, seed=7)

        self.assertEqual((len(split.train_y), len(split.validation_y), len(split.test_y)), (3, 1, 1))
        combined_targets = np.concatenate((split.train_y, split.validation_y, split.test_y))
        np.testing.assert_array_equal(np.sort(combined_targets), y)

    def test_hparam_tuning_does_not_run_an_extra_final_fit(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        y = np.linspace(-1.0, 1.0, num=6, dtype=np.float32)
        split = fm_torch.split_train_validation_test(x, y, seed=3)

        with mock.patch.object(fm_torch, "_fit_model_once") as fit_mock:
            fixed_hparams = fm_torch.tune_fm_hparams(
                split,
                optuna_trials=0,
                device="cpu",
                seed=3,
            )
        fit_mock.assert_not_called()
        self.assertIsInstance(fixed_hparams, fm_torch.FMHyperParams)

        trial_metrics = (
            {"train_loss": 1.0, "validation_loss": 2.0, "test_loss": 0.0},
            {"train_loss": 1.0, "validation_loss": 0.0, "test_loss": 2.0},
        )
        previous_verbosity = fm_torch.optuna.logging.get_verbosity()
        fm_torch.optuna.logging.set_verbosity(fm_torch.optuna.logging.WARNING)
        try:
            with mock.patch.object(
                fm_torch,
                "_fit_model_once",
                side_effect=[(object(), metrics) for metrics in trial_metrics],
            ) as trial_fit_mock:
                selected_hparams = fm_torch.tune_fm_hparams(
                    split,
                    optuna_trials=2,
                    device="cpu",
                    seed=3,
                )
        finally:
            fm_torch.optuna.logging.set_verbosity(previous_verbosity)
        self.assertEqual(trial_fit_mock.call_count, 2)
        self.assertEqual([call.args[3] for call in trial_fit_mock.call_args_list], [6, 7])
        self.assertEqual(
            [call.kwargs["include_test_loss"] for call in trial_fit_mock.call_args_list],
            [False, False],
        )
        self.assertEqual(selected_hparams, trial_fit_mock.call_args_list[1].args[1])

    def test_trial_fit_does_not_evaluate_test_loss(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        y = np.linspace(-1.0, 1.0, num=6, dtype=np.float32)
        split = fm_torch.split_train_validation_test(x, y, seed=3)
        hparams = fm_torch.FMHyperParams(init_std=0.05, l2_reg_w=1e-4, l2_reg_v=1e-4)

        with (
            mock.patch.object(fm_torch, "FM_MAX_STEPS", 1),
            mock.patch.object(fm_torch, "_evaluate_loss", wraps=fm_torch._evaluate_loss) as evaluate_mock,
        ):
            _, metrics = fm_torch._fit_model_once(
                split,
                hparams,
                "cpu",
                6,
                include_test_loss=False,
            )

        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertIn("validation_loss", metrics)
        self.assertNotIn("test_loss", metrics)

    def test_fit_torch_fm_runs_one_final_fit_after_tuning(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        y = np.linspace(-1.0, 1.0, num=6, dtype=np.float32)
        hparams = fm_torch.FMHyperParams(init_std=0.05, l2_reg_w=1e-4, l2_reg_v=1e-4)
        trained_model = object()
        metrics = {"train_loss": 0.3, "validation_loss": 0.4, "test_loss": 0.5}

        with (
            mock.patch.object(fm_torch, "tune_fm_hparams", return_value=hparams),
            mock.patch.object(
                fm_torch,
                "_fit_model_once",
                return_value=(trained_model, metrics),
            ) as final_fit_mock,
        ):
            model, metadata = fm_torch.fit_torch_fm(
                x,
                y,
                optuna_trials=0,
                device="cpu",
                seed=5,
            )

        final_fit_mock.assert_called_once()
        self.assertEqual(final_fit_mock.call_args.args[3], 8)
        self.assertTrue(final_fit_mock.call_args.kwargs["include_test_loss"])
        self.assertIs(model, trained_model)
        self.assertEqual(metadata["test_loss"], 0.5)
        self.assertEqual(
            metadata["seed_plan"],
            {
                "root": 5,
                "block_size": 4,
                "split": [5, 6],
                "tuner": 7,
                "trials": [],
                "final_fit": 8,
            },
        )

    def test_fm_to_qubo_drops_bias_and_matches_variable_energy(self) -> None:
        model = TorchFMRegressor(num_features=5, init_std=0.1)
        with torch.no_grad():
            model.w0.copy_(torch.tensor([0.7]))
            model.w.copy_(torch.tensor([0.2, -0.1, 0.4, 0.0, 0.3]))
            model.V.copy_(
                torch.tensor(
                    [
                        [0.5, -0.2, 0.1, 0.3, -0.1, 0.0],
                        [0.1, 0.3, -0.2, 0.4, 0.5, -0.2],
                        [-0.4, 0.2, 0.0, 0.1, 0.4, 0.3],
                        [0.0, 0.6, 0.3, -0.2, 0.1, 0.2],
                        [0.3, -0.1, 0.2, 0.0, -0.3, 0.4],
                    ],
                    dtype=torch.float32,
                )
            )
        q, bias = fm_to_qubo(model)
        self.assertEqual(bias, 0.0)

        model_bias = float(model.w0.detach().cpu().item())
        rng = np.random.default_rng(123)
        for _ in range(20):
            bits = rng.integers(0, 2, size=5).astype(np.float32)
            fm_value = float(model(torch.from_numpy(bits[None, :])).item())
            qubo_value = evaluate_qubo_energy(bits, q, bias)
            self.assertAlmostEqual(fm_value - model_bias, qubo_value, places=5)

    def test_qubo_normalization_scale_ignores_constant_bias(self) -> None:
        q = np.array([[2.0, 0.4], [0.4, -1.0]], dtype=np.float64)

        normalized_without_bias, bias_without_bias, scale_without_bias = normalize_qubo_term(
            q, 0.0, normalization_scheme=QUBO_NORMALIZATION_SCHEME
        )
        normalized_with_bias, normalized_bias, scale_with_bias = normalize_qubo_term(
            q, 100.0, normalization_scheme=QUBO_NORMALIZATION_SCHEME
        )

        np.testing.assert_allclose(normalized_without_bias, normalized_with_bias)
        self.assertEqual(scale_without_bias, 2.0)
        self.assertEqual(scale_with_bias, scale_without_bias)
        self.assertEqual(bias_without_bias, 0.0)
        self.assertEqual(normalized_bias, 50.0)

    def test_qubo_normalization_is_independent_of_matrix_storage(self) -> None:
        symmetric_q = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
        upper_triangular_q = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

        normalized_symmetric, _, symmetric_scale = normalize_qubo_term(
            symmetric_q, 0.0, normalization_scheme=QUBO_NORMALIZATION_SCHEME
        )
        normalized_upper, _, upper_scale = normalize_qubo_term(
            upper_triangular_q, 0.0, normalization_scheme=QUBO_NORMALIZATION_SCHEME
        )

        self.assertEqual(symmetric_scale, 1.0)
        self.assertEqual(upper_scale, symmetric_scale)
        bit_vectors = (
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.ones(2),
        )
        for bits in bit_vectors:
            self.assertEqual(
                evaluate_qubo_energy(bits, normalized_symmetric),
                evaluate_qubo_energy(bits, normalized_upper),
            )
        with self.assertRaisesRegex(ValueError, "normalization scheme"):
            normalize_qubo_term(symmetric_q, 0.0, normalization_scheme="unknown")

    def test_zero_value_uses_all_zero_block(self) -> None:
        encoding = create_iteration_encoding(num_levels=10)
        row = build_dataset_row(0, 0, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64))
        features = encode_single_objective_rows([row], encoding)
        self.assertTrue(np.all(features[0, :10] == 0.0))

    def test_bit_swap_curing_enforces_system_constraint(self) -> None:
        continuous = np.array([0.333, 0.333, 0.333, 0.001], dtype=np.float64)
        discrete = prepare_discrete_composition(continuous, num_levels=50)
        self.assertAlmostEqual(float(np.sum(discrete)), 1.0, places=12)
        scaled = discrete * 50
        self.assertTrue(np.allclose(scaled, np.rint(scaled)))

    def test_strategy_dimensions_and_penalty_policy(self) -> None:
        wo_strategy = get_setting_strategy("wo_cgfm")
        w_strategy = get_setting_strategy("w_cgfm")
        self.assertTrue(wo_strategy.include_system_penalty)
        self.assertFalse(w_strategy.include_system_penalty)

        direct_encoding = wo_strategy.create_encoding(10, seed=1)
        cgfm_encoding = w_strategy.create_encoding(10, seed=1)
        self.assertEqual(direct_encoding.num_blocks, 4)
        self.assertEqual(cgfm_encoding.num_blocks, 3)

        direct_q = np.zeros((4 * direct_encoding.num_levels, 4 * direct_encoding.num_levels), dtype=np.float64)
        cgfm_q = np.zeros((3 * cgfm_encoding.num_levels, 3 * cgfm_encoding.num_levels), dtype=np.float64)
        qubo_config = QuboConfig()
        direct_result = build_single_objective_qubo(
            direct_q, 0.0, direct_encoding, qubo_config, include_system_penalty=True
        )
        cgfm_result = build_single_objective_qubo(cgfm_q, 0.0, cgfm_encoding, qubo_config, include_system_penalty=False)

        self.assertEqual(direct_result.stats.num_variables, 40)
        self.assertEqual(direct_result.stats.system_penalty_weight, 650.0)
        self.assertEqual(cgfm_result.stats.num_variables, 30)
        self.assertEqual(cgfm_result.stats.system_penalty_weight, 0.0)

    def test_qubo_config_weights_and_zero_terms_are_reflected_in_stats(self) -> None:
        encoding = create_iteration_encoding(2)
        size = encoding.num_blocks * encoding.num_levels
        fm_q = np.zeros((size, size), dtype=np.float64)
        fm_q[0, 0] = 2.0
        result = build_single_objective_qubo(
            fm_q,
            4.0,
            encoding,
            QuboConfig(3.0, 0.0, 2.0),
            include_system_penalty=True,
        )
        normalized_fm_q, normalized_fm_bias, _ = normalize_qubo_term(
            fm_q,
            4.0,
            normalization_scheme=QUBO_NORMALIZATION_SCHEME,
        )
        one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)
        normalized_one_hot_q, normalized_one_hot_bias, _ = normalize_qubo_term(
            one_hot_q,
            one_hot_bias,
            normalization_scheme=QUBO_NORMALIZATION_SCHEME,
        )
        np.testing.assert_allclose(
            result.q,
            (3.0 * normalized_fm_q) + (2.0 * normalized_one_hot_q),
        )
        self.assertEqual(
            result.bias,
            (3.0 * normalized_fm_bias) + (2.0 * normalized_one_hot_bias),
        )
        self.assertEqual(result.stats.normalization_scheme, QUBO_NORMALIZATION_SCHEME)
        self.assertEqual(result.stats.fm_objective_weight, 3.0)
        self.assertEqual(result.stats.system_penalty_weight, 0.0)
        self.assertEqual(result.stats.system_scale, 0.0)
        self.assertEqual(result.stats.one_hot_penalty_weight, 2.0)

        zero_result = build_single_objective_qubo(
            fm_q,
            4.0,
            encoding,
            QuboConfig(0.0, 0.0, 0.0),
            include_system_penalty=True,
        )
        np.testing.assert_array_equal(zero_result.q, np.zeros_like(fm_q))
        self.assertEqual(zero_result.bias, 0.0)
        self.assertEqual(zero_result.stats.fm_scale, 0.0)
        self.assertEqual(zero_result.stats.system_scale, 0.0)
        self.assertEqual(zero_result.stats.one_hot_scale, 0.0)
        self.assertEqual(zero_result.stats.max_abs, 0.0)

    def test_one_hot_level_weights_follow_paper_eq18_and_are_decodable(self) -> None:
        encoding = create_iteration_encoding(10)
        expected_counts = np.arange(1, 11, dtype=np.int64)
        expected_bit_indices = np.arange(-1, 10, dtype=np.int64)
        for block_idx in range(4):
            np.testing.assert_array_equal(encoding.positive_count_by_bit[block_idx], expected_counts)
            np.testing.assert_array_equal(encoding.bit_index_by_count[block_idx], expected_bit_indices)

        row = build_dataset_row(0, 0, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64))
        bits = encode_single_objective_rows([row], encoding)[0]
        decoded = decode_candidate_bits_to_composition(bits, encoding)
        self.assertIsNotNone(decoded)
        self.assertTrue(np.allclose(decoded, np.array([0.0, 0.3, 0.2, 0.5], dtype=np.float64)))

    def test_cgfm_mapping_roundtrip_and_feature_shape(self) -> None:
        composition = np.array([0.12, 0.28, 0.35, 0.25], dtype=np.float64)
        encoding_a = create_cgfm_iteration_encoding(10, seed=11)
        encoding_b = create_cgfm_iteration_encoding(10, seed=11)
        encoding_c = create_cgfm_iteration_encoding(10, seed=12)
        self.assertEqual(encoding_a.num_blocks, 3)
        self.assertTrue(np.array_equal(encoding_a.phase_permutation, encoding_b.phase_permutation))
        self.assertFalse(np.array_equal(encoding_a.phase_permutation, encoding_c.phase_permutation))
        expected_counts = np.arange(1, 11, dtype=np.int64)
        for encoding in (encoding_a, encoding_b, encoding_c):
            for block_idx in range(encoding.num_blocks):
                np.testing.assert_array_equal(encoding.positive_count_by_bit[block_idx], expected_counts)

        angles = cgfm_composition_to_angles(composition, encoding_a.phase_permutation)
        decoded = cgfm_angles_to_composition(angles, encoding_a.phase_permutation)
        self.assertTrue(np.all(decoded >= -1e-12))
        self.assertAlmostEqual(float(np.sum(decoded)), 1.0, places=12)
        self.assertTrue(np.allclose(decoded, composition, atol=1e-10))

        row = build_dataset_row(0, 0, composition)
        features = encode_cgfm_rows([row], encoding_a)
        self.assertEqual(features.shape, (1, 30))

    def test_cgfm_decode_enforces_one_hot_and_sum_constraint(self) -> None:
        encoding = create_cgfm_iteration_encoding(5, seed=13)
        bits = np.zeros(3 * encoding.num_levels, dtype=np.float64)
        decoded = decode_candidate_bits_to_cgfm_composition(bits, encoding)
        self.assertIsNotNone(decoded)
        self.assertTrue(validate_candidate_composition(decoded))

        invalid = bits.copy()
        invalid[0] = 1.0
        invalid[1] = 1.0
        self.assertIsNone(decode_candidate_bits_to_cgfm_composition(invalid, encoding))

    def test_penalties_match_paper_zero_or_one_hot_semantics(self) -> None:
        encoding = create_iteration_encoding(num_levels=5)
        system_q, system_bias = build_system_penalty_matrix(encoding)
        one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)

        zero_block_row = build_dataset_row(0, 0, np.array([0.0, 0.2, 0.2, 0.6], dtype=np.float64))
        valid = encode_single_objective_rows([zero_block_row], encoding)[0].astype(np.float64)
        decoded_valid = decode_candidate_bits_to_composition(valid, encoding)
        self.assertIsNotNone(decoded_valid)
        self.assertTrue(validate_candidate_composition(decoded_valid))
        self.assertAlmostEqual(evaluate_qubo_energy(valid, one_hot_q, one_hot_bias), 0.0, places=12)

        invalid_one_hot = valid.copy()
        first_active = int(np.flatnonzero(invalid_one_hot > 0.5)[0])
        block_start = (first_active // encoding.num_levels) * encoding.num_levels
        invalid_index = block_start + ((first_active + 1) % encoding.num_levels)
        if invalid_index == first_active:
            invalid_index = block_start + ((first_active + 2) % encoding.num_levels)
        invalid_one_hot[invalid_index] = 1.0
        self.assertIsNone(decode_candidate_bits_to_composition(invalid_one_hot, encoding))
        self.assertGreater(evaluate_qubo_energy(invalid_one_hot, one_hot_q, one_hot_bias), 0.0)

        infeasible = valid.copy()
        infeasible[np.flatnonzero(infeasible > 0.5)[-1]] = 0.0
        decoded_infeasible = decode_candidate_bits_to_composition(infeasible, encoding)
        self.assertIsNotNone(decoded_infeasible)
        self.assertFalse(validate_candidate_composition(decoded_infeasible))

        feasible_row = build_dataset_row(1, 0, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        feasible = encode_single_objective_rows([feasible_row], encoding)[0].astype(np.float64)
        self.assertLess(
            evaluate_qubo_energy(feasible, system_q, system_bias),
            evaluate_qubo_energy(infeasible, system_q, system_bias),
        )

    def test_run_single_trajectory_grows_dataset(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=12, seed=3)
        config = Figure4ExperimentConfig(
            num_samples=12,
            iterations=2,
            encoding=EncodingConfig(num_levels=10),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=20, sweeps=6),
        )
        result = run_single_trajectory(rows, FIGURE4_OBJECTIVES[0], seed=3, config=config, setting="wo_cgfm")
        self.assertEqual(len(result.best_so_far), 2)
        self.assertEqual(result.final_dataset_size, 14)
        initial_best = max(
            float(
                build_dataset_row(
                    int(row["sample_id"]),
                    int(row["seed"]),
                    prepare_discrete_composition(
                        np.array([row["f1_norm"], row["f2_norm"], row["f3_norm"], row["f4_norm"]], dtype=np.float64),
                        10,
                    ),
                )["kappa"]
            )
            for row in rows
        )
        self.assertGreaterEqual(result.best_so_far[0], initial_best)

    def test_run_single_cgfm_trajectory_creates_valid_result(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=8, seed=7)
        config = Figure4ExperimentConfig(
            num_samples=8,
            iterations=1,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        q = np.zeros((3 * config.num_levels, 3 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("utils.figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("utils.figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch(
                "utils.figure4_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(np.zeros(3 * config.num_levels)),
            ),
        ):
            result = run_single_trajectory(rows, FIGURE4_OBJECTIVES[0], seed=7, config=config, setting="w_cgfm")
        self.assertEqual(result.setting, "w_cgfm")
        self.assertEqual(result.final_dataset_size, 9)
        self.assertEqual(result.completed_iterations, 1)
        self.assertIsNotNone(result.qubo_stats)
        self.assertEqual(result.qubo_stats.num_variables, 24)
        self.assertEqual(result.qubo_stats.system_penalty_weight, 0.0)

    def test_aggregate_includes_best_so_far_dispersion(self) -> None:
        results = [
            TrajectoryResult("kappa", "wo_cgfm", 0, [1.0, 3.0, 5.0], 3, "mock", {}, completed_iterations=3),
            TrajectoryResult("kappa", "wo_cgfm", 1, [3.0, 5.0, 7.0], 3, "mock", {}, completed_iterations=3),
        ]
        aggregated = aggregate_trajectory_results(results)
        stats = aggregated["kappa:wo_cgfm"]
        self.assertEqual(stats.best_so_far_mean, [2.0, 4.0, 6.0])
        self.assertEqual(stats.best_so_far_min, [1.0, 3.0, 5.0])
        self.assertEqual(stats.best_so_far_max, [3.0, 5.0, 7.0])
        self.assertEqual(stats.best_so_far_std, [1.0, 1.0, 1.0])
        self.assertEqual(stats.num_trajectories, 2)

    def test_infeasible_sa_samples_are_skipped_and_duplicates_are_replaced(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=8)
        config = Figure4ExperimentConfig(
            num_samples=6,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=1, sweeps=1),
        )
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)

        encoding = create_iteration_encoding(config.num_levels, num_blocks=4)
        valid_bits = encode_single_objective_rows(
            [build_dataset_row(0, 8, [1.0, 0.0, 0.0, 0.0])],
            encoding,
        )[0].astype(np.float64)
        with (
            mock.patch("utils.figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("utils.figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch(
                "utils.figure4_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(np.zeros(4 * config.num_levels), valid_bits),
            ),
            mock.patch("utils.figure4_pipeline.diagnose_no_feasible_candidate") as diagnostic_mock,
        ):
            selected_result = run_single_trajectory(
                rows, FIGURE4_OBJECTIVES[0], seed=8, config=config, setting="wo_cgfm"
            )
        diagnostic_mock.assert_not_called()
        self.assertEqual(selected_result.duplicate_replacements, 0)
        self.assertEqual(selected_result.accepted_sa_candidates, 1)
        self.assertEqual(selected_result.random_replacements, 0)
        self.assertEqual(selected_result.infeasible_sa_samples_skipped, 1)
        self.assertEqual(selected_result.max_feasible_candidate_rank, 2)

        duplicate_composition = discretize_rows_for_figure4(rows, config.num_levels)[0]
        duplicate_array = np.array(
            [
                duplicate_composition["f1_norm"],
                duplicate_composition["f2_norm"],
                duplicate_composition["f3_norm"],
                duplicate_composition["f4_norm"],
            ],
            dtype=np.float64,
        )
        with (
            mock.patch("utils.figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("utils.figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch(
                "utils.figure4_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(np.zeros(4 * config.num_levels)),
            ),
            mock.patch(
                "utils.figure4_setting_strategies.WOCGFMStrategy.decode_candidate", return_value=duplicate_array
            ),
        ):
            duplicate_result = run_single_trajectory(
                rows, FIGURE4_OBJECTIVES[0], seed=8, config=config, setting="wo_cgfm"
            )
        self.assertEqual(duplicate_result.duplicate_replacements, 1)
        self.assertEqual(duplicate_result.accepted_sa_candidates, 0)
        self.assertEqual(duplicate_result.random_replacements, 1)

    def test_empty_feasible_sa_batch_reports_trajectory_and_constraint_counts(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=8)
        config = Figure4ExperimentConfig(
            num_samples=6,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=1, sweeps=7),
        )
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)
        # 全零状态满足每个 block 至多一个 bit，但四相之和为 0，不是可行 composition。
        one_hot_only = np.zeros(4 * config.num_levels, dtype=np.float64)
        with (
            mock.patch("utils.figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("utils.figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch(
                "utils.figure4_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(one_hot_only),
            ),
            mock.patch(
                "utils.figure4_pipeline.diagnose_no_feasible_candidate",
                wraps=figure4_pipeline.diagnose_no_feasible_candidate,
            ) as diagnostic_mock,
        ):
            with self.assertRaises(RuntimeError) as raised:
                run_single_trajectory(
                    rows,
                    FIGURE4_OBJECTIVES[0],
                    seed=8,
                    config=config,
                    setting="wo_cgfm",
                )
        diagnostic_mock.assert_called_once()
        self.assertTrue(diagnostic_mock.call_args.kwargs["require_unit_sum"])
        detail = str(raised.exception)
        self.assertRegex(
            detail,
            (
                "setting='wo_cgfm'.*objective='kappa'.*seed=8.*iteration=1/1.*"
                "one_hot_valid=1/1.*fully_feasible=0/1.*num_levels=5.*sa_reads=1.*"
                "sa_sweeps=7.*classification=heuristic_sampling_failure.*"
                "lowest_sampled_infeasible_energy=.*exact_best_feasible_energy=.*"
                "exact_feasible_state_count=56.*unchanged --resume"
            ),
        )

    def test_random_replacement_is_forced_to_be_novel(self) -> None:
        duplicate = np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64)
        unique = np.array([0.0, 0.2, 0.2, 0.6], dtype=np.float64)
        rows = [
            build_dataset_row(0, 11, duplicate),
            build_dataset_row(1, 11, np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float64)),
        ]
        config = Figure4ExperimentConfig(
            num_samples=2,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=1, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "novel_replacement"
        checkpoint = figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 11)
        q = np.zeros((4 * config.num_levels, 4 * config.num_levels), dtype=np.float64)
        with (
            mock.patch("utils.figure4_pipeline.fit_torch_fm", return_value=(object(), {"mock": True})),
            mock.patch("utils.figure4_pipeline.fm_to_qubo", return_value=(q, 0.0)),
            mock.patch(
                "utils.figure4_pipeline.solve_qubo_with_sa",
                return_value=_sampling_result(np.zeros(4 * config.num_levels)),
            ),
            mock.patch("utils.figure4_setting_strategies.WOCGFMStrategy.decode_candidate", return_value=duplicate),
            mock.patch(
                "utils.figure4_pipeline.sample_single_objective_design",
                side_effect=[tuple(duplicate), tuple(unique)],
            ),
        ):
            result = run_single_trajectory(
                rows,
                FIGURE4_OBJECTIVES[0],
                seed=11,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
            )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        final_row = payload["state"]["rows"][-1]
        self.assertEqual(result.random_replacements, 1)
        self.assertEqual(payload["state"]["random_replacement_draws"], 2)
        self.assertEqual(
            [final_row["f1_norm"], final_row["f2_norm"], final_row["f3_norm"], final_row["f4_norm"]],
            unique.tolist(),
        )

    def test_resume_rejects_inconsistent_completed_state_before_skip(self) -> None:
        seed = 21
        initial_rows, _ = generate_initial_dataset_single_objective(num_samples=4, seed=seed)
        config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=2, sweeps=12),
        )
        prepared_rows = discretize_rows_for_figure4(initial_rows, config.num_levels)
        added_row = build_dataset_row(4, seed, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        rows = [*prepared_rows, added_row]
        state = TrajectoryState(
            rows=rows,
            best_so_far=[max(float(row["kappa"]) for row in rows)],
            fm_metadata=_fm_metadata(config, seed, 0),
            qubo_stats=_qubo_stats(config, 20),
            accepted_sa_candidates=1,
            max_feasible_candidate_rank=1,
        )
        checkpoint = figure4_output_layout(WORKSPACE_TMP_ROOT / "checkpoint_internal_validation").checkpoint_path(
            "wo_cgfm", "kappa", seed
        )
        valid_payload = checkpoint_payload(
            objective=FIGURE4_OBJECTIVES[0],
            setting="wo_cgfm",
            seed=seed,
            state=state,
            config=config,
        )

        corruptions = (
            (
                "missing state field",
                lambda payload: payload["state"].pop("random_replacement_draws"),
                "missing fields.*random_replacement_draws",
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
                "best curve",
                lambda payload: payload["state"]["best_so_far"].__setitem__(
                    0, payload["state"]["best_so_far"][0] + 1.0
                ),
                "best_so_far",
            ),
            (
                "candidate accounting",
                lambda payload: payload["state"].__setitem__("accepted_sa_candidates", 0),
                "accepted_sa_candidates",
            ),
            (
                "SA rank",
                lambda payload: payload["state"].__setitem__("max_feasible_candidate_rank", 3),
                "max_feasible_candidate_rank",
            ),
            (
                "FM seed plan",
                lambda payload: payload["state"]["fm_metadata"]["seed_plan"].__setitem__(
                    "root", payload["state"]["fm_metadata"]["seed_plan"]["root"] + 1
                ),
                "seed_plan",
            ),
            (
                "QUBO variable count",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("num_variables", 19),
                "num_variables",
            ),
            (
                "QUBO normalization scheme",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("normalization_scheme", "unknown"),
                "normalization_scheme",
            ),
            (
                "QUBO normalization scheme type",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("normalization_scheme", 1),
                "normalization_scheme.*string",
            ),
            (
                "missing QUBO field",
                lambda payload: payload["state"]["qubo_stats"].pop("one_hot_penalty_weight"),
                "missing.*one_hot_penalty_weight",
            ),
            (
                "QUBO objective weight",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("fm_objective_weight", 2.0),
                "fm.*weight",
            ),
            (
                "QUBO objective weight type",
                lambda payload: payload["state"]["qubo_stats"].__setitem__("fm_objective_weight", "1.0"),
                "fm_objective_weight.*finite number",
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
                with mock.patch("utils.figure4_pipeline._fit_and_solve_iteration") as fit_mock:
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        run_single_trajectory(
                            initial_rows,
                            FIGURE4_OBJECTIVES[0],
                            seed=seed,
                            config=config,
                            setting="wo_cgfm",
                            checkpoint_path=checkpoint,
                            resume=True,
                        )
                    fit_mock.assert_not_called()

    def test_resume_skips_completed_and_continues_partial_checkpoint(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=6, seed=9)
        config = Figure4ExperimentConfig(
            num_samples=6,
            iterations=3,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=2, sweeps=1),
        )
        output_dir = WORKSPACE_TMP_ROOT / "resume_output"
        checkpoint = figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 9)
        prepared_rows = discretize_rows_for_figure4(rows, config.num_levels)
        replacement_row = build_dataset_row(len(prepared_rows), 9, np.array([0.2, 0.2, 0.2, 0.4], dtype=np.float64))
        partial_rows = [*prepared_rows, replacement_row]
        partial_best = max(float(row["kappa"]) for row in partial_rows)
        state = TrajectoryState(
            rows=partial_rows,
            best_so_far=[partial_best],
            fm_metadata=_fm_metadata(config, 9, 0),
            qubo_stats=_qubo_stats(config, 20),
            duplicate_replacements=1,
            random_replacements=1,
            random_replacement_draws=1,
            infeasible_sa_samples_skipped=1,
            max_feasible_candidate_rank=2,
        )
        payload = checkpoint_payload(
            objective=FIGURE4_OBJECTIVES[0],
            setting="wo_cgfm",
            seed=9,
            state=state,
            config=config,
        )
        write_checkpoint(checkpoint, payload)
        loaded = load_checkpoint(
            checkpoint,
            objective=FIGURE4_OBJECTIVES[0],
            seed=9,
            config=config,
            setting="wo_cgfm",
        )
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        with mock.patch(
            "utils.figure4_pipeline._fit_and_solve_iteration",
            side_effect=[_training_result(config, 9, 1), _training_result(config, 9, 2)],
        ):
            resumed = run_single_trajectory(
                rows,
                FIGURE4_OBJECTIVES[0],
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
            skipped = run_single_trajectory(
                rows,
                FIGURE4_OBJECTIVES[0],
                seed=9,
                config=config,
                setting="wo_cgfm",
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(resumed.completed_iterations, 3)
        self.assertEqual(len(resumed.best_so_far), 3)
        self.assertEqual(resumed.final_dataset_size, config.num_samples + config.iterations)
        self.assertEqual(resumed.infeasible_sa_samples_skipped, 1)
        self.assertEqual(skipped.completed_iterations, 3)

    def test_checkpoint_config_mismatch_fails(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(num_samples=4, seed=12)
        config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=5),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        output_dir = WORKSPACE_TMP_ROOT / "config_mismatch"
        run_figure4_experiment(
            seed_list=[12],
            config=config,
            output_dir=output_dir,
            objectives=(FIGURE4_OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm",),
        )
        changed_config = Figure4ExperimentConfig(
            num_samples=4,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        with self.assertRaisesRegex(ValueError, "config"):
            run_single_trajectory(
                rows,
                FIGURE4_OBJECTIVES[0],
                seed=12,
                config=changed_config,
                setting="wo_cgfm",
                checkpoint_path=figure4_output_layout(output_dir).checkpoint_path("wo_cgfm", "kappa", 12),
                resume=True,
            )

    def test_experiment_summary_and_plot_generation(self) -> None:
        config = Figure4ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=20, sweeps=6),
        )
        output_dir = WORKSPACE_TMP_ROOT / "figure4_output"
        summary = run_figure4_experiment(
            seed_list=[0],
            config=config,
            output_dir=output_dir,
            objectives=(FIGURE4_OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm", "w_cgfm"),
        )
        layout = figure4_output_layout(output_dir)
        summary_path = layout.summary_path
        output_png = layout.figure_path
        self.assertTrue(summary_path.exists())
        self.assertEqual(summary.schema_version, 5)
        self.assertEqual(summary.training_backend, "pytorch_fm_lbfgs")
        self.assertEqual(summary.settings, ["wo_cgfm", "w_cgfm"])
        self.assertIn("kappa:wo_cgfm", summary.aggregated)
        self.assertIn("kappa:w_cgfm", summary.aggregated)
        self.assertTrue(summary.aggregated["kappa:wo_cgfm"].best_so_far_std)
        subprocess.run(
            [sys.executable, "-m", "utils.plot_figure4", "--summary", str(summary_path), "--output", str(output_png)],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
        )
        self.assertTrue(output_png.exists())
        self.assertTrue(layout.plot_log_path.exists())

    def test_plot_accepts_multiple_summaries(self) -> None:
        config = Figure4ExperimentConfig(
            num_samples=8,
            iterations=1,
            encoding=EncodingConfig(num_levels=6),
            fm=FMConfig(optuna_trials=0),
            sa=SAConfig(reads=64, sweeps=12),
        )
        wo_dir = WORKSPACE_TMP_ROOT / "figure4_multi_plot_wo"
        w_dir = WORKSPACE_TMP_ROOT / "figure4_multi_plot_w"
        run_figure4_experiment(
            seed_list=[1],
            config=config,
            output_dir=wo_dir,
            objectives=(FIGURE4_OBJECTIVES[0],),
            resume=True,
            settings=("wo_cgfm",),
        )
        run_figure4_experiment(
            seed_list=[1],
            config=config,
            output_dir=w_dir,
            objectives=(FIGURE4_OBJECTIVES[0],),
            resume=True,
            settings=("w_cgfm",),
        )
        output_png = WORKSPACE_TMP_ROOT / "figure4_multi_plot.png"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "utils.plot_figure4",
                "--summary",
                str(figure4_output_layout(wo_dir).summary_path),
                str(figure4_output_layout(w_dir).summary_path),
                "--output",
                str(output_png),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
        )
        self.assertTrue(output_png.exists())

    def test_plot_rejects_unsupported_or_empty_summary(self) -> None:
        invalid_dir = WORKSPACE_TMP_ROOT / "figure4_invalid_plot"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        old_schema_summary = invalid_dir / "old_schema.json"
        old_schema_summary.write_text(
            json.dumps({"schema_version": 4, "aggregated": {"kappa:wo_cgfm": {}}}),
            encoding="utf-8",
        )
        empty_summary = invalid_dir / "empty_summary.json"
        empty_summary.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "seed_derivation": SEED_DERIVATION_SCHEME,
                    "aggregated": {},
                }
            ),
            encoding="utf-8",
        )

        for summary_path in (old_schema_summary, empty_summary):
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "utils.plot_figure4",
                        "--summary",
                        str(summary_path),
                        "--output",
                        str(invalid_dir / f"{summary_path.stem}.png"),
                    ],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    unittest.main()
