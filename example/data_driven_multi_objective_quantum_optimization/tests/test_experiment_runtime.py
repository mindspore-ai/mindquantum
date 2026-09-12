from __future__ import annotations

import json
import logging
import subprocess
import sys
import unittest
from pathlib import Path

from utils.experiment_runtime import (
    FIGURE4_INITIAL_NAMESPACE,
    FIGURE4_REPLACEMENT_NAMESPACE,
    FIGURE4_SEED_INDEX,
    FIGURE5_INITIAL_NAMESPACE,
    FIGURE5_PREFERENCE_NAMESPACE,
    FIGURE5_REPLACEMENT_NAMESPACE,
    FIGURE5_SEED_INDEX,
    FM_SEED_MIN,
    NEAL_SEED_MAX,
    NUMPY_SEED_MAX,
    SEED_DERIVATION_SCHEME,
    configure_file_logging_path,
    derive_bounded_seed,
    derive_fm_seed_root,
    derive_python_seed,
    fm_seed_block_size,
    fm_seed_plan,
    require_integer,
    validate_seed_schedule,
    write_json_atomic,
)
from utils.figure4_experiment_config import (
    FIGURE4_PRESET_NUM_SEEDS,
)
from utils.figure4_experiment_config import (
    preset_config as figure4_preset_config,
)
from utils.figure5_experiment_config import (
    FIGURE5_PRESET_NUM_SEEDS,
)
from utils.figure5_experiment_config import (
    preset_config as figure5_preset_config,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test" / "runtime_mixed_radix_v1"


class ExperimentRuntimeTests(unittest.TestCase):
    def test_mixed_radix_schedules_are_unique_for_every_preset(self) -> None:
        layouts = (
            (
                FIGURE4_SEED_INDEX,
                10,
                2,
                1,
                figure4_preset_config,
                FIGURE4_PRESET_NUM_SEEDS,
            ),
            (
                FIGURE5_SEED_INDEX,
                2,
                1,
                3,
                figure5_preset_config,
                FIGURE5_PRESET_NUM_SEEDS,
            ),
        )
        for figure_index, trajectories, streams, models, resolver, seed_counts in layouts:
            for preset, seed_count in seed_counts.items():
                with self.subTest(figure=figure_index + 4, preset=preset):
                    config = resolver(preset, device="cpu")
                    seeds = list(range(seed_count))
                    block_size = fm_seed_block_size(config.optuna_trials)
                    self.assertEqual(
                        validate_seed_schedule(
                            seeds,
                            figure_index=figure_index,
                            trajectory_count=trajectories,
                            iterations=config.iterations,
                            bounded_stream_count=streams,
                            fm_model_count=models,
                            fm_seed_block_size=block_size,
                        ),
                        seeds,
                    )

                    bounded = {
                        derive_bounded_seed(
                            seed,
                            trajectory_count=trajectories,
                            trajectory_index=trajectory,
                            iterations=config.iterations,
                            iteration=iteration,
                            stream_count=streams,
                            stream_index=stream,
                            figure_index=figure_index,
                        )
                        for seed in seeds
                        for trajectory in range(trajectories)
                        for iteration in range(config.iterations)
                        for stream in range(streams)
                    }
                    self.assertEqual(
                        len(bounded),
                        seed_count * trajectories * config.iterations * streams,
                    )
                    self.assertLessEqual(max(bounded), NEAL_SEED_MAX)

                    roots = sorted(
                        derive_fm_seed_root(
                            seed,
                            trajectory_count=trajectories,
                            trajectory_index=trajectory,
                            iterations=config.iterations,
                            iteration=iteration,
                            model_count=models,
                            model_index=model,
                            figure_index=figure_index,
                            block_size=block_size,
                        )
                        for seed in seeds
                        for trajectory in range(trajectories)
                        for iteration in range(config.iterations)
                        for model in range(models)
                    )
                    self.assertEqual(len(roots), len(set(roots)))
                    self.assertTrue(all(left + block_size <= right for left, right in zip(roots, roots[1:])))
                    self.assertGreaterEqual(roots[0], FM_SEED_MIN)
                    self.assertGreater(roots[0], max(bounded))
                    self.assertLessEqual(roots[-1] + block_size - 1, NUMPY_SEED_MAX)

    def test_figure_and_stage_streams_are_isolated_but_intended_streams_are_shared(self) -> None:
        f4_bounded = derive_bounded_seed(
            0,
            trajectory_count=10,
            trajectory_index=0,
            iterations=2,
            iteration=0,
            stream_count=2,
            stream_index=0,
            figure_index=FIGURE4_SEED_INDEX,
        )
        f5_bounded = derive_bounded_seed(
            0,
            trajectory_count=2,
            trajectory_index=0,
            iterations=2,
            iteration=0,
            stream_count=1,
            stream_index=0,
            figure_index=FIGURE5_SEED_INDEX,
        )
        self.assertEqual(f4_bounded % 2, 0)
        self.assertEqual(f5_bounded % 2, 1)
        self.assertNotEqual(f4_bounded, f5_bounded)
        self.assertEqual(
            f4_bounded,
            derive_bounded_seed(
                0,
                trajectory_count=10,
                trajectory_index=0,
                iterations=2,
                iteration=0,
                stream_count=2,
                stream_index=0,
                figure_index=FIGURE4_SEED_INDEX,
            ),
        )

        block_size = fm_seed_block_size(3)
        f4_root = derive_fm_seed_root(
            0,
            trajectory_count=10,
            trajectory_index=0,
            iterations=2,
            iteration=0,
            model_count=1,
            model_index=0,
            figure_index=FIGURE4_SEED_INDEX,
            block_size=block_size,
        )
        f5_root = derive_fm_seed_root(
            0,
            trajectory_count=2,
            trajectory_index=0,
            iterations=2,
            iteration=0,
            model_count=3,
            model_index=0,
            figure_index=FIGURE5_SEED_INDEX,
            block_size=block_size,
        )
        self.assertTrue(set(range(f4_root, f4_root + block_size)).isdisjoint(range(f5_root, f5_root + block_size)))

        initial_f4 = derive_python_seed(FIGURE4_INITIAL_NAMESPACE, 7)
        initial_f5 = derive_python_seed(FIGURE5_INITIAL_NAMESPACE, 7)
        preference_a = derive_python_seed(FIGURE5_PREFERENCE_NAMESPACE, 7, iteration=3)
        preference_b = derive_python_seed(FIGURE5_PREFERENCE_NAMESPACE, 7, trajectory_index=0, iteration=3)
        self.assertEqual(preference_a, preference_b)
        self.assertNotEqual(initial_f4, initial_f5)
        self.assertNotEqual(
            derive_python_seed(FIGURE4_REPLACEMENT_NAMESPACE, 7, trajectory_index=0, iteration=1),
            derive_python_seed(FIGURE4_REPLACEMENT_NAMESPACE, 7, trajectory_index=1, iteration=1),
        )
        self.assertNotEqual(
            derive_python_seed(FIGURE5_REPLACEMENT_NAMESPACE, 7, trajectory_index=0, iteration=1),
            derive_python_seed(FIGURE5_REPLACEMENT_NAMESPACE, 7, trajectory_index=0, iteration=2),
        )

    def test_fm_seed_blocks_assign_every_stage_without_overlap(self) -> None:
        for trials in (0, 3):
            with self.subTest(optuna_trials=trials):
                plan = fm_seed_plan(100, trials)
                seeds = [*plan["split"], plan["tuner"], *plan["trials"], plan["final_fit"]]
                self.assertEqual(plan["block_size"], trials + 4)
                self.assertEqual(len(seeds), trials + 4)
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertEqual(seeds, list(range(100, 100 + trials + 4)))

    def test_schedule_bounds_and_strict_integer_contract(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "derived FM seed block"):
            validate_seed_schedule(
                [100_000_000],
                figure_index=FIGURE4_SEED_INDEX,
                trajectory_count=1,
                iterations=1,
                bounded_stream_count=1,
                fm_model_count=1,
                fm_seed_block_size=24,
            )
        self.assertIs(type(require_integer("value", 3)), int)
        for value in (True, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer"):
                    require_integer("value", value)  # type: ignore[arg-type]

    def test_atomic_json_is_compact_when_requested_and_cleans_failed_temporary_files(self) -> None:
        output = WORKSPACE_TMP_ROOT / "atomic" / "checkpoint.json"
        write_json_atomic(output, {"scheme": SEED_DERIVATION_SCHEME, "value": [1, 2]}, indent=None)
        rendered = output.read_text(encoding="utf-8")
        self.assertNotIn("\n  ", rendered)
        self.assertEqual(json.loads(rendered)["scheme"], SEED_DERIVATION_SCHEME)

        invalid = output.with_name("invalid.json")
        with self.assertRaises(ValueError):
            write_json_atomic(invalid, {"value": float("nan")}, indent=None)
        self.assertFalse(invalid.exists())
        self.assertEqual(list(invalid.parent.glob(f".{invalid.name}.*.tmp")), [])

    def test_logging_replaces_only_the_shared_experiment_handler(self) -> None:
        first = WORKSPACE_TMP_ROOT / "logging" / "first.log"
        second = WORKSPACE_TMP_ROOT / "logging" / "second.log"
        configure_file_logging_path(first)
        configure_file_logging_path(second)
        handlers = [
            handler for handler in logging.getLogger().handlers if getattr(handler, "_experiment_file_handler", False)
        ]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(Path(handlers[0].baseFilename), second.resolve())  # type: ignore[attr-defined]

    def test_dataset_and_validator_import_without_importing_torch(self) -> None:
        command = (
            "import sys; sys.modules['torch'] = None; "
            "from utils import alloy_dataset_generator, validate_reproduction; "
            "assert sys.modules['torch'] is None"
        )
        subprocess.run([sys.executable, "-c", command], check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
