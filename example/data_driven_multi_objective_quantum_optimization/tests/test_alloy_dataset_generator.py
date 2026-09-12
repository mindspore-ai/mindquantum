import csv
import math
import random
import unittest
from pathlib import Path

import numpy as np
from utils.alloy_dataset_generator import (
    CSV_FIELDNAMES,
    DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES,
    build_dataset_row,
    compute_dataset_statistics,
    compute_delta_t,
    compute_density,
    compute_properties,
    generate_initial_dataset_batch,
    generate_initial_dataset_multi_objective,
    generate_initial_dataset_multi_objective_batch,
    generate_initial_dataset_single_objective,
    normalized_to_volume_fractions,
    sample_multi_objective_design,
    sample_single_objective_design,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test"
WORKSPACE_TMP_ROOT.mkdir(exist_ok=True)


class AlloyDatasetGeneratorTests(unittest.TestCase):
    def test_sampling_produces_feasible_normalized_compositions(self) -> None:
        rng = random.Random(123)
        for _ in range(200):
            design = sample_single_objective_design(rng)
            self.assertTrue(all(value >= 0.0 for value in design))
            self.assertAlmostEqual(sum(design), 1.0, places=12)

    def test_multi_objective_sampling_produces_feasible_normalized_compositions(self) -> None:
        rng = random.Random(456)
        for _ in range(200):
            design = sample_multi_objective_design(rng)
            self.assertEqual(len(design), 4)
            self.assertTrue(all(value >= 0.0 for value in design))
            self.assertAlmostEqual(sum(design), 1.0, places=12)

    def test_generated_rows_preserve_volume_balance(self) -> None:
        rows, stats = generate_initial_dataset_single_objective(num_samples=100, seed=7)
        self.assertEqual(len(rows), 100)
        self.assertLessEqual(float(stats["max_sum_deviation"]), 1e-12)

        for row in rows:
            total_norm = sum(float(row[field]) for field in ("f1_norm", "f2_norm", "f3_norm", "f4_norm"))
            total_secondary = sum(float(row[field]) for field in ("f1_vol", "f2_vol", "f3_vol", "f4_vol"))
            self.assertAlmostEqual(total_norm, 1.0, places=12)
            self.assertAlmostEqual(total_secondary, 0.2, places=12)
            self.assertAlmostEqual(float(row["f_Al_vol"]) + total_secondary, 1.0, places=12)
            self.assertGreaterEqual(float(row["delta_alpha"]), 0.0)
            self.assertGreaterEqual(float(row["delta_T"]), 0.0)

    def test_multi_objective_generated_rows_preserve_volume_balance(self) -> None:
        rows, stats = generate_initial_dataset_multi_objective(seed=7)
        self.assertEqual(len(rows), DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES)
        self.assertEqual(stats["num_samples"], DEFAULT_MULTI_OBJECTIVE_NUM_SAMPLES)
        self.assertLessEqual(float(stats["max_sum_deviation"]), 1e-12)

        for row in rows:
            total_norm = sum(float(row[field]) for field in ("f1_norm", "f2_norm", "f3_norm", "f4_norm"))
            total_secondary = sum(float(row[field]) for field in ("f1_vol", "f2_vol", "f3_vol", "f4_vol"))
            self.assertAlmostEqual(total_norm, 1.0, places=12)
            self.assertAlmostEqual(total_secondary, 0.2, places=12)
            self.assertAlmostEqual(float(row["f_Al_vol"]) + total_secondary, 1.0, places=12)
            self.assertGreaterEqual(float(row["delta_alpha"]), 0.0)
            self.assertGreaterEqual(float(row["delta_T"]), 0.0)

    def test_density_matches_weighted_average(self) -> None:
        normalized = (0.25, 0.25, 0.25, 0.25)
        volume = normalized_to_volume_fractions(normalized)
        expected = (0.8 * 2.7) + (0.05 * 2.3) + (0.05 * 2.0) + (0.05 * 4.0) + (0.05 * 4.4)
        self.assertAlmostEqual(compute_density(volume), expected, places=12)

    def test_property_computation_stays_finite_for_extreme_designs(self) -> None:
        extreme_designs = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
        for normalized in extreme_designs:
            volume = normalized_to_volume_fractions(normalized)
            props = compute_properties(volume)
            for value in props.values():
                self.assertTrue(value == value)  # NaN check
                self.assertNotEqual(value, float("inf"))
                self.assertNotEqual(value, float("-inf"))

    def test_generation_is_deterministic_and_csv_is_written(self) -> None:
        rows_a, stats_a = generate_initial_dataset_single_objective(num_samples=10, seed=11)
        rows_b, stats_b = generate_initial_dataset_single_objective(num_samples=10, seed=11)
        rows_c, _ = generate_initial_dataset_single_objective(num_samples=10, seed=12)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(stats_a, stats_b)
        self.assertNotEqual(rows_a, rows_c)

        output_dir = WORKSPACE_TMP_ROOT / "dataset_generator_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset.csv"
        generate_initial_dataset_single_objective(num_samples=5, seed=22, output_path=output_path)
        self.assertTrue(output_path.exists())
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, CSV_FIELDNAMES)
            rows = list(reader)
        self.assertEqual(len(rows), 5)

    def test_multi_objective_generation_is_deterministic_and_csv_is_written(self) -> None:
        rows_a, stats_a = generate_initial_dataset_multi_objective(num_samples=10, seed=11)
        rows_b, stats_b = generate_initial_dataset_multi_objective(num_samples=10, seed=11)
        rows_c, _ = generate_initial_dataset_multi_objective(num_samples=10, seed=12)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(stats_a, stats_b)
        self.assertNotEqual(rows_a, rows_c)

        output_dir = WORKSPACE_TMP_ROOT / "multi_objective_dataset_generator_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset.csv"
        generate_initial_dataset_multi_objective(num_samples=5, seed=22, output_path=output_path)
        self.assertTrue(output_path.exists())
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, CSV_FIELDNAMES)
            rows = list(reader)
        self.assertEqual(len(rows), 5)

    def test_statistics_shape_matches_rows(self) -> None:
        rows, stats = generate_initial_dataset_single_objective(num_samples=20, seed=5)
        recomputed = compute_dataset_statistics(rows)
        self.assertEqual(stats, recomputed)
        self.assertEqual(stats["num_samples"], 20)
        self.assertIn("fraction_ranges", stats)
        self.assertIn("metric_statistics", stats)

    def test_integer_inputs_are_strict_and_numpy_integers_are_normalized(self) -> None:
        rows, _ = generate_initial_dataset_single_objective(
            num_samples=np.int64(2),
            seed=np.int64(3),
        )
        self.assertEqual(len(rows), 2)
        self.assertIs(type(rows[0]["sample_id"]), int)
        self.assertIs(type(rows[0]["seed"]), int)

        for invalid in (True, 2.0, "2"):
            with self.subTest(parameter="num_samples", value=invalid):
                with self.assertRaisesRegex(ValueError, "num_samples"):
                    generate_initial_dataset_single_objective(num_samples=invalid, seed=0)  # type: ignore[arg-type]
            with self.subTest(parameter="seed", value=invalid):
                with self.assertRaises(ValueError):
                    generate_initial_dataset_single_objective(num_samples=1, seed=invalid)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "sample_id"):
            build_dataset_row(1.0, 0, [1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]

    def test_dataset_batches_reject_empty_duplicate_and_invalid_seeds(self) -> None:
        for generator in (generate_initial_dataset_batch, generate_initial_dataset_multi_objective_batch):
            with self.subTest(generator=generator.__name__, case="empty"):
                with self.assertRaisesRegex(ValueError, "empty"):
                    generator([], num_samples=1)
            with self.subTest(generator=generator.__name__, case="duplicates"):
                with self.assertRaisesRegex(ValueError, "duplicates"):
                    generator([1, 1], num_samples=1)
            with self.subTest(generator=generator.__name__, case="type"):
                with self.assertRaisesRegex(ValueError, "integers"):
                    generator([1.0], num_samples=1)  # type: ignore[list-item]

    def test_material_inputs_require_valid_four_fraction_compositions(self) -> None:
        invalid_normalized = (
            [0.5, 0.5, 0.0],
            [math.nan, 0.2, 0.3, 0.5],
            [math.inf, 0.0, 0.0, 0.0],
            [-0.1, 0.2, 0.3, 0.6],
            [0.1, 0.2, 0.3, 0.3],
            [True, 0.0, 0.0, 0.0],
        )
        for composition in invalid_normalized:
            with self.subTest(composition=composition):
                with self.assertRaises(ValueError):
                    normalized_to_volume_fractions(composition)

        invalid_volume = (
            [0.05, 0.05, 0.05],
            [math.nan, 0.05, 0.05, 0.1],
            [-0.01, 0.05, 0.06, 0.1],
            [0.01, 0.02, 0.03, 0.04],
        )
        for composition in invalid_volume:
            with self.subTest(volume=composition):
                with self.assertRaises(ValueError):
                    compute_properties(composition)

        for invalid_si in (True, "0.1", math.nan, math.inf, -0.01, 0.21):
            with self.subTest(si_fraction=invalid_si):
                with self.assertRaisesRegex(ValueError, "si_fraction_volume"):
                    compute_delta_t(invalid_si)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
