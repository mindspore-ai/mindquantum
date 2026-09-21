import math
import random
import unittest

import numpy as np
from utils import figure5_scalarization as scalarization
from utils.figure5_scalarization import (
    FIGURE5_OBJECTIVE_SENSES,
    FIGURE5_OBJECTIVES,
    compute_ddts_targets,
    compute_individual_objective_targets,
    compute_weighted_sum_reference_targets,
    sample_preference_weights,
    validate_preference_weights,
)


class Figure5ScalarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"kappa": 10.0, "E": 100.0, "rho": 3.0},
            {"kappa": 20.0, "E": 90.0, "rho": 2.5},
            {"kappa": 15.0, "E": 110.0, "rho": 2.8},
        ]

    def test_public_objective_contract(self) -> None:
        self.assertEqual(FIGURE5_OBJECTIVES, ("kappa", "E", "rho"))
        self.assertEqual(
            FIGURE5_OBJECTIVE_SENSES,
            {"kappa": "maximize", "E": "maximize", "rho": "minimize"},
        )

    def test_weighted_sum_single_objective_weights_follow_objective_direction(self) -> None:
        kappa_only = compute_weighted_sum_reference_targets(self.rows, [1.0, 0.0, 0.0])
        e_only = compute_weighted_sum_reference_targets(self.rows, [0.0, 1.0, 0.0])
        rho_only = compute_weighted_sum_reference_targets(self.rows, [0.0, 0.0, 1.0])

        self.assertEqual(int(np.argmin(kappa_only.targets)), 1)
        self.assertEqual(int(np.argmin(e_only.targets)), 2)
        self.assertEqual(int(np.argmin(rho_only.targets)), 1)
        self.assertEqual(kappa_only.method, "weighted_sum")
        self.assertEqual(kappa_only.setting, "wo_ddts")
        self.assertEqual(kappa_only.targets.dtype, np.float32)

    def test_weighted_sum_matches_manual_zscore_dot_product(self) -> None:
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        result = compute_weighted_sum_reference_targets(self.rows, weights)

        raw = np.array([[10.0, 100.0, 3.0], [20.0, 90.0, 2.5], [15.0, 110.0, 2.8]], dtype=np.float64)
        transformed = np.column_stack((-raw[:, 0], -raw[:, 1], raw[:, 2]))
        means = np.mean(transformed, axis=0)
        scales = np.std(transformed, axis=0)
        expected = np.dot((transformed - means) / scales, weights).astype(np.float32)

        self.assertTrue(np.allclose(result.targets, expected))
        self.assertEqual(result.metadata["transformed_objectives"], {"kappa": "-kappa", "E": "-E", "rho": "rho"})
        self.assertEqual(result.metadata["zscore_mean"]["kappa"], -15.0)
        self.assertEqual(result.metadata["zscore_mean"]["E"], -100.0)
        self.assertAlmostEqual(result.metadata["zscore_mean"]["rho"], 2.7666666666666666)

    def test_individual_targets_are_direction_consistent_and_standardized(self) -> None:
        result = compute_individual_objective_targets(self.rows)

        self.assertEqual(int(np.argmin(result.targets["kappa"])), 1)
        self.assertEqual(int(np.argmin(result.targets["E"])), 2)
        self.assertEqual(int(np.argmin(result.targets["rho"])), 1)
        for objective in FIGURE5_OBJECTIVES:
            self.assertAlmostEqual(float(np.mean(result.targets[objective])), 0.0, places=6)
            self.assertAlmostEqual(float(np.std(result.targets[objective])), 1.0, places=6)

    def test_ddts_matches_paper_zscore_utopian_tchebycheff(self) -> None:
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        result = compute_ddts_targets(self.rows, weights)

        raw = np.array([[10.0, 100.0, 3.0], [20.0, 90.0, 2.5], [15.0, 110.0, 2.8]], dtype=np.float64)
        normalized = (raw - np.mean(raw, axis=0)) / np.std(raw, axis=0)
        utopian = np.array(
            [1.1 * np.max(normalized[:, 0]), 1.1 * np.max(normalized[:, 1]), 1.1 * np.min(normalized[:, 2])]
        )
        expected_distances = np.column_stack(
            (
                utopian[0] - normalized[:, 0],
                utopian[1] - normalized[:, 1],
                normalized[:, 2] - utopian[2],
            )
        )
        expected = np.max(expected_distances * weights, axis=1).astype(np.float32)

        self.assertEqual(result.method, "ddts")
        self.assertEqual(result.setting, "w_ddts")
        self.assertTrue(np.allclose(result.targets, expected))
        self.assertEqual(result.metadata["utopian_space"], "zscore")
        self.assertTrue(
            np.allclose(
                [result.metadata["utopian_point"][objective] for objective in FIGURE5_OBJECTIVES],
                utopian,
            )
        )

    def test_ddts_constant_columns_are_finite(self) -> None:
        rows = [
            {"kappa": 12.0, "E": 70.0, "rho": 2.7},
            {"kappa": 12.0, "E": 70.0, "rho": 2.7},
        ]
        result = compute_ddts_targets(rows, [0.2, 0.3, 0.5])

        self.assertTrue(np.all(np.isfinite(result.targets)))
        self.assertTrue(np.allclose(result.targets, np.zeros(2, dtype=np.float32)))
        self.assertEqual(result.metadata["zscore_scale"], {"kappa": 1.0, "E": 1.0, "rho": 1.0})
        self.assertEqual(result.metadata["utopian_point"], {"kappa": 0.0, "E": 0.0, "rho": 0.0})

    def test_preference_weight_sampling_is_deterministic_and_normalized(self) -> None:
        weights_a = sample_preference_weights(random.Random(123))
        weights_b = sample_preference_weights(random.Random(123))
        weights_c = sample_preference_weights(random.Random(124))

        self.assertEqual(weights_a, weights_b)
        self.assertNotEqual(weights_a, weights_c)
        self.assertEqual(len(weights_a), 3)
        self.assertTrue(all(weight >= 0.0 for weight in weights_a))
        self.assertAlmostEqual(sum(weights_a), 1.0, places=12)

    def test_preference_weight_validation_rejects_invalid_inputs(self) -> None:
        self.assertTrue(np.allclose(validate_preference_weights([0.2, 0.3, 0.5]), [0.2, 0.3, 0.5]))

        invalid_weights = [
            [0.2, 0.8],
            [0.2, -0.1, 0.9],
            [0.2, math.nan, 0.8],
            [0.2, math.inf, 0.8],
            [0.2, 0.3, 0.4],
        ]
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    validate_preference_weights(weights)

        for invalid_count in (0, True, 1.5, "3"):
            with self.subTest(num_objectives=invalid_count):
                with self.assertRaisesRegex(ValueError, "num_objectives"):
                    sample_preference_weights(  # type: ignore[arg-type]
                        random.Random(1), num_objectives=invalid_count
                    )

    def test_weighted_sum_reference_api_cannot_dispatch_training(self) -> None:
        self.assertIn("compute_weighted_sum_reference_targets", scalarization.__all__)
        self.assertNotIn("compute_weighted_sum_targets", scalarization.__all__)
        self.assertNotIn("scalarize_training_targets", scalarization.__all__)
        self.assertFalse(hasattr(scalarization, "scalarize_training_targets"))

    def test_invalid_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            compute_weighted_sum_reference_targets([], [0.2, 0.3, 0.5])
        with self.assertRaisesRegex(ValueError, "missing objective"):
            compute_weighted_sum_reference_targets([{"kappa": 1.0, "E": 2.0}], [0.2, 0.3, 0.5])
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_ddts_targets([{"kappa": 1.0, "E": 2.0, "rho": math.inf}], [0.2, 0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
