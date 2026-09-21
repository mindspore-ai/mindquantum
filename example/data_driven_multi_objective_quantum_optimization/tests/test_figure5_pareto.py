import math
import unittest

from utils.figure5_pareto import is_dominated, pareto_front


class Figure5ParetoTests(unittest.TestCase):
    def test_objective_directions_are_used_for_dominance(self) -> None:
        candidate = {"kappa": 100.0, "E": 70.0, "rho": 3.0}
        better = {"kappa": 101.0, "E": 71.0, "rho": 2.9}
        worse_kappa = {"kappa": 99.0, "E": 71.0, "rho": 2.9}
        worse_e = {"kappa": 101.0, "E": 69.0, "rho": 2.9}
        worse_rho = {"kappa": 101.0, "E": 71.0, "rho": 3.1}

        self.assertTrue(is_dominated(candidate, better))
        self.assertFalse(is_dominated(candidate, worse_kappa))
        self.assertFalse(is_dominated(candidate, worse_e))
        self.assertFalse(is_dominated(candidate, worse_rho))

    def test_identical_objective_points_are_deduplicated_without_changing_dominance(self) -> None:
        point_a = {"kappa": 120.0, "E": 90.0, "rho": 2.7}
        point_b = {"kappa": 120.0, "E": 90.0, "rho": 2.7}

        self.assertFalse(is_dominated(point_a, point_b))
        self.assertFalse(is_dominated(point_b, point_a))
        self.assertEqual(pareto_front([point_a, point_b]), [point_a])

    def test_equal_objectives_with_distinct_compositions_remain_distinct_solutions(self) -> None:
        point_a = {"composition": [1.0, 0.0, 0.0, 0.0], "kappa": 120.0, "E": 90.0, "rho": 2.7}
        point_b = {"composition": [0.0, 1.0, 0.0, 0.0], "kappa": 120.0, "E": 90.0, "rho": 2.7}

        self.assertEqual(pareto_front([point_a, point_b]), [point_a, point_b])

    def test_repeated_composition_keeps_the_first_record(self) -> None:
        first = {
            "iteration": 1,
            "composition": [0.25, 0.25, 0.25, 0.25],
            "kappa": 120.0,
            "E": 90.0,
            "rho": 2.7,
        }
        repeated = {**first, "iteration": 4}

        self.assertEqual(pareto_front([first, repeated]), [first])

    def test_pareto_front_removes_clearly_dominated_points(self) -> None:
        strong = {"id": "strong", "kappa": 130.0, "E": 100.0, "rho": 2.4}
        weak = {"id": "weak", "kappa": 120.0, "E": 90.0, "rho": 2.7}
        very_weak = {"id": "very_weak", "kappa": 110.0, "E": 80.0, "rho": 3.0}

        self.assertEqual(pareto_front([weak, strong, very_weak]), [strong])

    def test_tradeoff_boundary_and_nonconvex_points_are_preserved(self) -> None:
        high_kappa = {"id": "high_kappa", "kappa": 140.0, "E": 80.0, "rho": 3.0}
        high_e = {"id": "high_e", "kappa": 110.0, "E": 115.0, "rho": 2.8}
        low_rho = {"id": "low_rho", "kappa": 105.0, "E": 82.0, "rho": 2.2}
        dominated = {"id": "dominated", "kappa": 100.0, "E": 80.0, "rho": 3.2}

        front = pareto_front([dominated, high_kappa, high_e, low_rho])

        self.assertEqual(front, [high_kappa, high_e, low_rho])
        self.assertFalse(is_dominated(high_kappa, high_e))
        self.assertFalse(is_dominated(high_e, low_rho))
        self.assertFalse(is_dominated(low_rho, high_kappa))

    def test_pareto_front_preserves_input_order_and_metadata(self) -> None:
        point_a = {
            "setting": "w_ddts",
            "seed": 0,
            "iteration": 3,
            "composition": [0.1, 0.2, 0.3, 0.4],
            "kappa": 125.0,
            "E": 85.0,
            "rho": 2.6,
        }
        point_b = {
            "setting": "wo_ddts",
            "seed": 1,
            "iteration": 2,
            "composition": [0.4, 0.3, 0.2, 0.1],
            "kappa": 135.0,
            "E": 75.0,
            "rho": 2.7,
        }

        front = pareto_front([point_a, point_b])

        self.assertEqual(front, [point_a, point_b])
        self.assertIs(front[0], point_a)
        self.assertEqual(front[0]["composition"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(front[1]["setting"], "wo_ddts")

    def test_empty_input_returns_empty_front(self) -> None:
        self.assertEqual(pareto_front([]), [])

    def test_invalid_points_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing objective"):
            is_dominated({"kappa": 1.0, "E": 2.0}, {"kappa": 2.0, "E": 3.0, "rho": 2.5})
        with self.assertRaisesRegex(ValueError, "finite"):
            is_dominated(
                {"kappa": 1.0, "E": 2.0, "rho": math.nan},
                {"kappa": 2.0, "E": 3.0, "rho": 2.5},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            pareto_front([{"kappa": math.inf, "E": 2.0, "rho": 2.5}])


if __name__ == "__main__":
    unittest.main()
