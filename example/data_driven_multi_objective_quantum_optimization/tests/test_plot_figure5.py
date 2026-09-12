from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_figure5 as plot_module
from utils.experiment_runtime import SEED_DERIVATION_SCHEME
from utils.figure5_outputs import figure5_output_layout
from utils.figure5_pareto import pareto_front
from utils.plot_figure5 import (
    FIGURE5_AXIS_OBJECTIVES,
    default_iteration_boundaries,
    load_figure5_plot_data,
    parse_args,
    plot_figure5,
    validate_iteration_boundaries,
)

WORKSPACE_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test" / "plot_figure5"
WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _point(
    setting: str,
    seed: int,
    iteration: int,
    kappa: float,
    e_value: float,
    rho: float,
    composition: list[float] | None = None,
) -> dict[str, object]:
    return {
        "setting": setting,
        "seed": seed,
        "iteration": iteration,
        "status": "accepted",
        "weights": [0.2, 0.3, 0.5],
        "sample_id": 100 + iteration,
        "composition": [0.4, 0.3, 0.2, 0.1] if composition is None else composition,
        "kappa": kappa,
        "E": e_value,
        "rho": rho,
    }


def _summary(
    seed_list: list[int] | None = None,
    iterations: int = 150,
    settings: tuple[str, ...] = ("w_ddts", "wo_ddts"),
) -> dict[str, object]:
    seeds = [0] if seed_list is None else seed_list
    solutions: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for seed in seeds:
        seed_shift = float(seed) * 0.1
        for setting in settings:
            setting_shift = 0.0 if setting == "w_ddts" else -2.0
            points = [
                _point(setting, seed, 0, 100.0 + setting_shift + seed_shift, 70.0, 3.0, [1.0, 0.0, 0.0, 0.0]),
                _point(
                    setting,
                    seed,
                    min(60, iterations - 1),
                    110.0 + setting_shift + seed_shift,
                    80.0,
                    2.8,
                    [0.0, 1.0, 0.0, 0.0],
                ),
                _point(
                    setting,
                    seed,
                    min(120, iterations - 1),
                    120.0 + setting_shift + seed_shift,
                    72.0,
                    2.7,
                    [0.0, 0.0, 1.0, 0.0],
                ),
            ]
            duplicate = dict(points[1])
            duplicate["sample_id"] = 999
            duplicate["status"] = "duplicate_replacement"
            solutions.extend([*points, duplicate])

            replacement_iteration = min(60, iterations - 1)
            trajectories.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "iteration_records": [
                        {
                            "setting": setting,
                            "seed": seed,
                            "iteration": replacement_iteration,
                            "decision_status": "duplicate_replacement",
                            "added_solution": {
                                "sample_id": 500,
                                "composition": [0.25, 0.25, 0.25, 0.25],
                                "kappa": 90.0 + setting_shift,
                                "E": 65.0,
                                "rho": 3.2,
                            },
                        }
                    ],
                }
            )

    pareto_fronts = [
        {
            "setting": setting,
            "seed": seed,
            "solutions": pareto_front(
                [point for point in solutions if point["setting"] == setting and point["seed"] == seed]
            ),
        }
        for seed in seeds
        for setting in settings
    ]
    return {
        "schema_version": 5,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "training_backend": "pytorch_fm_lbfgs",
        "config": {"iterations": iterations},
        "seed_list": seeds,
        "objectives": ["kappa", "E", "rho"],
        "settings": list(settings),
        "trajectories": trajectories,
        "solutions": solutions,
        "pareto_fronts": pareto_fronts,
    }


def _write_summary(name: str, payload: dict[str, object]) -> Path:
    path = WORKSPACE_TMP_ROOT / name / "figure5_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class PlotFigure5Tests(unittest.TestCase):
    def test_paper_axis_order_maps_kappa_rho_and_e(self) -> None:
        point = _point("w_ddts", 0, 0, kappa=101.0, e_value=73.0, rho=2.8)
        x_values, y_values, z_values = plot_module._point_arrays([point])

        self.assertEqual(FIGURE5_AXIS_OBJECTIVES, ("kappa", "rho", "E"))
        np.testing.assert_array_equal(x_values, [101.0])
        np.testing.assert_array_equal(y_values, [2.8])
        np.testing.assert_array_equal(z_values, [73.0])

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        try:
            limits = {"kappa": (90.0, 120.0), "rho": (2.5, 3.2), "E": (65.0, 85.0)}
            plot_module._configure_axis(axis, limits)
            self.assertEqual(axis.get_xlim(), limits["kappa"])
            self.assertEqual(axis.get_ylim(), limits["rho"])
            self.assertEqual(axis.get_zlim(), limits["E"])
            self.assertIn(r"\rho", axis.get_ylabel())
            self.assertIn("$E$", axis.get_zlabel())
        finally:
            plt.close(figure)

    def test_default_and_custom_iteration_boundaries(self) -> None:
        self.assertEqual(default_iteration_boundaries(150), (0, 50, 100, 150))
        self.assertEqual(default_iteration_boundaries(1000), (0, 250, 500, 750, 1000))
        fallback = default_iteration_boundaries(7)
        self.assertEqual((fallback[0], fallback[-1], len(fallback)), (0, 7, 5))
        self.assertTrue(all(right > left for left, right in zip(fallback, fallback[1:])))
        self.assertEqual(validate_iteration_boundaries([0, 25, 75, 150], 150), (0, 25, 75, 150))

        invalid = ([0], [1, 150], [0, 50, 149], [0, 80, 60, 150], [0, 50.5, 150])
        for boundaries in invalid:
            with self.subTest(boundaries=boundaries):
                with self.assertRaises(ValueError):
                    validate_iteration_boundaries(boundaries, 150)

    def test_single_seed_is_automatic_and_stored_front_is_verified(self) -> None:
        summary_path = _write_summary("single_seed", _summary())
        data = load_figure5_plot_data(summary_path)

        self.assertEqual(data.seed, 0)
        self.assertEqual(data.settings, ("w_ddts", "wo_ddts"))
        self.assertEqual(len(data.solutions["w_ddts"]), 4)
        self.assertEqual(len(data.replacements["w_ddts"]), 1)
        self.assertEqual(sum(point["kappa"] == 110.0 for point in data.solutions["w_ddts"]), 2)
        self.assertTrue(all(point["kappa"] != 90.0 for point in data.solutions["w_ddts"]))
        self.assertGreater(len(data.pareto_fronts["w_ddts"]), 0)
        self.assertTrue(all(float(point["kappa"]) >= 110.0 for point in data.pareto_fronts["w_ddts"]))

    def test_multi_seed_requires_explicit_selection(self) -> None:
        summary_path = _write_summary("multi_seed", _summary(seed_list=[0, 1]))
        with self.assertRaisesRegex(ValueError, "--seed"):
            load_figure5_plot_data(summary_path)

        selected = load_figure5_plot_data(summary_path, seed=1)
        self.assertEqual(selected.seed, 1)
        self.assertTrue(all(point["seed"] == 1 for point in selected.solutions["w_ddts"]))
        with self.assertRaisesRegex(ValueError, "choices"):
            load_figure5_plot_data(summary_path, seed=9)

    def test_invalid_summary_data_is_rejected(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        wrong_schema = _summary()
        wrong_schema["schema_version"] = 4
        cases.append(("wrong_schema", wrong_schema, "schema"))

        non_finite = _summary()
        non_finite["solutions"][0]["kappa"] = math.inf  # type: ignore[index]
        cases.append(("non_finite", non_finite, "finite"))

        missing_objective = _summary()
        del missing_objective["solutions"][0]["rho"]  # type: ignore[index]
        cases.append(("missing_objective", missing_objective, "rho"))

        invalid_iteration = _summary()
        invalid_iteration["solutions"][0]["iteration"] = "0"  # type: ignore[index]
        cases.append(("invalid_iteration", invalid_iteration, "integer"))

        mismatched_record = _summary()
        mismatched_record["trajectories"][0]["iteration_records"][0]["seed"] = 3  # type: ignore[index]
        cases.append(("mismatched_record", mismatched_record, "does not match"))

        empty_setting = _summary()
        empty_setting["solutions"] = [
            point
            for point in empty_setting["solutions"]
            if point["setting"] == "w_ddts"  # type: ignore[index]
        ]
        cases.append(("empty_setting", empty_setting, "No proposed"))

        stale_front = _summary()
        stale_front["pareto_fronts"][0]["solutions"] = []  # type: ignore[index]
        cases.append(("stale_front", stale_front, "does not match"))

        stale_front_record = json.loads(json.dumps(_summary()))
        stale_front_record["pareto_fronts"][0]["solutions"][0]["status"] = "duplicate_replacement"
        cases.append(("stale_front_record", stale_front_record, "does not match exactly"))

        duplicate_front = _summary()
        duplicate_front["pareto_fronts"].append(duplicate_front["pareto_fronts"][0])  # type: ignore[index]
        cases.append(("duplicate_front", duplicate_front, "duplicate Pareto front"))

        for name, payload, message in cases:
            with self.subTest(name=name):
                path = _write_summary(name, payload)
                with self.assertRaisesRegex(ValueError, message):
                    load_figure5_plot_data(path)

    def test_single_setting_summary_can_render(self) -> None:
        summary_path = _write_summary(
            "single_setting",
            _summary(settings=("w_ddts",)),
        )
        data = load_figure5_plot_data(summary_path)
        output_path = summary_path.parent / "figure5_w_ddts.png"

        self.assertEqual(data.settings, ("w_ddts",))
        self.assertEqual(tuple(data.solutions), ("w_ddts",))
        self.assertEqual(plot_figure5(summary_path, output_path), output_path)
        self.assertGreater(output_path.stat().st_size, 10_000)

    def test_cli_parses_seed_replacements_and_boundaries(self) -> None:
        args = parse_args(
            [
                "--summary",
                "summary.json",
                "--output",
                "figure.png",
                "--seed",
                "4",
                "--include-replacements",
                "--iteration-boundaries",
                "0",
                "50",
                "150",
            ]
        )
        self.assertEqual(args.seed, 4)
        self.assertTrue(args.include_replacements)
        self.assertEqual(args.iteration_boundaries, [0, 50, 150])

    def test_synthetic_summary_generates_nonblank_png_and_log(self) -> None:
        summary_path = _write_summary("render", _summary())
        output_path = summary_path.parent / "figure5.png"
        result = plot_figure5(summary_path, output_path, include_replacements=True)
        without_replacements = plot_figure5(
            summary_path,
            summary_path.parent / "figure5_without_replacements.png",
        )

        self.assertEqual(result, output_path)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 10_000)
        image = np.asarray(mpimg.imread(output_path), dtype=np.float64)
        self.assertGreater(image.shape[0], 500)
        self.assertGreater(image.shape[1], 500)
        self.assertGreater(float(np.std(image)), 0.02)
        self.assertNotEqual(output_path.read_bytes(), without_replacements.read_bytes())
        self.assertTrue(figure5_output_layout(output_path.parent).plot_log_path.exists())

        with self.assertRaisesRegex(ValueError, "png"):
            plot_figure5(summary_path, output_path.with_suffix(".jpg"))


if __name__ == "__main__":
    unittest.main()
