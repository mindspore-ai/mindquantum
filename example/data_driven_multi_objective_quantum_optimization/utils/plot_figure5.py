"""Figure 5 multi-objective summary plotting entry point."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment_runtime import SEED_DERIVATION_SCHEME, configure_file_logging_path
from .figure5_outputs import SUMMARY_SCHEMA_VERSION, figure5_output_layout, load_summary
from .figure5_pareto import pareto_front
from .figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
)

FIGURE5_AXIS_OBJECTIVES = ("kappa", "rho", "E")
SETTING_LABELS = {"w_ddts": "w/ DDTS", "wo_ddts": "w/o DDTS"}
ALL_SOLUTION_COLOR = "#858b94"
PARETO_COLOR = "#1769aa"
REPLACEMENT_COLOR = "#d95f02"
TEMPORAL_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Figure5PlotData:
    """Store figure5 plot data data."""

    seed: int
    iterations: int
    settings: tuple[str, ...]
    solutions: dict[str, list[dict[str, Any]]]
    pareto_fronts: dict[str, list[Mapping[str, Any]]]
    replacements: dict[str, list[dict[str, Any]]]


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the Figure 5 multi-objective reproduction summary.")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help=f"Figure 5 summary schema v{SUMMARY_SCHEMA_VERSION} JSON file.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to plot. Required only when the summary contains multiple seeds.",
    )
    parser.add_argument(
        "--include-replacements",
        action="store_true",
        help="Overlay random replacement designs without adding them to the Pareto front.",
    )
    parser.add_argument(
        "--iteration-boundaries",
        type=int,
        nargs="+",
        default=None,
        help="Custom boundaries including 0 and the configured iteration count.",
    )
    return parser.parse_args(argv)


def default_iteration_boundaries(iterations: int) -> tuple[int, ...]:
    """Return paper/project windows or at most four evenly spaced fallback windows."""
    iterations = _strict_int(iterations, "iterations")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if iterations == 150:
        return (0, 50, 100, 150)
    if iterations == 1000:
        return (0, 250, 500, 750, 1000)

    num_windows = min(4, iterations)
    boundaries = np.linspace(0, iterations, num_windows + 1, dtype=np.int64)
    return tuple(int(value) for value in boundaries)


def validate_iteration_boundaries(
    boundaries: Sequence[int] | None,
    iterations: int,
) -> tuple[int, ...]:
    """Validate the requested Figure 5 plot windows."""
    iterations = _strict_int(iterations, "iterations")
    resolved = (
        default_iteration_boundaries(iterations)
        if boundaries is None
        else tuple(_strict_int(value, f"iteration boundary {index}") for index, value in enumerate(boundaries))
    )
    if len(resolved) < 2:
        raise ValueError("iteration boundaries must contain at least two values")
    if resolved[0] != 0 or resolved[-1] != iterations:
        raise ValueError(f"iteration boundaries must start at 0 and end at {iterations}")
    if any(right <= left for left, right in zip(resolved, resolved[1:])):
        raise ValueError("iteration boundaries must be strictly increasing")
    return resolved


def _selected_seed(seed_list: Any, requested_seed: int | None) -> int:
    if not isinstance(seed_list, list) or not seed_list:
        raise ValueError("Figure 5 summary seed_list must be a non-empty list")
    available = [_strict_int(seed, f"seed_list[{index}]") for index, seed in enumerate(seed_list)]
    if len(set(available)) != len(available):
        raise ValueError("Figure 5 summary seed_list contains duplicates")
    if requested_seed is None:
        if len(available) != 1:
            raise ValueError(f"Summary contains multiple seeds {available}; select one with --seed")
        return available[0]
    requested_seed = _strict_int(requested_seed, "requested seed")
    if requested_seed not in available:
        raise ValueError(f"Seed {requested_seed} is not available; choices: {available}")
    return requested_seed


def _finite_float(point: Mapping[str, Any], key: str, context: str) -> float:
    if key not in point:
        raise ValueError(f"{context} is missing {key!r}")
    raw_value = point[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{context} field {key!r} must be numeric and finite")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"{context} field {key!r} must be finite")
    return value


def _finite_numeric_value(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _iteration_value(point: Mapping[str, Any], iterations: int, context: str) -> int:
    if "iteration" not in point:
        raise ValueError(f"{context} is missing 'iteration'")
    value = _strict_int(point["iteration"], f"{context} iteration")
    if value < 0 or value >= iterations:
        raise ValueError(f"{context} iteration must be in [0, {iterations})")
    return value


def _validated_plot_point(
    point: Mapping[str, Any],
    *,
    setting: str,
    seed: int,
    iteration: int,
    context: str,
) -> dict[str, Any]:
    sample_id = _strict_int(point.get("sample_id"), f"{context} sample_id")
    raw_composition = point.get("composition")
    if not isinstance(raw_composition, list) or len(raw_composition) != 4:
        raise ValueError(f"{context} composition must contain four fractions")
    composition = [
        _finite_numeric_value(value, f"{context} composition[{index}]") for index, value in enumerate(raw_composition)
    ]
    if any(value < 0.0 for value in composition) or not math.isclose(sum(composition), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{context} composition must be non-negative and sum to 1")
    return {
        "setting": setting,
        "seed": int(seed),
        "iteration": int(iteration),
        "sample_id": sample_id,
        "composition": composition,
        "kappa": _finite_float(point, "kappa", context),
        "E": _finite_float(point, "E", context),
        "rho": _finite_float(point, "rho", context),
    }


def load_figure5_plot_data(summary_path: str | Path, seed: int | None = None) -> Figure5PlotData:
    """Validate summary schema and select exactly one seed for plotting."""
    summary = load_summary(summary_path)
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Figure 5 summary schema; expected schema_version {SUMMARY_SCHEMA_VERSION}")
    if summary.get("seed_derivation") != SEED_DERIVATION_SCHEME:
        raise ValueError(f"Unsupported Figure 5 seed_derivation; expected {SEED_DERIVATION_SCHEME!r}")
    if tuple(summary.get("objectives", ())) != FIGURE5_OBJECTIVES:
        raise ValueError(f"Figure 5 objectives must be {FIGURE5_OBJECTIVES}")
    raw_settings = summary.get("settings")
    if not isinstance(raw_settings, list) or not raw_settings:
        raise ValueError("Figure 5 summary settings must be a non-empty list")
    if any(not isinstance(setting, str) for setting in raw_settings):
        raise ValueError("Figure 5 summary settings must contain strings")
    if len(set(raw_settings)) != len(raw_settings) or any(setting not in FIGURE5_SETTINGS for setting in raw_settings):
        raise ValueError(f"Figure 5 summary settings must be a unique subset of {FIGURE5_SETTINGS}")
    selected_settings = tuple(setting for setting in FIGURE5_SETTINGS if setting in raw_settings)
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Figure 5 summary config must contain positive iterations")
    iterations = _strict_int(config.get("iterations"), "Figure 5 config iterations")
    if iterations <= 0:
        raise ValueError("Figure 5 summary config must contain positive iterations")
    selected_seed = _selected_seed(summary.get("seed_list"), seed)

    raw_solutions = summary.get("solutions")
    if not isinstance(raw_solutions, list):
        raise ValueError("Figure 5 summary solutions must be a list")
    available_seeds = [_strict_int(value, f"seed_list[{index}]") for index, value in enumerate(summary["seed_list"])]
    available_seed_set = set(available_seeds)
    all_solutions: dict[tuple[str, int], list[dict[str, Any]]] = {
        (setting, available_seed): [] for available_seed in available_seeds for setting in selected_settings
    }
    raw_solutions_by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = {key: [] for key in all_solutions}
    for index, raw_point in enumerate(raw_solutions):
        context = f"solution {index}"
        if not isinstance(raw_point, dict):
            raise ValueError(f"{context} must be an object")
        setting = str(raw_point.get("setting"))
        if setting not in selected_settings:
            raise ValueError(f"{context} setting {setting!r} is not declared in summary settings")
        point_seed = _strict_int(raw_point.get("seed"), f"{context} seed")
        if point_seed not in available_seed_set:
            raise ValueError(f"{context} seed {point_seed} is not declared in seed_list")
        iteration = _iteration_value(raw_point, iterations, context)
        point = _validated_plot_point(
            raw_point,
            setting=setting,
            seed=point_seed,
            iteration=iteration,
            context=context,
        )
        all_solutions[(setting, point_seed)].append(point)
        raw_solutions_by_key[(setting, point_seed)].append(raw_point)

    for (setting, point_seed), points in all_solutions.items():
        if not points:
            raise ValueError(f"No proposed QUBO solutions for setting {setting!r} and seed {point_seed}")
    solutions = {setting: all_solutions[(setting, selected_seed)] for setting in selected_settings}

    replacements: dict[str, list[dict[str, Any]]] = {setting: [] for setting in selected_settings}
    trajectories = summary.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("Figure 5 summary trajectories must be a list")
    for trajectory_index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            raise ValueError(f"trajectory {trajectory_index} must be an object")
        trajectory_setting = str(trajectory.get("setting"))
        trajectory_seed = _strict_int(trajectory.get("seed"), f"trajectory {trajectory_index} seed")
        if trajectory_setting not in selected_settings or trajectory_seed not in available_seed_set:
            raise ValueError(f"trajectory {trajectory_index} has invalid setting or seed")
        records = trajectory.get("iteration_records")
        if not isinstance(records, list):
            raise ValueError(f"trajectory {trajectory_index} iteration_records must be a list")
        for record_index, record in enumerate(records):
            context = f"trajectory {trajectory_index} record {record_index}"
            if not isinstance(record, dict):
                raise ValueError(f"{context} must be an object")
            record_setting = record.get("setting")
            record_seed = _strict_int(record.get("seed"), f"{context} seed")
            if record_setting != trajectory_setting or record_seed != trajectory_seed:
                raise ValueError(f"{context} setting/seed does not match its trajectory")
            record_iteration = _iteration_value(record, iterations, context)
            if trajectory_seed != selected_seed or record.get("decision_status") != "duplicate_replacement":
                continue
            added_solution = record.get("added_solution")
            if not isinstance(added_solution, dict):
                raise ValueError(f"{context} replacement is missing added_solution")
            replacements[trajectory_setting].append(
                _validated_plot_point(
                    added_solution,
                    setting=trajectory_setting,
                    seed=trajectory_seed,
                    iteration=record_iteration,
                    context=f"{context} added_solution",
                )
            )

    raw_fronts = summary.get("pareto_fronts")
    if not isinstance(raw_fronts, list):
        raise ValueError("Figure 5 summary pareto_fronts must be a list")
    expected_front_keys = set(all_solutions)
    stored_fronts: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for front_index, raw_front in enumerate(raw_fronts):
        context = f"pareto_fronts[{front_index}]"
        if not isinstance(raw_front, dict):
            raise ValueError(f"{context} must be an object")
        front_setting = raw_front.get("setting")
        if type(front_setting) is not str or front_setting not in selected_settings:
            raise ValueError(f"{context} has invalid setting")
        front_seed = _strict_int(raw_front.get("seed"), f"{context} seed")
        front_key = (front_setting, front_seed)
        if front_key not in expected_front_keys:
            raise ValueError(f"{context} setting/seed is not declared by the summary")
        if front_key in stored_fronts:
            raise ValueError(f"duplicate Pareto front for setting {front_setting!r} and seed {front_seed}")
        raw_front_solutions = raw_front.get("solutions")
        if not isinstance(raw_front_solutions, list):
            raise ValueError(f"{context}.solutions must be a list")
        validated_front: list[dict[str, Any]] = []
        for point_index, raw_point in enumerate(raw_front_solutions):
            point_context = f"{context}.solutions[{point_index}]"
            if not isinstance(raw_point, dict):
                raise ValueError(f"{point_context} must be an object")
            if raw_point.get("setting") != front_setting:
                raise ValueError(f"{point_context} setting does not match its front")
            point_seed = _strict_int(raw_point.get("seed"), f"{point_context} seed")
            if point_seed != front_seed:
                raise ValueError(f"{point_context} seed does not match its front")
            point_iteration = _iteration_value(raw_point, iterations, point_context)
            validated_front.append(
                _validated_plot_point(
                    raw_point,
                    setting=front_setting,
                    seed=front_seed,
                    iteration=point_iteration,
                    context=point_context,
                )
            )
        expected_raw_front = pareto_front(raw_solutions_by_key[front_key])
        if raw_front_solutions != expected_raw_front:
            raise ValueError(
                f"Stored Pareto front for setting {front_setting!r} and seed {front_seed} "
                "does not match exactly the proposed solutions"
            )
        stored_fronts[front_key] = validated_front

    if set(stored_fronts) != expected_front_keys:
        missing = sorted(expected_front_keys - set(stored_fronts))
        raise ValueError(f"Figure 5 summary is missing Pareto fronts: {missing}")
    for front_key, points in all_solutions.items():
        recomputed = pareto_front(points)
        if stored_fronts[front_key] != recomputed:
            raise ValueError(
                f"Stored Pareto front for setting {front_key[0]!r} and seed {front_key[1]} "
                "does not match proposed solutions"
            )
    fronts = {setting: stored_fronts[(setting, selected_seed)] for setting in selected_settings}
    return Figure5PlotData(
        seed=selected_seed,
        iterations=iterations,
        settings=selected_settings,
        solutions=solutions,
        pareto_fronts=fronts,
        replacements=replacements,
    )


def _point_arrays(points: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/z values in the axis order used by the paper's Figure 5."""
    return tuple(
        np.asarray([float(point[objective]) for point in points], dtype=np.float64)
        for objective in FIGURE5_AXIS_OBJECTIVES
    )  # type: ignore[return-value]


def _plot_limits(data: Figure5PlotData, include_replacements: bool) -> dict[str, tuple[float, float]]:
    points = [point for setting in data.settings for point in data.solutions[setting]]
    if include_replacements:
        points.extend(point for setting in data.settings for point in data.replacements[setting])
    limits: dict[str, tuple[float, float]] = {}
    for objective in FIGURE5_OBJECTIVES:
        values = np.asarray([float(point[objective]) for point in points], dtype=np.float64)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = maximum - minimum
        padding = (0.05 * span) if span > 0.0 else max(0.05 * abs(maximum), 0.05)
        limits[objective] = (minimum - padding, maximum + padding)
    return limits


def _configure_axis(axis: Any, limits: Mapping[str, tuple[float, float]], compact: bool = False) -> None:
    axis.set_xlim(*limits["kappa"])
    axis.set_ylim(*limits["rho"])
    axis.set_zlim(*limits["E"])
    axis.set_xlabel(r"$\kappa$ [W m$^{-1}$ K$^{-1}$]", fontsize=7 if compact else 9, labelpad=2)
    axis.set_ylabel(r"$\rho$ [g cm$^{-3}$]", fontsize=7 if compact else 9, labelpad=2)
    axis.set_zlabel(r"$E$ [GPa]", fontsize=7 if compact else 9, labelpad=2)
    axis.tick_params(labelsize=6 if compact else 8, pad=0)
    axis.view_init(elev=22, azim=-55)
    axis.grid(True, alpha=0.25)


def _scatter(
    axis: Any,
    points: Sequence[Mapping[str, Any]],
    *,
    color: str,
    marker: str = "o",
    size: float = 16.0,
    alpha: float = 0.7,
    label: str | None = None,
) -> None:
    if not points:
        return
    x_values, y_values, z_values = _point_arrays(points)
    axis.scatter(
        x_values,
        y_values,
        z_values,
        c=color,
        marker=marker,
        s=size,
        alpha=alpha,
        depthshade=False,
        linewidths=0.35 if marker == "o" else 0.9,
        label=label,
    )


def _window_points(points: Sequence[Mapping[str, Any]], start: int, end: int) -> list[Mapping[str, Any]]:
    return [point for point in points if start <= int(point["iteration"]) < end]


def render_figure5(
    data: Figure5PlotData,
    output_path: str | Path,
    *,
    include_replacements: bool = False,
    iteration_boundaries: Sequence[int] | None = None,
) -> Path:
    """Render the overview panels and time-resolved rows for one seed."""
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Figure 5 output must use a .png extension")
    boundaries = validate_iteration_boundaries(iteration_boundaries, data.iterations)
    num_windows = len(boundaries) - 1
    limits = _plot_limits(data, include_replacements)

    figure = plt.figure(figsize=(max(14.0, 3.8 * num_windows + 1.0), 12.0))
    outer_grid = figure.add_gridspec(
        1 + len(data.settings),
        1,
        height_ratios=(1.35, *([1.0] * len(data.settings))),
        hspace=0.18,
        left=0.035,
        right=0.985,
        bottom=0.035,
        top=0.965,
    )
    top_grid = outer_grid[0].subgridspec(1, len(data.settings), wspace=0.04)
    temporal_widths = (0.30, *([1.0] * num_windows))
    temporal_grids = [
        outer_grid[row_index + 1].subgridspec(
            1,
            num_windows + 1,
            width_ratios=temporal_widths,
            wspace=0.05,
        )
        for row_index in range(len(data.settings))
    ]

    for panel_index, setting in enumerate(data.settings):
        axis = figure.add_subplot(top_grid[0, panel_index], projection="3d")
        _scatter(
            axis,
            data.solutions[setting],
            color=ALL_SOLUTION_COLOR,
            size=17.0,
            alpha=0.48,
            label="QUBO solutions",
        )
        _scatter(
            axis,
            data.pareto_fronts[setting],
            color=PARETO_COLOR,
            size=30.0,
            alpha=0.95,
            label="Pareto front",
        )
        if include_replacements:
            _scatter(
                axis,
                data.replacements[setting],
                color=REPLACEMENT_COLOR,
                marker="x",
                size=32.0,
                alpha=0.9,
                label="Random replacements",
            )
        _configure_axis(axis, limits)
        panel_label = chr(ord("a") + panel_index)
        axis.set_title(f"{panel_label}) {SETTING_LABELS[setting]} - seed {data.seed}", pad=8)
        axis.legend(loc="upper left", fontsize=8, frameon=False)

    for setting_index, (row_grid, setting) in enumerate(zip(temporal_grids, data.settings)):
        panel_label = chr(ord("a") + len(data.settings) + setting_index)
        # 行标签使用独立栏位，避免 3D 子图标题参与布局后与相邻行重叠。
        label_axis = figure.add_subplot(row_grid[0, 0])
        label_axis.axis("off")
        label_axis.text(
            0.5,
            0.5,
            f"{panel_label})\n{SETTING_LABELS[setting]}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        for window_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            axis = figure.add_subplot(row_grid[0, window_index + 1], projection="3d")
            color = TEMPORAL_COLORS[window_index % len(TEMPORAL_COLORS)]
            _scatter(
                axis,
                _window_points(data.solutions[setting], start, end),
                color=color,
                size=18.0,
                alpha=0.75,
            )
            if include_replacements:
                _scatter(
                    axis,
                    _window_points(data.replacements[setting], start, end),
                    color=REPLACEMENT_COLOR,
                    marker="x",
                    size=26.0,
                    alpha=0.9,
                )
            _configure_axis(axis, limits, compact=True)
            axis.set_title(f"Iterations {start}-{end}", fontsize=9, pad=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def plot_figure5(
    summary_path: str | Path,
    output_path: str | Path,
    *,
    seed: int | None = None,
    include_replacements: bool = False,
    iteration_boundaries: Sequence[int] | None = None,
) -> Path:
    """Render the Figure 5 summary plot."""
    data = load_figure5_plot_data(summary_path, seed=seed)
    output = Path(output_path)
    output_layout = figure5_output_layout(output.parent)
    log_path = configure_file_logging_path(output_layout.plot_log_path)
    LOGGER.info(
        "Plotting Figure 5 summary=%s output=%s seed=%s replacements=%s",
        summary_path,
        output,
        data.seed,
        include_replacements,
    )
    result = render_figure5(
        data,
        output,
        include_replacements=include_replacements,
        iteration_boundaries=iteration_boundaries,
    )
    LOGGER.info("Wrote Figure 5 plot output=%s log=%s", result, log_path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = parse_args(argv)
    plot_figure5(
        args.summary,
        args.output,
        seed=args.seed,
        include_replacements=args.include_replacements,
        iteration_boundaries=args.iteration_boundaries,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FIGURE5_AXIS_OBJECTIVES",
    "Figure5PlotData",
    "default_iteration_boundaries",
    "load_figure5_plot_data",
    "main",
    "parse_args",
    "plot_figure5",
    "render_figure5",
    "validate_iteration_boundaries",
]
