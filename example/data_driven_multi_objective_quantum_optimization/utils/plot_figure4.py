"""Figure 4 summary 绘图入口。."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment_runtime import SEED_DERIVATION_SCHEME, configure_file_logging_path
from .figure4_experiment_config import FIGURE4_OBJECTIVES
from .figure4_outputs import SUMMARY_SCHEMA_VERSION, figure4_output_layout, load_summary

SETTING_LABELS = {
    "wo_cgfm": "w/o CGFM",
    "w_cgfm": "w/ CGFM",
}
SETTING_COLORS = {
    "wo_cgfm": "#d95f02",
    "w_cgfm": "#1f78b4",
}
LOG_OBJECTIVES = {"delta_alpha", "delta_T"}
LOG_EPSILON = 1e-12
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the Figure 4 reproduction summary.")
    parser.add_argument("--summary", type=Path, nargs="+", required=True, help="One or more summary files")
    parser.add_argument("--output", type=Path, required=True, help="Path to output PNG")
    return parser.parse_args()


def plot_values(objective_name: str, values: list[float] | np.ndarray) -> np.ndarray:
    """仅在绘图层 clamp log 轴目标，summary JSON 保留真实值。."""
    array = np.asarray(values, dtype=np.float64)
    if objective_name in LOG_OBJECTIVES:
        return np.maximum(array, LOG_EPSILON)
    return array


def _required_curve(stats: dict[str, Any], key: str) -> np.ndarray:
    if key not in stats:
        raise ValueError(f"Figure 4 aggregated entry is missing field {key!r}")
    try:
        curve = np.asarray(stats[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Figure 4 aggregated field {key!r} must be a numeric curve") from exc
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError(f"Figure 4 aggregated field {key!r} must be a non-empty 1-D curve")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"Figure 4 aggregated field {key!r} must contain finite values")
    return curve


def _merge_aggregated(summary_paths: list[Path]) -> dict[str, dict[str, Any]]:
    """合并一个或多个当前 schema summary 的 aggregated 曲线。."""
    merged: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        summary = load_summary(path)
        if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Figure 4 summary schema in {path}: expected schema_version {SUMMARY_SCHEMA_VERSION}"
            )
        if summary.get("seed_derivation") != SEED_DERIVATION_SCHEME:
            raise ValueError(f"Unsupported Figure 4 seed_derivation in {path}: expected {SEED_DERIVATION_SCHEME!r}")
        aggregated = summary.get("aggregated")
        if not isinstance(aggregated, dict) or not aggregated:
            raise ValueError(f"Figure 4 summary has no aggregated results: {path}")
        for key, value in aggregated.items():
            if key in merged:
                raise ValueError(f"Duplicate aggregated key {key!r} while merging {path}")
            if not isinstance(value, dict):
                raise ValueError(f"Figure 4 aggregated entry {key!r} in {path} must be an object")
            merged[key] = dict(value)
    return merged


def main() -> None:
    """绘制 Figure 4 五个 objective 的 w/o CGFM 与 w/ CGFM 曲线。."""
    args = parse_args()
    output_layout = figure4_output_layout(args.output.parent)
    log_path = configure_file_logging_path(output_layout.plot_log_path)
    LOGGER.info("Starting Figure 4 plot summaries=%s output=%s", [str(path) for path in args.summary], args.output)
    aggregated = _merge_aggregated(args.summary)
    skipped_keys: list[str] = []

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.flatten()

    for idx, objective in enumerate(FIGURE4_OBJECTIVES):
        axis = flat_axes[idx]
        for setting, color in SETTING_COLORS.items():
            key = f"{objective.name}:{setting}"
            if key not in aggregated:
                skipped_keys.append(key)
                continue
            stats = aggregated[key]
            best_so_far_mean = _required_curve(stats, "best_so_far_mean")
            best_so_far_std = _required_curve(stats, "best_so_far_std")
            best_so_far_min = _required_curve(stats, "best_so_far_min")
            best_so_far_max = _required_curve(stats, "best_so_far_max")
            curve_lengths = {
                len(best_so_far_mean),
                len(best_so_far_std),
                len(best_so_far_min),
                len(best_so_far_max),
            }
            if len(curve_lengths) != 1:
                raise ValueError(f"Figure 4 aggregated curves for {key!r} must have equal lengths")
            x_values = np.arange(1, len(best_so_far_mean) + 1)

            label = SETTING_LABELS.get(setting, setting)
            axis.fill_between(
                x_values,
                plot_values(objective.name, best_so_far_min),
                plot_values(objective.name, best_so_far_max),
                color=color,
                alpha=0.14,
                linewidth=0,
                label=f"{label} min-max",
            )
            axis.fill_between(
                x_values,
                plot_values(objective.name, best_so_far_mean - best_so_far_std),
                plot_values(objective.name, best_so_far_mean + best_so_far_std),
                color=color,
                alpha=0.24,
                linewidth=0,
                label=f"{label} std.",
            )
            axis.plot(
                x_values,
                plot_values(objective.name, best_so_far_mean),
                label=label,
                color=color,
                linewidth=2,
                marker="x" if setting == "w_cgfm" else None,
                markevery=max(1, len(best_so_far_mean) // 12),
                markersize=4,
            )
        axis.set_title(objective.name)
        axis.set_xlabel("Active learning iteration")
        axis.set_ylabel("Best-so-far objective")
        if objective.name in LOG_OBJECTIVES:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    flat_axes[-1].axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    LOGGER.info("Wrote Figure 4 plot output=%s log=%s", args.output, log_path)

    if skipped_keys:
        message = "Skipped missing summary keys: " + ", ".join(skipped_keys)
        LOGGER.info(message)
        print(message)


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args", "plot_values"]
