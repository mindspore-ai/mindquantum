"""Figure 4 命令行入口。.

Runner 只负责解析 CLI、检查依赖/CUDA、写 manifest 和调用 pipeline；实际 active
learning 逻辑保留在 `figure4_pipeline.py`。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .experiment_runtime import (
    FIGURE4_SEED_INDEX,
    SEED_DERIVATION_SCHEME,
    TRAINING_REQUIRED_MODULES,
    collect_runtime_metadata,
    configure_file_logging_path,
    ensure_compute_device_available,
    ensure_training_dependencies,
    fm_seed_block_size,
    resolve_contiguous_seeds,
    validate_seed_schedule,
    write_json_atomic,
)
from .figure4_experiment_config import (
    FIGURE4_OBJECTIVE_NAMES,
    FIGURE4_OBJECTIVES,
    FIGURE4_PRESET_NUM_SEEDS,
    FIGURE4_SETTINGS,
    SUPPORTED_PRESETS,
    Figure4ExperimentConfig,
    ObjectiveSpec,
    resolve_experiment_config,
)
from .figure4_outputs import (
    MANIFEST_SCHEMA_VERSION,
    Figure4OutputLayout,
    figure4_output_layout,
)
from .figure4_pipeline import run_figure4_experiment
from .figure4_setting_strategies import validate_settings

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Figure 4 PyTorch FM reproduction pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary outputs.")
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESETS,
        default="quick",
        help="Resolved defaults: paper, quick, or test. Explicit numeric options override the preset.",
    )
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of initial datasets / trajectories.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the contiguous seed list.")
    parser.add_argument("--iterations", type=int, default=None, help="Active learning iterations per trajectory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Initial dataset size per seed.")
    parser.add_argument("--num-levels", type=int, default=None, help="One-hot levels per variable block.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Number of Optuna trials for FM tuning.")
    parser.add_argument(
        "--sa-reads",
        "--sa-runs",
        dest="sa_reads",
        type=int,
        default=None,
        help="Simulated annealing read count (--sa-runs is a compatibility alias).",
    )
    parser.add_argument("--sa-sweeps", type=int, default=None, help="Simulated annealing sweep count.")
    parser.add_argument(
        "--fm-objective-weight",
        type=float,
        default=None,
        help="Weight applied to the normalized FM objective QUBO term.",
    )
    parser.add_argument(
        "--system-penalty-weight",
        type=float,
        default=None,
        help="Weight applied to the normalized system-constraint QUBO term.",
    )
    parser.add_argument(
        "--one-hot-penalty-weight",
        type=float,
        default=None,
        help="Weight applied to the normalized one-hot QUBO term.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch execution device.")
    parser.add_argument("--resume", action="store_true", help="Resume from per-trajectory checkpoints in output-dir.")
    parser.add_argument(
        "--settings",
        nargs="+",
        required=True,
        choices=FIGURE4_SETTINGS,
        help=f"Figure 4 settings to run. Choices: {' '.join(FIGURE4_SETTINGS)}",
    )
    parser.add_argument(
        "--objectives",
        nargs="*",
        default=None,
        help=f"Optional subset of objectives to run. Choices: {' '.join(FIGURE4_OBJECTIVE_NAMES)}",
    )
    return parser.parse_args(argv)


def _selected_objectives(requested_names: list[str] | None) -> tuple[ObjectiveSpec, ...]:
    """按 canonical `FIGURE4_OBJECTIVES` 顺序筛选 CLI 请求的 objective 子集。."""
    if not requested_names:
        return FIGURE4_OBJECTIVES
    requested = set(requested_names)
    selected = tuple(objective for objective in FIGURE4_OBJECTIVES if objective.name in requested)
    if not selected:
        raise ValueError("No valid objectives selected.")
    unknown = requested - set(FIGURE4_OBJECTIVE_NAMES)
    if unknown:
        raise ValueError(f"Unknown objectives: {', '.join(sorted(unknown))}")
    return selected


def _write_manifest(
    *,
    output_layout: Figure4OutputLayout,
    args: argparse.Namespace,
    resolved_config: Figure4ExperimentConfig,
    seed_list: list[int],
    objective_names: list[str],
    settings: list[str],
) -> Path:
    """记录本次运行命令、解释器、torch/cuda 状态和解析后的配置。."""
    workspace_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "figure": 4,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "command": [sys.executable, *sys.argv],
        "preset": args.preset,
        "runtime": collect_runtime_metadata(workspace_root, "references/paper_2512_11479.pdf"),
        "resolved_config": resolved_config.to_dict(),
        "output_paths": {
            "root": str(output_layout.root),
            "summary": str(output_layout.summary_path),
            "manifest": str(output_layout.manifest_path),
            "runner_log": str(output_layout.runner_log_path),
            "plot_log": str(output_layout.plot_log_path),
            "checkpoint_dir": str(output_layout.checkpoint_dir),
        },
        "resume": bool(args.resume),
        "seed_list": seed_list,
        "objectives": objective_names,
        "settings": settings,
    }
    return write_json_atomic(output_layout.manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """解析命令行并启动 Figure 4 实验。."""
    args = parse_args(argv)
    selected_objectives = _selected_objectives(args.objectives)
    selected_settings = validate_settings(args.settings)
    config = resolve_experiment_config(
        preset=args.preset,
        device=args.device,
        num_samples=args.num_samples,
        iterations=args.iterations,
        num_levels=args.num_levels,
        optuna_trials=args.optuna_trials,
        sa_reads=args.sa_reads,
        sa_sweeps=args.sa_sweeps,
        fm_objective_weight=args.fm_objective_weight,
        system_penalty_weight=args.system_penalty_weight,
        one_hot_penalty_weight=args.one_hot_penalty_weight,
    )
    ensure_training_dependencies(TRAINING_REQUIRED_MODULES)
    ensure_compute_device_available(config.device)
    seed_list = validate_seed_schedule(
        resolve_contiguous_seeds(FIGURE4_PRESET_NUM_SEEDS[args.preset], args.num_seeds, args.seed_start),
        figure_index=FIGURE4_SEED_INDEX,
        trajectory_count=len(FIGURE4_OBJECTIVES) * len(FIGURE4_SETTINGS),
        iterations=config.iterations,
        bounded_stream_count=2,
        fm_model_count=1,
        fm_seed_block_size=fm_seed_block_size(config.optuna_trials),
    )
    output_layout = figure4_output_layout(args.output_dir)
    log_path = configure_file_logging_path(output_layout.runner_log_path)
    LOGGER.info("Starting Figure 4 runner")
    LOGGER.info("Command: %s", " ".join(sys.argv))
    LOGGER.info("Resolved config: %s", json.dumps(config.to_dict(), sort_keys=True))
    LOGGER.info(
        "Run selection: preset=%s seeds=%s objectives=%s settings=%s resume=%s",
        args.preset,
        seed_list,
        [obj.name for obj in selected_objectives],
        list(selected_settings),
        bool(args.resume),
    )

    manifest_path = _write_manifest(
        output_layout=output_layout,
        args=args,
        resolved_config=config,
        seed_list=seed_list,
        objective_names=[obj.name for obj in selected_objectives],
        settings=list(selected_settings),
    )
    summary = run_figure4_experiment(
        seed_list=seed_list,
        config=config,
        output_dir=args.output_dir,
        objectives=selected_objectives,
        resume=args.resume,
        settings=selected_settings,
    )
    LOGGER.info("Finished Figure 4 runner; summary=%s manifest=%s", output_layout.summary_path, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preset": args.preset,
                "objectives": [obj.name for obj in selected_objectives],
                "settings": list(selected_settings),
                "training_backend": summary.training_backend,
                "manifest": str(manifest_path),
                "log": str(log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
