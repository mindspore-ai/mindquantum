"""Figure 5 命令行入口。.

Runner 只负责解析 CLI、检查训练环境、写 manifest 并调用 pipeline。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .experiment_runtime import (
    FIGURE5_SEED_INDEX,
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
from .figure5_experiment_config import (
    FIGURE5_PRESET_NUM_SEEDS,
    SUPPORTED_PRESETS,
    Figure5ExperimentConfig,
    resolve_experiment_config,
)
from .figure5_outputs import MANIFEST_SCHEMA_VERSION, Figure5OutputLayout, figure5_output_layout
from .figure5_pipeline import (
    run_figure5_experiment,
)
from .figure5_scalarization import (
    FIGURE5_OBJECTIVES,
    FIGURE5_SETTINGS,
    validate_settings,
)

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Figure 5 multi-objective FM+QUBO reproduction pipeline.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Figure 5 outputs.")
    parser.add_argument(
        "--preset",
        choices=SUPPORTED_PRESETS,
        default="quick",
        help="Run scale. Defaults to the project workflow scale quick.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds; every selected setting runs once for each seed.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the contiguous seed list.")
    parser.add_argument("--iterations", type=int, default=None, help="Active-learning iterations per trajectory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Initial multi-objective dataset size.")
    parser.add_argument("--num-levels", type=int, default=None, help="Direct one-hot levels per phase block.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Optuna trials per FM training.")
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
    parser.add_argument("--resume", action="store_true", help="Resume matching trajectory checkpoints.")
    parser.add_argument(
        "--settings",
        nargs="+",
        required=True,
        choices=FIGURE5_SETTINGS,
        help="Figure 5 comparison settings: w_ddts wo_ddts",
    )
    return parser.parse_args(argv)


def _write_manifest(
    *,
    output_layout: Figure5OutputLayout,
    args: argparse.Namespace,
    resolved_config: Figure5ExperimentConfig,
    seed_list: list[int],
    settings: list[str],
) -> Path:
    config_dict = resolved_config.to_dict()
    workspace_root = Path(__file__).resolve().parents[1]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "figure": 5,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "command": [sys.executable, *sys.argv],
        "preset": args.preset,
        "runtime": collect_runtime_metadata(workspace_root, "references/paper_2512_11479.pdf"),
        "resolved_config": config_dict,
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
        "objectives": list(FIGURE5_OBJECTIVES),
        "settings": settings,
    }
    return write_json_atomic(output_layout.manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = parse_args(argv)
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
        resolve_contiguous_seeds(
            FIGURE5_PRESET_NUM_SEEDS[args.preset],
            args.num_seeds,
            args.seed_start,
        ),
        figure_index=FIGURE5_SEED_INDEX,
        trajectory_count=len(FIGURE5_SETTINGS),
        iterations=config.iterations,
        bounded_stream_count=1,
        fm_model_count=len(FIGURE5_OBJECTIVES),
        fm_seed_block_size=fm_seed_block_size(config.optuna_trials),
    )
    output_layout = figure5_output_layout(args.output_dir)
    log_path = configure_file_logging_path(output_layout.runner_log_path)
    LOGGER.info("Starting Figure 5 runner")
    LOGGER.info(
        "Resolved Figure 5 config=%s seeds=%s settings=%s resume=%s",
        json.dumps(config.to_dict(), sort_keys=True),
        seed_list,
        list(selected_settings),
        bool(args.resume),
    )

    manifest_path = _write_manifest(
        output_layout=output_layout,
        args=args,
        resolved_config=config,
        seed_list=seed_list,
        settings=list(selected_settings),
    )
    summary = run_figure5_experiment(
        seed_list=seed_list,
        config=config,
        output_dir=args.output_dir,
        resume=args.resume,
        settings=selected_settings,
    )
    LOGGER.info("Finished Figure 5 runner summary=%s manifest=%s", output_layout.summary_path, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preset": args.preset,
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
