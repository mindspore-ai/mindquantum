"""Figure 5 输出布局、checkpoint I/O、summary 读取和日志配置。."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .experiment_runtime import (
    NEAL_SEED_MAX,
    SEED_DERIVATION_SCHEME,
    load_json_object,
    require_integer,
    same_json_value,
    write_json_atomic,
)

if TYPE_CHECKING:
    from .figure5_experiment_config import Figure5ExperimentConfig
    from .figure5_scalarization import Figure5Setting
else:
    Figure5ExperimentConfig = Any
    Figure5Setting = str


CHECKPOINT_SCHEMA_VERSION = 5
SUMMARY_SCHEMA_VERSION = 5
MANIFEST_SCHEMA_VERSION = 2
SUMMARY_FILENAME = "figure5_summary.json"
MANIFEST_FILENAME = "figure5_manifest.json"
DEFAULT_FIGURE_FILENAME = "figure5.png"
VALIDATION_REPORT_FILENAME = "figure5_validation.json"
LOG_DIR_NAME = "logs"
RUNNER_LOG_FILENAME = "figure5_runner.log"
PLOT_LOG_FILENAME = "plot_figure5.log"
CHECKPOINT_DIR_NAME = "trajectories"
_CHECKPOINT_FIELDS = frozenset({"schema_version", "seed_derivation", "setting", "seed", "config", "state"})


@dataclass(frozen=True)
class Figure5OutputLayout:
    """Figure 5 输出目录下的标准路径。."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def summary_path(self) -> Path:
        return self.root / SUMMARY_FILENAME

    @property
    def figure_path(self) -> Path:
        return self.root / DEFAULT_FIGURE_FILENAME

    @property
    def validation_report_path(self) -> Path:
        return self.root / VALIDATION_REPORT_FILENAME

    @property
    def logs_dir(self) -> Path:
        return self.root / LOG_DIR_NAME

    @property
    def runner_log_path(self) -> Path:
        return self.logs_dir / RUNNER_LOG_FILENAME

    @property
    def plot_log_path(self) -> Path:
        return self.logs_dir / PLOT_LOG_FILENAME

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / CHECKPOINT_DIR_NAME

    def checkpoint_filename(self, setting: Figure5Setting, seed: int) -> str:
        return checkpoint_filename(setting, seed)

    def checkpoint_path(self, setting: Figure5Setting, seed: int) -> Path:
        return self.checkpoint_dir / self.checkpoint_filename(setting, seed)


def figure5_output_layout(output_dir: str | Path) -> Figure5OutputLayout:
    """Build the Figure 5 output path layout."""
    return Figure5OutputLayout(root=Path(output_dir))


def checkpoint_filename(setting: Figure5Setting, seed: int) -> str:
    """Build a deterministic checkpoint filename."""
    normalized_seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    return f"{setting}_seed_{normalized_seed}.json"


def checkpoint_payload(
    *,
    setting: Figure5Setting,
    seed: int,
    state: object,
    config: Figure5ExperimentConfig,
) -> dict[str, Any]:
    """Build a checkpoint tied to one setting/seed/config trajectory."""
    normalized_seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "setting": setting,
        "seed": normalized_seed,
        "config": config.to_dict(),
        "state": asdict(state),
    }


def write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a validated checkpoint atomically."""
    return write_json_atomic(path, payload, indent=None)


def _checkpoint_error(checkpoint: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid checkpoint {checkpoint}: {detail}")


def load_checkpoint(
    path: str | Path,
    *,
    setting: Figure5Setting,
    seed: int,
    config: Figure5ExperimentConfig,
) -> dict[str, Any]:
    """Load a checkpoint after validating its outer schema and identity."""
    checkpoint = Path(path)
    requested_seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    payload = load_json_object(checkpoint, document_name="checkpoint")

    missing_fields = sorted(_CHECKPOINT_FIELDS - payload.keys())
    if missing_fields:
        raise _checkpoint_error(checkpoint, f"missing top-level fields: {', '.join(missing_fields)}")

    schema_version = payload["schema_version"]
    if type(schema_version) is not int:
        raise _checkpoint_error(checkpoint, "field 'schema_version' must be an integer")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_error(
            checkpoint,
            f"unsupported schema_version {schema_version!r}; expected {CHECKPOINT_SCHEMA_VERSION}",
        )

    seed_derivation = payload["seed_derivation"]
    if type(seed_derivation) is not str:
        raise _checkpoint_error(checkpoint, "field 'seed_derivation' must be a string")
    if seed_derivation != SEED_DERIVATION_SCHEME:
        raise _checkpoint_error(
            checkpoint,
            f"unsupported seed_derivation {seed_derivation!r}; expected {SEED_DERIVATION_SCHEME!r}",
        )

    checkpoint_setting = payload["setting"]
    checkpoint_seed = payload["seed"]
    if type(checkpoint_setting) is not str:
        raise _checkpoint_error(checkpoint, "field 'setting' must be a string")
    if type(checkpoint_seed) is not int:
        raise _checkpoint_error(checkpoint, "field 'seed' must be an integer")
    if checkpoint_setting != setting:
        raise _checkpoint_error(checkpoint, f"setting metadata does not match {setting!r}")
    if checkpoint_seed != requested_seed:
        raise _checkpoint_error(checkpoint, f"seed metadata does not match {requested_seed!r}")

    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, dict):
        raise _checkpoint_error(checkpoint, "field 'config' must be a JSON object")
    if not same_json_value(checkpoint_config, config.to_dict()):
        raise _checkpoint_error(
            checkpoint,
            "config does not match the current run; use a separate output directory for a different configuration",
        )
    if not isinstance(payload["state"], dict):
        raise _checkpoint_error(checkpoint, "field 'state' must be a JSON object")
    return payload


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a workflow summary JSON object."""
    return load_json_object(path, document_name="Figure 5 summary")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_FIGURE_FILENAME",
    "Figure5OutputLayout",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "VALIDATION_REPORT_FILENAME",
    "checkpoint_filename",
    "checkpoint_payload",
    "figure5_output_layout",
    "load_checkpoint",
    "load_summary",
    "write_checkpoint",
]
