"""Figure 4 输出布局、checkpoint I/O、summary 读取和日志文件配置。."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .experiment_runtime import (
    NEAL_SEED_MAX,
    SEED_DERIVATION_SCHEME,
    load_json_object,
    require_integer,
    same_json_value,
    write_json_atomic,
)

if TYPE_CHECKING:
    from .figure4_experiment_config import Figure4ExperimentConfig, Figure4Setting
else:
    Figure4ExperimentConfig = Any
    Figure4Setting = str


CHECKPOINT_SCHEMA_VERSION = 6
SUMMARY_SCHEMA_VERSION = 5
MANIFEST_SCHEMA_VERSION = 2
SUMMARY_FILENAME = "figure4_summary.json"
MANIFEST_FILENAME = "figure4_manifest.json"
DEFAULT_FIGURE_FILENAME = "figure4.png"
VALIDATION_REPORT_FILENAME = "figure4_validation.json"
LOG_DIR_NAME = "logs"
RUNNER_LOG_FILENAME = "figure4_runner.log"
PLOT_LOG_FILENAME = "plot_figure4.log"
CHECKPOINT_DIR_NAME = "trajectories"
_CHECKPOINT_FIELDS = frozenset({"schema_version", "seed_derivation", "objective", "setting", "seed", "config", "state"})


class ObjectiveLike(Protocol):
    """Store objective like data."""

    name: str


@dataclass(frozen=True)
class Figure4OutputLayout:
    """集中定义一个 Figure 4 输出目录下的所有标准路径。."""

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

    def checkpoint_filename(self, setting: Figure4Setting, objective_name: str, seed: int) -> str:
        return checkpoint_filename(setting, objective_name, seed)

    def checkpoint_path(self, setting: Figure4Setting, objective_name: str, seed: int) -> Path:
        return self.checkpoint_dir / self.checkpoint_filename(setting, objective_name, seed)


def figure4_output_layout(output_dir: str | Path) -> Figure4OutputLayout:
    """Build the Figure 4 output path layout."""
    return Figure4OutputLayout(root=Path(output_dir))


def checkpoint_filename(setting: Figure4Setting, objective_name: str, seed: int) -> str:
    """Build a deterministic checkpoint filename."""
    normalized_seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    return f"{setting}_{objective_name}_seed_{normalized_seed}.json"


def checkpoint_payload(
    *,
    objective: ObjectiveLike,
    setting: Figure4Setting,
    seed: int,
    state: object,
    config: Figure4ExperimentConfig,
) -> dict[str, Any]:
    """构造绑定 mixed-radix seed 方案的 checkpoint payload。.

    checkpoint 绑定 objective、setting、seed 和 config；恢复时这些字段必须一致。
    """
    normalized_seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "seed_derivation": SEED_DERIVATION_SCHEME,
        "objective": objective.name,
        "setting": setting,
        "seed": normalized_seed,
        "config": config.to_dict(),
        "state": asdict(state),
    }


def write_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """原子写入紧凑 checkpoint。."""
    return write_json_atomic(path, payload, indent=None)


def _checkpoint_error(checkpoint: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid checkpoint {checkpoint}: {detail}")


def load_checkpoint(
    path: str | Path,
    *,
    objective: ObjectiveLike,
    seed: int,
    config: Figure4ExperimentConfig,
    setting: Figure4Setting,
) -> dict[str, Any]:
    """读取 checkpoint，并校验外层 schema、类型和 trajectory 身份。."""
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

    checkpoint_objective = payload["objective"]
    checkpoint_setting = payload["setting"]
    checkpoint_seed = payload["seed"]
    if type(checkpoint_objective) is not str:
        raise _checkpoint_error(checkpoint, "field 'objective' must be a string")
    if type(checkpoint_setting) is not str:
        raise _checkpoint_error(checkpoint, "field 'setting' must be a string")
    if type(checkpoint_seed) is not int:
        raise _checkpoint_error(checkpoint, "field 'seed' must be an integer")
    if checkpoint_objective != objective.name:
        raise _checkpoint_error(checkpoint, f"objective metadata does not match {objective.name!r}")
    if checkpoint_seed != requested_seed:
        raise _checkpoint_error(checkpoint, f"seed metadata does not match {requested_seed!r}")
    if checkpoint_setting != setting:
        raise _checkpoint_error(checkpoint, f"setting metadata does not match {setting!r}")

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
    return load_json_object(path, document_name="Figure 4 summary")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_FIGURE_FILENAME",
    "Figure4OutputLayout",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "VALIDATION_REPORT_FILENAME",
    "checkpoint_filename",
    "checkpoint_payload",
    "figure4_output_layout",
    "load_checkpoint",
    "load_summary",
    "write_checkpoint",
]
