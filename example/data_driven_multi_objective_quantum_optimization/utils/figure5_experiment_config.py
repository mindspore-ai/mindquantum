"""Figure 5 多目标实验规模、preset seed 数和训练/求解配置。."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .experiment_config import EncodingConfig, FMConfig, QuboConfig, SAConfig
from .experiment_runtime import require_integer

Figure5RunScale = Literal["paper", "quick", "test"]
SUPPORTED_PRESETS: tuple[Figure5RunScale, ...] = ("paper", "quick", "test")
FIGURE5_PRESET_NUM_SEEDS: dict[Figure5RunScale, int] = {
    "paper": 1,
    "quick": 1,
    "test": 1,
}


@dataclass(frozen=True)
class Figure5ExperimentConfig:
    """Figure 5 一次实验的完整配置。.

    Figure 5 复用 Figure 4 的 FM、SA 和离散编码参数类，但使用
    独立的 preset，避免把 Figure 4 的 600 轮单目标默认值带入多目标实验。
    """

    num_samples: int = 500
    iterations: int = 1000
    encoding: EncodingConfig = field(default_factory=lambda: EncodingConfig(num_levels=25))
    fm: FMConfig = field(default_factory=FMConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    qubo: QuboConfig = field(default_factory=QuboConfig)

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        object.__setattr__(self, "num_samples", require_integer("num_samples", self.num_samples, minimum=1))
        object.__setattr__(self, "iterations", require_integer("iterations", self.iterations, minimum=1))

    @property
    def num_levels(self) -> int:
        return self.encoding.num_levels

    @property
    def optuna_trials(self) -> int:
        return self.fm.optuna_trials

    @property
    def device(self) -> str:
        return self.fm.device

    @property
    def sa_reads(self) -> int:
        return self.sa.reads

    @property
    def sa_sweeps(self) -> int:
        return self.sa.sweeps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_config(preset: Figure5RunScale, device: str = "cpu") -> Figure5ExperimentConfig:
    """Resolve the paper reference, project reproduction, or test scale."""
    if preset == "paper":
        return Figure5ExperimentConfig(
            fm=FMConfig(device=device),
            qubo=QuboConfig(one_hot_penalty_weight=1.0),
        )
    if preset == "quick":
        return Figure5ExperimentConfig(
            num_samples=500,
            iterations=150,
            encoding=EncodingConfig(num_levels=25),
            fm=FMConfig(optuna_trials=3, device=device),
            sa=SAConfig(reads=100, sweeps=500),
            qubo=QuboConfig(one_hot_penalty_weight=2.0),
        )
    if preset == "test":
        return Figure5ExperimentConfig(
            num_samples=10,
            iterations=2,
            encoding=EncodingConfig(num_levels=8),
            fm=FMConfig(optuna_trials=0, device=device),
            sa=SAConfig(reads=32, sweeps=12),
            qubo=QuboConfig(one_hot_penalty_weight=2.0),
        )
    raise ValueError(f"Unsupported preset {preset!r}. Choices: {', '.join(SUPPORTED_PRESETS)}")


def resolve_experiment_config(
    *,
    preset: Figure5RunScale,
    device: str,
    num_samples: int | None = None,
    iterations: int | None = None,
    num_levels: int | None = None,
    optuna_trials: int | None = None,
    sa_reads: int | None = None,
    sa_sweeps: int | None = None,
    fm_objective_weight: float | None = None,
    system_penalty_weight: float | None = None,
    one_hot_penalty_weight: float | None = None,
) -> Figure5ExperimentConfig:
    """Resolve a preset first, then apply explicit CLI overrides."""
    config = preset_config(preset, device=device)
    if num_samples is not None or iterations is not None:
        config = replace(
            config,
            num_samples=config.num_samples if num_samples is None else num_samples,
            iterations=config.iterations if iterations is None else iterations,
        )
    if num_levels is not None:
        config = replace(config, encoding=EncodingConfig(num_levels=num_levels))
    if optuna_trials is not None:
        config = replace(config, fm=replace(config.fm, optuna_trials=optuna_trials))
    if sa_reads is not None or sa_sweeps is not None:
        config = replace(
            config,
            sa=SAConfig(
                reads=config.sa.reads if sa_reads is None else sa_reads,
                sweeps=config.sa.sweeps if sa_sweeps is None else sa_sweeps,
            ),
        )
    if any(value is not None for value in (fm_objective_weight, system_penalty_weight, one_hot_penalty_weight)):
        config = replace(
            config,
            qubo=replace(
                config.qubo,
                fm_objective_weight=(
                    config.qubo.fm_objective_weight if fm_objective_weight is None else fm_objective_weight
                ),
                system_penalty_weight=(
                    config.qubo.system_penalty_weight if system_penalty_weight is None else system_penalty_weight
                ),
                one_hot_penalty_weight=(
                    config.qubo.one_hot_penalty_weight if one_hot_penalty_weight is None else one_hot_penalty_weight
                ),
            ),
        )
    return config


__all__ = [
    "FIGURE5_PRESET_NUM_SEEDS",
    "Figure5ExperimentConfig",
    "Figure5RunScale",
    "SUPPORTED_PRESETS",
    "preset_config",
    "resolve_experiment_config",
]
