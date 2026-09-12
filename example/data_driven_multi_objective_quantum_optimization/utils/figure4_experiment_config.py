"""Figure 4 实验配置和单目标优化方向定义。.

该模块保存 canonical objective/setting、preset seed 数及训练/SA 配置。
具体的编码策略在 `figure4_setting_strategies.py`，具体运行循环在 `figure4_pipeline.py`。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .experiment_config import EncodingConfig, FMConfig, QuboConfig, SAConfig
from .experiment_runtime import require_integer

Figure4RunScale = Literal["paper", "quick", "test"]
Figure4Setting = Literal["wo_cgfm", "w_cgfm"]
SUPPORTED_PRESETS: tuple[Figure4RunScale, ...] = ("paper", "quick", "test")
FIGURE4_SETTINGS: tuple[Figure4Setting, ...] = ("wo_cgfm", "w_cgfm")
FIGURE4_PRESET_NUM_SEEDS: dict[Figure4RunScale, int] = {
    "paper": 20,
    "quick": 3,
    "test": 1,
}


@dataclass(frozen=True)
class ObjectiveSpec:
    """描述一个 Figure 4 objective 的优化方向。.

    FM/QUBO 训练统一按“越小越好”处理，因此最大化目标会在训练前取负号；
    best-so-far 曲线仍按真实 objective 方向更新。
    """

    name: str
    maximize: bool

    def transform_for_training(self, values: Any) -> Any:
        return -values if self.maximize else values

    def initial_best(self) -> float:
        return -math.inf if self.maximize else math.inf

    def update_best(self, current_best: float, candidate: float) -> float:
        if self.maximize:
            return max(current_best, candidate)
        return min(current_best, candidate)


FIGURE4_OBJECTIVES: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("kappa", maximize=True),
    ObjectiveSpec("E", maximize=True),
    ObjectiveSpec("rho", maximize=False),
    ObjectiveSpec("delta_alpha", maximize=False),
    ObjectiveSpec("delta_T", maximize=False),
)
FIGURE4_OBJECTIVE_NAMES: tuple[str, ...] = tuple(objective.name for objective in FIGURE4_OBJECTIVES)


@dataclass(frozen=True)
class Figure4ExperimentConfig:
    """Figure 4 一次实验运行的完整配置。."""

    num_samples: int = 100
    iterations: int = 600
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
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


def preset_config(preset: Figure4RunScale, device: str = "cpu") -> Figure4ExperimentConfig:
    """把论文规模、快速规模和单元测试规模收敛到同一个配置 dataclass。."""
    if preset == "paper":
        return Figure4ExperimentConfig(
            fm=FMConfig(device=device),
            qubo=QuboConfig(one_hot_penalty_weight=1.0),
        )
    if preset == "quick":
        return Figure4ExperimentConfig(
            num_samples=50,
            iterations=100,
            # quick 只验收小规模流程。25 levels 对应 wo_cgfm 的 100-bit 和
            # w_cgfm 的 75-bit QUBO；直接编码有 C(28, 3)=3276 个可行组成。
            # 该缩小规模和 gamma=2 仍不保证有限 SA batch 必然包含可行终态。
            encoding=EncodingConfig(num_levels=25),
            fm=FMConfig(optuna_trials=3, device=device),
            sa=SAConfig(reads=250, sweeps=1000),
            qubo=QuboConfig(one_hot_penalty_weight=2.0),
        )
    if preset == "test":
        return Figure4ExperimentConfig(
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
    preset: Figure4RunScale,
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
) -> Figure4ExperimentConfig:
    """先解析 preset，再应用 CLI override，得到 runner 实际执行的配置。."""
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
    "FIGURE4_OBJECTIVE_NAMES",
    "FIGURE4_OBJECTIVES",
    "FIGURE4_PRESET_NUM_SEEDS",
    "FIGURE4_SETTINGS",
    "Figure4ExperimentConfig",
    "Figure4RunScale",
    "Figure4Setting",
    "ObjectiveSpec",
    "SUPPORTED_PRESETS",
    "preset_config",
    "resolve_experiment_config",
]
