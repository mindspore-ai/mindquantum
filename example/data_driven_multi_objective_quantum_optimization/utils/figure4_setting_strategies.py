"""Figure 4 两条流程的策略分发层。.

`wo_cgfm` 和 `w_cgfm` 共用同一套 active-learning 主循环；差异集中在
composition 如何编码成 FM/QUBO 特征、候选 bit vector 如何解码，以及是否需要
额外的 system penalty 来约束四个相分数总和。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .figure4_experiment_config import FIGURE4_SETTINGS, Figure4Setting
from .qubo_math import (
    IterationEncoding,
    create_cgfm_iteration_encoding,
    create_iteration_encoding,
    decode_candidate_bits_to_cgfm_composition,
    decode_candidate_bits_to_composition,
    encode_cgfm_rows,
    encode_single_objective_rows,
)


class SettingStrategy(Protocol):
    """主循环调用的最小策略接口，避免把两条 Figure 4 流程写成两套 pipeline。."""

    name: Figure4Setting
    include_system_penalty: bool

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding: ...

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray: ...

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None: ...


@dataclass(frozen=True)
class WOCGFMStrategy:
    """w/o CGFM：直接用四个相分数组成 block 编码。.

    每个 block 表示一个相的离散分数；候选解可能违反四相总和为 1 的系统约束，
    因此最终 QUBO 需要叠加 system penalty。
    """

    name: Figure4Setting = "wo_cgfm"
    include_system_penalty: bool = True

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding:
        # Eq. 18 的 alpha 层级固定；w/o CGFM 不使用逐轮随机 seed。
        return create_iteration_encoding(num_levels)

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray:
        return encode_single_objective_rows(rows, encoding)

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None:
        return decode_candidate_bits_to_composition(candidate_bits, encoding)


@dataclass(frozen=True)
class WCGFMStrategy:
    """w/ CGFM：用三个角度 block 参数化四相 simplex。.

    CGFM 解码会把三个角度映射回非负且总和为 1 的四相 composition，因此只保留
    one-hot penalty，不再加入直接 composition 编码所需的 system penalty。
    """

    name: Figure4Setting = "w_cgfm"
    include_system_penalty: bool = False

    def create_encoding(self, num_levels: int, seed: int) -> IterationEncoding:
        return create_cgfm_iteration_encoding(num_levels, seed)

    def encode_rows(self, rows: Sequence[dict[str, Any]], encoding: IterationEncoding) -> np.ndarray:
        return encode_cgfm_rows(rows, encoding)

    def decode_candidate(self, candidate_bits: np.ndarray, encoding: IterationEncoding) -> np.ndarray | None:
        return decode_candidate_bits_to_cgfm_composition(candidate_bits, encoding)


SETTING_STRATEGIES: dict[Figure4Setting, SettingStrategy] = {
    "wo_cgfm": WOCGFMStrategy(),
    "w_cgfm": WCGFMStrategy(),
}


def get_setting_strategy(setting: str) -> SettingStrategy:
    """Return the implementation for a Figure 4 setting."""
    if setting not in SETTING_STRATEGIES:
        raise ValueError(f"Unsupported setting {setting!r}. Choices: {', '.join(FIGURE4_SETTINGS)}")
    return SETTING_STRATEGIES[setting]  # type: ignore[index]


def validate_settings(settings: Sequence[str]) -> tuple[Figure4Setting, ...]:
    """Validate and normalize requested experiment settings."""
    if not settings:
        raise ValueError("At least one setting is required.")
    if any(not isinstance(setting, str) for setting in settings):
        raise ValueError("Figure 4 settings must contain strings")
    selected = tuple(get_setting_strategy(setting).name for setting in settings)
    if len(set(selected)) != len(selected):
        raise ValueError("Figure 4 settings must not contain duplicates")
    return selected
