"""Shared discrete encoding, CGFM mapping, QUBO constraints, and SA tools."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Real
from typing import Callable, Literal

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler

from .experiment_config import QUBO_NORMALIZATION_SCHEME, QuboConfig
from .experiment_runtime import NEAL_SEED_MAX, require_integer

DEFAULT_TOLERANCE = 1e-10
CGFM_ANGLE_MAX = np.pi / 2.0
MAX_EXACT_DIAGNOSTIC_STATES = 1_000_000

NoFeasibleCandidateClassification = Literal[
    "qubo_penalty_balance_failure",
    "heuristic_sampling_failure",
    "unclassified_no_feasible_candidate",
]


@dataclass(frozen=True)
class IterationEncoding:
    """一次 active-learning 迭代使用的离散编码表。.

    每个 block 对应一个可选离散值；count=0 不占用 bit，count=1..num_levels
    按论文 Eq. 18 固定对应 block 内索引为 count-1 的 one-hot bit，即
    alpha_i=i/num_levels。该数值映射不随 seed 或迭代改变。
    CGFM 模式下 `value_scale=pi/2` 且只有三个 block，表示三个角度而不是四个相分数。
    论文 S7/S8 所述的逐轮随机化只记录在 `phase_permutation`，表示相变量进入
    CGFM 正反映射的顺序。
    """

    num_levels: int
    positive_count_by_bit: tuple[np.ndarray, ...]
    bit_index_by_count: tuple[np.ndarray, ...]
    value_scale: float = 1.0
    phase_permutation: np.ndarray | None = None

    @property
    def num_blocks(self) -> int:
        return len(self.positive_count_by_bit)


@dataclass(frozen=True)
class QuboStats:
    """Store qubo stats data."""

    normalization_scheme: str
    fm_objective_weight: float
    system_penalty_weight: float
    one_hot_penalty_weight: float
    fm_scale: float
    system_scale: float
    one_hot_scale: float
    num_variables: int
    max_abs: float


@dataclass(frozen=True)
class QuboBuildResult:
    """Store qubo build result data."""

    q: np.ndarray
    bias: float
    stats: QuboStats


@dataclass(frozen=True)
class SASample:
    """One simulated-annealing sample in ascending energy order."""

    state: np.ndarray
    energy: float


@dataclass(frozen=True)
class SASamplingResult:
    """All samples returned by one D-Wave Ocean `neal` call."""

    samples: tuple[SASample, ...]

    @property
    def num_samples(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class FeasibleSASolution:
    """Lowest-energy sample accepted by a caller-provided feasibility check."""

    state: np.ndarray
    energy: float
    rank: int
    infeasible_samples_skipped: int


@dataclass(frozen=True)
class NoFeasibleCandidateDiagnostic:
    """Exact feasible-domain evidence for an SA batch containing no feasible state."""

    classification: NoFeasibleCandidateClassification
    lowest_sampled_infeasible_energy: float
    exact_best_feasible_energy: float | None
    exact_feasible_state_count: int
    max_exact_diagnostic_states: int

    def format_message(self) -> str:
        exact_energy = (
            "not_computed" if self.exact_best_feasible_energy is None else f"{self.exact_best_feasible_energy:.12g}"
        )
        evidence = (
            f"lowest_sampled_infeasible_energy={self.lowest_sampled_infeasible_energy:.12g}, "
            f"exact_best_feasible_energy={exact_energy}, "
            f"exact_feasible_state_count={self.exact_feasible_state_count}, "
            f"max_exact_diagnostic_states={self.max_exact_diagnostic_states}"
        )

        if self.classification == "qubo_penalty_balance_failure":
            return (
                "classification=qubo_penalty_balance_failure "
                "(QUBO penalty-balance failure); "
                f"{evidence}. At least one sampled infeasible state is lower in energy than "
                "every feasible encoded state; adjust the QUBO penalty balance rather than "
                "only increasing SA effort."
            )
        if self.classification == "heuristic_sampling_failure":
            return (
                "classification=heuristic_sampling_failure "
                "(heuristic-sampling failure); "
                f"{evidence}. The exact feasible optimum is lower in energy than every sampled "
                "infeasible state; adjust SA reads, sweeps, or solver configuration."
            )
        if self.exact_best_feasible_energy is None:
            reason = "exact comparison skipped because the feasible-domain state count exceeds the limit"
        else:
            reason = "the two energies are equal within rel_tol=abs_tol=1e-12"
        return (
            "classification=unclassified_no_feasible_candidate; "
            "no feasible SA candidate; cause not classified; "
            f"{evidence}. {reason}."
        )


def get_positive_level_values(num_levels: int) -> np.ndarray:
    """Return the positive discrete level indices."""
    num_levels = require_integer("num_levels", num_levels, minimum=2)
    return np.arange(1, num_levels + 1, dtype=np.int64)


def _create_fixed_one_hot_encoding(
    num_levels: int,
    num_blocks: int,
    value_scale: float,
    phase_permutation: np.ndarray | None,
) -> IterationEncoding:
    num_blocks = require_integer("num_blocks", num_blocks, minimum=1)

    counts = get_positive_level_values(num_levels)
    num_levels = int(num_levels)
    bit_indices = np.arange(-1, num_levels, dtype=np.int64)
    return IterationEncoding(
        num_levels=num_levels,
        positive_count_by_bit=tuple(counts.copy() for _ in range(num_blocks)),
        bit_index_by_count=tuple(bit_indices.copy() for _ in range(num_blocks)),
        value_scale=float(value_scale),
        phase_permutation=None if phase_permutation is None else np.asarray(phase_permutation, dtype=np.int64),
    )


def create_iteration_encoding(num_levels: int, num_blocks: int = 4) -> IterationEncoding:
    """创建使用 Eq. 18 固定 alpha 层级的直接 one-hot 编码。."""
    return _create_fixed_one_hot_encoding(
        num_levels=num_levels,
        num_blocks=num_blocks,
        value_scale=1.0,
        phase_permutation=None,
    )


def create_cgfm_iteration_encoding(num_levels: int, seed: int) -> IterationEncoding:
    """创建 CGFM 编码；seed 只随机化 S7/S8 中的相变量分配。."""
    seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    rng = np.random.default_rng(seed)
    phase_permutation = rng.permutation(4)
    return _create_fixed_one_hot_encoding(
        num_levels=num_levels,
        num_blocks=3,
        value_scale=CGFM_ANGLE_MAX,
        phase_permutation=phase_permutation,
    )


def evaluate_qubo_energy(bit_vector: np.ndarray, q: np.ndarray, bias: float = 0.0) -> float:
    """Evaluate the QUBO energy of a bit vector."""
    bits = np.asarray(bit_vector, dtype=np.float64)
    return float(bits @ q @ bits + bias)


def validate_candidate_composition(composition: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether a decoded composition is feasible."""
    return bool(np.all(composition >= -tolerance) and abs(float(np.sum(composition)) - 1.0) <= tolerance)


def _quantize_composition_to_counts(composition: np.ndarray, num_levels: int) -> np.ndarray:
    scaled = np.floor((np.asarray(composition, dtype=np.float64) * num_levels) + 0.5)
    return np.clip(scaled, 0, num_levels).astype(np.int64)


def _discrete_error(count: int, target: float, num_levels: int) -> float:
    return abs((count / num_levels) - target)


def cure_discrete_counts(composition: Sequence[float], num_levels: int) -> np.ndarray:
    """Round a composition to feasible discrete counts."""
    get_positive_level_values(num_levels)
    num_levels = int(num_levels)
    composition_array = np.asarray(composition, dtype=np.float64)
    counts = _quantize_composition_to_counts(composition_array, num_levels)
    deficit = int(num_levels - int(np.sum(counts)))

    while deficit != 0:
        if deficit > 0:
            candidates = [
                (
                    _discrete_error(int(counts[idx]) + 1, float(composition_array[idx]), num_levels)
                    - _discrete_error(int(counts[idx]), float(composition_array[idx]), num_levels),
                    idx,
                )
                for idx in range(4)
                if counts[idx] < num_levels
            ]
            if not candidates:
                break
            _, chosen_idx = min(candidates)
            counts[chosen_idx] += 1
            deficit -= 1
        else:
            candidates = [
                (
                    _discrete_error(int(counts[idx]) - 1, float(composition_array[idx]), num_levels)
                    - _discrete_error(int(counts[idx]), float(composition_array[idx]), num_levels),
                    idx,
                )
                for idx in range(4)
                if counts[idx] > 0
            ]
            if not candidates:
                break
            _, chosen_idx = min(candidates)
            counts[chosen_idx] -= 1
            deficit += 1

    if int(np.sum(counts)) != num_levels:
        raise RuntimeError("bit-SWAP curing failed to satisfy the system constraint")
    return counts


def counts_to_composition(counts: Sequence[int], num_levels: int) -> np.ndarray:
    """Convert discrete counts to a normalized composition."""
    get_positive_level_values(num_levels)
    num_levels = int(num_levels)
    return np.asarray(counts, dtype=np.float64) / float(num_levels)


def prepare_discrete_composition(composition: Sequence[float], num_levels: int) -> np.ndarray:
    """Prepare a composition for discrete encoding."""
    return counts_to_composition(cure_discrete_counts(composition, num_levels), num_levels)


def encode_discrete_values(values: Sequence[float], encoding: IterationEncoding) -> np.ndarray:
    """Encode discrete block values as one-hot bits."""
    value_array = np.asarray(values, dtype=np.float64)
    if value_array.shape[0] != encoding.num_blocks:
        raise ValueError(f"Expected {encoding.num_blocks} values, got {value_array.shape[0]}")
    counts = np.rint((value_array / encoding.value_scale) * encoding.num_levels).astype(np.int64)
    counts = np.clip(counts, 0, encoding.num_levels)
    bit_vector = np.zeros(encoding.num_blocks * encoding.num_levels, dtype=np.float32)
    for block_idx, count_value in enumerate(counts):
        # 零值不点亮任何 bit，避免为了表示 0 额外增加一个 one-hot 状态。
        if count_value <= 0:
            continue
        bit_idx = int(encoding.bit_index_by_count[block_idx][int(count_value)])
        if bit_idx < 0:
            raise RuntimeError("Missing bit assignment for discrete count")
        bit_vector[(block_idx * encoding.num_levels) + bit_idx] = 1.0
    return bit_vector


def encode_discrete_composition(composition: Sequence[float], encoding: IterationEncoding) -> np.ndarray:
    """Encode a discrete composition as one-hot bits."""
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("Direct composition encoding requires four unit-scale blocks")
    return encode_discrete_values(composition, encoding)


def cgfm_composition_to_angles(
    composition: Sequence[float],
    phase_permutation: Sequence[int],
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """把四相 composition 映射成三个 CGFM 角度。.

    这里使用类似 hyperspherical coordinates 的参数化：每个角度决定当前相
    与剩余 tail fraction 的比例。这样后续只需优化三个角度 block。
    """
    composition_array = np.asarray(composition, dtype=np.float64)
    permutation = np.asarray(phase_permutation, dtype=np.int64)
    if composition_array.shape[0] != 4 or permutation.shape[0] != 4:
        raise ValueError("CGFM requires four phase fractions and a four-element permutation")
    permuted = np.clip(composition_array[permutation], 0.0, None)
    total = float(np.sum(permuted))
    if total <= tolerance:
        raise ValueError("Composition total must be positive")
    permuted = permuted / total

    angles = np.zeros(3, dtype=np.float64)
    for angle_idx in range(3):
        current = float(permuted[angle_idx])
        tail = float(np.sum(permuted[angle_idx + 1 :]))
        if current <= tolerance and tail <= tolerance:
            angles[angle_idx] = 0.0
        else:
            angles[angle_idx] = np.arctan2(np.sqrt(max(tail, 0.0)), np.sqrt(max(current, 0.0)))
    return np.clip(angles, 0.0, CGFM_ANGLE_MAX)


def cgfm_angles_to_composition(
    angles: Sequence[float],
    phase_permutation: Sequence[int],
) -> np.ndarray:
    """把三个 CGFM 角度解码回四相 composition。.

    `cos^2/sin^2` 形式保证每一项非负；归一化和反 permutation 后得到原始相顺序。
    """
    angle_array = np.clip(np.asarray(angles, dtype=np.float64), 0.0, CGFM_ANGLE_MAX)
    permutation = np.asarray(phase_permutation, dtype=np.int64)
    if angle_array.shape[0] != 3 or permutation.shape[0] != 4:
        raise ValueError("CGFM requires three angles and a four-element permutation")

    s1, s2, s3 = np.sin(angle_array)
    c1, c2, c3 = np.cos(angle_array)
    permuted = np.array(
        [
            c1 * c1,
            (s1 * s1) * (c2 * c2),
            (s1 * s1) * (s2 * s2) * (c3 * c3),
            (s1 * s1) * (s2 * s2) * (s3 * s3),
        ],
        dtype=np.float64,
    )
    total = float(np.sum(permuted))
    if total > 0.0:
        permuted /= total
    composition = np.zeros(4, dtype=np.float64)
    composition[permutation] = permuted
    return composition


def encode_single_objective_rows(
    rows: Sequence[dict[str, float | int]],
    encoding: IterationEncoding,
) -> np.ndarray:
    """w/o CGFM 特征：四个 block 直接表示四个 normalized phase fractions。."""
    features = np.zeros((len(rows), encoding.num_blocks * encoding.num_levels), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        composition = np.array(
            [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
            dtype=np.float64,
        )
        features[row_idx] = encode_discrete_composition(composition, encoding)
    return features


def encode_cgfm_rows(
    rows: Sequence[dict[str, float | int]],
    encoding: IterationEncoding,
) -> np.ndarray:
    """w/ CGFM 特征：先把四相 composition 转成三个角度，再做离散 bit 编码。."""
    if encoding.num_blocks != 3 or encoding.phase_permutation is None:
        raise ValueError("CGFM encoding requires three blocks and a phase permutation")
    features = np.zeros((len(rows), encoding.num_blocks * encoding.num_levels), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        composition = np.array(
            [float(row["f1_norm"]), float(row["f2_norm"]), float(row["f3_norm"]), float(row["f4_norm"])],
            dtype=np.float64,
        )
        angles = cgfm_composition_to_angles(composition, encoding.phase_permutation)
        features[row_idx] = encode_discrete_values(angles, encoding)
    return features


def build_system_penalty_matrix(encoding: IterationEncoding) -> tuple[np.ndarray, float]:
    """约束直接编码下四个相分数的总和接近 1。.

    该项只适用于 w/o CGFM 的四 block composition 编码；CGFM 角度解码已经天然满足
    simplex 总和约束，不应再叠加这个 penalty。
    """
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("System penalty is defined only for direct four-fraction encoding")
    alpha = np.concatenate(
        [block.astype(np.float64) / float(encoding.num_levels) for block in encoding.positive_count_by_bit]
    )
    q = np.outer(alpha, alpha)
    q[np.diag_indices_from(q)] -= 2.0 * alpha
    return q, 1.0


def build_one_hot_penalty_matrix(encoding: IterationEncoding) -> tuple[np.ndarray, float]:
    """约束每个 block 最多选择一个 positive count。."""
    size = encoding.num_blocks * encoding.num_levels
    q = np.zeros((size, size), dtype=np.float64)
    for block_idx in range(encoding.num_blocks):
        start = block_idx * encoding.num_levels
        end = start + encoding.num_levels
        q[start:end, start:end] += 1.0
        q[np.arange(start, end), np.arange(start, end)] = 0.0
    return q, 0.0


def _maximum_abs_qubo_coefficient(q: np.ndarray) -> float:
    """返回二元多项式中的最大绝对系数，与 Q 矩阵存储形式无关。."""
    q_array = np.asarray(q, dtype=np.float64)
    if q_array.ndim != 2 or q_array.shape[0] != q_array.shape[1]:
        raise ValueError("QUBO matrix must be square")
    if q_array.shape[0] == 0:
        return 0.0

    linear_scale = float(np.max(np.abs(np.diag(q_array))))
    row_indices, column_indices = np.triu_indices(q_array.shape[0], k=1)
    if row_indices.size == 0:
        return linear_scale
    quadratic_coefficients = q_array[row_indices, column_indices] + q_array[column_indices, row_indices]
    quadratic_scale = float(np.max(np.abs(quadratic_coefficients)))
    return max(linear_scale, quadratic_scale)


def normalize_qubo_term(
    q: np.ndarray,
    bias: float,
    *,
    normalization_scheme: str,
) -> tuple[np.ndarray, float, float]:
    """按实际线性/二次多项式系数归一化，常数偏置不参与尺度计算。."""
    if normalization_scheme != QUBO_NORMALIZATION_SCHEME:
        raise ValueError(f"Unsupported QUBO normalization scheme: {normalization_scheme!r}")
    q_array = np.asarray(q, dtype=np.float64)
    scale = _maximum_abs_qubo_coefficient(q_array)
    if scale <= 0.0:
        return q_array.copy(), float(bias), 1.0
    return q_array / scale, float(bias) / scale, scale


def build_single_objective_qubo(
    fm_q: np.ndarray,
    fm_bias: float,
    encoding: IterationEncoding,
    qubo_config: QuboConfig,
    include_system_penalty: bool = True,
) -> QuboBuildResult:
    """合成最终交给 SA 的 QUBO。.

    FM 负责学习目标函数；one-hot penalty 保证每个 block 的离散选择合法；
    system penalty 只在直接四相分数编码时加入，用来惩罚四相总和偏离 1。
    """
    fm_q_array = np.asarray(fm_q, dtype=np.float64)
    expected_size = encoding.num_blocks * encoding.num_levels
    if fm_q_array.shape != (expected_size, expected_size):
        raise ValueError(f"FM QUBO matrix shape must be {(expected_size, expected_size)} for this encoding")
    total_q = np.zeros_like(fm_q_array)
    total_bias = 0.0

    fm_scale = 0.0
    if qubo_config.fm_objective_weight > 0.0:
        normalized_fm_q, normalized_fm_bias, fm_scale = normalize_qubo_term(
            fm_q_array,
            fm_bias,
            normalization_scheme=qubo_config.normalization_scheme,
        )
        total_q += qubo_config.fm_objective_weight * normalized_fm_q
        total_bias += qubo_config.fm_objective_weight * normalized_fm_bias

    one_hot_scale = 0.0
    if qubo_config.one_hot_penalty_weight > 0.0:
        one_hot_q, one_hot_bias = build_one_hot_penalty_matrix(encoding)
        normalized_one_hot_q, normalized_one_hot_bias, one_hot_scale = normalize_qubo_term(
            one_hot_q,
            one_hot_bias,
            normalization_scheme=qubo_config.normalization_scheme,
        )
        total_q += qubo_config.one_hot_penalty_weight * normalized_one_hot_q
        total_bias += qubo_config.one_hot_penalty_weight * normalized_one_hot_bias

    system_scale = 0.0
    system_penalty_weight = 0.0
    if include_system_penalty and qubo_config.system_penalty_weight > 0.0:
        system_q, system_bias = build_system_penalty_matrix(encoding)
        normalized_system_q, normalized_system_bias, system_scale = normalize_qubo_term(
            system_q,
            system_bias,
            normalization_scheme=qubo_config.normalization_scheme,
        )
        total_q += qubo_config.system_penalty_weight * normalized_system_q
        total_bias += qubo_config.system_penalty_weight * normalized_system_bias
        system_penalty_weight = qubo_config.system_penalty_weight

    stats = QuboStats(
        normalization_scheme=qubo_config.normalization_scheme,
        fm_objective_weight=qubo_config.fm_objective_weight,
        system_penalty_weight=float(system_penalty_weight),
        one_hot_penalty_weight=qubo_config.one_hot_penalty_weight,
        fm_scale=float(fm_scale),
        system_scale=float(system_scale),
        one_hot_scale=float(one_hot_scale),
        num_variables=int(total_q.shape[0]),
        max_abs=_maximum_abs_qubo_coefficient(total_q),
    )
    return QuboBuildResult(q=total_q, bias=float(total_bias), stats=stats)


def decode_candidate_bits_to_values(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> np.ndarray | None:
    """Decode one-hot candidate bits to block values."""
    bit_array = np.asarray(bit_vector, dtype=np.float64)
    if bit_array.shape[0] != encoding.num_blocks * encoding.num_levels:
        raise ValueError("Unexpected bit vector length")
    values = np.zeros(encoding.num_blocks, dtype=np.float64)
    for block_idx in range(encoding.num_blocks):
        start = block_idx * encoding.num_levels
        end = start + encoding.num_levels
        block = bit_array[start:end]
        active_indices = np.flatnonzero(block > 0.5)
        if len(active_indices) > 1:
            return None
        if len(active_indices) == 1:
            count_value = int(encoding.positive_count_by_bit[block_idx][int(active_indices[0])])
            values[block_idx] = (count_value / float(encoding.num_levels)) * encoding.value_scale
    return values


def decode_candidate_bits_to_composition(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> np.ndarray | None:
    """Decode candidate bits to a direct composition."""
    values = decode_candidate_bits_to_values(bit_vector, encoding)
    if values is None:
        return None
    if encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE:
        raise ValueError("Direct composition decoding requires four unit-scale blocks")
    return values


def decode_candidate_bits_to_cgfm_composition(
    bit_vector: np.ndarray,
    encoding: IterationEncoding,
) -> np.ndarray | None:
    """Decode candidate bits through the CGFM mapping."""
    values = decode_candidate_bits_to_values(bit_vector, encoding)
    if values is None:
        return None
    if encoding.num_blocks != 3 or encoding.phase_permutation is None:
        raise ValueError("CGFM decoding requires three blocks and a phase permutation")
    return cgfm_angles_to_composition(values, encoding.phase_permutation)


def _qubo_to_bqm(q: np.ndarray, bias: float) -> dimod.BinaryQuadraticModel:
    linear = {idx: float(q[idx, idx]) for idx in range(q.shape[0])}
    quadratic = {}
    for i in range(q.shape[0]):
        for j in range(i + 1, q.shape[1]):
            coefficient = float(q[i, j] + q[j, i])
            if abs(coefficient) > 0.0:
                quadratic[(i, j)] = coefficient
    return dimod.BinaryQuadraticModel(linear, quadratic, float(bias), dimod.BINARY)


def solve_qubo_with_sa(
    q: np.ndarray,
    bias: float,
    reads: int,
    sweeps: int,
    seed: int,
) -> SASamplingResult:
    """Return every SA read, sorted by energy, instead of discarding alternatives.

    The lowest-energy unconstrained sample is not necessarily a legal one-hot/system
    state. Keeping the full sample batch lets the pipeline select the lowest-energy
    feasible state without replacing an optimization result with random data.
    """
    reads = require_integer("reads", reads, minimum=1)
    sweeps = require_integer("sweeps", sweeps, minimum=1)
    seed = require_integer("seed", seed, minimum=0, maximum=NEAL_SEED_MAX)
    bqm = _qubo_to_bqm(q, bias)
    sampler = SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=reads, num_sweeps=sweeps, seed=seed)
    variables = [int(variable) for variable in sample_set.variables]
    order = np.argsort(sample_set.record.energy, kind="stable")
    samples = []
    for record_index in order:
        state = np.zeros(q.shape[0], dtype=np.float64)
        raw_state = sample_set.record.sample[int(record_index)]
        for column_index, variable in enumerate(variables):
            state[variable] = float(raw_state[column_index])
        samples.append(
            SASample(
                state=state,
                energy=float(sample_set.record.energy[int(record_index)]),
            )
        )
    if not samples:
        raise RuntimeError("Simulated annealing returned no samples")
    return SASamplingResult(samples=tuple(samples))


def select_lowest_energy_feasible_sample(
    sampling: SASamplingResult,
    is_feasible: Callable[[np.ndarray], bool],
) -> FeasibleSASolution:
    """Select the first feasible state from an energy-sorted SA sample batch."""
    for sample_index, sample in enumerate(sampling.samples):
        if is_feasible(sample.state):
            return FeasibleSASolution(
                state=sample.state.copy(),
                energy=float(sample.energy),
                rank=sample_index + 1,
                infeasible_samples_skipped=sample_index,
            )
    raise RuntimeError(f"No feasible QUBO candidate found among {sampling.num_samples} simulated-annealing reads")


def _iter_unit_sum_counts(num_blocks: int, remaining: int) -> Iterator[tuple[int, ...]]:
    if num_blocks == 1:
        yield (remaining,)
        return
    for count in range(remaining + 1):
        for suffix in _iter_unit_sum_counts(num_blocks - 1, remaining - count):
            yield (count, *suffix)


def _active_qubo_energy(q: np.ndarray, bias: float, active_indices: Sequence[int]) -> float:
    energy = float(bias) + sum(float(q[index, index]) for index in active_indices)
    for left_position, left_index in enumerate(active_indices):
        for right_index in active_indices[left_position + 1 :]:
            energy += float(q[left_index, right_index]) + float(q[right_index, left_index])
    if not math.isfinite(energy):
        raise ValueError("QUBO energy must remain finite for every diagnostic state")
    return energy


def _state_is_feasible_for_encoding(
    state: np.ndarray,
    encoding: IterationEncoding,
    *,
    require_unit_sum: bool,
) -> bool:
    total_count = 0
    for block_index in range(encoding.num_blocks):
        start = block_index * encoding.num_levels
        block = state[start : start + encoding.num_levels]
        active = np.flatnonzero(block > 0.5)
        if len(active) > 1:
            return False
        if len(active) == 1:
            total_count += int(encoding.positive_count_by_bit[block_index][int(active[0])])
    return not require_unit_sum or total_count == encoding.num_levels


def diagnose_no_feasible_candidate(
    q: np.ndarray,
    bias: float,
    sampling: SASamplingResult,
    encoding: IterationEncoding,
    *,
    require_unit_sum: bool,
    max_exact_diagnostic_states: int = MAX_EXACT_DIAGNOSTIC_STATES,
) -> NoFeasibleCandidateDiagnostic:
    """Classify an all-infeasible SA batch by exact feasible-domain enumeration.

    This function is diagnostic only: it deliberately returns energies and a
    classification, never the exact feasible state itself.
    """
    max_states = require_integer(
        "max_exact_diagnostic_states",
        max_exact_diagnostic_states,
        minimum=1,
    )
    if type(require_unit_sum) is not bool:
        raise ValueError("require_unit_sum must be a boolean")
    if isinstance(bias, bool) or not isinstance(bias, Real) or not math.isfinite(float(bias)):
        raise ValueError("QUBO bias must be a finite real number")

    raw_q = np.asarray(q)
    if raw_q.dtype.kind not in "iuf":
        raise ValueError("QUBO matrix must contain real numeric coefficients")
    q_array = raw_q.astype(np.float64, copy=False)
    num_variables = encoding.num_blocks * encoding.num_levels
    if q_array.shape != (num_variables, num_variables):
        raise ValueError(f"QUBO matrix shape must be {(num_variables, num_variables)} for this encoding")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("QUBO matrix coefficients must be finite")
    if require_unit_sum and (encoding.num_blocks != 4 or abs(encoding.value_scale - 1.0) > DEFAULT_TOLERANCE):
        raise ValueError("Unit-sum diagnostics require direct four-fraction encoding")
    if not sampling.samples:
        raise ValueError("sampling must contain at least one infeasible state")

    sampled_energies: list[float] = []
    for sample_index, sample in enumerate(sampling.samples):
        raw_state = np.asarray(sample.state)
        if raw_state.dtype.kind not in "iuf":
            raise ValueError(f"sampling.samples[{sample_index}].state must contain numeric binary values")
        state = raw_state.astype(np.float64, copy=False)
        if state.shape != (num_variables,):
            raise ValueError(f"sampling.samples[{sample_index}].state must have shape {(num_variables,)}")
        if not np.all(np.isfinite(state)) or not np.all((state == 0.0) | (state == 1.0)):
            raise ValueError(f"sampling.samples[{sample_index}].state must contain finite binary values")
        if _state_is_feasible_for_encoding(
            state,
            encoding,
            require_unit_sum=require_unit_sum,
        ):
            raise ValueError("sampling must contain no feasible encoded state")
        sampled_energies.append(
            _active_qubo_energy(
                q_array,
                float(bias),
                [int(index) for index in np.flatnonzero(state)],
            )
        )
    lowest_sampled_energy = min(sampled_energies)

    if require_unit_sum:
        feasible_state_count = math.comb(
            encoding.num_levels + encoding.num_blocks - 1,
            encoding.num_blocks - 1,
        )
    else:
        feasible_state_count = (encoding.num_levels + 1) ** encoding.num_blocks
    if feasible_state_count > max_states:
        return NoFeasibleCandidateDiagnostic(
            classification="unclassified_no_feasible_candidate",
            lowest_sampled_infeasible_energy=lowest_sampled_energy,
            exact_best_feasible_energy=None,
            exact_feasible_state_count=feasible_state_count,
            max_exact_diagnostic_states=max_states,
        )

    if require_unit_sum:
        count_candidates = _iter_unit_sum_counts(
            encoding.num_blocks,
            encoding.num_levels,
        )
    else:
        count_candidates = product(
            range(encoding.num_levels + 1),
            repeat=encoding.num_blocks,
        )

    exact_best_energy = float("inf")
    enumerated_states = 0
    for counts in count_candidates:
        enumerated_states += 1
        active_indices = [
            (block_index * encoding.num_levels) + int(encoding.bit_index_by_count[block_index][count])
            for block_index, count in enumerate(counts)
            if count > 0
        ]
        exact_best_energy = min(
            exact_best_energy,
            _active_qubo_energy(q_array, float(bias), active_indices),
        )
    if enumerated_states != feasible_state_count or not math.isfinite(exact_best_energy):
        raise RuntimeError("Internal error while enumerating the exact feasible encoding domain")

    if math.isclose(
        lowest_sampled_energy,
        exact_best_energy,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        classification: NoFeasibleCandidateClassification = "unclassified_no_feasible_candidate"
    elif lowest_sampled_energy < exact_best_energy:
        classification = "qubo_penalty_balance_failure"
    else:
        classification = "heuristic_sampling_failure"
    return NoFeasibleCandidateDiagnostic(
        classification=classification,
        lowest_sampled_infeasible_energy=lowest_sampled_energy,
        exact_best_feasible_energy=exact_best_energy,
        exact_feasible_state_count=feasible_state_count,
        max_exact_diagnostic_states=max_states,
    )


__all__ = [
    "CGFM_ANGLE_MAX",
    "DEFAULT_TOLERANCE",
    "FeasibleSASolution",
    "IterationEncoding",
    "MAX_EXACT_DIAGNOSTIC_STATES",
    "NoFeasibleCandidateDiagnostic",
    "QuboBuildResult",
    "QuboStats",
    "SASample",
    "SASamplingResult",
    "build_one_hot_penalty_matrix",
    "build_single_objective_qubo",
    "build_system_penalty_matrix",
    "cgfm_angles_to_composition",
    "cgfm_composition_to_angles",
    "counts_to_composition",
    "create_cgfm_iteration_encoding",
    "create_iteration_encoding",
    "cure_discrete_counts",
    "decode_candidate_bits_to_cgfm_composition",
    "decode_candidate_bits_to_composition",
    "decode_candidate_bits_to_values",
    "diagnose_no_feasible_candidate",
    "encode_cgfm_rows",
    "encode_discrete_composition",
    "encode_discrete_values",
    "encode_single_objective_rows",
    "evaluate_qubo_energy",
    "get_positive_level_values",
    "normalize_qubo_term",
    "prepare_discrete_composition",
    "select_lowest_energy_feasible_sample",
    "solve_qubo_with_sa",
    "validate_candidate_composition",
]
