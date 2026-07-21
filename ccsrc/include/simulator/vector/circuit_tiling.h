/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_H_
#define INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_H_

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

#include "core/mq_base_types.h"
#include "ops/basic_gate.h"

namespace mindquantum::sim::vector::detail::circuit_tiling {
// A 2^8-amplitude tile is the measured cache/performance optimum.  This is an
// execution policy, not a user-facing tuning parameter.
inline constexpr size_t kMaxTileQubits = 8;

struct IndexPair {
    size_t lhs = 0;
    size_t rhs = 0;
};

template <typename value_t>
struct PauliPair : IndexPair {
    value_t lhs_from_rhs = 0;
    value_t rhs_from_lhs = 0;
};

template <typename value_t>
struct Scale {
    size_t basis = 0;
    value_t factor = 1;
};

template <typename value_t>
struct PauliKernel {
    index_t external_z_mask = 0;
    std::vector<Scale<value_t>> scales;
    std::vector<PauliPair<value_t>> pairs;
};

template <typename value_t>
struct PauliRotationKernel {
    using calc_type = typename value_t::value_type;

    index_t external_z_mask = 0;
    calc_type cosine = 1;
    calc_type sine = 0;

    // With no external Z operand the builder resolves the complete local
    // coefficients once.  Otherwise the executor applies the state-dependent
    // external parity sign.
    bool coefficients_prepared = false;
    std::vector<Scale<value_t>> scales;
    std::vector<PauliPair<value_t>> pairs;
};

template <typename value_t>
struct DiagonalKernel {
    std::vector<Scale<value_t>> scales;
};

struct HadamardKernel {
    std::vector<IndexPair> pairs;
};

template <typename value_t>
struct PairMatrixKernel {
    std::array<value_t, 4> matrix{value_t{1, 0}, value_t{0, 0}, value_t{0, 0}, value_t{1, 0}};
    std::vector<IndexPair> pairs;
};

struct SwapKernel {
    std::vector<IndexPair> pairs;
};

template <typename value_t>
struct ISwapKernel {
    value_t phase = value_t{0, 1};
    std::vector<IndexPair> pairs;
};

struct FSimIndices {
    size_t lhs = 0;
    size_t rhs = 0;
    size_t both = 0;
};

template <typename value_t>
struct FSimKernel {
    std::array<value_t, 4> exchange{value_t{1, 0}, value_t{0, 0}, value_t{0, 0}, value_t{1, 0}};
    value_t phase = 1;
    std::vector<FSimIndices> indices;
};

template <typename value_t>
struct DenseMatrixKernel {
    size_t matrix_dim = 0;
    // Consecutive matrix_dim entries describe one independent local basis
    // group.  Indices are fully projected during lowering, so the hot loop
    // performs no bit insertion or masking.
    std::vector<size_t> indices;
    std::vector<value_t> matrix;
};

template <typename value_t>
using Kernel = std::variant<PauliKernel<value_t>, PauliRotationKernel<value_t>, DiagonalKernel<value_t>, HadamardKernel,
                            PairMatrixKernel<value_t>, SwapKernel, ISwapKernel<value_t>, FSimKernel<value_t>,
                            DenseMatrixKernel<value_t>>;

template <typename value_t>
struct Operation {
    index_t external_control_mask = 0;
    Kernel<value_t> kernel;
};

template <typename value_t>
using DerivativeKernel = std::variant<PauliRotationKernel<value_t>, DiagonalKernel<value_t>, PairMatrixKernel<value_t>,
                                      DenseMatrixKernel<value_t>>;

template <typename value_t>
struct GradientTarget {
    size_t output_index = 0;
    value_t coefficient = 0;
};

template <typename value_t>
struct Derivative {
    index_t external_control_mask = 0;
    DerivativeKernel<value_t> kernel;
    std::vector<GradientTarget<value_t>> targets;
};

template <typename value_t>
struct AdjointOperation {
    // Adjoint-circuit order is significant: evolve the shared bra, evaluate
    // derivatives against the current ket states, then evolve those kets.
    Operation<value_t> evolution;
    std::vector<Derivative<value_t>> derivatives;
};

struct TileLayout {
    qbits_t qubits;
    size_t dimension = 0;
    index_t tile_count = 0;
    std::vector<index_t> state_offsets;
    size_t workspace_size = 0;
};

template <typename value_t>
struct PreparedTile {
    TileLayout layout;
    std::vector<Operation<value_t>> operations;
};

template <typename value_t>
struct PreparedAdjointTile {
    TileLayout layout;
    std::vector<AdjointOperation<value_t>> operations;
};

template <typename value_t>
struct AdjointProgram {
    size_t output_size = 0;
    // The initial bra-ket overlaps can share the first tile's state gather.
    bool compute_overlap = false;
    std::vector<PreparedAdjointTile<value_t>> tiles;
};

// A scheduling-level tile: every gate that mixes amplitudes acts only on
// `qubits`.  Controls and diagonal operands may remain external.
struct CircuitTile {
    qbits_t qubits;
    std::vector<std::shared_ptr<BasicGate>> gates;
};

using ScheduleEntry = std::variant<std::shared_ptr<BasicGate>, CircuitTile>;
using Schedule = std::vector<ScheduleEntry>;
}  // namespace mindquantum::sim::vector::detail::circuit_tiling

#endif  // INCLUDE_SIMULATOR_VECTOR_CIRCUIT_TILING_H_
