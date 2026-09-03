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

#ifndef INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_SCALAR_RUNTIME_H_
#define INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_SCALAR_RUNTIME_H_

#include <cmath>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "simulator/circuit_tiling/circuit_tiling.h"
#include "simulator/state_index.h"

namespace mindquantum::sim::circuit_tiling::detail {
struct ScalarBlockContext {
    index_t mirror_base = 0;
    bool mirrored = false;
};

// Enumerates the blocks of a tile.  A Hermitian-symmetric tile only visits
// the blocks on or below its block diagonal: every block strictly below the
// diagonal is paired with the conjugate transposed mirror block above it,
// which the runtime writes instead of evolving.
class ScalarBlockTraversal {
 public:
    explicit ScalarBlockTraversal(const TileLayout& layout)
        : expander_(layout.qubits), mirror_shift_(layout.mirror_shift), count_(layout.tile_count) {
        if (mirror_shift_ != 0) {
            // Excluded qubits come in image pairs, so a compressed block
            // index splits into equal column and row halves.
            half_bits_ = static_cast<size_t>(mirror_shift_) - layout.qubits.size() / 2;
            const auto rows = index_t{1} << half_bits_;
            count_ = rows * (rows + 1) / 2;
        }
    }

    [[nodiscard]] auto Count() const noexcept -> index_t {
        return count_;
    }

    [[nodiscard]] auto Block(index_t ordinal, ScalarBlockContext* context) const noexcept -> index_t {
        if (mirror_shift_ == 0) {
            *context = {};
            return expander_.Expand(ordinal);
        }
        const auto [row, column] = TriangularCoordinates(ordinal);
        const auto base = expander_.Expand(column | (row << half_bits_));
        const auto column_mask = (index_t{1} << mirror_shift_) - 1;
        context->mirror_base = ((base & column_mask) << mirror_shift_) | (base >> mirror_shift_);
        context->mirrored = row != column;
        return base;
    }

 private:
    // The row-major coordinates of the ordinal-th entry of a lower
    // triangle: the largest row with row * (row + 1) / 2 <= ordinal.
    [[nodiscard]] static auto TriangularCoordinates(index_t ordinal) noexcept -> std::pair<index_t, index_t> {
        auto row = static_cast<index_t>((std::sqrt(8.0 * static_cast<double>(ordinal) + 1.0) - 1.0) / 2.0);
        while (row * (row + 1) / 2 > ordinal) {
            --row;
        }
        while ((row + 1) * (row + 2) / 2 <= ordinal) {
            ++row;
        }
        return {row, ordinal - row * (row + 1) / 2};
    }

    state_index::BitExpander expander_;
    qbit_t mirror_shift_;
    size_t half_bits_ = 0;
    index_t count_;
};

// The portable algebra: one complex amplitude per operation.
template <typename value_t_>
struct ScalarTileAlgebra {
    using value_t = value_t_;
    using calc_type = typename value_t::value_type;
    using vector_t = value_t;
    using predicate_t = bool;
    using state_index_t = index_t;
    using index_context_t = ScalarBlockContext;
    using storage_t = value_t;

    [[nodiscard]] static auto ExternalControls(predicate_t active, state_index_t state_base, const index_context_t&,
                                               index_t mask) -> predicate_t {
        return active && (state_base & mask) == mask;
    }

    [[nodiscard]] static auto ExternalParity(predicate_t, state_index_t state_base, const index_context_t&,
                                             index_t mask) -> predicate_t {
        return (CountOne(static_cast<std::uint64_t>(state_base & mask)) & 1U) != 0;
    }

    [[nodiscard]] static auto Skip(predicate_t predicate) -> bool {
        return !predicate;
    }

    [[nodiscard]] static auto SignedCoefficient(value_t coefficient, predicate_t negative) -> vector_t {
        return negative ? -coefficient : coefficient;
    }

    [[nodiscard]] static auto Zero() -> vector_t {
        return 0;
    }

    [[nodiscard]] static auto Broadcast(value_t value) -> vector_t {
        return value;
    }

    [[nodiscard]] static auto Load(const storage_t* cache, size_t basis) -> vector_t {
        return cache[basis];
    }

    static void Store(storage_t* cache, size_t basis, predicate_t predicate, vector_t value) {
        if (predicate) {
            cache[basis] = value;
        }
    }

    [[nodiscard]] static auto Add(predicate_t, vector_t lhs, vector_t rhs) -> vector_t {
        return lhs + rhs;
    }

    [[nodiscard]] static auto Subtract(predicate_t, vector_t lhs, vector_t rhs) -> vector_t {
        return lhs - rhs;
    }

    [[nodiscard]] static auto Multiply(predicate_t, vector_t value, calc_type scalar) -> vector_t {
        return value * scalar;
    }

    [[nodiscard]] static auto MultiplyCoefficient(predicate_t, vector_t value, value_t coefficient) -> vector_t {
        return value * coefficient;
    }

    [[nodiscard]] static auto MultiplyDynamic(predicate_t, vector_t value, vector_t coefficient) -> vector_t {
        return value * coefficient;
    }

    [[nodiscard]] static auto MultiplyAddCoefficient(predicate_t, vector_t accumulator, vector_t value,
                                                     value_t coefficient) -> vector_t {
        return accumulator + value * coefficient;
    }

    [[nodiscard]] static auto MultiplyAddDynamic(predicate_t, vector_t accumulator, vector_t value,
                                                 vector_t coefficient) -> vector_t {
        return accumulator + value * coefficient;
    }

    [[nodiscard]] static auto InnerProductAdd(predicate_t, vector_t accumulator, vector_t bra, vector_t ket)
        -> vector_t {
        return accumulator + std::conj(bra) * ket;
    }

    [[nodiscard]] static auto Reduce(predicate_t, vector_t value) -> value_t {
        return value;
    }
};

template <typename value_t>
struct ScalarTileRuntime {
    using algebra_t = ScalarTileAlgebra<value_t>;
    using storage_t = typename algebra_t::storage_t;
    using traversal_t = ScalarBlockTraversal;

    // Hermitian-symmetric tiles evolve only their lower block triangle.
    static constexpr bool kMirrorsHermitianTiles = true;

    [[nodiscard]] static auto MakeTraversal(const TileLayout& layout) -> traversal_t {
        return traversal_t(layout);
    }

    [[nodiscard]] static auto ThreadScratch(size_t size) -> storage_t* {
        thread_local std::vector<storage_t> scratch;
        if (scratch.size() < size) {
            scratch.resize(size);
        }
        return scratch.data();
    }

    [[nodiscard]] static auto Offset(storage_t* storage, size_t logical_offset) -> storage_t* {
        return storage + logical_offset;
    }

    [[nodiscard]] static auto Offset(const storage_t* storage, size_t logical_offset) -> const storage_t* {
        return storage + logical_offset;
    }

    template <typename function_t>
    static void ForEach(const traversal_t& traversal, const function_t& function) {
        ScalarBlockContext context;
        for (index_t block = 0; block < traversal.Count(); ++block) {
            const auto state_base = traversal.Block(block, &context);
            function(true, state_base, context);
        }
    }

    template <typename function_t>
    static void ForEachInParallel(const traversal_t& traversal, const function_t& function) {
        MQ_DO_PRAGMA(omp for schedule(static))
        for (omp::idx_t block = 0; block < static_cast<omp::idx_t>(traversal.Count()); ++block) {
            ScalarBlockContext context;
            const auto state_base = traversal.Block(static_cast<index_t>(block), &context);
            function(true, state_base, context);
        }
    }

    static void Gather(const value_t* state, const TileLayout& layout, bool, index_t state_base,
                       const ScalarBlockContext&, storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            tile[basis] = state[state_base | layout.state_offsets[basis]];
        }
    }

    static void Scatter(value_t* state, const TileLayout& layout, bool, index_t state_base,
                        const ScalarBlockContext& context, const storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            state[state_base | layout.state_offsets[basis]] = tile[basis];
        }
        if (context.mirrored) {
            for (size_t basis = 0; basis < layout.dimension; ++basis) {
                state[context.mirror_base | layout.state_offsets[MirrorBasis(layout, basis)]] = std::conj(tile[basis]);
            }
        }
    }

    [[nodiscard]] static auto Mirrored(const ScalarBlockContext& context) noexcept -> bool {
        return context.mirrored;
    }

    [[nodiscard]] static auto MirrorBase(const ScalarBlockContext& context) noexcept -> index_t {
        return context.mirror_base;
    }

    // The conjugate transpose of a local block: the mirror block's content.
    static void MirrorBlock(const TileLayout& layout, const storage_t* block, storage_t* mirror) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            mirror[MirrorBasis(layout, basis)] = std::conj(block[basis]);
        }
    }

 private:
    // Swaps the column and row image halves of a local basis index.
    [[nodiscard]] static auto MirrorBasis(const TileLayout& layout, size_t basis) noexcept -> size_t {
        const auto half = layout.qubits.size() / 2;
        const auto low = (size_t{1} << half) - 1;
        return ((basis & low) << half) | (basis >> half);
    }
};
}  // namespace mindquantum::sim::circuit_tiling::detail

#endif  // INCLUDE_SIMULATOR_CIRCUIT_TILING_DETAIL_SCALAR_RUNTIME_H_
