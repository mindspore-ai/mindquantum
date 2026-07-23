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

#include <arm_sve.h>

#include <complex>
#include <cstdint>
#include <vector>

#include "cpu_vector_arm_backend.h"
#include "cpu_vector_sve_utils.h"
#include "core/utils.h"
#include "simulator/vector/detail/circuit_tiling_executor.h"

#if !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_COMPLEX)
#    error "The Arm SVE circuit-tiling backend requires SVE FCMLA support."
#endif

namespace mindquantum::sim::vector::detail {
namespace {
struct SveIndexContext {
    index_t window_origin = 0;
    index_t local_mask = 0;
};

template <typename calc_type_>
struct SveTileAlgebra {
    using calc_type = calc_type_;
    using value_t = std::complex<calc_type>;
    using vector_t = sve::vector_t<calc_type>;
    using predicate_t = svbool_t;
    using state_index_t = sve::index_vector_t<calc_type>;
    using index_context_t = SveIndexContext;
    using storage_t = calc_type;

    [[nodiscard]] static auto ExternalControls(predicate_t active, state_index_t state_base,
                                               const index_context_t& context, index_t mask) -> predicate_t {
        if (mask == 0) {
            return active;
        }
        const auto fixed_mask = mask & ~context.local_mask;
        if ((context.window_origin & fixed_mask) != fixed_mask) {
            return sve::FalsePredicate();
        }
        const auto local_mask = mask & context.local_mask;
        return local_mask == 0 ? active
                               : sve::IndexEquals<calc_type>(
                                     active, sve::AndIndex<calc_type>(active, state_base, local_mask), local_mask);
    }

    [[nodiscard]] static auto ExternalParity(predicate_t predicate, state_index_t state_base,
                                             const index_context_t& context, index_t mask) -> predicate_t {
        const auto fixed_mask = mask & ~context.local_mask;
        const auto fixed_parity
            = (CountOne(static_cast<std::uint64_t>(context.window_origin & fixed_mask)) & 1U) != 0;
        auto parity = sve::DuplicateIndex<calc_type>(fixed_parity ? 1 : 0);
        auto local_mask = mask & context.local_mask;
        while (local_mask != 0) {
            const auto bit = local_mask & (~local_mask + 1);
            const auto set = sve::IndexNotEquals<calc_type>(predicate,
                                                            sve::AndIndex<calc_type>(predicate, state_base, bit), 0);
            parity = sve::XorIndex<calc_type>(
                predicate, parity,
                sve::SelectIndex<calc_type>(set, sve::DuplicateIndex<calc_type>(1), sve::DuplicateIndex<calc_type>(0)));
            local_mask &= local_mask - 1;
        }
        return sve::IndexNotEquals<calc_type>(predicate, parity, 0);
    }

    // SVE arithmetic is predicated already.  Testing every operation for an
    // empty predicate costs more than allowing the masked instructions to run.
    [[nodiscard]] static auto Skip(predicate_t) -> bool {
        return false;
    }

    [[nodiscard]] static auto SignedCoefficient(value_t coefficient, predicate_t negative) -> vector_t {
        const auto predicate = sve::TruePredicate<calc_type>();
        const auto positive = sve::BroadcastComplex<calc_type>(coefficient);
        return sve::Select<calc_type>(negative, sve::Negate<calc_type>(predicate, positive), positive);
    }

    [[nodiscard]] static auto Zero() -> vector_t {
        return sve::Zero<calc_type>();
    }

    [[nodiscard]] static auto Broadcast(value_t value) -> vector_t {
        return sve::BroadcastComplex<calc_type>(value);
    }

    [[nodiscard]] static auto Load(const storage_t* cache, size_t basis) -> vector_t {
        return sve::LoadCachedComplex<calc_type>(cache, basis);
    }

    static void Store(storage_t* cache, size_t basis, predicate_t predicate, vector_t value) {
        sve::StoreCachedComplex<calc_type>(cache, basis, predicate, value);
    }

    [[nodiscard]] static auto Add(predicate_t predicate, vector_t lhs, vector_t rhs) -> vector_t {
        return sve::Add<calc_type>(predicate, lhs, rhs);
    }

    [[nodiscard]] static auto Subtract(predicate_t predicate, vector_t lhs, vector_t rhs) -> vector_t {
        return sve::Subtract<calc_type>(predicate, lhs, rhs);
    }

    [[nodiscard]] static auto Multiply(predicate_t predicate, vector_t value, calc_type scalar) -> vector_t {
        return sve::Multiply<calc_type>(predicate, value, scalar);
    }

    [[nodiscard]] static auto MultiplyCoefficient(predicate_t predicate, vector_t value, value_t coefficient)
        -> vector_t {
        return sve::ComplexMultiply<calc_type>(predicate, value, coefficient);
    }

    [[nodiscard]] static auto MultiplyDynamic(predicate_t predicate, vector_t value, vector_t coefficient) -> vector_t {
        return sve::ComplexMultiply<calc_type>(predicate, value, coefficient);
    }

    [[nodiscard]] static auto MultiplyAddCoefficient(predicate_t predicate, vector_t accumulator, vector_t value,
                                                     value_t coefficient) -> vector_t {
        return sve::CoefficientMulAdd<calc_type>(predicate, accumulator, value, coefficient);
    }

    [[nodiscard]] static auto MultiplyAddDynamic(predicate_t predicate, vector_t accumulator, vector_t value,
                                                 vector_t coefficient) -> vector_t {
        return sve::ComplexMulAdd<calc_type>(predicate, accumulator, value, coefficient);
    }

    [[nodiscard]] static auto Conjugate(predicate_t predicate, vector_t value) -> vector_t {
        const auto imaginary_lane = sve::IndexNotEquals<calc_type>(
            predicate, sve::AndIndex<calc_type>(predicate, sve::LaneIndex<calc_type>(), 1), 0);
        return sve::Select<calc_type>(imaginary_lane, sve::Negate<calc_type>(predicate, value), value);
    }

    [[nodiscard]] static auto InnerProductAdd(predicate_t predicate, vector_t accumulator, vector_t bra, vector_t ket)
        -> vector_t {
        return sve::ComplexMulAdd<calc_type>(predicate, accumulator, Conjugate(predicate, bra), ket);
    }

    [[nodiscard]] static auto Reduce(predicate_t predicate, vector_t value) -> value_t {
        const auto lane_parity = sve::AndIndex<calc_type>(predicate, sve::LaneIndex<calc_type>(),
                                                          static_cast<index_t>(1));
        const auto real = sve::IndexEquals<calc_type>(predicate, lane_parity, 0);
        const auto imaginary = sve::IndexNotEquals<calc_type>(predicate, lane_parity, 0);
        return {sve::ReduceAdd<calc_type>(real, value), sve::ReduceAdd<calc_type>(imaginary, value)};
    }
};

template <typename policy_t>
struct SveTileRuntime {
    using calc_type = typename policy_t::calc_type;
    using algebra_t = SveTileAlgebra<calc_type>;
    using storage_t = typename algebra_t::storage_t;
    using traversal_t = state_index::StateIndexWindow;

    [[nodiscard]] static auto MakeTraversal(const circuit_tiling::TileLayout& layout) -> traversal_t {
        return {sve::LocalStateIndexBits<calc_type>(), layout.qubits};
    }

    [[nodiscard]] static auto ThreadScratch(size_t logical_size) -> storage_t* {
        return sve::ThreadScratch<calc_type>(logical_size * sve::Traits<calc_type>::ScalarLanes());
    }

    [[nodiscard]] static auto Offset(storage_t* storage, size_t logical_offset) -> storage_t* {
        return storage + logical_offset * sve::Traits<calc_type>::ScalarLanes();
    }

    [[nodiscard]] static auto Offset(const storage_t* storage, size_t logical_offset) -> const storage_t* {
        return storage + logical_offset * sve::Traits<calc_type>::ScalarLanes();
    }

    template <typename function_t>
    static void ForEachInParallel(const traversal_t& traversal, index_t tile_count, const function_t& function) {
        sve::ForEachStateIndexBatchInParallel<calc_type>(
            traversal, tile_count, [&](const state_index::StateIndexBatch& batch, index_t state_window_origin) {
                const auto loop_index = sve::LoopIndex<calc_type>(batch.local_compressed_start);
                const auto active = sve::ActiveComplexPredicate<calc_type>(loop_index, batch.local_compressed_count);
                const auto state_base = sve::ExpandBaseIndex<calc_type>(active, loop_index,
                                                                        traversal.LocalExcludedQubits());
                function(active, state_base, SveIndexContext{state_window_origin, traversal.LocalStateMask()});
            });
    }

    static void Gather(const typename policy_t::qs_data_p_t state, const circuit_tiling::TileLayout& layout,
                       svbool_t active, sve::index_vector_t<calc_type> state_base, const SveIndexContext& index_context,
                       storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            sve::StoreCachedComplex<calc_type>(
                tile, basis, active,
                sve::GatherComplexAt<calc_type>(state, index_context.window_origin + layout.state_offsets[basis],
                                                active, state_base));
        }
    }

    static void Scatter(typename policy_t::qs_data_p_t state, const circuit_tiling::TileLayout& layout, svbool_t active,
                        sve::index_vector_t<calc_type> state_base, const SveIndexContext& index_context,
                        const storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            sve::ScatterComplexAt<calc_type>(state, index_context.window_origin + layout.state_offsets[basis], active,
                                             state_base, sve::LoadCachedComplex<calc_type>(tile, basis));
        }
    }
};
}  // namespace

template <typename policy_t>
void arm_sve::ApplyCircuitTile(typename policy_t::qs_data_p_t* state,
                               const circuit_tiling::PreparedTile<typename policy_t::py_qs_data_t>& tile, index_t dim) {
    circuit_tiling::ApplyTile<policy_t, SveTileRuntime<policy_t>>(state, tile, dim);
}

template <typename policy_t>
void arm_sve::ApplyAdjointTileProgram(typename policy_t::qs_data_p_t* shared_state,
                                      const std::vector<typename policy_t::qs_data_p_t*>& batch_states,
                                      const circuit_tiling::AdjointProgram<typename policy_t::py_qs_data_t>& program,
                                      index_t dim, VVT<typename policy_t::py_qs_data_t>* output,
                                      bool conjugate_output) {
    circuit_tiling::ApplyAdjointProgram<policy_t, SveTileRuntime<policy_t>>(shared_state, batch_states, program, dim,
                                                                            output, conjugate_output);
}

template void arm_sve::ApplyCircuitTile<CPUVectorPolicyArmFloat>(
    CPUVectorPolicyArmFloat::qs_data_p_t*, const circuit_tiling::PreparedTile<CPUVectorPolicyArmFloat::py_qs_data_t>&,
    index_t);
template void arm_sve::ApplyCircuitTile<CPUVectorPolicyArmDouble>(
    CPUVectorPolicyArmDouble::qs_data_p_t*, const circuit_tiling::PreparedTile<CPUVectorPolicyArmDouble::py_qs_data_t>&,
    index_t);
template void arm_sve::ApplyAdjointTileProgram<CPUVectorPolicyArmFloat>(
    CPUVectorPolicyArmFloat::qs_data_p_t*, const std::vector<CPUVectorPolicyArmFloat::qs_data_p_t*>&,
    const circuit_tiling::AdjointProgram<CPUVectorPolicyArmFloat::py_qs_data_t>&, index_t,
    VVT<CPUVectorPolicyArmFloat::py_qs_data_t>*, bool);
template void arm_sve::ApplyAdjointTileProgram<CPUVectorPolicyArmDouble>(
    CPUVectorPolicyArmDouble::qs_data_p_t*, const std::vector<CPUVectorPolicyArmDouble::qs_data_p_t*>&,
    const circuit_tiling::AdjointProgram<CPUVectorPolicyArmDouble::py_qs_data_t>&, index_t,
    VVT<CPUVectorPolicyArmDouble::py_qs_data_t>*, bool);
}  // namespace mindquantum::sim::vector::detail
