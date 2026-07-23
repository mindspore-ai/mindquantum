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

#include <complex>
#include <cstdint>
#include <vector>

#include "config/openmp.h"
#include "core/utils.h"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.h"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.h"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.h"
#endif
#include "simulator/vector/detail/circuit_tiling_executor.h"
#include "simulator/vector/detail/cpu_vector_policy.h"
#include "simulator/vector/detail/state_index.h"

namespace mindquantum::sim::vector::detail {
namespace {
struct ScalarIndexContext {};

template <typename value_t_>
struct ScalarTileAlgebra {
    using value_t = value_t_;
    using calc_type = typename value_t::value_type;
    using vector_t = value_t;
    using predicate_t = bool;
    using state_index_t = index_t;
    using index_context_t = ScalarIndexContext;
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

template <typename policy_t>
struct ScalarTileRuntime {
    using algebra_t = ScalarTileAlgebra<typename policy_t::py_qs_data_t>;
    using storage_t = typename algebra_t::storage_t;
    using traversal_t = state_index::BitExpander;

    [[nodiscard]] static auto MakeTraversal(const circuit_tiling::TileLayout& layout) -> traversal_t {
        return traversal_t(layout.qubits);
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
    static void ForEachInParallel(const traversal_t& traversal, index_t tile_count, const function_t& function) {
        MQ_DO_PRAGMA(omp for schedule(static))
        for (omp::idx_t tile = 0; tile < static_cast<omp::idx_t>(tile_count); ++tile) {
            function(true, traversal.Expand(static_cast<index_t>(tile)), ScalarIndexContext{});
        }
    }

    static void Gather(const typename policy_t::qs_data_p_t state, const circuit_tiling::TileLayout& layout, bool,
                       index_t state_base, const ScalarIndexContext&, storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            tile[basis] = state[state_base | layout.state_offsets[basis]];
        }
    }

    static void Scatter(typename policy_t::qs_data_p_t state, const circuit_tiling::TileLayout& layout, bool,
                        index_t state_base, const ScalarIndexContext&, const storage_t* tile) {
        for (size_t basis = 0; basis < layout.dimension; ++basis) {
            state[state_base | layout.state_offsets[basis]] = tile[basis];
        }
    }
};
}  // namespace

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyCircuitTile(qs_data_p_t* state,
                                                                 const circuit_tiling::PreparedTile<py_qs_data_t>& tile,
                                                                 index_t dim) {
    circuit_tiling::ApplyTile<derived, ScalarTileRuntime<derived>>(state, tile, dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyAdjointTileProgram(
    qs_data_p_t* shared_state, const std::vector<qs_data_p_t*>& batch_states,
    const circuit_tiling::AdjointProgram<py_qs_data_t>& program, index_t dim, VVT<py_qs_data_t>* output,
    bool conjugate_output) {
    circuit_tiling::ApplyAdjointProgram<derived, ScalarTileRuntime<derived>>(shared_state, batch_states, program, dim,
                                                                             output, conjugate_output);
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
