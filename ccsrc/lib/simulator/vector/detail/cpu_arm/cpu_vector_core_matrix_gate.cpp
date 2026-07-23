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

#include <algorithm>

#include "cpu_vector_arm_backend.h"
#include "cpu_vector_sve_utils.h"
#include "simulator/vector/detail/cpu_vector_matrix_gate_utils.h"

#if !defined(__ARM_FEATURE_SVE)
#    error "The Arm SVE matrix-gate backend requires SVE FCMLA support."
#endif

namespace mindquantum::sim::vector::detail {
namespace {
template <typename policy_t>
inline void ApplySmallRows(svbool_t pg, const typename policy_t::qs_data_t* matrix, size_t gate_dim, size_t row_start,
                           const typename policy_t::calc_type* input, typename policy_t::qs_data_t* state,
                           sve::index_vector_t<typename policy_t::calc_type> local_base, index_t fixed_offset,
                           const index_t* state_offsets) {
    using calc_type = typename policy_t::calc_type;
    const auto row_count = std::min<size_t>(4, gate_dim - row_start);
    const auto* row0 = matrix + row_start * gate_dim;
    const auto* row1 = matrix + (row_start + 1) * gate_dim;
    const auto* row2 = row_count > 2 ? matrix + (row_start + 2) * gate_dim : nullptr;
    const auto* row3 = row_count > 3 ? matrix + (row_start + 3) * gate_dim : nullptr;
    auto out0 = sve::Zero<calc_type>();
    auto out1 = sve::Zero<calc_type>();
    auto out2 = sve::Zero<calc_type>();
    auto out3 = sve::Zero<calc_type>();

    for (size_t col = 0; col < gate_dim; ++col) {
        const auto in = sve::LoadCachedComplex<calc_type>(input, col);
        out0 = sve::CoefficientMulAdd<calc_type>(pg, out0, in, row0[col]);
        out1 = sve::CoefficientMulAdd<calc_type>(pg, out1, in, row1[col]);
        if (row_count > 2) {
            out2 = sve::CoefficientMulAdd<calc_type>(pg, out2, in, row2[col]);
        }
        if (row_count > 3) {
            out3 = sve::CoefficientMulAdd<calc_type>(pg, out3, in, row3[col]);
        }
    }
    sve::ScatterComplexAt<calc_type>(state, fixed_offset + state_offsets[row_start], pg, local_base, out0);
    sve::ScatterComplexAt<calc_type>(state, fixed_offset + state_offsets[row_start + 1], pg, local_base, out1);
    if (row_count > 2) {
        sve::ScatterComplexAt<calc_type>(state, fixed_offset + state_offsets[row_start + 2], pg, local_base, out2);
    }
    if (row_count > 3) {
        sve::ScatterComplexAt<calc_type>(state, fixed_offset + state_offsets[row_start + 3], pg, local_base, out3);
    }
}

template <typename policy_t>
inline void ApplyLargeRows(svbool_t pg, const typename policy_t::qs_data_t* matrix, size_t gate_dim, size_t row_start,
                           const typename policy_t::calc_type* input, typename policy_t::qs_data_t* state,
                           sve::index_vector_t<typename policy_t::calc_type> local_base, index_t fixed_offset,
                           const index_t* state_offsets) {
    using calc_type = typename policy_t::calc_type;
    const auto* row0 = matrix + (row_start + 0) * gate_dim;
    const auto* row1 = matrix + (row_start + 1) * gate_dim;
    const auto* row2 = matrix + (row_start + 2) * gate_dim;
    const auto* row3 = matrix + (row_start + 3) * gate_dim;
    const auto* row4 = matrix + (row_start + 4) * gate_dim;
    const auto* row5 = matrix + (row_start + 5) * gate_dim;
    const auto* row6 = matrix + (row_start + 6) * gate_dim;
    const auto* row7 = matrix + (row_start + 7) * gate_dim;
    auto out0 = sve::Zero<calc_type>();
    auto out1 = sve::Zero<calc_type>();
    auto out2 = sve::Zero<calc_type>();
    auto out3 = sve::Zero<calc_type>();
    auto out4 = sve::Zero<calc_type>();
    auto out5 = sve::Zero<calc_type>();
    auto out6 = sve::Zero<calc_type>();
    auto out7 = sve::Zero<calc_type>();

    for (size_t col = 0; col < gate_dim; ++col) {
        const auto in = sve::LoadCachedComplex<calc_type>(input, col);
        out0 = sve::CoefficientMulAdd<calc_type>(pg, out0, in, row0[col]);
        out1 = sve::CoefficientMulAdd<calc_type>(pg, out1, in, row1[col]);
        out2 = sve::CoefficientMulAdd<calc_type>(pg, out2, in, row2[col]);
        out3 = sve::CoefficientMulAdd<calc_type>(pg, out3, in, row3[col]);
        out4 = sve::CoefficientMulAdd<calc_type>(pg, out4, in, row4[col]);
        out5 = sve::CoefficientMulAdd<calc_type>(pg, out5, in, row5[col]);
        out6 = sve::CoefficientMulAdd<calc_type>(pg, out6, in, row6[col]);
        out7 = sve::CoefficientMulAdd<calc_type>(pg, out7, in, row7[col]);
    }
    const auto scatter = [&](size_t row, sve::vector_t<calc_type> value) {
        sve::ScatterComplexAt<calc_type>(state, fixed_offset + state_offsets[row], pg, local_base, value);
    };
    scatter(row_start + 0, out0);
    scatter(row_start + 1, out1);
    scatter(row_start + 2, out2);
    scatter(row_start + 3, out3);
    scatter(row_start + 4, out4);
    scatter(row_start + 5, out5);
    scatter(row_start + 6, out6);
    scatter(row_start + 7, out7);
}

template <typename policy_t>
void ApplyMatrixImpl(const typename policy_t::qs_data_p_t& src_out, typename policy_t::qs_data_p_t* des_p,
                     const qbits_t& objs, const qbits_t& ctrls, const arm_matrix_t<policy_t>& matrix, index_t dim) {
    using calc_type = typename policy_t::calc_type;
    using value_t = typename policy_t::qs_data_t;
    matrix_gate::ValidateNQubitsMatrixInputs(objs, ctrls, matrix, dim);
    const auto prepared = matrix_gate::PrepareMatrixGate<0, value_t>(objs, ctrls, matrix, dim);

    auto& des = *des_p;
    if (des == nullptr) {
        des = policy_t::InitState(dim);
    }
    auto src = src_out;
    const auto owns_src = src == nullptr;
    if (owns_src) {
        src = policy_t::InitState(dim);
    }

    const state_index::StateIndexWindow index_window(sve::LocalStateIndexBits<calc_type>(), prepared.occupied_qubits);
    sve::ForEachStateIndexBatchWithScratch<calc_type>(
        dim, policy_t::DimTh, index_window, prepared.block_count, prepared.gate_dim,
        [&](const state_index::StateIndexBatch& batch, index_t state_window_origin, calc_type* input) {
            const auto loop_index = sve::LoopIndex<calc_type>(batch.local_compressed_start);
            const auto pg = sve::ActiveComplexPredicate<calc_type>(loop_index, batch.local_compressed_count);
            const auto local_base = sve::ExpandBaseIndex<calc_type>(pg, loop_index, index_window.LocalExcludedQubits());
            const auto fixed_offset = state_window_origin + prepared.ctrl_mask;

            for (size_t basis = 0; basis < prepared.gate_dim; ++basis) {
                sve::StoreCachedComplex<calc_type>(
                    input, basis, pg,
                    sve::GatherComplexAt<calc_type>(src, fixed_offset + prepared.obj_masks[basis], pg, local_base));
            }

            if (objs.size() < 5) {
                for (size_t row = 0; row < prepared.gate_dim; row += 4) {
                    ApplySmallRows<policy_t>(pg, prepared.gate_flat.data(), prepared.gate_dim, row, input, des,
                                             local_base, fixed_offset, prepared.obj_masks.data());
                }
            } else {
                for (size_t row = 0; row < prepared.gate_dim; row += 8) {
                    ApplyLargeRows<policy_t>(pg, prepared.gate_flat.data(), prepared.gate_dim, row, input, des,
                                             local_base, fixed_offset, prepared.obj_masks.data());
                }
            }
        });
    if (owns_src) {
        policy_t::FreeState(&src);
    }
}
}  // namespace

template <typename policy_t>
void arm_sve::ApplyMatrix(const typename policy_t::qs_data_p_t& src, typename policy_t::qs_data_p_t* des_p,
                          const qbits_t& objs, const qbits_t& ctrls, const arm_matrix_t<policy_t>& matrix,
                          index_t dim) {
    ApplyMatrixImpl<policy_t>(src, des_p, objs, ctrls, matrix, dim);
}

template void arm_sve::ApplyMatrix<CPUVectorPolicyArmFloat>(const CPUVectorPolicyArmFloat::qs_data_p_t&,
                                                            CPUVectorPolicyArmFloat::qs_data_p_t*, const qbits_t&,
                                                            const qbits_t&,
                                                            const arm_matrix_t<CPUVectorPolicyArmFloat>&, index_t);
template void arm_sve::ApplyMatrix<CPUVectorPolicyArmDouble>(const CPUVectorPolicyArmDouble::qs_data_p_t&,
                                                             CPUVectorPolicyArmDouble::qs_data_p_t*, const qbits_t&,
                                                             const qbits_t&,
                                                             const arm_matrix_t<CPUVectorPolicyArmDouble>&, index_t);
}  // namespace mindquantum::sim::vector::detail
