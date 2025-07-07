/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_POLICY_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_POLICY_H

#include <cmath>

#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "simulator/chemistry/detail/ci_basis.h"

namespace mindquantum::sim::chem::detail {

struct cpu_ci_vector_float_policy;
struct cpu_ci_vector_double_policy;

// Base policy for CPU CI-vector operations
template <typename derived_, typename calc_t_>
struct cpu_ci_vector_policy_base {
    using derived = derived_;
    using calc_type = calc_t_;
    using qs_data_t = ci_basis::CIVector<calc_type>;
    using qs_data_p_t = qs_data_t*;
    using py_qs_data_t = calc_type;

    static qs_data_p_t AllocateState(qbit_t n_qubits, int n_electrons, unsigned /*seed*/) {
        auto state = new qs_data_t(n_qubits, n_electrons);
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits, n_electrons);
        auto hf_mask = ci_basis::SlaterDeterminant::HartreeFock(n_qubits, n_electrons).occupied_orbitals;
        auto hf_idx = indexer->rank(hf_mask);
        state->add_amplitude_by_index(hf_idx, static_cast<calc_type>(1));
        return state;
    }
    static void FreeState(qs_data_p_t state) {
        delete state;
    }
    static void CopyState(qs_data_p_t src, qs_data_p_t dst, qbit_t, int) {
        *dst = *src;
    }
    static std::vector<std::pair<uint64_t, py_qs_data_t>> GetState(qs_data_p_t state, qbit_t n_qubits,
                                                                   int n_electrons) {
        std::vector<std::pair<uint64_t, py_qs_data_t>> repr;
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits, n_electrons);
        const auto& data = state->data();
        for (size_t idx = 0; idx < data.size(); ++idx) {
            auto amp = data[idx];
            if (amp != calc_type(0)) {
                uint64_t mask = indexer->unrank(idx);
                repr.emplace_back(mask, amp);
            }
        }
        return repr;
    }
    static void SetState(qs_data_p_t state, const std::vector<std::pair<uint64_t, py_qs_data_t>>& qs_repr,
                         qbit_t n_qubits, int n_electrons) {
        state->clear();
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits, n_electrons);
        for (auto const& [mask, amp] : qs_repr) {
            auto idx = indexer->rank(mask);
            state->add_amplitude_by_index(idx, amp);
        }
    }

    template <typename ExcOp>
    static void ApplyUCCGate(qs_data_p_t qs, const ExcOp& op, double theta, qbit_t n_qubits, int n_electrons) {
        if (!op.is_valid_) {
            return;  // Not a gate we can optimize.
        }

        auto& data = qs->data();
        const auto cos_th = static_cast<calc_type>(std::cos(theta));
        const auto sin_th = static_cast<calc_type>(std::sin(theta));
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits, n_electrons);
        const size_t dim = qs->dimension();

        THRESHOLD_OMP_FOR(
            dim, 1UL << 13, for (omp::idx_t i = 0; i < dim; i++) {
                uint64_t mask_i = indexer->unrank(i);
                if ((mask_i & op.excit_mask_) == op.ket_bra_mask_) {
                    uint64_t mask_j = mask_i ^ op.flip_mask_;
                    size_t j = indexer->rank(mask_j);
                    double phase = 1.0;
                    uint64_t current_mask = mask_i;
                    for (size_t k = 0; k < op.op_sequence_.size(); ++k) {
                        const auto& [orb_idx, is_creation] = op.op_sequence_[k];
                        const auto& parity_mask = op.op_parity_masks_[k];
                        if (ci_basis::SlaterDeterminant::count_set_bits(current_mask & parity_mask) & 1) {
                            phase = -phase;
                        }
                        current_mask ^= (1ULL << orb_idx);
                    }
                    auto c1 = data[i];
                    auto c2 = data[j];
                    data[i] = cos_th * c1 - static_cast<calc_type>(phase) * sin_th * c2;
                    data[j] = static_cast<calc_type>(phase) * sin_th * c1 + cos_th * c2;
                }
            });
    }
};

}  // namespace mindquantum::sim::chem::detail
#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPU_CI_VECTOR_POLICY_H
