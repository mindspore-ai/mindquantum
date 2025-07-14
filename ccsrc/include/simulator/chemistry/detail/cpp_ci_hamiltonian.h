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

#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_CI_HAMILTONIAN_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_CI_HAMILTONIAN_H

#include <complex>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "simulator/chemistry/detail/ci_basis.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"  // For FermionOpData types

namespace mindquantum::sim::chem::detail {

/// Structure to hold the action of a single Hamiltonian term.
template <typename calc_t>
struct HamiltonianTermAction {
    uint64_t creation_mask;
    uint64_t annihilation_mask;
    uint64_t flip_mask;          // XOR mask to get from input to output state
    calc_t coeff;
    std::vector<uint64_t> parity_masks;
};

template <typename calc_t>
class CppCIHamiltonian {
 public:
    using CIVector = ci_basis::CIVector<calc_t>;
    using FermionOpData = typename CppExcitationOperator<calc_t>::FermionOpData;

    CppCIHamiltonian(const FermionOpData& ham_data, qbit_t n_qubits, int n_electrons)
        : n_qubits_(n_qubits), n_electrons_(n_electrons) {
        for (const auto& [ops, val] : ham_data) {
            HamiltonianTermAction<calc_t> term_action;
            term_action.creation_mask = 0;
            term_action.annihilation_mask = 0;
            term_action.coeff = val;

            for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
                const auto& op = *it;
                const int orb_idx = op.first;
                const bool is_creation = op.second;
                const uint64_t bit = 1ULL << orb_idx;

                term_action.parity_masks.push_back(bit - 1);

                if (is_creation) {
                    term_action.creation_mask |= bit;
                } else {
                    term_action.annihilation_mask |= bit;
                }
            }
            term_action.flip_mask = term_action.creation_mask ^ term_action.annihilation_mask;
            hamiltonian_terms_.push_back(std::move(term_action));
        }
    }

    CppCIHamiltonian(const CppCIHamiltonian& other) = default;

    /// Apply H to a CI vector: returns H|psi>
    CIVector apply_to_civector(const CIVector& civec) const {
        CIVector result(civec.n_qubits(), civec.n_electrons());
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits_, n_electrons_);
        const auto& data_in = civec.data();
        auto& data_out = result.data();
        const size_t dim = data_in.size();
#pragma omp parallel for schedule(dynamic)
        for (omp::idx_t i = 0; i < dim; ++i) {
            const auto amp_in = data_in[i];
            if (amp_in == calc_t(0)) {
                continue;
            }
            uint64_t mask_in = indexer->unrank(i);
            for (const auto& term : hamiltonian_terms_) {
                if ((mask_in & term.annihilation_mask) != term.annihilation_mask) {
                    continue;
                }
                if (((mask_in ^ term.annihilation_mask) & term.creation_mask) != 0) {
                    continue;
                }
                uint64_t current_mask = mask_in;
                int sign = 1;
                for (const auto& parity_mask : term.parity_masks) {
                    if (ci_basis::SlaterDeterminant::count_set_bits(current_mask & parity_mask) & 1) {
                        sign = -sign;
                    }
                    current_mask ^= (parity_mask + 1) & ~parity_mask;
                }
                uint64_t mask_out = mask_in ^ term.flip_mask;
                size_t idx_out = indexer->rank(mask_out);
#pragma omp atomic
                data_out[idx_out] += static_cast<calc_t>(sign) * term.coeff * amp_in;
            }
        }
        return result;
    }

    calc_t get_expectation_value(const CIVector& civec) const {
        CIVector Hpsi = apply_to_civector(civec);
        std::complex<calc_t> exp_val{};
        const auto& data_l = civec.data();
        const auto& data_r = Hpsi.data();
        size_t dim = data_l.size();
        for (size_t i = 0; i < dim; ++i) {
            auto amp_l = data_l[i];
            if (amp_l == calc_t(0)) {
                continue;
            }
            std::complex<calc_t> cl = std::conj(static_cast<std::complex<calc_t>>(amp_l));
            std::complex<calc_t> cr = static_cast<std::complex<calc_t>>(data_r[i]);
            exp_val += cl * cr;
        }
        return std::real(exp_val);
    }

 private:
    std::vector<HamiltonianTermAction<calc_t>> hamiltonian_terms_;
    qbit_t n_qubits_;
    int n_electrons_;
};

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_CI_HAMILTONIAN_H
