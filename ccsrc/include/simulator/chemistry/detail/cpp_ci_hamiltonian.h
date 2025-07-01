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
#include "core/utils.h"
#include "simulator/chemistry/detail/ci_basis.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"  // For FermionTerm and FermionOpData types

namespace mindquantum::sim::chem::detail {

/// Structure to hold precomputed information for applying a single Hamiltonian term to a basis state.
template <typename calc_t>
struct PrecomputedTerm {
    size_t idx_in;   // Index of the input basis state
    size_t idx_out;  // Index of the output basis state
    calc_t coeff;    // Total coefficient for this transition
};

/// CI-basis Hamiltonian: wraps excitation-operator application and expectation
template <typename calc_t>
class CppCIHamiltonian {
 public:
    using SlaterDeterminant = ci_basis::SlaterDeterminant;
    using CIVector = ci_basis::CIVector<calc_t>;
    using FermionOpData = typename CppExcitationOperator<calc_t>::FermionOpData;

    /// Construct from fermionic Hamiltonian data, precomputing the operator's action.
    CppCIHamiltonian(const FermionOpData& ham_data, qbit_t n_qubits, int n_electrons)
        : n_qubits_(n_qubits), n_electrons_(n_electrons) {
        auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits_, n_electrons_);

        for (const auto& [ops, val] : ham_data) {
            std::vector<std::pair<int, bool>> term_ops;
            for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
                term_ops.emplace_back(it->first, it->second);
            }

            uint64_t creation_mask = 0;
            uint64_t annihilation_mask = 0;
            for (const auto& op : term_ops) {
                if (op.second) {  // is_creation
                    creation_mask |= (1ULL << op.first);
                } else {
                    annihilation_mask |= (1ULL << op.first);
                }
            }

            uint64_t involved_mask = creation_mask | annihilation_mask;
            int n_involved = ci_basis::SlaterDeterminant::count_set_bits(involved_mask);
            int n_annihilated = ci_basis::SlaterDeterminant::count_set_bits(annihilation_mask);

            int n_spec_q = n_qubits_ - n_involved;
            int n_spec_e = n_electrons_ - n_annihilated;

            if (n_spec_e < 0 || n_spec_e > n_spec_q) {
                continue;
            }

            size_t n_spec_comb = ci_basis::Combinatorics::get(n_spec_q, n_spec_e);

            for (size_t i = 0; i < n_spec_comb; i++) {
                uint64_t spec_mask_small = ci_basis::unrank_lexicographical(i, n_spec_q, n_spec_e);

                uint64_t spectator_mask = 0;
                int orb_idx = 0;
                for (int j = 0; j < n_spec_q; j++) {
                    while ((involved_mask >> orb_idx) & 1) {
                        orb_idx++;
                    }
                    if ((spec_mask_small >> j) & 1) {
                        spectator_mask |= (1ULL << orb_idx);
                    }
                    orb_idx++;
                }

                uint64_t mask_in = spectator_mask | annihilation_mask;
                size_t idx_in = indexer->rank(mask_in);

                uint64_t current_mask = mask_in;
                calc_t total_coeff = val;

                for (const auto& op : term_ops) {
                    const int orb_idx_op = op.first;
                    const uint64_t bit = 1ULL << orb_idx_op;

                    if (ci_basis::SlaterDeterminant::count_set_bits(current_mask & (bit - 1)) & 1) {
                        total_coeff = -total_coeff;
                    }
                    current_mask ^= bit;
                }

                const size_t idx_out = indexer->rank(current_mask);
                precomputed_terms_.push_back({idx_in, idx_out, total_coeff});
            }
        }
    }

    /// Default copy constructor is sufficient.
    CppCIHamiltonian(const CppCIHamiltonian& other) = default;

    /// Apply H to a CI vector using precomputed data: returns H|psi>
    CIVector apply_to_civector(const CIVector& civec) const {
        CIVector result(civec.n_qubits(), civec.n_electrons());
        const auto& data_in = civec.data();
        auto& data_out = result.data();

        THRESHOLD_OMP_FOR(
            precomputed_terms_.size(), 1UL << 13, for (omp::idx_t i = 0; i < precomputed_terms_.size(); ++i) {
                const auto& term = precomputed_terms_[i];
                const auto amp_in = data_in[term.idx_in];
                data_out[term.idx_out] += term.coeff * amp_in;
            });
        return result;
    }

    /// Compute ⟨psi_bra | H_psi_ket ⟩
    calc_t get_expectation_value(const CIVector& psi_bra, const CIVector& psi_ket_H_applied) const {
        std::complex<calc_t> exp_val{};
        const auto& data_l = psi_bra.data();
        const auto& data_r = psi_ket_H_applied.data();
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

    /// Compute ⟨psi | H | psi⟩ by applying H internally
    calc_t get_expectation_value(const CIVector& civec) const {
        CIVector Hpsi = apply_to_civector(civec);
        return get_expectation_value(civec, Hpsi);
    }

 private:
    std::vector<PrecomputedTerm<calc_t>> precomputed_terms_;
    qbit_t n_qubits_;
    int n_electrons_;
};

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_CI_HAMILTONIAN_H
