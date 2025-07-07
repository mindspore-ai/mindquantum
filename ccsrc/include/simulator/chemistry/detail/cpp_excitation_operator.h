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

#ifndef INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_EXCITATION_OPERATOR_H
#define INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_EXCITATION_OPERATOR_H

#include <algorithm>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/chemistry/detail/ci_basis.h"

namespace mindquantum::sim::chem::detail {

/// A term in a fermionic excitation operator: a string of creation/annihilation ops with a coefficient.
template <typename calc_t>
struct FermionTerm {
    /// Sequence of (orbital_index, is_creation_op) pairs.
    std::vector<std::pair<int, bool>> ops;
    calc_t coefficient;
};

/// Functor applying a UCC excitation operator G to CI vectors.
template <typename calc_t>
class CppExcitationOperator {
 public:
    using calc_type = calc_t;
    using SlaterDeterminant = ci_basis::SlaterDeterminant;
    using CIVector = ci_basis::CIVector<calc_type>;
    /// Binding data: list of [(idx, is_creation)] terms.
    using FermionTermData = std::vector<std::pair<int, bool>>;
    /// Binding data: list of (term ops, raw generator coefficient) pairs.
    using FermionOpData = std::vector<std::pair<FermionTermData, calc_type>>;

    /// Construct from fermionic operator data, number of spin-orbitals, electrons, and parameter resolver.
    explicit CppExcitationOperator(const FermionOpData& term_data, qbit_t n_qubits, int n_electrons,
                                   const parameter::ParameterResolver& pr = {});

    /// Copy constructor
    CppExcitationOperator(const CppExcitationOperator& other);

    /// Move constructor
    CppExcitationOperator(CppExcitationOperator&& other) = default;

    /*!
     * @brief Get the excitation indices from the operator terms.
     * @return A tuple containing p_indices (creation), q_indices (annihilation), and a boolean indicating validity.
     */
    std::tuple<std::vector<int>, std::vector<int>, bool> get_excitation_indices() const {
        std::vector<int> p_indices, q_indices;
        if (terms_.size() != 2) {
            return {p_indices, q_indices, false};
        }
        for (const auto& [idx, is_creation] : terms_[0].ops) {
            (is_creation ? p_indices : q_indices).push_back(idx);
        }
        std::sort(p_indices.begin(), p_indices.end(), std::greater<int>());
        std::sort(q_indices.begin(), q_indices.end(), std::greater<int>());
        if (!((p_indices.size() == 1 && q_indices.size() == 1) || (p_indices.size() == 2 && q_indices.size() == 2))) {
            return {{}, {}, false};
        }
        return {p_indices, q_indices, true};
    }

    parameter::ParameterResolver coeff;
    std::vector<FermionTerm<calc_type>> terms_;
    qbit_t n_qubits_;
    int n_electrons_;

    // Pre-calculated masks for fast gate application
    bool is_valid_ = false;
    uint64_t excit_mask_ = 0;    // Mask of all active orbitals
    uint64_t ket_bra_mask_ = 0;  // Mask to identify the 'ket' part of the pair
    uint64_t flip_mask_ = 0;     // Mask to flip between ket and bra states
    std::vector<std::pair<int, bool>> op_sequence_;
    std::vector<uint64_t> op_parity_masks_;

 private:
    void precompute_mask();
};

// Implementation
template <typename calc_t>
void CppExcitationOperator<calc_t>::precompute_mask() {
    auto [p_indices, q_indices, is_valid] = get_excitation_indices();
    is_valid_ = is_valid;
    if (!is_valid_)
        return;

    uint64_t p_mask = 0, q_mask = 0;
    for (auto p : p_indices)
        p_mask |= (1ULL << p);
    for (auto q : q_indices)
        q_mask |= (1ULL << q);

    excit_mask_ = p_mask | q_mask;
    ket_bra_mask_ = q_mask;
    flip_mask_ = excit_mask_;

    const auto& term_ops = terms_[0].ops;
    op_sequence_.reserve(term_ops.size());
    op_parity_masks_.reserve(term_ops.size());
    for (const auto& op : term_ops) {
        op_sequence_.push_back(op);
        op_parity_masks_.push_back((1ULL << op.first) - 1);
    }
}

template <typename calc_t>
CppExcitationOperator<calc_t>::CppExcitationOperator(const FermionOpData& term_data, qbit_t n_qubits, int n_electrons,
                                                     const parameter::ParameterResolver& pr)
    : coeff(pr), n_qubits_(n_qubits), n_electrons_(n_electrons) {
    for (auto const& term : term_data) {
        FermionTerm<calc_t> ft;
        const auto& ops = term.first;
        const auto& val = term.second;
        ft.coefficient = val;
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            ft.ops.emplace_back(it->first, it->second);
        }
        terms_.push_back(std::move(ft));
    }
    precompute_mask();
}

template <typename calc_t>
CppExcitationOperator<calc_t>::CppExcitationOperator(const CppExcitationOperator<calc_t>& other)
    : coeff(other.coeff), terms_(other.terms_), n_qubits_(other.n_qubits_), n_electrons_(other.n_electrons_) {
    precompute_mask();
}

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_EXCITATION_OPERATOR_H
