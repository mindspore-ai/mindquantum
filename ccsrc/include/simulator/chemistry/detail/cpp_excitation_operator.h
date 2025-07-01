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
#include <mutex>
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

    /// Struct to hold pre-calculated information for a coupled group.
    struct GateGroupInfo {
        size_t idx1;
        size_t idx2;
        double phase;
    };

    /// Construct from fermionic operator data, number of spin-orbitals, electrons, and parameter resolver.
    explicit CppExcitationOperator(const FermionOpData& term_data, qbit_t n_qubits, int n_electrons,
                                   const parameter::ParameterResolver& pr = {});

    /// Copy constructor to handle std::mutex member
    CppExcitationOperator(const CppExcitationOperator& other)
        : coeff(other.coeff)
        , terms_(other.terms_)
        , n_qubits_(other.n_qubits_)
        , n_electrons_(other.n_electrons_)
        , group_info_(other.group_info_)
        , group_info_populated_(other.group_info_populated_) {
    }

    /// Ensure the group info cache is populated. This is a one-time operation.
    void EnsureGroupInfoPopulated() const;

    /// Parameter resolver for generator coefficient (e.g., rotation angle expression).
    parameter::ParameterResolver coeff;
    std::vector<FermionTerm<calc_type>> terms_;
    qbit_t n_qubits_;
    int n_electrons_;

    // Cache for pre-calculated group information
    mutable std::vector<GateGroupInfo> group_info_;
    mutable bool group_info_populated_ = false;
    mutable std::mutex mutex_;  // To make cache population thread-safe

 private:
    /// Populate group_info_ for single excitations.
    void PopulateSingleExcitationGroupInfo(const std::vector<int>& p_indices, const std::vector<int>& q_indices) const;

    /// Populate group_info_ for double excitations.
    void PopulateDoubleExcitationGroupInfo(const std::vector<int>& p_indices, const std::vector<int>& q_indices) const;
};

// Implementation
template <typename calc_t>
CppExcitationOperator<calc_t>::CppExcitationOperator(const FermionOpData& term_data, qbit_t n_qubits, int n_electrons,
                                                     const parameter::ParameterResolver& pr)
    : coeff(pr), n_qubits_(n_qubits), n_electrons_(n_electrons) {
    for (auto const& term : term_data) {
        FermionTerm<calc_t> ft;
        // Raw generator coefficient and ops are provided per term
        const auto& ops = term.first;
        const auto& val = term.second;
        ft.coefficient = val;
        for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
            ft.ops.emplace_back(it->first, it->second);
        }
        terms_.push_back(std::move(ft));
    }
}

template <typename calc_t>
void CppExcitationOperator<calc_t>::PopulateSingleExcitationGroupInfo(const std::vector<int>& p_indices,
                                                                      const std::vector<int>& q_indices) const {
    auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits_, n_electrons_);
    // Single excitation
    int p = p_indices[0], q = q_indices[0];
    int n_spec_q = n_qubits_ - 2, n_spec_e = n_electrons_ - 1;
    if (n_spec_e < 0 || n_spec_e > n_spec_q) {
        return;
    }
    size_t n_spec_comb = ci_basis::Combinatorics::get(n_spec_q, n_spec_e);
    group_info_.reserve(n_spec_comb);
    uint64_t p_mask = 1ULL << p, q_mask = 1ULL << q;

    for (size_t i = 0; i < n_spec_comb; i++) {
        uint64_t spec_mask_small = ci_basis::unrank_lexicographical(i, n_spec_q, n_spec_e);
        uint64_t base_mask = 0;
        int orb_idx = 0;
        for (int j = 0; j < n_spec_q; j++) {
            while (orb_idx == q || orb_idx == p)
                orb_idx++;
            if ((spec_mask_small >> j) & 1)
                base_mask |= (1ULL << orb_idx);
            orb_idx++;
        }
        uint64_t mask1 = base_mask | q_mask, mask2 = base_mask | p_mask;
        size_t idx1 = indexer->rank(mask1), idx2 = indexer->rank(mask2);
        int phase_q_ = ci_basis::SlaterDeterminant::count_set_bits(mask1 & ((1ULL << q) - 1));
        uint64_t mask_tmp1 = mask1 ^ q_mask;
        int phase_p_ = ci_basis::SlaterDeterminant::count_set_bits(mask_tmp1 & ((1ULL << p) - 1));
        double phase = ((phase_q_ + phase_p_) & 1) ? -1.0 : 1.0;
        group_info_.push_back({idx1, idx2, phase});
    }
}

template <typename calc_t>
void CppExcitationOperator<calc_t>::PopulateDoubleExcitationGroupInfo(const std::vector<int>& p_indices,
                                                                      const std::vector<int>& q_indices) const {
    auto indexer = ci_basis::IndexingManager::GetIndexer(n_qubits_, n_electrons_);
    // Double excitation
    int p = p_indices[0], q = p_indices[1];
    int r = q_indices[0], s = q_indices[1];
    int n_spec_q = n_qubits_ - 4, n_spec_e = n_electrons_ - 2;
    if (n_spec_e < 0 || n_spec_e > n_spec_q) {
        return;
    }
    size_t n_spec_comb = ci_basis::Combinatorics::get(n_spec_q, n_spec_e);
    group_info_.reserve(n_spec_comb);
    uint64_t p_mask = 1ULL << p, q_mask = 1ULL << q;
    uint64_t r_mask = 1ULL << r, s_mask = 1ULL << s;

    for (size_t i = 0; i < n_spec_comb; i++) {
        uint64_t spec_mask_small = ci_basis::unrank_lexicographical(i, n_spec_q, n_spec_e);
        uint64_t base_mask = 0;
        int orb_idx = 0;
        for (int j = 0; j < n_spec_q; j++) {
            while (orb_idx == p || orb_idx == q || orb_idx == r || orb_idx == s)
                orb_idx++;
            if ((spec_mask_small >> j) & 1)
                base_mask |= (1ULL << orb_idx);
            orb_idx++;
        }
        uint64_t mask1 = base_mask | r_mask | s_mask;
        uint64_t mask2 = base_mask | p_mask | q_mask;
        size_t idx1 = indexer->rank(mask1), idx2 = indexer->rank(mask2);
        int phase_s_ = ci_basis::SlaterDeterminant::count_set_bits(mask1 & ((1ULL << s) - 1));
        uint64_t mask_tmp1 = mask1 ^ s_mask;
        int phase_r_ = ci_basis::SlaterDeterminant::count_set_bits(mask_tmp1 & ((1ULL << r) - 1));
        uint64_t mask_tmp2 = mask_tmp1 ^ r_mask;
        int phase_q_ = ci_basis::SlaterDeterminant::count_set_bits(mask_tmp2 & ((1ULL << q) - 1));
        uint64_t mask_tmp3 = mask_tmp2 | q_mask;
        int phase_p_ = ci_basis::SlaterDeterminant::count_set_bits(mask_tmp3 & ((1ULL << p) - 1));
        double phase = ((phase_s_ + phase_r_ + phase_q_ + phase_p_) & 1) ? -1.0 : 1.0;
        group_info_.push_back({idx1, idx2, phase});
    }
}

template <typename calc_t>
void CppExcitationOperator<calc_t>::EnsureGroupInfoPopulated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (group_info_populated_) {
        return;
    }

    if (terms_.size() != 2) {
        // Not a standard UCC gate with G = T - T_dagger, cannot pre-compute.
        group_info_populated_ = true;  // Mark as populated to avoid re-checking
        return;
    }

    std::vector<int> p_indices, q_indices;
    for (const auto& [idx, is_creation] : terms_[0].ops) {
        (is_creation ? p_indices : q_indices).push_back(idx);
    }
    std::sort(p_indices.begin(), p_indices.end(), std::greater<int>());
    std::sort(q_indices.begin(), q_indices.end(), std::greater<int>());

    if (p_indices.size() == 1 && q_indices.size() == 1) {
        PopulateSingleExcitationGroupInfo(p_indices, q_indices);
    } else if (p_indices.size() == 2 && q_indices.size() == 2) {
        PopulateDoubleExcitationGroupInfo(p_indices, q_indices);
    }
    group_info_populated_ = true;
}

}  // namespace mindquantum::sim::chem::detail

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_DETAIL_CPP_EXCITATION_OPERATOR_H
