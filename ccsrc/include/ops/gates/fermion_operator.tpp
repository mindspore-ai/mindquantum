//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef FERMION_OPERATOR_TPP
#define FERMION_OPERATOR_TPP

#include <algorithm>
#include <utility>
#include <vector>

#include "config/format/eigen_matrices.hpp"
#include "config/format/std_optional.hpp"
#include "config/logging.hpp"

#include "ops/gates/details/fermion_operator_helper_functions.hpp"
#include "ops/gates/fermion_operator.hpp"

#if MQ_HAS_CXX20_SPAN
#    include <span>
namespace mindquantum::compat {
using std::span;
}  // namespace mindquantum::compat
#else
#    include <boost/core/span.hpp>
namespace mindquantum::compat {
using boost::span;
}  // namespace mindquantum::compat
#endif  // MQ_HAS_CXX20_SPAN

namespace mindquantum::ops {

// =============================================================================

template <typename coeff_t>
auto FermionOperator<coeff_t>::matrix() const -> std::optional<matrix_t> {
    MQ_TRACE("Calling FermionOperator<{}>::matrix()", get_type_name<coeff_t>());
    return std::nullopt;
}

// =============================================================================

template <typename coeff_t>
auto FermionOperator<coeff_t>::sparse_matrix(std::optional<uint32_t> n_qubits) const -> std::optional<sparse_matrix_t> {
    MQ_TRACE("Calling FermionOperator<{}>::sparse_matrix({})", get_type_name<coeff_t>(), n_qubits);
    using scalar_t = typename sparse_matrix_t::Scalar;
    if (std::empty(base_t::terms_)) {
        return std::nullopt;
    }

    const auto& n_qubits_local = base_t::num_targets();

    if (n_qubits_local == 0UL && !n_qubits) {
        MQ_ERROR("You should specify n_qubits for converting an identity qubit operator.");
        return std::nullopt;
    }

    if (n_qubits && n_qubits.value() < n_qubits_local) {
        MQ_ERROR("Given n_qubits {} is smaller than the number of qubits of this qubit operator, which is {}.\n",
                 n_qubits.value(), n_qubits_local);
        return std::nullopt;
    }

    const auto n_qubits_value = n_qubits.value_or(n_qubits_local);

    const auto process_term = [n_qubits_value](const auto& local_ops) -> sparse_matrix_t {
        MQ_TRACE("process_term({})", local_ops);
        if (std::empty(local_ops)) {
            return details::n_identity<scalar_t>(n_qubits_value);
        }

        constexpr auto size_groups = 2UL;
        std::vector<compat::span<const term_t>> groups;
        groups.reserve(std::size(local_ops) / size_groups + 1);

        const auto local_ops_end = end(local_ops);
        auto num_to_copy = size_groups;
        for (auto it(begin(local_ops)); it != local_ops_end; std::advance(it, num_to_copy)) {
            num_to_copy = std::min(static_cast<decltype(size_groups)>(std::distance(it, local_ops_end)), size_groups);
            groups.emplace_back(&*it, num_to_copy);
        }

        auto process_group = [n_qubits_value](const auto& group) constexpr {
            MQ_TRACE("process_group({})", group);
            assert(group.size() == 1 || group.size() == 2);
            if (std::size(group) == 2) {
                MQ_TRACE("2FW:, {}, {}, {}, {}, {}", group[0].first, group[0].second == TermValue::adg, group[1].first,
                         group[1].second == TermValue::adg, n_qubits_value);
                return details::two_fermion_word<scalar_t>(group[0].first, group[0].second == TermValue::adg,
                                                           group[1].first, group[1].second == TermValue::adg,
                                                           n_qubits_value);
            }
            MQ_TRACE("1FW:, {}, {}, {}", group[0].first, group[0].second == TermValue::adg, n_qubits_value);
            return details::single_fermion_word<scalar_t>(group[0].first, group[0].second == TermValue::adg,
                                                          n_qubits_value);
        };

        assert(!std::empty(groups));

        auto tmp = process_group(groups.front());
        for (auto it(begin(groups) + 1); it != end(groups); ++it) {
            tmp = (tmp * process_group(*it)).eval();
            if (tmp.nonZeros() == 0) {
                return decltype(tmp){tmp.rows(), tmp.cols()};
            }
        }

        MQ_TRACE("tmp has {} non-zero coefficients", tmp.nonZeros());
        return tmp;
    };

    // NB: if the coefficient type is always constant (e.g. float, double), then the compiler should be able to remove
    //     both if() below.

    auto it = begin(base_t::terms_);
    if (!coeff_policy_t::is_const(it->second)) {
        MQ_ERROR("Coeff is not const! ({})", it->second);
        return {};
    }
    auto result = (process_term(it->first) * coeff_policy_t::get_num(it->second)).eval();
    ++it;
    for (; it != end(base_t::terms_); ++it) {
        if (!coeff_policy_t::is_const(it->second)) {
            MQ_ERROR("Coeff is not const! ({})", it->second);
            return {};
        }
        result = (result + process_term(it->first) * coeff_policy_t::get_num(it->second)).eval();
    }

    return result;
}

// =============================================================================

template <typename coeff_t>
auto FermionOperator<coeff_t>::normal_ordered() const -> self_t {
    FermionOperator ordered_op;
    for (const auto& [local_ops, coeff] : base_t::terms_) {
        ordered_op += normal_ordered_term_(local_ops, coeff);
    }
    return ordered_op;
}

// =============================================================================

template <typename coeff_t>
auto FermionOperator<coeff_t>::normal_ordered_term_(terms_t local_ops, coefficient_t coeff) -> self_t {
    auto ordered_term = FermionOperator{};

    if (std::empty(local_ops)) {
        return self_t(local_ops, coeff);
    }

    for (auto it(begin(local_ops) + 1); it != end(local_ops); ++it) {
        for (auto it_jm1(std::make_reverse_iterator(it)), it_j(std::make_reverse_iterator(it) - 1);
             it_jm1 != rend(local_ops); ++it_jm1, ++it_j) {
            // Swap operators if left operator is a and right operator is a^\dagger
            if (it_jm1->second == TermValue::a && it_j->second == TermValue::adg) {
                std::iter_swap(it_jm1, it_j);
                coeff *= -1.;
                // If indice are same, employ the anti-commutation relationship and generate the new term
                if (it_jm1->first == it_j->first) {
                    // NB: we need to skip skip elements j-1 and j. Since it_jm1 and it_j are reverse iterators:
                    //     (it_j + 1).base() is actually the j-1 element
                    //     it_j.base() is actually the j+1 element
                    auto new_terms = terms_t(begin(local_ops), (it_jm1 + 1).base());
                    new_terms.reserve(std::size(local_ops) - 2);
                    std::copy(it_j.base(), end(local_ops), std::back_inserter(new_terms));
                    ordered_term += normal_ordered_term_(std::move(new_terms), -coeff);
                }
            } else if (it_jm1->second == it_j->second) {
                // If indices are the same, evaluate to zero
                if (it_jm1->first == it_j->first) {
                    return ordered_term;
                }
                // Swap them if the same operator but lower index on the left
                if (it_jm1->first < it_j->first) {
                    std::iter_swap(it_jm1, it_j);
                    coeff *= -1.;
                }
            }
        }
    }
    return ordered_term += self_t(local_ops, coeff);
}

}  // namespace mindquantum::ops

#endif /* FERMION_OPERATOR_TPP */
