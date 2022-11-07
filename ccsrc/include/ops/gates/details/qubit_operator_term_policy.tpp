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

#ifndef DETAILS_QUBIT_OPERATOR_TERM_POLICY_TPP
#define DETAILS_QUBIT_OPERATOR_TERM_POLICY_TPP

#include <tuple>
#include <utility>
#include <vector>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

#include "config/logging.hpp"

#include "ops/gates/details/qubit_operator_term_policy.hpp"

namespace mindquantum::ops::details {

template <typename number_t>
constexpr std::tuple<number_t, TermValue> pauli_products_real(const TermValue& left_op, const TermValue& right_op) {
    if (left_op == TermValue::I && right_op == TermValue::X) {
        return {1., TermValue::X};
    }
    if (left_op == TermValue::X && right_op == TermValue::I) {
        return {1., TermValue::X};
    }
    if (left_op == TermValue::I && right_op == TermValue::Y) {
        return {1., TermValue::Y};
    }
    if (left_op == TermValue::Y && right_op == TermValue::I) {
        return {1., TermValue::Y};
    }
    if (left_op == TermValue::I && right_op == TermValue::Z) {
        return {1., TermValue::Z};
    }
    if (left_op == TermValue::Z && right_op == TermValue::I) {
        return {1., TermValue::Z};
    }
    if (left_op == TermValue::X && right_op == TermValue::X) {
        return {1., TermValue::I};
    }
    if (left_op == TermValue::Y && right_op == TermValue::Y) {
        return {1., TermValue::I};
    }
    if (left_op == TermValue::Z && right_op == TermValue::Z) {
        return {1., TermValue::I};
    }

    return {1., TermValue::I};
}

constexpr std::tuple<std::complex<double>, TermValue> pauli_products(const TermValue& left_op,
                                                                     const TermValue& right_op) {
    if (left_op == TermValue::X && right_op == TermValue::Y) {
        return {{0, 1.}, TermValue::Z};
    }
    if (left_op == TermValue::X && right_op == TermValue::Z) {
        return {{0, -1.}, TermValue::Y};
    }
    if (left_op == TermValue::Y && right_op == TermValue::X) {
        return {{0, -1.}, TermValue::Z};
    }
    if (left_op == TermValue::Y && right_op == TermValue::Z) {
        return {{0, 1.}, TermValue::X};
    }
    if (left_op == TermValue::Z && right_op == TermValue::X) {
        return {{0, 1.}, TermValue::Y};
    }
    if (left_op == TermValue::Z && right_op == TermValue::Y) {
        return {{0, -1.}, TermValue::X};
    }

    return pauli_products_real<std::complex<double>>(left_op, right_op);
}

// =============================================================================

template <typename coefficient_t>
auto QubitOperatorTermPolicy<coefficient_t>::simplify(terms_t terms, coefficient_t coeff)
    -> std::tuple<terms_t, coefficient_t> {
    if (std::empty(terms)) {
        return {terms_t{}, coeff};
    }
    std::stable_sort(
        begin(terms), end(terms), [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });

    terms_t reduced_terms;
    auto left_term = terms.front();
    for (auto it(begin(terms) + 1); it != end(terms); ++it) {
        const auto& [left_qubit_id, left_operator] = left_term;
        const auto& [right_qubit_id, right_operator] = *it;
        if constexpr (traits::is_complex_v<coefficient_t>) {
            if (left_qubit_id == right_qubit_id) {
                const auto [new_coeff, new_op] = pauli_products(left_operator, right_operator);
                MQ_TRACE("{} x {} = {} ({})", left_operator, right_operator, new_op, new_coeff);
                left_term = term_t{left_qubit_id, new_op};
                MQ_TRACE("({}) * ({}) | ({})", coeff, static_cast<coefficient_t>(new_coeff),
                         coeff * static_cast<coefficient_t>(new_coeff));
                coeff *= static_cast<coefficient_t>(new_coeff);
                MQ_TRACE("{} ({})", left_term, coeff);
            } else {
                if (left_operator != TermValue::I) {
                    MQ_TRACE("appending {}", left_term);
                    reduced_terms.emplace_back(left_term);
                }
                left_term = *it;
            }
        } else {
            if (left_qubit_id == right_qubit_id) {
                if ((left_operator == TermValue::X && right_operator == TermValue::Y)
                    || (left_operator == TermValue::X && right_operator == TermValue::Z)
                    || (left_operator == TermValue::Y && right_operator == TermValue::X)
                    || (left_operator == TermValue::Y && right_operator == TermValue::Z)
                    || (left_operator == TermValue::Z && right_operator == TermValue::X)
                    || (left_operator == TermValue::Z && right_operator == TermValue::Y)) {
                    MQ_INFO(
                        "Cannot simplify a real-valued QubitOperator with terms {}-{}. "
                        "Please cast to a complex-valued operator and retry.",
                        left_operator, right_operator);
                    if (left_operator != TermValue::I) {
                        MQ_TRACE("appending {}", left_term);
                        reduced_terms.emplace_back(left_term);
                    }
                    left_term = *it;
                } else {
                    const auto [new_coeff, new_op] = pauli_products_real<double>(left_operator, right_operator);
                    MQ_TRACE("{} x {} = {} ({})", left_operator, right_operator, new_op, new_coeff);
                    left_term = term_t{left_qubit_id, new_op};
                    coeff *= static_cast<coefficient_t>(new_coeff);
                }
            } else {
                if (left_operator != TermValue::I) {
                    MQ_TRACE("appending {}", left_term);
                    reduced_terms.emplace_back(left_term);
                }
                left_term = *it;
            }
        }
    }
    if (left_term.second != TermValue::I) {
        MQ_TRACE("appending {}", left_term);
        reduced_terms.emplace_back(left_term);
    }
    MQ_TRACE("simplified: {}, {}", coeff, reduced_terms);
    return {std::move(reduced_terms), coeff};
}

// -----------------------------------------------------------------------------

template <typename coefficient_t>
auto QubitOperatorTermPolicy<coefficient_t>::simplify(const py_terms_t& py_terms, coefficient_t coeff)
    -> std::tuple<terms_t, coefficient_t> {
    terms_t terms;
    terms.reserve(std::size(py_terms));
    boost::range::push_back(
        terms, py_terms | boost::adaptors::transformed([](const auto& value) -> term_t {
                   return {std::get<0>(value), static_cast<mindquantum::ops::TermValue>(std::get<1>(value))};
               }));

    return simplify(terms, coeff);
}

// =============================================================================

template <typename coefficient_t>
auto QubitOperatorTermPolicy<coefficient_t>::sort_terms(terms_t terms, coefficient_t coeff)
    -> std::pair<terms_t, coefficient_t> {
    std::stable_sort(
        begin(terms), end(terms), [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });
    return {std::move(terms), coeff};  // TODO(dnguyen): Should we move? or can (N)RVO take care of that?
}

// =============================================================================

}  // namespace mindquantum::ops::details

#endif /* DETAILS_QUBIT_OPERATOR_TERM_POLICY_TPP */
