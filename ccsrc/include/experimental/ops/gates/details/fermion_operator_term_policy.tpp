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

#ifndef DETAILS_FERMION_OPERATOR_TERM_POLICY_TPP
#define DETAILS_FERMION_OPERATOR_TERM_POLICY_TPP

#include <tuple>
#include <utility>
#include <vector>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

#include "experimental/ops/gates/details/fermion_operator_term_policy.hpp"

namespace mindquantum::ops::details {

// =============================================================================

template <typename coefficient_t>
auto FermionOperatorTermPolicy<coefficient_t>::simplify(terms_t local_ops, coefficient_t coeff)
    -> std::tuple<std::vector<term_t>, coefficient_t> {
    return sort_terms(std::move(local_ops), coeff);
}

// -----------------------------------------------------------------------------

template <typename coefficient_t>
auto FermionOperatorTermPolicy<coefficient_t>::simplify(py_terms_t py_terms, coefficient_t coeff)
    -> std::tuple<terms_t, coefficient_t> {
    terms_t terms;
    terms.reserve(std::size(py_terms));
    boost::range::push_back(
        terms, py_terms | boost::adaptors::transformed([](const auto& value) -> term_t {
                   return {std::get<0>(value), static_cast<mindquantum::ops::TermValue>(std::get<1>(value))};
               }));

    return simplify_(terms, coeff);
}

// =============================================================================

template <typename coefficient_t>
auto FermionOperatorTermPolicy<coefficient_t>::sort_terms(terms_t local_ops, coefficient_t coeff)
    -> std::pair<terms_t, coefficient_t> {
    // std::sort(rbegin(local_ops), rend(local_ops));
    return {std::move(local_ops), coeff};
}

}  // namespace mindquantum::ops::details

#endif /* DETAILS_FERMION_OPERATOR_TERM_POLICY_TPP */
