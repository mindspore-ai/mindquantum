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

#ifndef DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP
#define DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP

#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "core/parameter_resolver.hpp"
#include "ops/gates/term_value.hpp"

namespace mindquantum::ops::details {

struct FermionOpteratorTermPolicyBase {
    static auto to_string(const TermValue& value) {
        using namespace std::literals::string_literals;
        if (value == TermValue::adg) {
            return "^"s;
        }
        return ""s;
    }

    //! Convert a (fermion) term to its hermitian conjugate
    static auto hermitian(term_t term) -> term_t;

    //! Convert a list of terms to its hermitian conjugate
    static auto hermitian(const terms_t& terms) -> terms_t;

    //! Convert a term to a string
    /*!
     * \param term A term to convert to a string
     */
    static auto to_string(const term_t& term) -> std::string;

    //! Parse a string containing FermionOperator local operators
    /*!
     * \param terms_string A string
     */
    static auto parse_terms_string(std::string_view terms_string) -> terms_t;
};

// -----------------------------------------------------------------------------

template <typename coefficient_t>
struct FermionOperatorTermPolicy : FermionOpteratorTermPolicyBase {
    //! Simplify the list of local operators
    static std::tuple<std::vector<term_t>, coefficient_t> simplify(terms_t terms, coefficient_t coeff = 1.);

    //! Simplify the list of local operators
    static std::tuple<std::vector<term_t>, coefficient_t> simplify(const py_terms_t& py_terms,
                                                                   coefficient_t coeff = 1.);

    //! Sort a list of local operators
    /*!
     * \param terms A list of local operators
     * \param coeff A coefficient
     * \note Potentially called by the TermsOperator constructor
     * \note Defaults to normal ordering by calling \sa normal_ordered_term_
     */
    static std::pair<terms_t, coefficient_t> sort_terms(terms_t terms, coefficient_t coeff);
};
}  // namespace mindquantum::ops::details

#include "fermion_operator_term_policy.tpp"  // NOLINT

#endif /* DETAILS_FERMION_OPERATOR_TERM_POLICY_HPP */
