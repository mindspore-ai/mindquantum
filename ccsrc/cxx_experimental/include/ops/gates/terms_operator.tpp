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

#ifndef TERMS_OPERATOR_TPP
#define TERMS_OPERATOR_TPP

#include <cmath>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "ops/gates/terms_operator.hpp"
#include "ops/meta/dagger.hpp"

namespace mindquantum::ops {

template <typename derived_t>
TermsOperator<derived_t>::TermsOperator(term_t term, coefficient_t coefficient)
    : terms_{{{std::move(term)}, coefficient}} {
}

template <typename derived_t>
TermsOperator<derived_t>::TermsOperator(complex_term_dict_t terms) : terms_{std::move(terms)} {
}

// =============================================================================

template <typename derived_t>
operator_t TermsOperator<derived_t>::adjoint() const noexcept {
    return DaggerOperation(*this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::compress(double abs_tol) -> self_t& {
    const auto end_it = end(terms_);
    // NOLINTNEXTLINE(altera-id-dependent-backward-branch)
    for (auto it = begin(terms_); it != end_it; ++it) {
        if (std::abs(it->second) < abs_tol) {
            terms_.erase(it);
        } else if (std::abs(it->second.imag()) < abs_tol) {
            it->second = it->second.real();
        } else if (std::abs(it->second.real()) < abs_tol) {
            it->second = coefficient_t{0., it->second.imag()};
        }
    }
    return *this;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::pow(uint32_t exponent) const -> self_t {
    derived_t result;
    for (auto i(0UL); i < exponent; ++i) {
        result *= *this;
    }
    return result;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator+=(const self_t& other) -> self_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { lhs += rhs; },
        [](const coefficient_t& coefficient) constexpr { return coefficient; });
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator+=(const number_t& number) -> self_t& {
    *this += self_t{} * number;
    return *this;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator-=(const self_t& other) -> self_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { lhs -= rhs; },
        [](const coefficient_t& coefficient) constexpr { return -coefficient; });
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator-=(const number_t& number) -> self_t& {
    *this -= self_t{} * number;
    return *this;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator-() const -> self_t {
    return (*this *= -1);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator*=(const self_t& other) -> self_t& {
    complex_term_dict_t product_results;
    for (const auto& [left_op, left_coeff] : terms_) {
        for (const auto& [right_op, right_coeff] : other.terms_) {
            auto new_op = std::vector<term_t>{left_op};
            new_op.insert(end(new_op), begin(right_op), end(right_op));
            const auto [new_terms, new_coeff] = derived_t::simplify_(new_op, left_coeff * right_coeff);
            if (auto it = product_results.find(new_terms); it != end(product_results)) {
                it->second += new_coeff;
            } else {
                product_results.emplace(std::move(new_op), std::move(new_coeff));
            }
        }
    }
    terms_ = std::move(product_results);
    return *this;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator*=(const number_t& number) -> self_t& {
    for (auto& [term, coeff] : terms_) {
        coeff *= number;
    }
    return *this;
}

// =============================================================================

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator/=(const number_t& number) -> self_t& {
    *this *= 1 / number;
    return *this;
}

// =============================================================================

template <typename derived_t>
bool TermsOperator<derived_t>::operator==(self_t& other) {
    this->compress();
    other.compress();
    return *this == other;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
bool TermsOperator<derived_t>::operator==(const self_t& other) const {
    std::vector<complex_term_dict_t::value_type> intersection;
    std::vector<complex_term_dict_t::value_type> symmetric_differences;

    std::set_intersection(begin(terms_), end(terms_), begin(other.terms_), end(other.terms_),
                          std::back_inserter(intersection),
                          [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (const auto& term : intersection) {
        const auto& left = terms_.at(term.first);
        const auto& right = other.terms_.at(term.first);
        if (std::abs(left - right)
            <= std::max(EQ_TOLERANCE, EQ_TOLERANCE * std::max(std::abs(left), std::abs(right)))) {
            return false;
        }
    }

    std::set_symmetric_difference(begin(terms_), end(terms_), begin(other.terms_), end(other.terms_),
                                  std::back_inserter(symmetric_differences),
                                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return std::empty(symmetric_differences);
}

// =============================================================================

template <typename derived_t>
template <typename assign_modify_op_t, typename coeff_unary_op_t>
auto TermsOperator<derived_t>::add_sub_impl_(const self_t& other, assign_modify_op_t&& assign_modify_op,
                                             coeff_unary_op_t&& coeff_unary_op) -> self_t& {
    for (const auto& [term, coeff] : other.terms_) {
        auto it = terms_.find(term);
        if (it != terms_.end()) {
            assign_modify_op(it->second, coeff);
        } else {
            it = terms_.emplace(term, coeff_unary_op(coeff)).first;
        }

        if (std::abs(it->second) < EQ_TOLERANCE) {
            terms_.erase(it);
        }
    }

    return *this;
}

// =============================================================================

}  // namespace mindquantum::ops

#undef TYPENAME_NUMBER
#undef TYPENAME_NUMBER_CONSTRAINTS_DEF
#undef TYPENAME_NUMBER_CONSTRAINTS_IMPL

#endif /* TERMS_OPERATOR_TPP */
