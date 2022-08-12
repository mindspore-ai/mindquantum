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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/logging.hpp"
#include "ops/gates/terms_operator.hpp"
#include "ops/meta/dagger.hpp"

namespace mindquantum::ops {

template <typename derived_t>
TermsOperator<derived_t>::TermsOperator(term_t term, coefficient_t coeff)
    : TermsOperator(complex_term_dict_t{{{std::move(term)}, coeff}}) {
}

// -----------------------------------------------------------------------------

template <typename derived_t>
TermsOperator<derived_t>::TermsOperator(const terms_t& term, coefficient_t coeff) {
    const auto [new_terms, new_coeff] = derived_t::simplify_(term, coeff);
    terms_.emplace(new_terms, new_coeff);
    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t>
TermsOperator<derived_t>::TermsOperator(const complex_term_dict_t& terms) {
    for (const auto& [local_ops, coeff] : terms) {
        terms_.emplace(derived_t::sort_terms_(local_ops, coeff));
    }

    calculate_num_targets_();
}

// =============================================================================

template <typename derived_t>
uint32_t TermsOperator<derived_t>::num_targets() const noexcept {
    return num_targets_;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::empty() const noexcept {
    return std::empty(terms_);
}
// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::size() const noexcept {
    return std::size(terms_);
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::get_terms() const noexcept -> const complex_term_dict_t& {
    return terms_;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
bool TermsOperator<derived_t>::is_identity(double abs_tol) const noexcept {
#if MQ_HAS_CXX20_RANGES
    return std::ranges::all_of(
        terms_, [abs_tol](const auto& term) constexpr {
            return std::empty(term.first)
                   || std::ranges::all_of(
                       term.first, [](const auto& local_op) constexpr { return local_op.second == TermValue::I; })
                   || std::abs(term.second) <= abs_tol;
        });
#else
    for (const auto& [term, coeff] : terms_) {
        if (!std::empty(term)
            && std::any_of(
                std::begin(term), std::end(term),
                [](const auto& local_op) constexpr { return local_op.second != TermValue::I; })
            && std::abs(coeff) > abs_tol) {
            return false;
        }
    }
#endif  // MQ_HAS_CXX20_RANGES
    return true;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::count_qubits() const noexcept -> term_t::first_type {
    term_t::first_type num_qubits{0};
    for (const auto& [local_ops, coeff] : terms_) {
        if (std::empty(local_ops)) {
            num_qubits = std::max(decltype(num_qubits){1}, num_qubits);
        } else {
            num_qubits = std::max(static_cast<decltype(num_qubits)>(std::max_element(
                                                                        begin(local_ops), end(local_ops),
                                                                        [](const auto& lhs, const auto& rhs) constexpr {
                                                                            return lhs.first < rhs.first;
                                                                        })
                                                                        ->first
                                                                    + 1),  // NB: qubit_ids are 0-based
                                  num_qubits);
        }
    }
    return num_qubits;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::identity() -> derived_t {
    return derived_t{terms_t{}};
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::adjoint() const noexcept -> operator_t {
    return DaggerOperation(*this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::constant() const noexcept -> coefficient_t {
    if (const auto it = terms_.find({}); it != end(terms_)) {
        assert(std::empty(it->first));
        return it->second;
    }
    return 0.0;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::constant(const coefficient_t& coeff) -> void {
    terms_[{}] = coeff;
}

// =============================================================================

template <typename derived_t>
bool TermsOperator<derived_t>::is_singlet() const noexcept {
    return size() == 1;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::singlet() const noexcept -> std::vector<derived_t> {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }

    std::vector<derived_t> words;
    for (const auto& local_op : begin(terms_)->first) {
        words.emplace_back(term_t{local_op}, 1.);
    }

    return words;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::singlet_coeff() const noexcept -> coefficient_t {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }
    return begin(terms_)->second;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::real() const noexcept -> derived_t {
    auto out(*static_cast<const derived_t*>(this));
    for (auto& [local_ops, coeff] : out.terms_) {
        coeff = coeff.real();
    }
    return out;
}

// -----------------------------------------------------------------------------

template <typename derived_t>
auto TermsOperator<derived_t>::imag() const noexcept -> derived_t {
    auto out(*static_cast<const derived_t*>(this));
    for (auto& [local_ops, coeff] : out.terms_) {
        coeff = coeff.imag();
    }
    return out;
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::compress(double abs_tol) -> derived_t& {
    const auto end_it = end(terms_);
    for (auto it = begin(terms_); it != end_it;) {
        if (std::abs(it->second) <= abs_tol) {
            it = terms_.erase(it);
            continue;
        }
        if (std::abs(it->second.imag()) <= abs_tol) {
            it->second = it->second.real();
        } else if (std::abs(it->second.real()) <= abs_tol) {
            it->second = coefficient_t{0., it->second.imag()};
        }
        ++it;
    }
    calculate_num_targets_();
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator+=(const derived_t& other) -> derived_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { lhs += rhs; },
        [](const coefficient_t& coefficient) constexpr { return coefficient; });
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator+=(const number_t& number) -> derived_t& {
    *this += derived_t::identity() * number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator-=(const derived_t& other) -> derived_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { lhs -= rhs; },
        [](const coefficient_t& coefficient) constexpr { return -coefficient; });
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator-=(const number_t& number) -> derived_t& {
    *this -= derived_t::identity() * number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator-() const -> derived_t {
    return (*static_cast<const derived_t*>(this) * -1.);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::operator*=(const derived_t& other) -> derived_t& {
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
    calculate_num_targets_();
    return *static_cast<derived_t*>(this);
}

// -----------------------------------------------------------------------------

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator*=(const number_t& number) -> derived_t& {
    for (auto& [term, coeff] : terms_) {
        coeff *= number;
    }
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
template <TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t>::operator/=(const number_t& number) -> derived_t& {
    *this *= 1. / number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
auto TermsOperator<derived_t>::pow(uint32_t exponent) const -> derived_t {
    derived_t result = identity();
    for (auto i(0UL); i < exponent; ++i) {
        result *= *static_cast<const derived_t*>(this);
    }
    return result;
}

// =============================================================================

// template <typename derived_t>
// bool TermsOperator<derived_t>::operator==(derived_t& other) {
//     this->compress();
//     other.compress();
//     return *this == other;
// }

// -----------------------------------------------------------------------------

template <typename derived_t>
bool TermsOperator<derived_t>::operator==(const derived_t& other) const {
    std::vector<complex_term_dict_t::value_type> intersection;
    std::vector<complex_term_dict_t::value_type> symmetric_differences;

    std::set_intersection(
        begin(terms_), end(terms_), begin(other.terms_), end(other.terms_), std::back_inserter(intersection),
        [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });

    for (const auto& term : intersection) {
        const auto& left = terms_.at(term.first);
        const auto& right = other.terms_.at(term.first);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(left)>, coefficient_t>);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(right)>, coefficient_t>);
        if (std::abs(left - right) > std::max(EQ_TOLERANCE, EQ_TOLERANCE * std::max(std::abs(left), std::abs(right)))) {
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
auto TermsOperator<derived_t>::add_sub_impl_(const derived_t& other, assign_modify_op_t&& assign_modify_op,
                                             coeff_unary_op_t&& coeff_unary_op) -> derived_t& {
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

    calculate_num_targets_();

    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t>
void TermsOperator<derived_t>::calculate_num_targets_() noexcept {
    num_targets_ = count_qubits();
}

// =============================================================================

template <typename derived_t, TYPENAME_NUMBER number_t TYPENAME_NUMBER_CONSTRAINTS_IMPL>
auto operator-(const number_t& number, const TermsOperator<derived_t>& other) {
    return number + (-other);
}

// =============================================================================

}  // namespace mindquantum::ops

#undef TYPENAME_NUMBER
#undef TYPENAME_NUMBER_CONSTRAINTS_DEF
#undef TYPENAME_NUMBER_CONSTRAINTS_IMPL

#endif /* TERMS_OPERATOR_TPP */
