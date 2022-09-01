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
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/any_range.hpp>

#include "config/format/std_complex.hpp"
#include "config/real_cast.hpp"

#include "experimental/core/logging.hpp"
#include "experimental/ops/gates/terms_operator.hpp"
#include "experimental/ops/meta/dagger.hpp"

namespace mindquantum::ops {

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(term_t term, coefficient_t coeff)
    : TermsOperator(coeff_term_dict_t{{{std::move(term)}, coeff}}) {
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(const terms_t& terms, coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace(new_terms, new_coeff);
    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(const py_terms_t& terms, coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace(new_terms, new_coeff);
    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(const coeff_term_dict_t& terms) {
    for (const auto& [local_ops, coeff] : terms) {
        terms_.emplace(term_policy_t::sort_terms(local_ops, coeff));
    }

    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(std::string_view terms_string,
                                                                        coefficient_t coeff)
    : TermsOperator(term_policy_t::parse_terms_string(terms_string), coeff) {
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::TermsOperator(coeff_term_dict_t terms,
                                                                        sorted_constructor_t /* unused */)
    : terms_{std::move(terms)} {
    calculate_num_targets_();
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
uint32_t TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::num_targets() const {
    return num_targets_;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::empty() const noexcept {
    return std::empty(terms_);
}
// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::size() const {
    return std::size(terms_);
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::get_terms() const -> const coeff_term_dict_t& {
    return terms_;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::get_terms_pair() const -> py_coeff_term_list_t {
    py_coeff_term_list_t out;
    for (const auto& [local_ops, coeff] : terms_) {
        terms_t py_local_ops;
        for (const auto& [idx, term_value] : local_ops) {
            py_local_ops.emplace_back(std::make_pair(idx, term_value));
        }
        out.emplace_back(std::make_pair(py_local_ops, coeff));
    }
    return out;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::get_coeff(const terms_t& term) const -> coefficient_t {
    return terms_.at(term);
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
bool TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::is_identity(double abs_tol) const {
#if MQ_HAS_CXX20_RANGES
    return std::ranges::all_of(
        terms_, [abs_tol](const auto& term) constexpr {
            return std::empty(term.first)
                   || std::ranges::all_of(
                       term.first, [](const auto& local_op) constexpr { return local_op.second == TermValue::I; })
                   || coeff_policy_t::is_zero(term.second, abs_tol);
        });
#else
    for (const auto& [term, coeff] : terms_) {
        if (!std::empty(term)
            && std::any_of(
                std::begin(term), std::end(term),
                [](const auto& local_op) constexpr { return local_op.second != TermValue::I; })
            && !coeff_policy_t::is_zero(coeff, abs_tol)) {
            return false;
        }
    }
#endif  // MQ_HAS_CXX20_RANGES
    return true;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::count_qubits() const -> term_t::first_type {
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

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::identity() -> derived_t {
    return derived_t{terms_t{}};
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::hermitian() const -> derived_t {
    coeff_term_dict_t terms;
    for (auto& [local_ops, coeff] : terms_) {
        terms.emplace(std::move(term_policy_t::hermitian(local_ops)), std::move(coeff_policy_t::conjugate(coeff)));
    }
    return derived_t(std::move(terms));
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::adjoint() const noexcept -> operator_t {
    return DaggerOperation(*this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::constant() const noexcept -> coefficient_t {
    if (const auto it = terms_.find({}); it != end(terms_)) {
        assert(std::empty(it->first));
        return it->second;
    }
    return coefficient_t{0.0};
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::constant(const coefficient_t& coeff) -> void {
    terms_[{}] = coeff;
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
bool TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::is_singlet() const {
    return size() == 1;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::singlet() const -> std::vector<derived_t> {
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

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::singlet_coeff() const -> coefficient_t {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }
    return begin(terms_)->second;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::split() const
    -> std::vector<std::pair<coefficient_t, derived_t>> {
    std::vector<std::pair<coefficient_t, derived_t>> result;
    for (const auto& [local_ops, coeff] : terms_) {
        result.emplace_back(local_ops, coeff);
    }
    return result;
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::real() const -> derived_t {
    auto out(*static_cast<const derived_t*>(this));
    for (auto& [local_ops, coeff] : out.terms_) {
        // TODO(dnguyen): Finish this!
        // coeff = real_cast<RealCastType::REAL>(coeff);
    }
    return out;
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::imag() const -> derived_t {
    auto out(*static_cast<const derived_t*>(this));
    for (auto& [local_ops, coeff] : out.terms_) {
        // TODO(dnguyen): Finish this!
        // coeff = real_cast<RealCastType::IMAG>(coeff);
    }
    return out;
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::compress(double abs_tol) -> derived_t& {
    // TODO(dnguyen): Fix this! std::remove uses std::move(*it)... but *it is const!
    auto new_end = std::remove(begin(terms_), end(terms_), [abs_tol](const auto& term) -> bool {
        return coeff_policy_t::is_zero(term.second, abs_tol);
    });
    terms_.erase(new_end, end(terms_));

    for (auto it(begin(terms_)), it_end(end(terms_)); it != new_end; ++it) {
        coeff_policy_t::compress(it.value(), abs_tol);
    }

    // NB: cannot use this with tsl::ordered_map since iterators are invalidated after calling erase()
    // const auto end_it = end(terms_);
    // for (auto it = begin(terms_); it != end_it;) {
    //     if (coeff_policy_t::is_zero(it->second, abs_tol)) {
    //         it = terms_.erase(it);
    //         continue;
    //     }
    //     coeff_policy_t::compress(it.value(), abs_tol);
    //     ++it;
    // }
    calculate_num_targets_();
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::to_string() const noexcept -> std::string {
    using namespace std::literals::string_literals;
    if (std::empty(terms_)) {
        return "0"s;
    }
    return boost::algorithm::join(
        terms_
            | boost::adaptors::transformed(static_cast<std::string (*)(const typename coeff_term_dict_t::value_type&)>(
                details::stringize<derived_t>::to_string)),
        "\n"s);
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::dumps(std::size_t indent) const -> std::string {
    nlohmann::json json(*this);
    return json.dump(static_cast<int>(indent), ' ', true);
}

// -------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::loads(std::string_view string_data)
    -> std::optional<derived_t> {
    return nlohmann::json::parse(string_data).get<derived_t>();
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator+=(const derived_t& other) -> derived_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { coeff_policy_t::iadd(lhs, rhs); },
        [](const coefficient_t& coefficient) constexpr { return coefficient; });
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
template <TYPENAME_COEFFICIENT number_t TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator+=(const number_t& number) -> derived_t& {
    *this += derived_t::identity() * number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator-=(const derived_t& other) -> derived_t& {
    return add_sub_impl_(
        other, [](coefficient_t & lhs, const coefficient_t& rhs) constexpr { coeff_policy_t::isub(lhs, rhs); },
        [](const coefficient_t& coefficient) constexpr { return coeff_policy_t::uminus(coefficient); });
}

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
template <TYPENAME_COEFFICIENT number_t TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator-=(const number_t& number) -> derived_t& {
    *this -= derived_t::identity() * number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator-() const -> derived_t {
    return (*static_cast<const derived_t*>(this) * -1.);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator*=(const derived_t& other) -> derived_t& {
    coeff_term_dict_t product_results;
    for (const auto& [left_op, left_coeff] : terms_) {
        for (const auto& [right_op, right_coeff] : other.terms_) {
            auto new_op = std::vector<term_t>{left_op};
            new_op.insert(end(new_op), begin(right_op), end(right_op));
            const auto [new_terms, new_coeff] = term_policy_t::simplify(new_op,
                                                                        coeff_policy_t::mul(left_coeff, right_coeff));
            if (auto it = product_results.find(new_terms); it != end(product_results)) {
                coeff_policy_t::iadd(it.value(), new_coeff);
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

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
template <TYPENAME_COEFFICIENT number_t TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator*=(const number_t& number) -> derived_t& {
    // NB: cannot use the usual range-for loop since that uses operator*() implicitly and using tsl::ordered_map the
    //     values accessed in this way are constants
    for (auto it(begin(terms_)), it_end(end(terms_)); it != it_end; ++it) {
        coeff_policy_t::imul(it.value(), number);
    }
    // NB: This would work for normal std::map/std::unordered_map
    // for (auto& [term, coeff] : terms_) {
    //     coeff_policy_t::imul(coeff.value(), number);
    // }
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
template <TYPENAME_COEFFICIENT number_t TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator/=(const number_t& number) -> derived_t& {
    *this *= 1. / number;
    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::pow(uint32_t exponent) const -> derived_t {
    derived_t result = identity();
    for (auto i(0UL); i < exponent; ++i) {
        result *= *static_cast<const derived_t*>(this);
    }
    return result;
}

// =============================================================================

// template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
// bool TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator==(derived_t& other) {
//     this->compress();
//     other.compress();
//     return *this == other;
// }

// -----------------------------------------------------------------------------

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
bool TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::operator==(const derived_t& other) const {
    using policy_t = coeff_policy_t;
    std::vector<typename coeff_term_dict_t::value_type> intersection;
    std::vector<typename coeff_term_dict_t::value_type> symmetric_differences;

    std::set_intersection(
        begin(terms_), end(terms_), begin(other.terms_), end(other.terms_), std::back_inserter(intersection),
        [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });

    for (const auto& term : intersection) {
        const auto& left = terms_.at(term.first);
        const auto& right = other.terms_.at(term.first);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(left)>, coefficient_t>);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(right)>, coefficient_t>);
        if (!policy_t::equal(left, right)) {
            return false;
        }
    }

    std::set_symmetric_difference(begin(terms_), end(terms_), begin(other.terms_), end(other.terms_),
                                  std::back_inserter(symmetric_differences),
                                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return std::empty(symmetric_differences);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
template <typename assign_modify_op_t, typename coeff_unary_op_t>
auto TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::add_sub_impl_(const derived_t& other,
                                                                             assign_modify_op_t&& assign_modify_op,
                                                                             coeff_unary_op_t&& coeff_unary_op)
    -> derived_t& {
    for (const auto& [term, coeff] : other.terms_) {
        auto it = terms_.find(term);
        if (it != terms_.end()) {
            assign_modify_op(it.value(), coeff);
        } else {
            it = terms_.emplace(term, coeff_unary_op(coeff)).first;
        }

        if (coeff_policy_t::is_zero(it->second)) {
            terms_.erase(it);
        }
    }

    calculate_num_targets_();

    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <typename derived_t, template <typename coeff_t> class term_policy_t_, typename coeff_policy_t>
void TermsOperator<derived_t, term_policy_t_, coeff_policy_t>::calculate_num_targets_() noexcept {
    num_targets_ = count_qubits();
}

// =============================================================================

template <TYPENAME_COEFFICIENT number_t, typename derived_t TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL>
auto operator-(const number_t& number, const derived_t& other) {
    return number + (-other);
}

// =============================================================================

}  // namespace mindquantum::ops

#undef TYPENAME_COEFFICIENT
#undef TYPENAME_COEFFICIENT_CONSTRAINTS_DEF
#undef TYPENAME_COEFFICIENT_CONSTRAINTS_DEF_ADD
#undef TYPENAME_COEFFICIENT_CONSTRAINTS_IMPL

#endif /* TERMS_OPERATOR_TPP */
