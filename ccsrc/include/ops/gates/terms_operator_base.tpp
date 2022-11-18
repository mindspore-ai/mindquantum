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

#ifndef TERMS_OPERATOR_BASE_TPP
#define TERMS_OPERATOR_BASE_TPP

#include <cmath>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/any_range.hpp>

#include <fmt/ranges.h>

#include "config/constexpr_type_name.hpp"
#include "config/conversion.hpp"
#include "config/format/std_complex.hpp"
#include "config/logging.hpp"
#include "config/real_cast.hpp"

#include "ops/gates/terms_operator_base.hpp"
#include "ops/gates/traits.hpp"
// #include "ops/meta/dagger.hpp"

// =============================================================================

namespace mindquantum::ops {
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(term_t term, coefficient_t coeff)
    : TermsOperatorBase(coeff_term_dict_t{{{std::move(term)}, coeff}}) {
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(const terms_t& terms,
                                                                                 coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace_back(new_terms, new_coeff);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(const py_terms_t& terms,
                                                                                 coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace_back(new_terms, new_coeff);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(const coeff_term_dict_t& terms) {
    for (const auto& [local_ops, coeff] : terms) {
        // NB: not calling simplify() here... assuming that the terms are simplified already.
        terms_.emplace_back(term_policy_t::sort_terms(local_ops, coeff));
    }
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(coeff_term_dict_t&& terms)
    : terms_(std::move(terms)) {
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(std::string_view terms_string,
                                                                                 coefficient_t coeff)
    : TermsOperatorBase(term_policy_t::parse_terms_string(terms_string), coeff) {
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(coeff_term_dict_t terms,
                                                                                 sorted_constructor_t /* unused */)
    : terms_{std::move(terms)} {
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename other_coeff_t>
TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::TermsOperatorBase(const derived_t_<other_coeff_t>& other)
    : num_targets_(other.num_targets_) {
    using conv_helper_t = traits::conversion_helper<coefficient_t>;
    std::transform(begin(other.terms_), end(other.terms_), std::back_inserter(terms_), [](const auto& term) {
        return typename coeff_term_dict_t::value_type{term.first, conv_helper_t::apply(term.second)};
    });
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
constexpr auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::kind() -> std::string_view {
    return get_type_name<derived_t>();
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
uint32_t TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::num_targets() const {
    calculate_num_targets_();
    return num_targets_;
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::empty() const noexcept {
    return std::empty(terms_);
}
// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::size() const {
    return std::size(terms_);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::get_terms() const -> const coeff_term_dict_t& {
    return terms_;
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::get_terms_pair() const -> py_coeff_term_list_t {
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

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::get_coeff(const terms_t& term) const
    -> coefficient_t {
    auto& hashed_index = terms_.template get<order::hashed>();
    if (auto it = hashed_index.find(term); it != hashed_index.cend()) {
        return it->second;
    }
    throw std::out_of_range(fmt::format("Term not found in TermsOperator: {}", term));
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
bool TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::is_identity(double abs_tol) const {
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

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::count_qubits() const -> term_t::first_type {
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

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::identity() -> derived_t {
    return derived_t{terms_t{}};
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::identity(const coefficient_t& coeff) -> derived_t {
    return derived_t{{terms_t{}, coeff}};
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::subs(
    const details::CoeffSubsProxy<coefficient_t>& subs_params) const -> derived_t {
    using value_t = typename coeff_term_dict_t::value_type;

    auto out(*static_cast<const derived_t*>(this));

    auto& ordered_index = out.terms_.template get<order::insertion>();
    for (auto it = begin(ordered_index); it != end(ordered_index); ++it) {
        ordered_index.modify(it, [&subs_params](value_t& value) { subs_params.apply(value.second); });
    }
    return out;
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::hermitian() const -> derived_t {
    coeff_term_dict_t terms;
    for (auto& [local_ops, coeff] : terms_) {
        terms.emplace_back(std::move(term_policy_t::hermitian(local_ops)), coeff_policy_t::conjugate(coeff));
    }
    return derived_t{std::move(terms)};
}

// =============================================================================

// template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
//           template <typename coeff_t> class term_policy_t_>
// auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::adjoint() const noexcept -> operator_t {
//     if constexpr (is_real_valued) {
//         return *static_cast<const derived_t*>(this);
//     } else {
//         return DaggerOperation(*static_cast<const derived_t*>(this));
//     }
// }

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::constant() const noexcept -> coefficient_t {
    auto& hashed_index = terms_.template get<order::hashed>();
    if (const auto it = hashed_index.find(terms_t{}); it != hashed_index.cend()) {
        assert(std::empty(it->first));
        return it->second;
    }
    return coefficient_t{0.0};
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::constant(const coefficient_t& coeff) -> void {
    using value_t = typename coeff_term_dict_t::value_type;

    auto& hashed_index = terms_.template get<order::hashed>();
    if (const auto it = hashed_index.find(terms_t{}); it != hashed_index.cend()) {
        assert(std::empty(it->first));
        hashed_index.modify(it, [&coeff](value_t& value) { value.second = coeff; });
    } else {
        terms_.emplace_back(terms_t{}, coeff);
    }
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
bool TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::is_singlet() const {
    return size() == 1;
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::singlet() const -> std::vector<derived_t> {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }

    std::vector<derived_t> words;
    for (const auto& local_op : begin(terms_)->first) {
        words.emplace_back(term_t{local_op}, coeff_policy_t::one);
    }

    return words;
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::singlet_coeff() const -> coefficient_t {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }
    return begin(terms_)->second;
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::split() const
    -> std::vector<std::pair<coefficient_t, derived_t>> {
    std::vector<std::pair<coefficient_t, derived_t>> result;
    for (const auto& [local_ops, coeff] : terms_) {
        result.emplace_back(std::make_pair(coeff, derived_t(local_ops, coeff_policy_t::one)));
    }
    return result;
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <mindquantum::concepts::scalar scalar_t>
#else
template <typename scalar_t, typename>
#endif  // MQ_HAS_CONCEPTS && !(defined _MSC_VER)
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::cast() const -> derived_t_<scalar_t> {
    return cast<derived_t_<scalar_t>>();
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <mindquantum::concepts::terms_op op_t>
#else
template <typename op_t, typename>
#endif  // MQ_HAS_CONCEPTS && !(defined _MSC_VER)
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::cast() const
    -> derived_t_<typename op_t::coefficient_t> {
    using other_coeff_t = typename op_t::coefficient_t;

    if constexpr (std::is_same_v<coefficient_t, other_coeff_t>) {
        return *static_cast<const derived_t*>(this);
    } else {
        return cast_<derived_t_<other_coeff_t>>(
            [](const coefficient_t& coeff) constexpr { return static_cast<other_coeff_t>(coeff); });
    }
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::real() const -> derived_real_t {
    if constexpr (is_real_valued) {
        return *static_cast<const derived_t*>(this);
    }

    // NB: Perhaps GCC 9.4 still fails (not tested; but 9.3 fails and 9.5 seems ok)
#if defined(__GNUC__) && (__GNUC__ < 9) || (__GNUC__ == 9 && __GNUC_MINOR__ < 4)
    return cast_<derived_real_t, decltype(&mindquantum::real_cast<RealCastType::REAL, const coefficient_t&>)>
#else
    return cast_<derived_real_t>
#endif
        (&mindquantum::real_cast<RealCastType::REAL, const coefficient_t&>);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::imag() const -> derived_real_t {
    if constexpr (is_real_valued) {
        return *static_cast<const derived_t*>(this);
    }
    // NB: Perhaps GCC 9.4 still fails (not tested; but 9.3 fails and 9.5 seems ok)
#if defined(__GNUC__) && (__GNUC__ < 9) || (__GNUC__ == 9 && __GNUC_MINOR__ < 4)
    return cast_<derived_real_t, decltype(&mindquantum::real_cast<RealCastType::IMAG, const coefficient_t&>)>
#else
    return cast_<derived_real_t>
#endif
        (&mindquantum::real_cast<RealCastType::IMAG, const coefficient_t&>);
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::compress(double abs_tol) -> derived_t& {
    // terms_.remove_if([abs_tol](const auto& term) -> bool { return coeff_policy_t::is_zero(term.second, abs_tol); });

    // auto ordered_index = terms_.template get<order::insertion>();
    // for (auto it(ordered_index.begin()), it_end(ordered_index.end()); it != it_end; ++it) {
    //     coeff_policy_t::compress(it.value(), abs_tol);
    // }

    using value_t = typename coeff_term_dict_t::value_type;

    auto& ordered_index = terms_.template get<order::insertion>();
    const auto end_it = ordered_index.cend();
    for (auto it = ordered_index.begin(); it != end_it;) {
        if (coeff_policy_t::is_zero(it->second, abs_tol)) {
            // NB: this is safe as long as the elements are completely unique (which they are by design)
            it = ordered_index.erase(it);
            continue;
        }

        ordered_index.modify(it, [abs_tol](value_t& term) { coeff_policy_t::compress(term.second, abs_tol); });
        ++it;
    }
    return *static_cast<derived_t*>(this);
}

// =============================================================================

namespace details {
using namespace std::literals::string_literals;  // NOLINT
template <typename TermsOperatorBase>
struct stringize {
    using term_t = typename TermsOperatorBase::term_t;
    using terms_t = typename TermsOperatorBase::terms_t;
    using term_policy_t = typename TermsOperatorBase::term_policy_t;
    using coeff_terms_t = typename TermsOperatorBase::coeff_term_dict_t::value_type;
    using self_t = stringize<TermsOperatorBase>;

    static auto to_string(const terms_t& local_ops) -> std::string {
        return boost::algorithm::join(
            local_ops
                | boost::adaptors::transformed(static_cast<std::string (*)(const term_t&)>(term_policy_t::to_string)),
            " ");
    }
    static auto to_string(const coeff_terms_t& coeff_terms) -> std::string {
        const auto& [local_ops, coeff] = coeff_terms;
        return fmt::format("{} [{}]", coeff, to_string(local_ops));
    }
    static auto to_json(const coeff_terms_t& coeff_terms, std::size_t indent) {
        const auto& [local_ops, coeff] = coeff_terms;
        return fmt::format(
            R"({}"{}": "{}")", std::string(indent, ' '),
            boost::algorithm::join(local_ops
                                       | boost::adaptors::transformed(
                                           static_cast<std::string (*)(const term_t&)>(term_policy_t::to_string)),
                                   " "),
            coeff);
    }
};
}  // namespace details

// -------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::to_string() const noexcept -> std::string {
    using namespace std::literals::string_literals;  // NOLINT
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

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::dumps(std::size_t indent) const -> std::string {
    nlohmann::json json(*this);
    return json.dump(static_cast<int>(indent), ' ', true);
}

// -------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::loads(std::string_view string_data)
    -> std::optional<derived_t> {
    return nlohmann::json::parse(string_data).get<derived_t>();
}

// =============================================================================
// Addition

#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op<derived_t_<coefficient_t_>> op_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator+=(const op_t& other) -> derived_t& {
    return add_sub_impl_(
        other, coeff_policy_t::iadd, [](const coefficient_t& coefficient) constexpr { return coefficient; });
}

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op_scalar<derived_t_<coefficient_t_>> scalar_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator+=(const scalar_t& scalar) -> derived_t& {
    *this += (derived_t::identity() *= scalar);
    return *static_cast<derived_t*>(this);
}
#else
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename type_t, typename>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator+=(const type_t& op_or_scalar)
    -> derived_t& {
    if constexpr (traits::is_terms_operator_v<type_t>) {
        return add_sub_impl_(
            op_or_scalar, coeff_policy_t::iadd, [](const coefficient_t& coefficient) constexpr { return coefficient; });
    } else {
        *this += (derived_t::identity() *= op_or_scalar);
        return *static_cast<derived_t*>(this);
    }
}
#endif  // MQ_HAS_CONCEPTS && !_MSC_VER

// =============================================================================¨
// Subtraction

#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op<derived_t_<coefficient_t_>> op_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator-=(const op_t& other) -> derived_t& {
    return add_sub_impl_(other, coeff_policy_t::isub, coeff_policy_t::uminus);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op_scalar<derived_t_<coefficient_t_>> scalar_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator-=(const scalar_t& scalar) -> derived_t& {
    *this -= (derived_t::identity() *= scalar);
    return *static_cast<derived_t*>(this);
}
#else
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename type_t, typename>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator-=(const type_t& op_or_scalar)
    -> derived_t& {
    if constexpr (traits::is_terms_operator_v<type_t>) {
        return add_sub_impl_(op_or_scalar, coeff_policy_t::isub, coeff_policy_t::uminus);
    } else {
        *this -= (derived_t::identity() *= op_or_scalar);
        return *static_cast<derived_t*>(this);
    }
}
#endif  // MQ_HAS_CONCEPTS && !_MSC_VER

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator-() const -> derived_t {
    return (*static_cast<const derived_t*>(this) * static_cast<coefficient_t>(-1.));
}

// =============================================================================
// Multiplication

#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op<derived_t_<coefficient_t_>> op_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator*=(const op_t& other) -> derived_t& {
    return mul_impl_(other);
}

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <mindquantum::concepts::compat_terms_op_scalar<derived_t_<coefficient_t_>> scalar_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator*=(const scalar_t& scalar) -> derived_t& {
    using conv_helper_t = traits::conversion_helper<coefficient_t>;
    using value_t = typename coeff_term_dict_t::value_type;

    auto& ordered_index = terms_.template get<order::insertion>();
    for (auto it = begin(ordered_index); it != end(ordered_index); ++it) {
        ordered_index.modify(it, [scalar = conv_helper_t::apply(scalar)](value_t& value) {
            coeff_policy_t::imul(value.second, scalar);
        });
    }
    return *static_cast<derived_t*>(this);
}
#else
template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename type_t, typename>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator*=(const type_t& op_or_scalar)
    -> derived_t& {
    if constexpr (traits::is_terms_operator_v<type_t>) {
        return mul_impl_(op_or_scalar);
    } else {
        using conv_helper_t = traits::conversion_helper<coefficient_t>;
        using value_t = typename coeff_term_dict_t::value_type;
        auto& ordered_index = terms_.template get<order::insertion>();
        for (auto it = begin(ordered_index); it != end(ordered_index); ++it) {
            ordered_index.modify(it, [scalar = conv_helper_t::apply(op_or_scalar)](value_t& value) {
                coeff_policy_t::imul(value.second, scalar);
            });
        }
        return *static_cast<derived_t*>(this);
    }
}
#endif  // MQ_HAS_CONCEPTS && !_MSC_VER

// =============================================================================
// Division

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <mindquantum::concepts::compat_terms_op_scalar<derived_t_<coefficient_t_>> scalar_t>
#else
template <typename scalar_t, typename>
#endif  // MQ_HAS_CONCEPTS && !_MSC_VER
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::operator/=(const scalar_t& scalar) -> derived_t& {
    using conv_helper_t = traits::conversion_helper<coefficient_t>;
    return *this *= static_cast<coefficient_t>(1.) / conv_helper_t::apply(scalar);
}

// =============================================================================
// Other mathematical member functions

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::pow(uint32_t exponent) const -> derived_t {
    derived_t result = identity();
    for (auto i(0UL); i < exponent; ++i) {
        result *= *static_cast<const derived_t*>(this);
    }
    return result;
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
#if MQ_HAS_CONCEPTS && !(defined _MSC_VER)
template <mindquantum::concepts::terms_op op_t>
#else
template <typename op_t, typename>
#endif  // MQ_HAS_CONCEPTS && !(defined _MSC_VER)
bool TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::is_equal(const op_t& other) const {
    using policy_t = coeff_policy_t;
    using conv_helper_t = traits::conversion_helper<coefficient_t>;

    std::vector<typename coeff_term_dict_t::value_type> intersection;
    std::vector<typename coeff_term_dict_t::value_type> symmetric_differences;

    std::set_intersection(
        begin(terms_), end(terms_), begin(other.terms_), end(other.terms_), std::back_inserter(intersection),
        [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });
    MQ_DEBUG("Set intersection: {}", intersection);

    for (const auto& term : intersection) {
        const auto& left = terms_.template get<order::hashed>().find(term.first)->second;
        const auto& right = other.get_terms().template get<order::hashed>().find(term.first)->second;
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(left)>, coefficient_t>);
        static_assert(std::is_same_v<std::remove_cvref_t<decltype(right)>, coefficient_t>);
        if (!policy_t::equal(left, conv_helper_t::apply(right))) {
            MQ_DEBUG("{} != {}", left, conv_helper_t::apply(right));
            return false;
        }
    }

    std::set_symmetric_difference(begin(terms_), end(terms_), begin(other.terms_), end(other.terms_),
                                  std::back_inserter(symmetric_differences),
                                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    MQ_DEBUG("Symmetric differences: {}", symmetric_differences);
    return std::empty(symmetric_differences);
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename return_t, typename cast_func_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::cast_(const cast_func_t& cast_func) const
    -> return_t {
    using other_coeff_t = typename return_t::coefficient_t;
    using other_terms_t = typename return_t::coeff_term_dict_t;
    using other_value_t = typename other_terms_t::value_type;

    other_terms_t real_terms;
    auto out(*static_cast<const derived_t*>(this));
    std::transform(begin(terms_), end(terms_), std::back_inserter(real_terms), [&cast_func](const auto& term) {
        return other_value_t{term.first, cast_func(term.second)};
    });
    return return_t{real_terms};
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename other_t, typename assign_modify_op_t, typename coeff_unary_op_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::add_sub_impl_(const other_t& other,
                                                                                  assign_modify_op_t&& assign_modify_op,
                                                                                  coeff_unary_op_t&& coeff_unary_op)
    -> derived_t& {
    using conv_helper_t = traits::conversion_helper<coefficient_t>;
    using value_t = typename coeff_term_dict_t::value_type;

    for (const auto& [term, coeff] : other.terms_) {
        auto conv_coeff = conv_helper_t::apply(coeff);
        auto& hashed_index = terms_.template get<order::hashed>();
        if (auto it = hashed_index.find(term); it != hashed_index.cend()) {
            hashed_index.modify(it, [&assign_modify_op, &conv_coeff](value_t& term_coeff) {
                assign_modify_op(term_coeff.second, conv_coeff);
            });
            if (coeff_policy_t::is_zero(it->second)) {
                hashed_index.erase(it);
            }
        } else if (!coeff_policy_t::is_zero(conv_coeff)) {
            terms_.emplace_back(term, coeff_unary_op(conv_coeff));
        }
    }

    return *static_cast<derived_t*>(this);
}

// -----------------------------------------------------------------------------

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
template <typename other_t>
auto TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::mul_impl_(const other_t& other) -> derived_t& {
    using conv_helper_t = traits::conversion_helper<coefficient_t>;
    using value_t = typename coeff_term_dict_t::value_type;

    coeff_term_dict_t product_results;
    for (const auto& [left_op, left_coeff] : terms_) {
        for (const auto& [right_op, right_coeff] : other.get_terms()) {
            auto new_op = std::vector<term_t>{left_op};
            new_op.insert(end(new_op), begin(right_op), end(right_op));
            const auto [new_terms, new_coeff] = term_policy_t::simplify(
                new_op, coeff_policy_t::mul(left_coeff, conv_helper_t::apply(right_coeff)));
            auto& hashed_index = product_results.template get<order::hashed>();
            if (auto it = hashed_index.find(new_terms); it != hashed_index.cend()) {
                hashed_index.modify(it, [new_coeff = new_coeff](value_t& term_coeff) {
                    coeff_policy_t::iadd(term_coeff.second, new_coeff);
                });
            } else {
                product_results.emplace_back(std::move(new_terms), std::move(new_coeff));
            }
        }
    }
    terms_ = std::move(product_results);

    return *static_cast<derived_t*>(this);
}

// =============================================================================

template <template <typename coeff_t> class derived_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_>
void TermsOperatorBase<derived_t_, coefficient_t_, term_policy_t_>::calculate_num_targets_() const noexcept {
    num_targets_ = count_qubits();
}

// =============================================================================

}  // namespace mindquantum::ops
#endif /* TERMS_OPERATOR_BASE_TPP */
