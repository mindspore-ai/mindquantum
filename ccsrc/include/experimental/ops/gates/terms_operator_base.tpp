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
#include "experimental/ops/gates/terms_operator_base.hpp"
#include "experimental/ops/meta/dagger.hpp"

// =============================================================================

namespace mindquantum::ops {
template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    term_t term, coefficient_t coeff)
    : TermsOperatorBase(coeff_term_dict_t{{{std::move(term)}, coeff}}) {
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    const terms_t& terms, coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace(new_terms, new_coeff);
    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    const py_terms_t& terms, coefficient_t coeff) {
    const auto [new_terms, new_coeff] = term_policy_t::simplify(terms, coeff);
    terms_.emplace(new_terms, new_coeff);
    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    const coeff_term_dict_t& terms) {
    for (const auto& [local_ops, coeff] : terms) {
        terms_.emplace(term_policy_t::sort_terms(local_ops, coeff));
    }

    calculate_num_targets_();
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    std::string_view terms_string, coefficient_t coeff)
    : TermsOperatorBase(term_policy_t::parse_terms_string(terms_string), coeff) {
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::TermsOperatorBase(
    coeff_term_dict_t terms, sorted_constructor_t /* unused */)
    : terms_{std::move(terms)} {
    calculate_num_targets_();
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
uint32_t TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::num_targets()
    const {
    return num_targets_;
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::empty()
    const noexcept {
    return std::empty(terms_);
}
// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::size() const {
    return std::size(terms_);
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::get_terms() const
    -> const coeff_term_dict_t& {
    return terms_;
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::get_terms_pair()
    const -> py_coeff_term_list_t {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::get_coeff(
    const terms_t& term) const -> coefficient_t {
    return terms_.at(term);
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
bool TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::is_identity(
    double abs_tol) const {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::count_qubits()
    const -> term_t::first_type {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::identity()
    -> derived_t {
    return derived_t{terms_t{}};
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::hermitian() const
    -> derived_t {
    coeff_term_dict_t terms;
    for (auto& [local_ops, coeff] : terms_) {
        terms.emplace(std::move(term_policy_t::hermitian(local_ops)), std::move(coeff_policy_t::conjugate(coeff)));
    }
    return derived_t(std::move(terms));
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::adjoint()
    const noexcept -> operator_t {
    return *static_cast<derived_t*>(*this);
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::constant()
    const noexcept -> coefficient_t {
    if (const auto it = terms_.find({}); it != end(terms_)) {
        assert(std::empty(it->first));
        return it->second;
    }
    return coefficient_t{0.0};
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::constant(
    const coefficient_t& coeff) -> void {
    terms_[{}] = coeff;
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
bool TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::is_singlet()
    const {
    return size() == 1;
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::singlet() const
    -> std::vector<derived_t> {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::singlet_coeff()
    const -> coefficient_t {
    if (!is_singlet()) {
        MQ_ERROR("Operator is not a singlet!");
        return {};
    }
    return begin(terms_)->second;
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::split() const
    -> std::vector<std::pair<coefficient_t, derived_t>> {
    std::vector<std::pair<coefficient_t, derived_t>> result;
    for (const auto& [local_ops, coeff] : terms_) {
        result.emplace_back(std::make_pair(coeff, derived_t(local_ops, coeff_policy_t::one)));
    }
    return result;
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::real() const
    -> derived_real_t {
    return real_cast_<RealCastType::REAL>();
}

// -----------------------------------------------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::imag() const
    -> derived_real_t {
    return real_cast_<RealCastType::REAL>();
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::compress(
    double abs_tol) -> derived_t& {
    auto new_end = tsl::remove_if(begin(terms_), end(terms_), [abs_tol](const auto& term) -> bool {
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

namespace details {
using namespace std::literals::string_literals;
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::to_string()
    const noexcept -> std::string {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::dumps(
    std::size_t indent) const -> std::string {
    nlohmann::json json(*this);
    return json.dump(static_cast<int>(indent), ' ', true);
}

// -------------------------------------

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::loads(
    std::string_view string_data) -> std::optional<derived_t> {
    return nlohmann::json::parse(string_data).get<derived_t>();
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::operator-() const
    -> derived_t {
    return (*static_cast<const derived_t*>(this) * -1.);
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::pow(
    uint32_t exponent) const -> derived_t {
    derived_t result = identity();
    for (auto i(0UL); i < exponent; ++i) {
        result *= *static_cast<const derived_t*>(this);
    }
    return result;
}

// =============================================================================

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
bool TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::operator==(
    const derived_t& other) const {
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

template <typename derived_t, typename derived_real_t_, typename coefficient_t_,
          template <typename coeff_t> class term_policy_t_, template <typename coeff_t> class coeff_policy_t_>
template <RealCastType cast_type>
auto TermsOperatorBase<derived_t, derived_real_t_, coefficient_t_, term_policy_t_, coeff_policy_t_>::real_cast_() const
    -> derived_real_t {
    using real_terms_t = typename derived_real_t::coeff_term_dict_t;
    using real_value_t = typename real_terms_t::value_type;

    real_terms_t real_terms;
    auto out(*static_cast<const derived_t*>(this));
    std::transform(begin(terms_), end(terms_), std::inserter(real_terms, end(real_terms)), [](const auto& term) {
        return real_value_t{term.first, real_cast<cast_type>(term.second)};
    });
    return derived_real_t{real_terms};
}

// =============================================================================

}  // namespace mindquantum::ops
#endif /* TERMS_OPERATOR_BASE_TPP */
