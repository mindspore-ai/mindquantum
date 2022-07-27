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

#include "ops/gates/fermion_operator.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>

#include <fmt/format.h>
#include <lru_cache/lru_cache.h>

#ifdef ENABLE_LOGGING
#    include <spdlog/spdlog.h>
#endif  // ENABLE_LOGGING

#include "core/parser/boost_x3_error_handler.hpp"
#include "details/boost_x3_parse_term.hpp"
#include "details/eigen_diagonal_identity.hpp"
#include "ops/gates.hpp"
#include "ops/gates/terms_operator.hpp"

// -----------------------------------------------------------------------------

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

#if MQ_HAS_ABSEIL_CPP
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define DECLARE_MEMOIZE_CACHE(name, cache_size, function)                                                          \
        static auto name = lru_cache::node::memoize_function(cache_size, function)
#else
#    include "details/cache_impl.hpp"
   // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define DECLARE_MEMOIZE_CACHE(name, cache_size, function)                                                          \
        static auto name = lru_cache::staticc::memoize_function<cache_size>(function)
#endif  // MQ_HAS_ABSEIL_CPP

// =============================================================================

namespace {
constexpr auto cache_size = 100UL;
using csr_matrix_t = mindquantum::ops::FermionOperator::csr_matrix_t;

auto n_identity(std::size_t n) -> csr_matrix_t {
    using scalar_t = csr_matrix_t::Scalar;
    // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MQ_CASE_FOR_NQUBITS(n)                                                                                         \
    case n:                                                                                                            \
        return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1U << (n)>()};                                         \
        break

    switch (n) {
        MQ_CASE_FOR_NQUBITS(0U);
        MQ_CASE_FOR_NQUBITS(1U);
        MQ_CASE_FOR_NQUBITS(2U);
        MQ_CASE_FOR_NQUBITS(3U);
        MQ_CASE_FOR_NQUBITS(4U);
        MQ_CASE_FOR_NQUBITS(5U);
        MQ_CASE_FOR_NQUBITS(6U);
        default:
            auto tmp = Eigen::DiagonalMatrix<csr_matrix_t::Scalar, Eigen::Dynamic>(1U << n);
            tmp.setIdentity();
            return csr_matrix_t{tmp};
            break;
    }
#undef MQ_CASE_FOR_NQUBITS
}

// =============================================================================

auto n_sz_impl(std::size_t n) -> csr_matrix_t {
    using scalar_t = csr_matrix_t::Scalar;

    if (n == 0) {
        return n_identity(0);
    }

    auto result = csr_matrix_t{2, 2};
    result.insert(0, 0) = 1;
    result.insert(1, 1) = -1;
    const auto tmp = result;

    --n;
    for (auto i(0UL); i < n; ++i) {
        result = Eigen::kroneckerProduct(result, tmp).eval();
    }

    return result;
}

// -----------------------------------------------------------------------------

auto n_sz(std::size_t n) {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, n_sz_impl);
    return cache_(n);
}

// =============================================================================

auto single_fermion_word_impl(const std::tuple<std::size_t, bool, std::size_t>& data) -> csr_matrix_t {
    const auto& [idx, is_adg, n_qubits] = data;
    auto result = csr_matrix_t{2, 2};
    if (is_adg) {
        result.insert(0, 1) = 1;
    } else {
        result.insert(1, 0) = 1;
    }
    return Eigen::kroneckerProduct(n_identity(n_qubits - 1 - idx), Eigen::kroneckerProduct(result, n_sz(idx)).eval());
}

// -----------------------------------------------------------------------------

auto single_fermion_word(std::size_t idx, bool is_adg, std::size_t n_qubits) {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, single_fermion_word_impl);
    return cache_(std::make_tuple(idx, is_adg, n_qubits));
}

// =============================================================================

auto two_fermion_word_impl(const std::tuple<std::size_t, bool, std::size_t, bool, std::size_t>& data) -> csr_matrix_t {
    const auto& [idx1, is_adg1, idx2, is_adg2, n_qubits] = data;
    return single_fermion_word(idx1, is_adg1, n_qubits) * single_fermion_word(idx2, is_adg2, n_qubits);
}

// -----------------------------------------------------------------------------

auto two_fermion_word(std::size_t idx1, bool is_adg1, std::size_t idx2, bool is_adg2, std::size_t n_qubits) {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, two_fermion_word_impl);
    return cache_(std::make_tuple(idx1, is_adg1, idx2, is_adg2, n_qubits));
}
// =============================================================================
}  // namespace

namespace mindquantum::ops {

FermionOperator::FermionOperator(term_t term, coefficient_t coeff) : TermsOperator(std::move(term), coeff) {
}

// -----------------------------------------------------------------------------

FermionOperator::FermionOperator(const std::vector<term_t>& term, coefficient_t coeff) : TermsOperator(term, coeff) {
}

// -----------------------------------------------------------------------------

FermionOperator::FermionOperator(complex_term_dict_t terms) : TermsOperator(std::move(terms)) {
}

// =============================================================================

auto FermionOperator::matrix(std::optional<uint32_t> n_qubits) const -> std::optional<csr_matrix_t> {
    if (std::empty(terms_)) {
        return std::nullopt;
    }

    // NB: this is assuming that num_targets_ is up-to-date
    const auto& n_qubits_local = num_targets_;

    if (n_qubits_local == 0UL && !n_qubits) {
        std::cerr << "You should specify n_qubits for converting an identity qubit operator.";
        return std::nullopt;
    }

    if (n_qubits && n_qubits.value() < n_qubits_local) {
        std::cerr << fmt::format(
            "Given n_qubits {} is smaller than the number of qubits of this qubit operator, which is {}.\n",
            n_qubits.value(), n_qubits_local);
        return std::nullopt;
    }

    const auto n_qubits_value = n_qubits.value_or(n_qubits_local);

    const auto process_term = [n_qubits_value](const auto& local_ops) -> csr_matrix_t {
        if (std::empty(local_ops)) {
            return (Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(
                        1U << n_qubits_value, 1U << n_qubits_value))
                .sparseView();
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
            if (std::size(group) == 2) {
                return ::two_fermion_word(group[0].first, group[0].second == TermValue::adg, group[1].first,
                                          group[1].second == TermValue::adg, n_qubits_value);
            }
            return ::single_fermion_word(group[0].first, group[0].second == TermValue::adg, n_qubits_value);
        };

        assert(!std::empty(groups));
        auto tmp = process_group(groups.front());

        for (auto it(begin(groups) + 1); it != end(groups); ++it) {
            tmp *= process_group(*it);
        }

        return tmp;
    };

    auto it = begin(terms_);
    auto result = process_term(it->first) * it->second;
    ++it;
    for (; it != end(terms_); ++it) {
        result *= process_term(it->first) * it->second;
    }

    return result;
}

// =============================================================================

auto FermionOperator::split() const noexcept -> std::vector<FermionOperator> {
    std::vector<FermionOperator> result;
    for (const auto& [local_ops, coeff] : terms_) {
        result.emplace_back(local_ops, coeff);
    }
    return result;
}

// -----------------------------------------------------------------------------

auto FermionOperator::normal_ordered() const -> self_t {
    auto ordered_op{*this};
    for (const auto& [local_ops, coeff] : terms_) {
        ordered_op += normal_ordered_term_(local_ops, coeff);
    }
    return ordered_op;
}

// =============================================================================

auto FermionOperator::simplify_(std::vector<term_t> terms, coefficient_t coeff)
    -> std::tuple<std::vector<term_t>, coefficient_t> {
    return {std::move(terms), coeff};
}

// =============================================================================

auto FermionOperator::normal_ordered_term_(std::vector<term_t> terms, coefficient_t coeff) -> FermionOperator {
    FermionOperator ordered_term;

    for (auto it(begin(terms) + 1); it != end(terms); ++it) {
        for (auto it_jm1(std::make_reverse_iterator(it) - 1), it_j(std::make_reverse_iterator(it));
             it_jm1 != rend(terms); ++it_jm1, ++it_j) {
            // Swap operators if left operator is a and right operator is a^\dagger
            if (it_jm1->second == TermValue::a && it_j->second == TermValue::adg) {
                std::iter_swap(it_jm1, it_j);
                coeff *= -1.;
                // If indice are same, employ the anti-commutation relationship and generate the new term
                if (it_jm1->first == it_j->first) {
                    // NB: we need to skip skip elements j-1 and j. Since it_jm1 and it_j are reverse iterators:
                    //     (it_j + 1).base() is actually the j-1 element
                    //     it_jm1.base() is actually the j+1 element
                    auto new_terms = std::vector<term_t>(begin(terms), (it_j + 1).base());
                    new_terms.reserve(std::size(terms) - 1);
                    std::copy(it_jm1.base(), end(terms), std::back_inserter(new_terms));
                    ordered_term += normal_ordered_term_(new_terms, -1. * coeff);
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

    // for (auto i(1UL); i < std::size(terms); ++i) {
    //     for (auto j(i); j < i; --i) {
    //         const auto& left_sub_term = terms[j - 1];
    //         const auto& right_sub_term = terms[j];
    //         // Swap operators if left operator is a and right operator is a^\dagger
    //         if (left_sub_term.second == TermValue::a && right_sub_term.second == TermValue::adg) {
    //             std::swap(terms[j - 1], terms[j]);
    //             coeff *= -1.;
    //             // If indice are same, employ the anti-commutation relationship and generate the new term
    //             if (left_sub_term.first == right_sub_term.first) {
    //                 // NB: skip elements j-1 and j
    //                 auto new_term = std::vector<term_t>(begin(terms), begin(terms) + j - 1);
    //                 new_term.reserve(std::size(terms) - 1);
    //                 std::copy(begin(terms) + j + 1, end(terms), std::back_inserter(new_term));
    //                 ordered_term += normal_ordered_term_(new_term, -1. * coeff);
    //             }
    //         } else if (left_sub_term.second == right_sub_term.second) {
    //             // TODO(dnguyen): Check that this is really ok? What if ordered_term is not zero?
    //             // If indices are the same, evaluate to zero
    //             if (left_sub_term.first == right_sub_term.first) {
    //                 return ordered_term;
    //             }
    //             // Swap them if the same operator but lower index on the left
    //             if (left_sub_term.first < right_sub_term.first) {
    //                 std::swap(terms[j - 1], terms[j]);
    //                 coeff *= -1.;
    //             }
    //         }
    //     }
    // }

    // Add the terms and return
    ordered_term += FermionOperator(terms, coeff);
    return ordered_term;
}
}  // namespace mindquantum::ops

// =============================================================================

namespace x3 = boost::spirit::x3;

namespace ast {
using mindquantum::ops::TermValue;
using term_t = mindquantum::ops::FermionOperator::term_t;

struct TermOp : x3::symbols<TermValue> {
    TermOp() {
        add("^", TermValue::adg);
    }
} const term_op;

}  // namespace ast

// -----------------------------------------------------------------------------

namespace parser {
struct local_op_sym_class {};
x3::rule<local_op_sym_class, ast::TermValue> const local_op_sym = "ladder operator (a: '', adg: '^')";
static const auto local_op_sym_def = ast::term_op | x3::attr(ast::TermValue::a);

struct unsigned_value_class {};
x3::rule<unsigned_value_class, uint32_t> const unsigned_value = "unsigned int";
static const auto unsigned_value_def = x3::uint_;

struct term_class : mindquantum::parser::x3::rule::error_handler {};
x3::rule<term_class, ast::term_t> const term = "fermion_operator term";
static const auto term_def = x3::eps > unsigned_value > local_op_sym > x3::omit[*x3::space];

BOOST_SPIRIT_DEFINE(local_op_sym, unsigned_value, term);

// -------------------------------------

static const auto terms = +term;
}  // namespace parser

// -----------------------------------------------------------------------------

namespace mindquantum::ops {
auto FermionOperator::parse_string_(std::string_view terms_string) -> std::vector<term_t> {
    using terms_t = std::vector<term_t>;
    if (terms_t terms; parser::parse_term(begin(terms_string), end(terms_string), terms, ::parser::terms)) {
        return terms;
    }
    return {};
}

// =============================================================================

}  // namespace mindquantum::ops
