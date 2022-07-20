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
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <lru_cache/lru_cache.h>
#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>

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

#include "ops/gates.hpp"
#include "ops/gates/terms_operator.hpp"

#if !MQ_HAS_ABSEIL_CPP
#    include "details/cache_impl.hpp"
#endif  // MQ_HAS_ABSEIL_CPP

namespace {
static constexpr auto cache_size = 100UL;
using csr_matrix_t = mindquantum::ops::FermionOperator::csr_matrix_t;

// =============================================================================

// Merge two integer sequences
template <typename lhs_t, typename rhs_t>
struct merge;

template <typename int_t, int_t... lhs, int_t... rhs>
struct merge<std::integer_sequence<int_t, lhs...>, std::integer_sequence<int_t, rhs...>> {
    using type = std::integer_sequence<int_t, lhs..., rhs...>;
};

template <typename int_t, typename N>
struct log_make_sequence {
    using L = std::integral_constant<int_t, N::value / 2>;
    using R = std::integral_constant<int_t, N::value - L::value>;
    using type =
        typename merge<typename log_make_sequence<int_t, L>::type, typename log_make_sequence<int_t, R>::type>::type;
};

template <typename int_t>
struct log_make_sequence<int_t, std::integral_constant<int_t, 0>> {
    using type = std::integer_sequence<int_t>;
};

template <typename int_t>
struct log_make_sequence<int_t, std::integral_constant<int_t, 1>> {
    using type = std::integer_sequence<int_t, 1>;
};

template <std::size_t N>
using make_ones_sequence = typename log_make_sequence<std::size_t, std::integral_constant<std::size_t, N>>::type;

template <typename scalar_t, typename int_t, int_t... ints>
auto generate_eigen_diagonal_impl(std::integer_sequence<int_t, ints...>) {
    return Eigen::DiagonalMatrix<scalar_t, sizeof...(ints)>{ints...};
}

template <typename scalar_t, std::size_t N>
auto generate_eigen_diagonal() {
    return generate_eigen_diagonal_impl<scalar_t>(make_ones_sequence<N>{});
}

static auto n_identity(std::size_t n) -> csr_matrix_t {
    using scalar_t = csr_matrix_t::Scalar;
    switch (n) {
        case 0:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 0>()};
            break;
        case 1:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 1>()};
            break;
        case 2:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 2>()};
            break;
        case 3:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 3>()};
            break;
        case 4:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 4>()};
            break;
        case 5:
            return csr_matrix_t{::generate_eigen_diagonal<scalar_t, 1 << 5>()};
            break;
        default:
            auto tmp = Eigen::DiagonalMatrix<csr_matrix_t::Scalar, Eigen::Dynamic>(1UL << n);
            tmp.setIdentity();
            return csr_matrix_t{tmp};
            break;
    }
}

static auto n_sz_impl(std::size_t n) -> csr_matrix_t {
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
static auto n_sz(std::size_t n) -> csr_matrix_t {
#if MQ_HAS_ABSEIL_CPP
    static auto cache_ = lru_cache::node::memoize_function(cache_size, n_sz_impl);
#else
    static auto cache_ = lru_cache::staticc::memoize_function<cache_size>(n_sz_impl);
#endif  // MQ_HAS_ABSEIL_CPP
    return cache_(n);
}

static auto single_fermion_word(std::size_t idx, bool dag, std::size_t n_qubits) -> csr_matrix_t {
#if MQ_HAS_ABSEIL_CPP
    static auto cache_ = lru_cache::node::make_cache<std::tuple<std::size_t, bool, std::size_t>, csr_matrix_t>(
        cache_size);
#else
    static auto cache_
        = lru_cache::staticc::make_cache<cache_size, std::tuple<std::size_t, bool, std::size_t>, csr_matrix_t>();
#endif  // MQ_HAS_ABSEIL_CPP

    if (auto value_or_null = cache_.get_or_null(std::make_tuple(idx, dag, n_qubits)); value_or_null != nullptr) {
        return *value_or_null;
    }

    auto result = csr_matrix_t{2, 2};
    if (dag) {
        result.insert(0, 1) = 1;
    } else {
        result.insert(1, 0) = 1;
    }
    return cache_.insert(
        std::make_tuple(idx, dag, n_qubits),
        Eigen::kroneckerProduct(n_identity(n_qubits - 1 - idx), Eigen::kroneckerProduct(result, n_sz(idx)).eval()));
}

static auto two_fermion_word(std::size_t idx1, bool dag1, std::size_t idx2, bool dag2, std::size_t n_qubits)
    -> csr_matrix_t {
    return single_fermion_word(idx1, dag1, n_qubits) * single_fermion_word(idx2, dag2, n_qubits);
}
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
    csr_matrix_t result;
    for (const auto& [local_ops, coeff] : terms_) {
        if (std::empty(local_ops)) {
            result += (Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(
                           1U << n_qubits_value, 1U << n_qubits_value))
                          .sparseView()
                      * coeff;
        } else {
            static constexpr auto size_groups = 2UL;
            std::vector<compat::span<const term_t>> groups;
            groups.reserve(std::size(local_ops) / size_groups + 1);

            const auto local_ops_end = end(local_ops);
            auto num_to_copy = size_groups;
            for (auto it(begin(local_ops)); it != local_ops_end; std::advance(it, num_to_copy)) {
                num_to_copy = std::min(static_cast<decltype(size_groups)>(std::distance(it, local_ops_end)),
                                       size_groups);
                groups.emplace_back(&*it, size_groups);
            }

            auto tmp = n_identity(1);
            for (const auto& local_ops_span : groups) {
                if (!std::empty(local_ops_span)) {
                    if (std::size(local_ops_span) == 2) {
                    } else {
                    }
                }
            }
        }
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

// =============================================================================

}  // namespace mindquantum::ops
