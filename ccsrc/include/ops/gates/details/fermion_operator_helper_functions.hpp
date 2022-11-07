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

#ifndef DETAILS_FERMION_OPERATOR_HELPER_FUNCTIONS_HPP
#define DETAILS_FERMION_OPERATOR_HELPER_FUNCTIONS_HPP

#include <cstdint>
#include <tuple>
#include <utility>

#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>

#include <lru_cache/lru_cache.h>

#include "ops/gates/details/eigen_sparse_identity.hpp"

// -----------------------------------------------------------------------------

#if MQ_HAS_ABSEIL_CPP
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define DECLARE_MEMOIZE_CACHE(name, cache_size, function)                                                          \
        static auto name = lru_cache::node::memoize_function(cache_size, function)
#else
#    include "ops/gates/details/cache_impl.hpp"
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define DECLARE_MEMOIZE_CACHE(name, cache_size, function)                                                          \
        static auto name = lru_cache::staticc::memoize_function<cache_size>(function)
#endif  // MQ_HAS_ABSEIL_CPP

// =============================================================================

namespace mindquantum::ops::details {
constexpr auto cache_size = 100UL;
template <typename coeff_t>
types::csr_matrix_t<coeff_t> n_identity(std::size_t n_qubits);
template <typename coeff_t>
types::csr_matrix_t<coeff_t> n_sz(std::size_t n);
template <typename coeff_t>
types::csr_matrix_t<coeff_t> single_fermion_word(std::size_t idx, bool is_adg, std::size_t n_qubits);
template <typename coeff_t>
types::csr_matrix_t<coeff_t> two_fermion_word(std::size_t idx1, bool is_adg1, std::size_t idx2, bool is_adg2,
                                              std::size_t n_qubits);
}  // namespace mindquantum::ops::details

// =============================================================================
// =============================================================================

namespace mindquantum::ops::details::impl {
template <typename coeff_t>
auto n_sz(std::size_t n) -> types::csr_matrix_t<coeff_t> {
    using matrix_t = types::csr_matrix_t<coeff_t>;
    using mindquantum::ops::details::n_identity;
    using scalar_t = typename matrix_t::Scalar;

    if (n == 0) {
        return n_identity<coeff_t>(0);
    }

    auto result = matrix_t{2, 2};
    result.insert(0, 0) = static_cast<coeff_t>(1.);
    result.insert(1, 1) = static_cast<coeff_t>(-1.);
    const auto tmp = result;

    --n;
    for (auto i(0UL); i < n; ++i) {
        result = Eigen::kroneckerProduct(result, tmp).eval();
    }

    return result;
}

// -----------------------------------------------------------------------------

template <typename coeff_t>
auto single_fermion_word(const std::tuple<std::size_t, bool, std::size_t>& data) -> types::csr_matrix_t<coeff_t> {
    using matrix_t = types::csr_matrix_t<coeff_t>;
    using mindquantum::ops::details::n_identity;
    using mindquantum::ops::details::n_sz;
    const auto& [idx, is_adg, n_qubits] = data;
    auto result = matrix_t{2, 2};
    if (is_adg) {
        result.insert(1, 0) = static_cast<coeff_t>(1.);
    } else {
        result.insert(0, 1) = static_cast<coeff_t>(1.);
    }
    return Eigen::kroneckerProduct(n_identity<coeff_t>(n_qubits - 1 - idx),
                                   Eigen::kroneckerProduct(result, n_sz<coeff_t>(idx)).eval())
        .eval();
}

// -----------------------------------------------------------------------------

template <typename coeff_t>
auto two_fermion_word(const std::tuple<std::size_t, bool, std::size_t, bool, std::size_t>& data)
    -> types::csr_matrix_t<coeff_t> {
    const auto& [idx1, is_adg1, idx2, is_adg2, n_qubits] = data;
    return mindquantum::ops::details::single_fermion_word<coeff_t>(idx1, is_adg1, n_qubits)
           * mindquantum::ops::details::single_fermion_word<coeff_t>(idx2, is_adg2, n_qubits);
}
}  // namespace mindquantum::ops::details::impl

// =============================================================================

namespace mindquantum::ops::details {
template <typename coeff_t>
auto n_sz(std::size_t n) -> types::csr_matrix_t<coeff_t> {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, impl::n_sz<coeff_t>);
    return cache_(n);
}

template <typename coeff_t>
auto single_fermion_word(std::size_t idx, bool is_adg, std::size_t n_qubits) -> types::csr_matrix_t<coeff_t> {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, impl::single_fermion_word<coeff_t>);
    return cache_(std::make_tuple(idx, is_adg, n_qubits));
}

template <typename coeff_t>
auto two_fermion_word(std::size_t idx1, bool is_adg1, std::size_t idx2, bool is_adg2, std::size_t n_qubits)
    -> types::csr_matrix_t<coeff_t> {
    DECLARE_MEMOIZE_CACHE(cache_, cache_size, impl::two_fermion_word<coeff_t>);
    return cache_(std::make_tuple(idx1, is_adg1, idx2, is_adg2, n_qubits));
}

}  // namespace mindquantum::ops::details

// =============================================================================

#endif /* DETAILS_FERMION_OPERATOR_HELPER_FUNCTIONS_HPP */
