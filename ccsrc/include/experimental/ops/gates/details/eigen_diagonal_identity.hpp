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

#ifndef EIGEN_DIAGONAL_IDENTITY_HPP
#define EIGEN_DIAGONAL_IDENTITY_HPP

#include <type_traits>

#include <Eigen/Core>

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

#endif /* EIGEN_DIAGONAL_IDENTITY_HPP */
