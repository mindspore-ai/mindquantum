//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef CORE_TRAITS_HPP
#define CORE_TRAITS_HPP

#include <complex>
#include <tuple>
#include <type_traits>

#include "experimental/core/config.hpp"
#include "experimental/core/traits.hpp"

namespace mindquantum::traits {

// ---------------------------------

template <typename... Ts>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename... Ts>
inline constexpr auto is_tuple_v = is_tuple<Ts...>::value;

// ---------------------------------

//! C++ type-traits that is true if and only if the list of types passed in as argument only contains unique types.
template <typename...>
inline constexpr auto is_unique = std::true_type{};

template <typename T, typename... Ts>
inline constexpr auto is_unique<T, Ts...> = std::bool_constant<
    std::conjunction_v<std::negation<std::is_same<T, Ts>>...> && is_unique<Ts...>>{};

// ---------------------------------

//! C++ type-traits that is true if and only the type passed in argument can be found inside the type list of the tuple
template <typename T, typename Tuple>
inline constexpr auto tuple_contains = false;

template <typename T, typename... Us>
inline constexpr auto tuple_contains<T, std::tuple<Us...>> = std::disjunction_v<std::is_same<T, Us>...>;

// =============================================================================

template <typename float_t, typename = void>
struct to_real_type;

template <typename float_t>
struct to_real_type<float_t, std::enable_if_t<std::is_floating_point_v<float_t>>> {
    using type = float_t;
};

template <typename complex_t>
struct to_real_type<complex_t, std::enable_if_t<is_std_complex_v<complex_t>>> {
    using type = typename complex_t::value_type;
};

template <typename float_t>
using to_real_type_t = typename to_real_type<float_t>::type;

// -----------------------------------------------------------------------------

template <typename float_t, typename = void>
struct to_cmplx_type;

template <typename float_t>
struct to_cmplx_type<float_t, std::enable_if_t<std::is_floating_point_v<float_t>>> {
    using type = std::complex<float_t>;
};

template <typename complex_t>
struct to_cmplx_type<complex_t, std::enable_if_t<is_std_complex_v<complex_t>>> {
    using type = complex_t;
};

template <typename float_t>
using to_cmplx_type_t = typename to_cmplx_type<float_t>::type;

}  // namespace mindquantum::traits

#endif /* CORE_TRAITS_HPP */
