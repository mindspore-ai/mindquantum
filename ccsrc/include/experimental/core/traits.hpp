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

#include <tuple>
#include <type_traits>

#include "config/type_traits.hpp"

#include "experimental/core/config.hpp"

namespace mindquantum::traits {
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
}  // namespace mindquantum::traits

#endif /* CORE_TRAITS_HPP */
