/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPS_GATES_TRAITS_HPP
#define OPS_GATES_TRAITS_HPP

#include <complex>
#include <type_traits>

#include "config/type_traits.h"

namespace mindquantum::traits {
template <typename scalar_t, bool is_real, typename = void>
struct is_compatible_scalar : std::false_type {};

// NB: If the ref coefficient is complex, then we accept all scalar types
template <typename scalar_t>
struct is_compatible_scalar<scalar_t, false, std::enable_if_t<is_scalar_v<scalar_t>>> : std::true_type {};

// NB: If the ref coefficient is real-valued, then we accept only real-valued scalar types
template <typename scalar_t>
struct is_compatible_scalar<scalar_t, true, std::enable_if_t<is_scalar_v<scalar_t> && !is_complex_v<scalar_t>>>
    : std::true_type {};

template <typename scalar_t, bool is_real>
inline constexpr auto is_compatible_scalar_v = is_compatible_scalar<scalar_t, is_real>::value;

template <typename scalar_t, bool is_real>
inline constexpr auto is_compatible_scalar_decay_v = is_compatible_scalar_v<std::remove_cvref_t<scalar_t>, is_real>;

// Real numbers
static_assert(is_compatible_scalar_v<float, true>);
static_assert(is_compatible_scalar_v<double, true>);
static_assert(!is_compatible_scalar_v<std::complex<float>, true>);
static_assert(!is_compatible_scalar_v<std::complex<double>, true>);
// Complex numbers
static_assert(is_compatible_scalar_v<float, false>);
static_assert(is_compatible_scalar_v<double, false>);
static_assert(is_compatible_scalar_v<std::complex<float>, false>);
static_assert(is_compatible_scalar_v<std::complex<double>, false>);
}  // namespace mindquantum::traits

#endif /* OPS_GATES_TRAITS_HPP */
