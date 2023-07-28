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

#ifndef MQ_CONFIG_CONCEPTS_HPP
#define MQ_CONFIG_CONCEPTS_HPP

#include <concepts>
#include <tuple>

#include "config/config.h"
#include "config/type_traits.h"

namespace mindquantum::concepts {
template <typename T, typename U>
concept same_decay_as = std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<U>>;

template <typename T>
concept scalar = traits::is_scalar_decay_v<T>;

template <typename T, typename... Ts>
concept tuple_contains = traits::tuple_contains<T, std::tuple<Ts...>>;

template <typename T>
concept std_complex = traits::is_std_complex_v<T>;

template <typename T>
concept real_number = std::integral<T> || std::floating_point<T>;
template <typename T>
concept complex_number = std::same_as<std::complex<double>, T>;

template <typename T>
concept number = real_number<std::remove_cvref_t<T>> || complex_number<std::remove_cvref_t<T>>;
}  // namespace mindquantum::concepts

#endif /* MQ_CONFIG_CONCEPTS_HPP */
