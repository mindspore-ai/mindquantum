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

#ifndef MQ_CONFIG_COMMON_TYPE_HPP
#define MQ_CONFIG_COMMON_TYPE_HPP

// =============================================================================

namespace mindquantum::traits {
template <typename... args_t>
struct common_type;

// Delegate to std::common_type
template <typename T>
struct common_type<T> : std::common_type_t<T> {};

template <typename T, typename U>
struct common_type<T, U> : std::common_type<T, U> {};

// -----------------------------------------------------------------------------
// Special cases involving std::complex

template <typename float_t, typename U>
struct common_type<std::complex<float_t>, U> {
    using type = std::complex<std::common_type_t<float_t, U>>;
};
template <typename T, typename float_t>
struct common_type<T, std::complex<float_t>> {
    using type = std::complex<std::common_type_t<T, float_t>>;
};
template <typename float_t, typename float2_t>
struct common_type<std::complex<float_t>, std::complex<float2_t>> {
    using type = std::complex<std::common_type_t<float_t, float2_t>>;
};

// -----------------------------------------------------------------------------
// Magic to support multiple types and use the std::complex<...> logic above

//! Helper type to implement pair-wise common_type logic
template <typename, typename, typename = void>
struct common_type_fold;

//! Default specialization if no common type can be found
template <typename common_t, typename arg_t>
struct common_type_fold<common_t, arg_t, void> {};

//! Placeholder type to store a template argument pack
template <typename... args_t>
struct common_type_pack {};

//! Main specialization to perform chained common_type computation
template <typename common_t, typename... args_t>
struct common_type_fold<common_t, common_type_pack<args_t...>, std::void_t<typename common_t::type>>
    : public common_type<typename common_t::type, args_t...> {};

// -----------------------------------------------------------------------------

//! Implement pair-wise common_type logic (like std::common_type)
template <typename T, typename U, typename... args_t>
struct common_type<T, U, args_t...> : common_type_fold<common_type<T, U>, common_type_pack<args_t...>> {};

// =============================================================================

template <typename... args_t>
using common_type_t = typename common_type<args_t...>::type;
}  // namespace mindquantum::traits

// =============================================================================

#endif /* MQ_CONFIG_COMMON_TYPE_HPP */
