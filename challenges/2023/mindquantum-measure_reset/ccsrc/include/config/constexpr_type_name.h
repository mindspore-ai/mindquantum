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

#ifndef MQ_CONFIG_CONSTEXPR_TYPE_NAME
#define MQ_CONFIG_CONSTEXPR_TYPE_NAME

#include <algorithm>
#include <string_view>

#include "config/config.h"

// =============================================================================

/*
 * The code below work by finding the prefix/suffix offsets in the function signature of wrapped_type_name<...> where
 * the template parameter appears. This is done using the default type (ie. void).
 * For example (on Clang/GCC):
 *   - std::string_view details::wrapped_type_name() [T = void]
 * -> prefix_offset = 50, suffix_offset = 1
 *
 * With these two offsets, we can then extract the substring containing the type name for any other instantiation of
 * wrapped_type_name<...>().
 *
 * NB: on Windows with MSVC, we adjust the prefix offset in the case of classes/structs since otherwise the types would
 *     be reported as 'class A' or 'struct B'.
 */

namespace mindquantum {
template <typename T>
constexpr std::string_view get_type_name();

template <>
constexpr std::string_view get_type_name<void>() {
    return "void";
}

namespace details {
using type_name_prober = void;

template <typename T>
constexpr std::string_view wrapped_type_name() {
#ifdef __clang__
    return __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
#    error "Unsupported compiler"
#endif
}

constexpr std::size_t wrapped_type_name_prefix_length() {
    return wrapped_type_name<type_name_prober>().find(get_type_name<type_name_prober>());
}

constexpr std::size_t wrapped_type_name_suffix_length() {
    return wrapped_type_name<type_name_prober>().length() - wrapped_type_name_prefix_length()
           - get_type_name<type_name_prober>().length();
}
}  // namespace details

// =============================================================================

template <typename T>
constexpr std::string_view get_type_name() {
    constexpr auto wrapped_name = details::wrapped_type_name<T>();
#if defined(_MSC_VER)
    /* Correct prefix length by 6 (length of 'class ') and then find first non-space char since length of 'struct ' is
     * 7.
     */
    constexpr auto prefix_length
        = details::wrapped_type_name_prefix_length()
          + std::conditional_t<std::is_class_v<std::decay_t<T>>, std::integral_constant<int, 6>,
                               std::integral_constant<int, 0>>::value;
    constexpr auto real_prefix_length = wrapped_name.find_first_not_of(' ', prefix_length);
#else
    constexpr auto real_prefix_length = details::wrapped_type_name_prefix_length();
#endif
    constexpr auto suffix_length = details::wrapped_type_name_suffix_length();
    constexpr auto type_name_length = wrapped_name.length() - real_prefix_length - suffix_length;
    return wrapped_name.substr(real_prefix_length, type_name_length);
}
}  // namespace mindquantum

// =============================================================================

#endif /* MQ_CONFIG_CONSTEXPR_TYPE_NAME */
