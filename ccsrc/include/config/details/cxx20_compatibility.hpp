/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef CORE_CXX20_COMPATIBILITY_HPP
#define CORE_CXX20_COMPATIBILITY_HPP

#include <type_traits>
#ifdef __has_include
#    if __has_include(<version>)
#        include <version>
#    endif
#endif

#if MQ_HAS_CONCEPTS
#    include <concepts>
#endif

namespace std {
#if MQ_HAS_CONCEPTS && !MQ_HAS_CONCEPT_LIBRARY && !(defined(_MSC_VER) && __cpp_lib_concepts == 201907L)
// clang-format off
template <class T>
concept integral = is_integral_v<T>;
template <class T>
concept signed_integral = integral<T> && is_signed_v<T>;

template <class T>
concept unsigned_integral = integral<T> && !signed_integral<T>;
template <class T>
concept floating_point = is_floating_point_v<T>;

template <class Derived, class Base>
concept derived_from = is_base_of_v<Base, Derived> && is_convertible_v<const volatile Derived*,
                                                                                 const volatile Base*>;

#if !MQ_HAS_CONCEPT_DESTRUCTIBLE
template <class T>
concept destructible = is_nothrow_destructible_v<T>;
#endif /* !MQ_HAS_CONCEPT_DESTRUCTIBLE */

template <class T, class... Args>
concept constructible_from = destructible<T>&& is_constructible_v<T, Args...>;

template <class T>
concept default_initializable = constructible_from<T>&& requires {
    T{};
}
&&requires {
    ::new (static_cast<void*>(nullptr)) T;
};

template <class From, class To>
concept convertible_to = is_convertible_v<From, To>&& requires(add_rvalue_reference_t<From> (&f)()) {
    static_cast<To>(f());
}

template <class T>
concept move_constructible = constructible_from<T, T>&& convertible_to<T, T>;

template <class T>
concept copy_constructible = move_constructible<T>&& constructible_from<T, T&>&& convertible_to<T&, T>&&
    constructible_from<T, const T&>&& convertible_to<const T&, T>&& constructible_from<T, const T>&&
        convertible_to<const T, T>;

// clang-format on
#endif  // MQ_HAS_CONCEPTS && !MQ_HAS_CONCEPT_LIBRARY

#if !MQ_HAS_REMOVE_CVREF_T
template <class T>
struct remove_cvref {
    using type = remove_cv_t<remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif  // !MQ_HAS_REMOVE_CVREF_T

#if !MQ_HAS_STD_LAUNDER
template <typename T>
constexpr T* launder(T* t) noexcept {
    return t;
}
#endif  // !MQ_HAS_STD_LAUNDER

}  // namespace std

#endif /* CORE_CXX20_COMPATIBILITY_HPP */
