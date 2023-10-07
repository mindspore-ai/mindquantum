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

#ifndef MQ_CONFIG_TYPE_PROMOTION_HPP
#define MQ_CONFIG_TYPE_PROMOTION_HPP

#include <complex>
#include <type_traits>
#include <utility>

#include "config/config.h"
#include "config/type_traits.h"

namespace mindquantum {
namespace traits {
template <typename T>
struct type_promotion;

// -------------------------------------

namespace details {
//! Helper type traits class
/*!
 * Makes for easier template specialization for classes that encapsulate a floating point type (e.g.
 * std::complex<float_t>).
 */
template <typename T, template <typename float_t> class type_t>
struct type_promotion_encapsulated_fp {
    using down_cast_t = type_t<typename type_promotion<T>::down_cast_t>;
    using up_cast_t = type_t<typename type_promotion<T>::up_cast_t>;
};
}  // namespace details

// -------------------------------------
// float -> double -> long double (if the latter is enabled)

#if MQ_HAS_LONG_DOUBLE
template <>
struct type_promotion<long double> {
    using down_cast_t = double;
    using up_cast_t = long double;
};
#endif  // MQ_HAS_LONG_DOUBLE

template <>
struct type_promotion<double> {
    using down_cast_t = float;
#if MQ_HAS_LONG_DOUBLE
    using up_cast_t = long double;
#else
    using up_cast_t = double;
#endif  // MQ_HAS_LONG_DOUBLE
};

template <>
struct type_promotion<float> {
    using down_cast_t = float;
    using up_cast_t = double;
};

// -------------------------------------
// std::complex follow the same logic as the floating point types

template <typename T>
struct type_promotion<std::complex<T>> : details::type_promotion_encapsulated_fp<T, std::complex> {};
}  // namespace traits

//! Cast from a floating point representation (or similar) to a larger type representation.
/*!
 * This function uses static_cast<> to perform the actual conversion.
 *
 * For example: \c float -> \c double
 *
 * \note The helper trait type_promotion<> is used to "calculate" the resulting type
 */
template <typename float_t>
auto up_cast(float_t&& value) {
    return static_cast<typename traits::type_promotion<std::remove_cvref_t<float_t>>::up_cast_t>(
        std::forward<float_t>(value));
}

template <typename float_t, std::enable_if_t<traits::is_complex_decay_v<float_t>>>
auto up_cast(float_t&& value) {
    using up_type = typename traits::type_promotion<std::remove_cvref_t<float_t>>::up_cast_t;
    return up_type(up_cast(std::real(value)), up_cast(std::imag(value)));
}

//! Cast from a floating point representation (or similar) to a smaller type representation.
/*!
 * This function uses static_cast<> to perform the actual conversion.
 *
 * For example: \c double -> \c float
 *
 * \note The helper trait type_promotion<> is used to "calculate" the resulting type
 */
template <typename float_t>
auto down_cast(float_t&& value) {
    return static_cast<typename traits::type_promotion<std::remove_cvref_t<float_t>>::down_cast_t>(
        std::forward<float_t>(value));
}

template <typename float_t, std::enable_if_t<traits::is_complex_decay_v<float_t>>>
auto down_cast(float_t&& value) {
    using down_type = typename traits::type_promotion<std::remove_cvref_t<float_t>>::up_cast_t;
    return down_type(down_cast(std::real(value)), up_cast(std::imag(value)));
}

template <typename src, typename des, typename = void>
struct ComplexCast {
    static auto apply(const src& value) {
        return des(std::real(value), std::imag(value));
    }
};

template <typename src, typename des>
struct ComplexCast<src, des, std::enable_if_t<std::is_same_v<typename traits::type_promotion<src>::up_cast_t, des>>> {
    static auto apply(const std::complex<src>& value) {
        return up_cast(value);
    }
};

template <typename src, typename des>
struct ComplexCast<src, des, std::enable_if_t<std::is_same_v<typename traits::type_promotion<src>::down_cast_t, des>>> {
    static auto apply(const std::complex<src>& value) {
        return down_cast(value);
    }
};
}  // namespace mindquantum

#endif /* MQ_CONFIG_TYPE_PROMOTION_HPP */
