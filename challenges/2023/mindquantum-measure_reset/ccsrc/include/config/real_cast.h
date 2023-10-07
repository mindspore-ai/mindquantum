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

#ifndef MQ_CONFIG_REAL_CAST_HPP
#define MQ_CONFIG_REAL_CAST_HPP

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "config/config.h"
#include "config/type_traits.h"

namespace mindquantum {
template <typename coeff_t>
struct ParameterResolver;

enum class RealCastType { REAL, IMAG };
enum class VType { Float, Double, Complex64, Complex128, Invalid };

template <VType v>
std::string_view VTypeToString() {
    switch (v) {
        case VType::Float:
            return "float";
        case VType::Double:
            return "double";
        case VType::Complex64:
            return "complex64";
        case VType::Complex128:
            return "complex128";
        case VType::Invalid:
        default:
            throw std::runtime_error("Unknown vtype.");
    }
}

template <typename T>
struct VTypeMap {
    constexpr static VType v = VType::Invalid;
};

template <>
struct VTypeMap<float> {
    constexpr static VType v = VType::Float;
};

template <>
struct VTypeMap<double> {
    constexpr static VType v = VType::Double;
};

template <>
struct VTypeMap<std::complex<float>> {
    constexpr static VType v = VType::Complex64;
};

template <>
struct VTypeMap<std::complex<double>> {
    constexpr static VType v = VType::Complex128;
};

template <>
struct VTypeMap<ParameterResolver<float>> {
    constexpr static VType v = VType::Float;
};

template <>
struct VTypeMap<ParameterResolver<double>> {
    constexpr static VType v = VType::Double;
};

template <>
struct VTypeMap<ParameterResolver<std::complex<float>>> {
    constexpr static VType v = VType::Complex64;
};

template <>
struct VTypeMap<ParameterResolver<std::complex<double>>> {
    constexpr static VType v = VType::Complex128;
};

namespace details {
template <RealCastType type, typename complex_t, typename = void>
struct real_cast_impl;

template <RealCastType cast_type, typename float_t>
struct real_cast_impl<cast_type, float_t, std::enable_if_t<std::is_floating_point_v<float_t>>> {
    static constexpr auto apply(const float_t& number) {
        return number;
    }
};

template <RealCastType cast_type, typename float_t>
struct real_cast_impl<cast_type, std::complex<float_t>> {
    using type = std::complex<float_t>;
    static constexpr auto apply(const type& number) {
        if constexpr (cast_type == RealCastType::REAL) {
            return number.real();
        }
        return number.imag();
    }
};
}  // namespace details

template <RealCastType type, typename complex_t>
constexpr auto real_cast(complex_t&& number) {
    return details::real_cast_impl<type, std::remove_cvref_t<complex_t>>::apply(std::forward<complex_t>(number));
}
}  // namespace mindquantum

#endif /* MQ_CONFIG_REAL_CAST_HPP */
