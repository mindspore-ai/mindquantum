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

#ifndef MQ_CONFIG_REAL_CAST_HPP
#define MQ_CONFIG_REAL_CAST_HPP

#include <complex>
#include <type_traits>
#include <utility>

#include "config/config.hpp"
#include "config/type_traits.hpp"

namespace mindquantum {
template <typename coeff_t>
class ParameterResolver;

enum class RealCastType { REAL, IMAG };

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
