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

#ifndef MQ_CONFIG_COMPLEX_CAST_HPP
#define MQ_CONFIG_COMPLEX_CAST_HPP

#include <complex>
#include <type_traits>
#include <utility>

#include "config/type_traits.h"

namespace mindquantum {
template <typename coeff_t>
class ParameterResolver;

namespace details {
template <typename float_t, typename = void>
struct complex_cast_impl;

template <typename float_t>
struct complex_cast_impl<std::complex<float_t>> {
    using type = std::complex<float_t>;
    static constexpr auto apply(const type& number) {
        return number;
    }
};

template <typename float_t>
struct complex_cast_impl<float_t, std::enable_if_t<std::is_floating_point_v<float_t>>> {
    static constexpr auto apply(const float_t& number) {
        return std::complex<float_t>{number};
    }
};
}  // namespace details

template <typename real_t>
constexpr auto complex_cast(real_t&& number) {
    return details::complex_cast_impl<std::remove_cvref_t<real_t>>::apply(std::forward<real_t>(number));
}
}  // namespace mindquantum

#endif /* MQ_CONFIG_COMPLEX_CAST_HPP */
