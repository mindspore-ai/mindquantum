/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef MATH_TENSOR_OPS_CPU_UTILS_HPP_
#define MATH_TENSOR_OPS_CPU_UTILS_HPP_

#include <complex>
#include <type_traits>

#include "math/tensor/traits.h"

namespace tensor {
template <typename T>
struct is_complex {
    static constexpr bool v = false;
};

template <typename T>
struct is_complex<std::complex<T>> {
    static constexpr bool v = true;
};

template <typename T>
static constexpr bool is_complex_v = is_complex<T>::v;

// -----------------------------------------------------------------------------

template <typename src_t, typename des_t>
struct cast_value {
    using real_des_t = to_device_t<to_real_dtype_t<to_dtype_v<des_t>>>;
    des_t operator()(const src_t& a) const {
        if constexpr (std::is_same_v<src_t, des_t>) {
            return a;
        } else if constexpr (!is_complex_v<des_t> && is_complex_v<src_t>) {
            return std::real(a);
        } else if constexpr (is_complex_v<des_t> && is_complex_v<src_t>) {
            return {static_cast<real_des_t>(std::real(a)), static_cast<real_des_t>(std::imag(a))};
        } else {
            return a;
        }
    }
};
}  // namespace tensor

#endif /* MATH_TENSOR_OPS_CPU_UTILS_HPP_ */
