//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_
#define MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_
#include <iostream>
#include <type_traits>

#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/ops_cpu/utils.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
template <typename src_t, typename des_t>
struct cast_value {
    des_t operator()(const src_t& a) const {
        if constexpr (std::is_same_v<src_t, des_t>) {
            return a;
        } else if constexpr (!is_complex_v<des_t> && is_complex_v<src_t>) {
            return std::real(a);
        } else if constexpr (is_complex_v<des_t> && is_complex_v<src_t>) {
            return {std::real(a), std::imag(a)};
        } else {
            return a;
        }
    }
};

template <TDtype src, typename T, template <typename ops_t = void> class binary_ops>
void inplace_binary(void* data, size_t len, T a) {
    using calc_t = to_device_t<src>;
    auto c_data = reinterpret_cast<calc_t*>(data);
    auto ops = binary_ops<>();
    auto caster = cast_value<T, calc_t>();
    for (size_t i = 0; i < len; i++) {
        c_data[i] = ops(c_data[i], caster(a));
    }
}

template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary(void* data, TDtype src, size_t len, T a) {
    switch (src) {
        case TDtype::Float32: {
            cpu::inplace_binary<TDtype::Float32, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Float64: {
            cpu::inplace_binary<TDtype::Float64, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Complex64: {
            cpu::inplace_binary<TDtype::Complex64, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Complex128: {
            cpu::inplace_binary<TDtype::Complex128, T, binary_ops>(data, len, a);
            break;
        }
    }
}

template <TDtype src, typename T, template <typename ops_t = void> class binary_ops>
void inplace_binary_rev(void* data, size_t len, T a) {
    using calc_t = to_device_t<src>;
    auto c_data = reinterpret_cast<calc_t*>(data);
    auto ops = binary_ops<>();
    auto caster = cast_value<T, calc_t>();
    for (size_t i = 0; i < len; i++) {
        c_data[i] = ops(caster(a), c_data[i]);
    }
}

template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary_rev(void* data, TDtype src, size_t len, T a) {
    switch (src) {
        case TDtype::Float32: {
            cpu::inplace_binary_rev<TDtype::Float32, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Float64: {
            cpu::inplace_binary_rev<TDtype::Float64, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Complex64: {
            cpu::inplace_binary_rev<TDtype::Complex64, T, binary_ops>(data, len, a);
            break;
        }
        case TDtype::Complex128: {
            cpu::inplace_binary_rev<TDtype::Complex128, T, binary_ops>(data, len, a);
            break;
        }
    }
}

// -----------------------------------------------------------------------------

template <TDtype des_t, typename T, template <typename ops_t = void> class binary_ops>
Tensor tensor_binary(void* data, size_t len, T a) {
    constexpr TDtype src_t = to_dtype_v<T>;
    constexpr TDtype upper_t = upper_type<des_t, src_t>::get();
    Tensor out = init<upper_t>(len);
    auto c_des = reinterpret_cast<to_device_t<upper_t>*>(out.data);
    auto c_src = reinterpret_cast<to_device_t<des_t>*>(data);
    auto ops = binary_ops<>();
    auto caster = cast_value<T, to_device_t<des_t>>();
    for (size_t i = 0; i < len; i++) {
        c_des[i] = ops(c_src[i], caster(a));
    }
    return out;
}

template <TDtype des_t, typename T, template <typename ops_t = void> class binary_ops>
Tensor tensor_binary_rev(void* data, size_t len, T a) {
    constexpr TDtype src_t = to_dtype_v<T>;
    constexpr TDtype upper_t = upper_type<des_t, src_t>::get();
    Tensor out = init<upper_t>(len);
    auto c_des = reinterpret_cast<to_device_t<upper_t>*>(out.data);
    auto c_src = reinterpret_cast<to_device_t<des_t>*>(data);
    auto ops = binary_ops<>();
    auto caster = cast_value<T, to_device_t<des_t>>();
    for (size_t i = 0; i < len; i++) {
        c_des[i] = ops(caster(a), c_src[i]);
    }
    return out;
}

template <typename T, template <typename ops_t = void> class binary_ops>
Tensor tensor_binary_rev(void* data, TDtype dtype, size_t len, T a) {
    switch (dtype) {
        case (TDtype::Float32): {
            return tensor_binary_rev<TDtype::Float32, T, binary_ops>(data, len, a);
        }
        case (TDtype::Float64): {
            return tensor_binary_rev<TDtype::Float64, T, binary_ops>(data, len, a);
        }
        case (TDtype::Complex128): {
            return tensor_binary_rev<TDtype::Complex128, T, binary_ops>(data, len, a);
        }
        case (TDtype::Complex64): {
            return tensor_binary_rev<TDtype::Complex64, T, binary_ops>(data, len, a);
        }
    }
}

template <typename T, template <typename ops_t = void> class binary_ops>
Tensor tensor_binary(void* data, TDtype dtype, size_t len, T a) {
    switch (dtype) {
        case (TDtype::Float32): {
            return tensor_binary<TDtype::Float32, T, binary_ops>(data, len, a);
        }
        case (TDtype::Float64): {
            return tensor_binary<TDtype::Float64, T, binary_ops>(data, len, a);
        }
        case (TDtype::Complex128): {
            return tensor_binary<TDtype::Complex128, T, binary_ops>(data, len, a);
        }
        case (TDtype::Complex64): {
            return tensor_binary<TDtype::Complex64, T, binary_ops>(data, len, a);
        }
    }
}
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_ */
