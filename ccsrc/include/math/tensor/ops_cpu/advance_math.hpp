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

#ifndef MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_
#define MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_
#include <algorithm>

#include "math/tensor/ops_cpu/basic_math.hpp"
#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/ops_cpu/utils.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"
namespace tensor::ops::cpu {

template <TDtype dtype>
Tensor real(void* data, size_t len) {
    constexpr TDtype real_t = to_real_dtype_t<dtype>;
    if constexpr (dtype == real_t) {
        return cpu::copy<dtype>(data, len);
    } else {
        auto out = ops::cpu::init<real_t>(len);
        auto c_data = reinterpret_cast<to_device_t<dtype>*>(data);
        auto c_out = reinterpret_cast<to_device_t<real_t>*>(out.data);
        for (size_t i = 0; i < len; i++) {
            c_out[i] = std::real(c_data[i]);
        }
        return out;
    }
}

template <TDtype dtype>
Tensor imag(void* data, size_t len) {
    constexpr TDtype real_t = to_real_dtype_t<dtype>;
    if constexpr (dtype == real_t) {
        return cpu::copy<dtype>(data, len);
    } else {
        auto out = ops::cpu::init<real_t>(len);
        auto c_data = reinterpret_cast<to_device_t<dtype>*>(data);
        auto c_out = reinterpret_cast<to_device_t<real_t>*>(out.data);
        for (size_t i = 0; i < len; i++) {
            c_out[i] = std::imag(c_data[i]);
        }
        return out;
    }
}

Tensor real(const Tensor& t);
Tensor imag(const Tensor& t);

// -----------------------------------------------------------------------------

template <TDtype dtype>
Tensor conj(void* data, size_t len) {
    constexpr TDtype real_t = to_real_dtype_t<dtype>;
    if constexpr (dtype == real_t) {
        return cpu::copy<dtype>(data, len);
    } else {
        auto out = ops::cpu::init<dtype>(len);
        auto c_data = reinterpret_cast<to_device_t<dtype>*>(data);
        auto c_out = reinterpret_cast<to_device_t<dtype>*>(out.data);
        for (size_t i = 0; i < len; i++) {
            c_out[i] = std::conj(c_data[i]);
        }
        return out;
    }
}

Tensor conj(const Tensor& t);

// -----------------------------------------------------------------------------

template <TDtype bra_dtype, TDtype ket_dtype>
Tensor vdot(void* bra, size_t len, void* ket) {
    using bra_t = to_device_t<bra_dtype>;
    using ket_t = to_device_t<ket_dtype>;
    constexpr TDtype upper_dtype = upper_type<bra_dtype, ket_dtype>::get();
    using upper_t = to_device_t<upper_dtype>;

    auto c_bra = reinterpret_cast<bra_t*>(bra);
    auto c_ket = reinterpret_cast<ket_t*>(ket);
    auto caster_bra = cast_value<bra_t, upper_t>();
    auto caster_ket = cast_value<ket_t, upper_t>();
    upper_t value = 0;
    for (size_t i = 0; i < len; i++) {
        if constexpr (is_complex_v<upper_t>) {
            value += std::conj(caster_bra(c_bra[i])) * caster_ket(c_ket[i]);
        } else {
            value += caster_bra(c_bra[i]) * caster_ket(c_ket[i]);
        }
    }
    return ops::cpu::init_with_value<upper_t>(value);
}

Tensor vdot(const Tensor& bra, const Tensor& ket);
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_ */
