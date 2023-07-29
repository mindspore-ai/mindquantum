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

#ifndef MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_
#define MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "math/tensor/ops/advance_math.h"
#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/ops_cpu/basic_math.h"
#include "math/tensor/ops_cpu/concrete_tensor.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/ops_cpu/utils.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
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
        if (data == nullptr) {
            throw std::runtime_error("data cannot be nullptr.");
        }
        return cpu::zeros(len, dtype);
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

// -----------------------------------------------------------------------------

template <TDtype dtype>
bool is_all_zero(void* data, size_t len) {
    auto c_data = reinterpret_cast<to_device_t<dtype>*>(data);
    for (size_t i = 0; i < len; i++) {
        if constexpr (is_complex_v<to_device_t<dtype>>) {
            if ((std::real(c_data[i]) > 0) || (std::real(c_data[i]) < 0) || (std::imag(c_data[i]) > 0)
                || (std::imag(c_data[i]) < 0)) {
                return false;
            }
        } else {
            if ((c_data[i] > 0) || (c_data[i] < 0)) {
                return false;
            }
        }
    }
    return true;
}
bool is_all_zero(const Tensor& t);

// -----------------------------------------------------------------------------
template <typename T1, typename T2>
bool operator==(T1 a, const std::complex<T2>& b) {
    return (!(a > b.real())) && (!(a < b.real()));
}

template <typename T1, typename T2>
bool operator==(const std::complex<T1>& a, T2 b) {
    return (!(a.real() > b)) && (!(a.real() < b));
}

template <typename T1, typename T2>
bool operator==(const std::complex<T1>& a, const std::complex<T2> b) {
    return !(a.real() > b.real()) && !(a.real() < b.real()) && !(a.imag() > b.imag()) && !(a.imag() < b.imag());
}

template <TDtype lhs_dtype, TDtype rhs_dtype>
std::vector<bool> is_equal_to(void* lhs, size_t lhs_len, void* rhs, size_t rhs_len) {
    using lhs_t = to_device_t<lhs_dtype>;
    using rhs_t = to_device_t<rhs_dtype>;
    if (lhs_len != rhs_len) {
        throw std::runtime_error("Dimension mismatch for compare tow tensors.");
    }
    auto c_lhs = reinterpret_cast<lhs_t*>(lhs);
    auto c_rhs = reinterpret_cast<rhs_t*>(rhs);
    std::vector<bool> out;
    if constexpr (!is_complex_dtype_v<lhs_dtype> && !is_complex_dtype_v<rhs_dtype>) {
        if (lhs_len == 1) {
            for (size_t i = 0; i < rhs_len; i++) {
                out.push_back(!(c_lhs[0] > c_rhs[i]) && !(c_lhs[0] < c_rhs[i]));
            }
        } else if (rhs_len == 1) {
            for (size_t i = 0; i < lhs_len; i++) {
                out.push_back(!(c_lhs[i] > c_rhs[0]) && !(c_lhs[i] < c_rhs[0]));
            }
        } else {
            for (size_t i = 0; i < lhs_len; i++) {
                out.push_back(!(c_lhs[i] > c_rhs[i]) && !(c_lhs[i] < c_rhs[i]));
            }
        }
    } else {
        if (lhs_len == 1) {
            for (size_t i = 0; i < rhs_len; i++) {
                out.push_back(c_lhs[0] == c_rhs[i]);
            }
        } else if (rhs_len == 1) {
            for (size_t i = 0; i < lhs_len; i++) {
                out.push_back(c_lhs[i] == c_rhs[0]);
            }
        } else {
            for (size_t i = 0; i < lhs_len; i++) {
                out.push_back(c_lhs[i] == c_rhs[i]);
            }
        }
    }

    return out;
}

std::vector<bool> is_equal_to(const Tensor& lhs, const Tensor& rhs);

template <TDtype out_dtype, TDtype src_dtype, typename F>
Tensor ElementFunc(void* data, size_t len, F&& func) {
    auto c_data = reinterpret_cast<to_device_t<src_dtype>*>(data);
    auto out = init(len, out_dtype);
    auto out_data = reinterpret_cast<to_device_t<out_dtype>*>(out.data);
    for (size_t i = 0; i < len; i++) {
        if constexpr (is_complex_dtype_v<src_dtype> && !is_complex_dtype_v<out_dtype>) {
            out_data[i] = std::real(func(c_data[i]));
        } else {
            out_data[i] = func(c_data[i]);
        }
    }
    return out;
}

template <typename F>
Tensor ElementFunc(const Tensor& t, TDtype out_dtype, F&& func) {
    auto& data = t.data;
    auto len = t.dim;
    auto src_dtype = t.dtype;

    switch (out_dtype) {
        case TDtype::Float32: {
            switch (src_dtype) {
                case TDtype::Float32:
                    return ElementFunc<TDtype::Float32, TDtype::Float32>(data, len, func);
                case TDtype::Float64:
                    return ElementFunc<TDtype::Float32, TDtype::Float64>(data, len, func);
                case TDtype::Complex64:
                    return ElementFunc<TDtype::Float32, TDtype::Complex64>(data, len, func);
                case TDtype::Complex128:
                    return ElementFunc<TDtype::Float32, TDtype::Complex128>(data, len, func);
            }
        } break;
        case TDtype::Float64: {
            switch (src_dtype) {
                case TDtype::Float32:
                    return ElementFunc<TDtype::Float64, TDtype::Float32>(data, len, func);
                case TDtype::Float64:
                    return ElementFunc<TDtype::Float64, TDtype::Float64>(data, len, func);
                case TDtype::Complex64:
                    return ElementFunc<TDtype::Float64, TDtype::Complex64>(data, len, func);
                case TDtype::Complex128:
                    return ElementFunc<TDtype::Float64, TDtype::Complex128>(data, len, func);
            }
        } break;
        case TDtype::Complex64: {
            switch (src_dtype) {
                case TDtype::Float32:
                    return ElementFunc<TDtype::Complex64, TDtype::Float32>(data, len, func);
                case TDtype::Float64:
                    return ElementFunc<TDtype::Complex64, TDtype::Float64>(data, len, func);
                case TDtype::Complex64:
                    return ElementFunc<TDtype::Complex64, TDtype::Complex64>(data, len, func);
                case TDtype::Complex128:
                    return ElementFunc<TDtype::Complex64, TDtype::Complex128>(data, len, func);
            }
        } break;
        case TDtype::Complex128: {
            switch (src_dtype) {
                case TDtype::Float32:
                    return ElementFunc<TDtype::Complex128, TDtype::Float32>(data, len, func);
                case TDtype::Float64:
                    return ElementFunc<TDtype::Complex128, TDtype::Float64>(data, len, func);
                case TDtype::Complex64:
                    return ElementFunc<TDtype::Complex128, TDtype::Complex64>(data, len, func);
                case TDtype::Complex128:
                    return ElementFunc<TDtype::Complex128, TDtype::Complex128>(data, len, func);
            }
        } break;
    }
    return Tensor();
}

// -----------------------------------------------------------------------------

Tensor Gather(const std::vector<Tensor>& tensors);
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_ADVANCE_MATH_HPP_ */
