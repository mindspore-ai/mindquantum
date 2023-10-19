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

#ifndef MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_
#define MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "config/openmp.h"
#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/tensor/csr_matrix.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops_cpu/concrete_tensor.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/ops_cpu/utils.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops::cpu {
template <TDtype lhs_dtype, TDtype other_dtype, bool is_array = false, bool reverse = false,
          template <typename ops_t = void> class binary_ops>
void InplaceBinary(void* data, size_t len, void* other) {
    using calc_t = to_device_t<lhs_dtype>;
    using other_t = to_device_t<other_dtype>;
    auto c_data = reinterpret_cast<calc_t*>(data);
    auto c_other = reinterpret_cast<other_t*>(other);
    auto ops = binary_ops<>();
    auto caster = cast_value<other_t, calc_t>();
    for (size_t i = 0; i < len; i++) {
        if constexpr (is_array) {
            if constexpr (reverse) {
                c_data[i] = ops(caster(c_other[i]), c_data[i]);
            } else {
                c_data[i] = ops(c_data[i], caster(c_other[i]));
            }
        } else {
            if constexpr (reverse) {
                c_data[i] = ops(caster(c_other[0]), c_data[i]);
            } else {
                c_data[i] = ops(c_data[i], caster(c_other[0]));
            }
        }
    }
}

template <TDtype lhs_dtype, TDtype other_dtype, bool is_array = false, bool reverse = false,
          template <typename ops_t = void> class binary_ops>
Tensor GenerateBinary(void* data, size_t len, void* other) {
    constexpr TDtype upper_t = upper_type<lhs_dtype, other_dtype>::get();
    Tensor out = init<upper_t>(len);
    auto c_des = reinterpret_cast<to_device_t<upper_t>*>(out.data);
    auto c_data = reinterpret_cast<to_device_t<lhs_dtype>*>(data);
    auto c_other = reinterpret_cast<to_device_t<other_dtype>*>(other);
    auto ops = binary_ops<>();
    auto caster0 = cast_value<to_device_t<lhs_dtype>, to_device_t<upper_t>>();
    auto caster1 = cast_value<to_device_t<other_dtype>, to_device_t<upper_t>>();
    for (size_t i = 0; i < len; i++) {
        if constexpr (is_array) {
            if constexpr (reverse) {
                c_des[i] = ops(caster1(c_other[i]), caster0(c_data[i]));
            } else {
                c_des[i] = ops(caster0(c_data[i]), caster1(c_other[i]));
            }
        } else {
            if constexpr (reverse) {
                c_des[i] = ops(caster1(c_other[0]), caster0(c_data[i]));
            } else {
                c_des[i] = ops(caster0(c_data[i]), caster1(c_other[0]));
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// vector -> vector + number
template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary(void* data, TDtype src, size_t len, T a) {
    auto other = reinterpret_cast<void*>(&a);
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (src) {
        case TDtype::Float32: {
            cpu::InplaceBinary<TDtype::Float32, other_dtype, false, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Float64: {
            cpu::InplaceBinary<TDtype::Float64, other_dtype, false, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex64: {
            cpu::InplaceBinary<TDtype::Complex64, other_dtype, false, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex128: {
            cpu::InplaceBinary<TDtype::Complex128, other_dtype, false, false, binary_ops>(data, len, other);
            break;
        }
    }
}

// vector -> number + vector
template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary_rev(void* data, TDtype src, size_t len, T a) {
    auto other = reinterpret_cast<void*>(&a);
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (src) {
        case TDtype::Float32: {
            cpu::InplaceBinary<TDtype::Float32, other_dtype, false, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Float64: {
            cpu::InplaceBinary<TDtype::Float64, other_dtype, false, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex64: {
            cpu::InplaceBinary<TDtype::Complex64, other_dtype, false, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex128: {
            cpu::InplaceBinary<TDtype::Complex128, other_dtype, false, true, binary_ops>(data, len, other);
            break;
        }
    }
}

// -----------------------------------------------------------------------------

// vector = vector + number
template <typename T, template <typename ops_t = void> class binary_ops>
Tensor generate_binary(void* data, TDtype dtype, size_t len, T a) {
    auto other = reinterpret_cast<void*>(&a);
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (dtype) {
        case (TDtype::Float32): {
            return GenerateBinary<TDtype::Float32, other_dtype, false, false, binary_ops>(data, len, other);
        }
        case (TDtype::Float64): {
            return GenerateBinary<TDtype::Float64, other_dtype, false, false, binary_ops>(data, len, other);
        }
        case (TDtype::Complex128): {
            return GenerateBinary<TDtype::Complex128, other_dtype, false, false, binary_ops>(data, len, other);
        }
        case (TDtype::Complex64): {
            return GenerateBinary<TDtype::Complex64, other_dtype, false, false, binary_ops>(data, len, other);
        }
    }
    return Tensor();
}

// vector = number + vector
template <typename T, template <typename ops_t = void> class binary_ops>
Tensor generate_binary_rev(void* data, TDtype dtype, size_t len, T a) {
    auto other = reinterpret_cast<void*>(&a);
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (dtype) {
        case (TDtype::Float32): {
            return GenerateBinary<TDtype::Float32, other_dtype, false, true, binary_ops>(data, len, other);
        }
        case (TDtype::Float64): {
            return GenerateBinary<TDtype::Float64, other_dtype, false, true, binary_ops>(data, len, other);
        }
        case (TDtype::Complex128): {
            return GenerateBinary<TDtype::Complex128, other_dtype, false, true, binary_ops>(data, len, other);
        }
        case (TDtype::Complex64): {
            return GenerateBinary<TDtype::Complex64, other_dtype, false, true, binary_ops>(data, len, other);
        }
    }
    return Tensor();
}

// -----------------------------------------------------------------------------

// vector -> vector1 + vector2
template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary_array(void* data, TDtype src, size_t len, void* other) {
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (src) {
        case TDtype::Float32: {
            cpu::InplaceBinary<TDtype::Float32, other_dtype, true, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Float64: {
            cpu::InplaceBinary<TDtype::Float64, other_dtype, true, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex64: {
            cpu::InplaceBinary<TDtype::Complex64, other_dtype, true, false, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex128: {
            cpu::InplaceBinary<TDtype::Complex128, other_dtype, true, false, binary_ops>(data, len, other);
            break;
        }
    }
}

// vector -> vector2 + vector1
template <typename T, template <typename ops_t> class binary_ops>
void inplace_binary_array_rev(void* data, TDtype src, size_t len, void* other) {
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (src) {
        case TDtype::Float32: {
            cpu::InplaceBinary<TDtype::Float32, other_dtype, true, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Float64: {
            cpu::InplaceBinary<TDtype::Float64, other_dtype, true, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex64: {
            cpu::InplaceBinary<TDtype::Complex64, other_dtype, true, true, binary_ops>(data, len, other);
            break;
        }
        case TDtype::Complex128: {
            cpu::InplaceBinary<TDtype::Complex128, other_dtype, true, true, binary_ops>(data, len, other);
            break;
        }
    }
}

// -----------------------------------------------------------------------------

// vector = vector1 + vector2
template <typename T, template <typename ops_t = void> class binary_ops>
Tensor generate_binary_array(void* data, TDtype dtype, size_t len, void* other) {
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (dtype) {
        case (TDtype::Float32): {
            return GenerateBinary<TDtype::Float32, other_dtype, true, false, binary_ops>(data, len, other);
        }
        case (TDtype::Float64): {
            return GenerateBinary<TDtype::Float64, other_dtype, true, false, binary_ops>(data, len, other);
        }
        case (TDtype::Complex128): {
            return GenerateBinary<TDtype::Complex128, other_dtype, true, false, binary_ops>(data, len, other);
        }
        case (TDtype::Complex64): {
            return GenerateBinary<TDtype::Complex64, other_dtype, true, false, binary_ops>(data, len, other);
        }
    }
    return Tensor();
}

// vector = vector2 + vector1
template <typename T, template <typename ops_t = void> class binary_ops>
Tensor generate_binary_array_rev(void* data, TDtype dtype, size_t len, void* other) {
    constexpr TDtype other_dtype = to_dtype_v<T>;
    switch (dtype) {
        case (TDtype::Float32): {
            return GenerateBinary<TDtype::Float32, other_dtype, true, true, binary_ops>(data, len, other);
        }
        case (TDtype::Float64): {
            return GenerateBinary<TDtype::Float64, other_dtype, true, true, binary_ops>(data, len, other);
        }
        case (TDtype::Complex128): {
            return GenerateBinary<TDtype::Complex128, other_dtype, true, true, binary_ops>(data, len, other);
        }
        case (TDtype::Complex64): {
            return GenerateBinary<TDtype::Complex64, other_dtype, true, true, binary_ops>(data, len, other);
        }
    }
    return Tensor();
}

// -----------------------------------------------------------------------------

//
template <template <typename ops_t = void> class binary_ops>
void inplace_binary_array(void* data, TDtype src, size_t len, const Tensor& a) {
    if (a.device != TDevice::CPU) {
        throw std::runtime_error("Need a tensor in cpu.");
    }
    if (a.dim == 1) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
        }
    } else if (a.dim == len) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                inplace_binary_array<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                inplace_binary_array<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                inplace_binary_array<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                inplace_binary_array<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
        }
    } else {
        throw std::runtime_error("Dimension miss match.");
    }
}

template <template <typename ops_t = void> class binary_ops>
void inplace_binary_array_rev(void* data, TDtype src, size_t len, const Tensor& a) {
    if (a.device != TDevice::CPU) {
        throw std::runtime_error("Need a tensor in cpu.");
    }
    if (a.dim == 1) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                inplace_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
                break;
            }
        }
    } else if (a.dim == len) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                inplace_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                inplace_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                inplace_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                inplace_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
                break;
            }
        }
    } else {
        throw std::runtime_error("Dimension miss match.");
    }
}

template <template <typename ops_t = void> class binary_ops>
Tensor generate_binary_array(void* data, TDtype src, size_t len, const Tensor& a) {
    if (a.device != TDevice::CPU) {
        throw std::runtime_error("Need a tensor in cpu.");
    }
    if (a.dim == 1) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
        }
    } else if (a.dim == len) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                return generate_binary_array<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                return generate_binary_array<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                return generate_binary_array<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                return generate_binary_array<calc_t, binary_ops>(data, src, len, a.data);
            }
        }
    } else if (len == 1) {
        switch (src) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                auto b = reinterpret_cast<calc_t*>(data)[0];
                return ops::cpu::generate_binary_rev<calc_t, binary_ops>(a.data, a.dtype, a.dim, b);
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                auto b = reinterpret_cast<calc_t*>(data)[0];
                return ops::cpu::generate_binary_rev<calc_t, binary_ops>(a.data, a.dtype, a.dim, b);
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                auto b = reinterpret_cast<calc_t*>(data)[0];
                return ops::cpu::generate_binary_rev<calc_t, binary_ops>(a.data, a.dtype, a.dim, b);
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                auto b = reinterpret_cast<calc_t*>(data)[0];
                return ops::cpu::generate_binary_rev<calc_t, binary_ops>(a.data, a.dtype, a.dim, b);
            }
        }
    } else {
        throw std::runtime_error("Dimension miss match.");
    }
    return Tensor();
}

template <template <typename ops_t = void> class binary_ops>
Tensor generate_binary_array_rev(void* data, TDtype src, size_t len, const Tensor& a) {
    if (a.device != TDevice::CPU) {
        throw std::runtime_error("Need a tensor in cpu.");
    }
    if (a.dim == 1) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                auto c_a = reinterpret_cast<calc_t*>(a.data);
                return generate_binary_rev<calc_t, binary_ops>(data, src, len, c_a[0]);
            }
        }
    } else if (a.dim == len) {
        switch (a.dtype) {
            case TDtype::Float32: {
                using calc_t = to_device_t<TDtype::Float32>;
                return generate_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Float64: {
                using calc_t = to_device_t<TDtype::Float64>;
                return generate_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Complex64: {
                using calc_t = to_device_t<TDtype::Complex64>;
                return generate_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
            }
            case TDtype::Complex128: {
                using calc_t = to_device_t<TDtype::Complex128>;
                return generate_binary_array_rev<calc_t, binary_ops>(data, src, len, a.data);
            }
        }
    } else {
        throw std::runtime_error("Dimension miss match.");
    }
    return Tensor();
}

// -----------------------------------------------------------------------------

template <TDtype m1_dtype, TDtype m2_dtype>
Matrix MatMul(void* m1, size_t m1_row, size_t m1_col, void* m2, size_t m2_row, size_t m2_col) {
    if (m1_col != m2_row) {
        throw std::runtime_error("Dimension mismatch of multiply two matrix.");
    }
    using m1_t = to_device_t<m1_dtype>;
    using m2_t = to_device_t<m2_dtype>;
    constexpr TDtype upper_dtype = upper_type<m1_dtype, m2_dtype>::get();
    using upper_t = to_device_t<upper_dtype>;
    auto c_m1 = reinterpret_cast<m1_t*>(m1);
    auto c_m2 = reinterpret_cast<m2_t*>(m2);
    auto out = zeros(m1_row * m2_col, upper_dtype);
    auto out_data = reinterpret_cast<upper_t*>(out.data);
    for (size_t i = 0; i < m1_row; i++) {
        for (size_t j = 0; j < m2_col; j++) {
            for (size_t k = 0; k < m1_col; k++) {
                if constexpr (m1_dtype == m2_dtype) {
                    out_data[i * m2_col + j] += c_m1[i * m1_col + k] * c_m2[k * m2_col + j];
                } else if constexpr (!is_complex_dtype_v<m1_dtype> && is_complex_dtype_v<m2_dtype>) {
                    out_data[i * m2_col + j] += upper_t{c_m1[i * m1_col + k] * std::real(c_m2[k * m2_col + j]),
                                                        c_m1[i * m1_col + k] * std::imag(c_m2[k * m2_col + j])};
                } else if constexpr (!is_complex_dtype_v<m2_dtype> && is_complex_dtype_v<m1_dtype>) {
                    out_data[i * m2_col + j] += upper_t{c_m2[k * m2_col + j] * std::real(c_m1[i * m1_col + k]),
                                                        c_m2[k * m2_col + j] * std::imag(c_m1[i * m1_col + k])};
                } else if constexpr (is_complex_dtype_v<m1_dtype> && is_complex_dtype_v<m2_dtype>) {
                    out_data[i * m2_col + j] += upper_t{
                        std::real(c_m1[i * m1_col + k]) * std::real(c_m2[k * m2_col + j])
                            - std::imag(c_m1[i * m1_col + k]) * std::imag(c_m2[k * m2_col + j]),
                        std::real(c_m1[i * m1_col + k]) * std::imag(c_m2[k * m2_col + j])
                            + std::imag(c_m1[i * m1_col + k]) * std::real(c_m2[k * m2_col + j]),
                    };
                } else {
                    out_data[i * m2_col + j] += c_m1[i * m1_col + k] * c_m2[k * m2_col + j];
                }
            }
        }
    }
    return Matrix(std::move(out), m1_row, m2_col);
}

Matrix MatMul(const Matrix& m1, const Matrix& m2);

// -----------------------------------------------------------------------------

template <TDtype m1_dtype, TDtype m2_dtype>
Tensor MatMul(void* m1, size_t* indptr, size_t* indices, size_t n_row, size_t n_col, void* m2, size_t len) {
    if (n_col != len) {
        throw std::runtime_error("Dimension mismatch: cannot multiply matrix and vector.");
    }
    using m1_t = to_device_t<m1_dtype>;
    using m2_t = to_device_t<m2_dtype>;
    constexpr TDtype upper_dtype = upper_type<m1_dtype, m2_dtype>::get();
    using upper_t = to_device_t<upper_dtype>;
    auto c_m1 = reinterpret_cast<m1_t*>(m1);
    auto c_m2 = reinterpret_cast<m2_t*>(m2);
    auto out = init(n_row, upper_dtype);
    auto c_out = reinterpret_cast<upper_t*>(out.data);

    THRESHOLD_OMP_FOR(
        len, static_cast<uint64_t>(1) << mindquantum::nQubitTh,
        for (omp::idx_t i = 0; i < static_cast<omp::idx_t>(n_row); i++) {
            upper_t sum = 0.0;
            for (omp::idx_t j = indptr[i]; j < static_cast<omp::idx_t>(indptr[i + 1]); j++) {
                if constexpr (m1_dtype == m2_dtype) {
                    sum += c_m1[j] * c_m2[indices[j]];
                } else if constexpr (is_complex_dtype_v<m1_dtype> && !is_complex_dtype_v<m2_dtype>) {
                    sum += upper_t{std::real(c_m1[j]) * c_m2[indices[j]], std::imag(c_m1[j]) * c_m2[indices[j]]};
                } else if constexpr (!is_complex_dtype_v<m1_dtype> && is_complex_dtype_v<m2_dtype>) {
                    sum += upper_t{c_m1[j] * std::real(c_m2[indices[j]]), c_m1[j] * std::imag(c_m2[indices[j]])};
                } else if constexpr (is_complex_dtype_v<m1_dtype> && is_complex_dtype_v<m2_dtype>) {
                    sum += upper_t{std::real(c_m1[j]) * std::real(c_m2[indices[j]])
                                       - std::imag(c_m1[j]) * std::imag(c_m2[indices[j]]),
                                   std::real(c_m1[j]) * std::imag(c_m2[indices[j]])
                                       + std::imag(c_m1[j]) * std::real(c_m2[indices[j]])};
                } else {
                    sum += c_m1[j] * c_m2[indices[j]];
                }
            }
            c_out[i] = sum;
        })
    return out;
}

Tensor MatMul(const CsrMatrix& m1, const Tensor& m2);
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_BASIC_MATH_HPP_ */
