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

#ifndef MATH_TENSOR_OPS_CPU_HPP_
#define MATH_TENSOR_OPS_CPU_HPP_
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "core/utils.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/ops_cpu/utils.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops::cpu {
template <TDtype dtype>
Tensor init(size_t len) {
    using calc_t = to_device_t<dtype>;
    void* data = nullptr;
    if (len != 0) {
        data = reinterpret_cast<void*>(malloc(sizeof(calc_t) * len));
    }
    if (data == nullptr) {
        throw std::runtime_error("malloc memory error.");
    }
    return Tensor{dtype, TDevice::CPU, data, len};
}

Tensor init(size_t len, TDtype dtype);

void destroy(Tensor* t);
// -----------------------------------------------------------------------------

template <TDtype src, TDtype des>
Tensor cast_to(const void* data, size_t len) {
    using d_src = to_device_t<src>;
    using d_des = to_device_t<des>;
    auto c_data = reinterpret_cast<const d_src*>(data);
    auto out = cpu::init<des>(len);
    auto c_out = reinterpret_cast<d_des*>(out.data);
    auto caster = cast_value<to_device_t<src>, to_device_t<des>>();
    for (size_t i = 0; i < len; i++) {
        c_out[i] = caster(c_data[i]);
    }
    return out;
}

Tensor cast_to(const Tensor& t, TDtype des);

// -----------------------------------------------------------------------------

template <TDtype dtype>
std::string to_string(const void* data, size_t dim, bool simplify = false) {
    std::string out = "";
    if (!simplify) {
        out = "array(dtype: " + dtype_to_string(dtype) + ", device: " + device_to_string(TDevice::CPU) + ", data: [";
    }
    using calc_t = to_device_t<dtype>;
    const calc_t* data_ = reinterpret_cast<const calc_t*>(data);
    for (size_t i = 0; i < dim; i++) {
        if constexpr (is_complex_v<calc_t>) {
            out += "(" + std::to_string(data_[i].real()) + ", " + std::to_string(data_[i].imag()) + ")";
        } else {
            out += std::to_string(data_[i]);
        }
        if (i != dim - 1) {
            out += ", ";
        }
    }
    if (!simplify) {
        out += "])";
    }
    return out;
}

std::string to_string(const Tensor& t, bool simplify = false);

// -----------------------------------------------------------------------------

template <typename T, typename = std::enable_if_t<is_arithmetic_v<T>>>
Tensor init_with_value(T a) {
    constexpr auto dtype = to_dtype_v<T>;
    auto out = cpu::init<dtype>(1);
    auto c_data = reinterpret_cast<T*>(out.data);
    c_data[0] = a;
    return out;
}

template <typename T>
Tensor init_with_vector(const std::vector<T>& a) {
    constexpr auto dtype = to_dtype_v<T>;
    auto out = cpu::init<dtype>(a.size());
    mindquantum::safe_copy(out.data, sizeof(T) * a.size(), a.data(), sizeof(T) * a.size());
    return out;
}

// -----------------------------------------------------------------------------

template <TDtype dtype>
Tensor copy(const void* data, size_t len) {
    using calc_t = to_device_t<dtype>;
    auto out = init<dtype>(len);
    mindquantum::safe_copy(out.data, sizeof(calc_t) * len, data, sizeof(calc_t) * len);
    return out;
}

Tensor copy(const Tensor& t);

template <TDtype dtype>
void* copy_mem(const void* data, size_t len) {
    using calc_t = to_device_t<dtype>;
    auto res = reinterpret_cast<void*>(malloc(sizeof(calc_t) * len));
    if (res == nullptr) {
        throw std::runtime_error("malloc memory error.");
    }
    mindquantum::safe_copy(res, sizeof(calc_t) * len, data, sizeof(calc_t) * len);
    return res;
}
void* copy_mem(const void* data, TDtype dtype, size_t len);

// -----------------------------------------------------------------------------
template <typename src, typename T>
void set(void* data, size_t len, T a, size_t idx) {
    if (idx >= len) {
        throw std::runtime_error("index " + std::to_string(idx) + " out of range: " + std::to_string(len));
    }
    auto c_data = reinterpret_cast<src*>(data);
    if constexpr (is_complex_v<T> && !is_complex_v<src>) {
        c_data[idx] = std::real(a);
    } else {
        c_data[idx] = a;
    }
}

template <typename T>
void set(void* data, TDtype dtype, T a, size_t dim, size_t idx) {
    switch (dtype) {
        case TDtype::Float32:
            set<to_device_t<TDtype::Float32>, T>(data, dim, a, idx);
            break;
        case TDtype::Float64:
            set<to_device_t<TDtype::Float64>, T>(data, dim, a, idx);
            break;
        case TDtype::Complex128:
            set<to_device_t<TDtype::Complex128>, T>(data, dim, a, idx);
            break;
        case TDtype::Complex64:
            set<to_device_t<TDtype::Complex64>, T>(data, dim, a, idx);
            break;
    }
}

template <typename des_t, typename src_t>
void set(void* des, void* src, size_t len, size_t idx) {
    auto c_src = reinterpret_cast<src_t*>(src);
    cpu::set<des_t, src_t>(des, len, c_src[0], idx);
}

void set(Tensor* t, const Tensor& source, size_t idx);

// -----------------------------------------------------------------------------
Tensor get(const Tensor& t, size_t idx);

// -----------------------------------------------------------------------------

template <TDtype src_dtype>
std::vector<to_device_t<src_dtype>> to_vector(const void* data, size_t len) {
    auto c_data = reinterpret_cast<const to_device_t<src_dtype>*>(data);
    std::vector<to_device_t<src_dtype>> out;
    for (size_t i = 0; i < len; i++) {
        out.push_back(c_data[i]);
    }
    return out;
}

template <typename T>
std::vector<T> to_vector(const Tensor& ori) {
    auto t = ori;
    if (t.dtype != to_dtype_v<T>) {
        t = t.astype(to_dtype_v<T>);
    }
    return to_vector<to_dtype_v<T>>(t.data, t.dim);
}

template <TDtype src_dtype>
std::vector<std::vector<to_device_t<src_dtype>>> to_vector(const void* data, size_t n_row, size_t n_col) {
    auto c_data = reinterpret_cast<const to_device_t<src_dtype>*>(data);
    std::vector<std::vector<to_device_t<src_dtype>>> out;
    for (size_t i = 0; i < n_row; i++) {
        std::vector<to_device_t<src_dtype>> tmp;
        for (size_t j = 0; j < n_col; j++) {
            tmp.push_back(c_data[i * n_col + j]);
        }
        out.push_back(tmp);
    }
    return out;
}

template <typename T>
std::vector<std::vector<T>> to_vector(const Matrix& t) {
    Matrix m = t;
    if (t.dtype != to_dtype_v<T>) {
        m = Matrix(t.astype(to_dtype_v<T>), t.n_row, t.n_col);
    }
    return to_vector<to_dtype_v<T>>(m.data, m.n_row, m.n_col);
}
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_HPP_ */
