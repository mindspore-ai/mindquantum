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

#ifndef MATH_TENSOR_TRAITS_HPP_
#define MATH_TENSOR_TRAITS_HPP_

#include <complex>
#include <stdexcept>
#include <string>
namespace tensor {
enum class TDevice : int {
    CPU,
    GPU,
};

enum class TDtype : int {
    Float32,
    Float64,
    Complex64,
    Complex128,
};

// -----------------------------------------------------------------------------

template <TDtype dtype>
struct to_device;

template <TDtype dtype>
using to_device_t = typename to_device<dtype>::type;

template <typename T>
struct to_dtype;

template <typename T>
static constexpr TDtype to_dtype_v = to_dtype<T>::dtype;

template <TDtype dtype>
struct to_real_dtype {
    static constexpr TDtype t = dtype;
};

template <TDtype dtype>
static constexpr TDtype to_real_dtype_t = to_real_dtype<dtype>::t;

TDtype ToRealType(TDtype dtype);

template <TDtype dtype>
struct to_complex_dtype {
    static constexpr TDtype t = dtype;
};

template <TDtype dtype>
static constexpr TDtype to_complex_dtype_t = to_complex_dtype<dtype>::t;

TDtype ToComplexType(TDtype dtype);

template <typename T>
struct is_arithmetic {
    static constexpr bool v = false;
};

template <typename T>
static constexpr bool is_arithmetic_v = is_arithmetic<T>::v;

template <TDtype dtype>
struct is_complex_dtype {
    static constexpr bool v = true;
};

template <TDtype dtype>
static constexpr bool is_complex_dtype_v = is_complex_dtype<dtype>::v;

bool IsRealType(TDtype dtype);

template <TDtype dtype>
struct is_real_dtype {
    static constexpr bool v = false;
};

template <TDtype dtype>
static constexpr bool is_real_dtype_v = is_real_dtype<dtype>::v;

bool IsComplexType(TDtype dtype);
// -----------------------------------------------------------------------------

template <>
struct to_device<TDtype::Float32> {
    using type = float;
};

template <>
struct to_device<TDtype::Float64> {
    using type = double;
};

template <>
struct to_device<TDtype::Complex64> {
    using type = std::complex<float>;
};

template <>
struct to_device<TDtype::Complex128> {
    using type = std::complex<double>;
};

template <>
struct to_dtype<float> {
    static constexpr auto dtype = TDtype::Float32;
};

template <>
struct to_dtype<double> {
    static constexpr auto dtype = TDtype::Float64;
};

template <>
struct to_dtype<std::complex<float>> {
    static constexpr auto dtype = TDtype::Complex64;
};

template <>
struct to_dtype<std::complex<double>> {
    static constexpr auto dtype = TDtype::Complex128;
};

template <>
struct is_arithmetic<float> {
    static constexpr bool v = true;
};

template <>
struct is_arithmetic<double> {
    static constexpr bool v = true;
};

template <>
struct is_arithmetic<std::complex<float>> {
    static constexpr bool v = true;
};
template <>
struct is_arithmetic<std::complex<double>> {
    static constexpr bool v = true;
};

// -----------------------------------------------------------------------------

template <>
struct to_real_dtype<TDtype::Complex128> {
    static constexpr TDtype t = TDtype::Float64;
};

template <>
struct to_real_dtype<TDtype::Complex64> {
    static constexpr TDtype t = TDtype::Float32;
};

template <>
struct to_complex_dtype<TDtype::Float64> {
    static constexpr TDtype t = TDtype::Complex128;
};

template <>
struct to_complex_dtype<TDtype::Float32> {
    static constexpr TDtype t = TDtype::Complex64;
};

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------

template <TDtype dtype>
std::string dtype_to_string() {
    if constexpr (dtype == TDtype::Float32) {
        return "float32";
    } else if constexpr (dtype == TDtype::Float64) {
        return "float64";
    } else if constexpr (dtype == TDtype::Complex64) {
        return "complex64";
    } else if constexpr (dtype == TDtype::Complex128) {
        return "complex128";
    } else {
        throw std::runtime_error("Unknown dtype.");
    }
}

std::string dtype_to_string(TDtype dtype);

// -----------------------------------------------------------------------------

template <>
struct is_complex_dtype<TDtype::Float32> {
    static constexpr bool v = false;
};

template <>
struct is_complex_dtype<TDtype::Float64> {
    static constexpr bool v = false;
};

template <>
struct is_real_dtype<TDtype::Float32> {
    static constexpr bool v = true;
};

template <>
struct is_real_dtype<TDtype::Float64> {
    static constexpr bool v = true;
};

// -----------------------------------------------------------------------------

template <TDevice device>
std::string device_to_string() {
    if constexpr (device == TDevice::CPU) {
        return "CPU";
    } else if constexpr (device == TDevice::GPU) {
        return "GPU";
    } else {
        throw std::runtime_error("Unknown device.");
    }
}

std::string device_to_string(TDevice device);

template <typename T>
std::string to_string(const std::complex<T>& a) {
    return "(" + std::to_string(a.real()) + ", " + std::to_string(a.imag()) + ")";
}

// -----------------------------------------------------------------------------

int bit_size(TDtype dtype);

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------

template <TDtype t1, TDtype t2>
struct upper_type {
    static constexpr TDtype get() {
        if constexpr (t1 == t2) {
            return t1;
        }
        if constexpr (t1 == TDtype::Complex128 || t2 == TDtype::Complex128) {
            return TDtype::Complex128;
        }
        if constexpr (t1 == TDtype::Float32) {
            return t2;
        }
        if constexpr (t2 == TDtype::Float32) {
            return t1;
        }
        if constexpr (t1 == TDtype::Float64 && t2 == TDtype::Complex64) {
            return TDtype::Complex128;
        }
        if constexpr (t1 == TDtype::Complex64 && t2 == TDtype::Float64) {
            return TDtype::Complex128;
        }
    }
};
TDtype upper_type_v(TDtype t1, TDtype t2);
}  // namespace tensor
#endif
