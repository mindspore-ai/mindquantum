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

#ifndef MATH_TENSOR_TENSOR_HPP_
#define MATH_TENSOR_TENSOR_HPP_

#include <cstddef>

#include "math/tensor/traits.hpp"

namespace tensor {
struct Tensor {
    TDtype dtype = TDtype::Float64;
    TDevice device = TDevice::CPU;
    void* data = nullptr;
    size_t dim = 0;

    // -----------------------------------------------------------------------------

    ~Tensor();
    Tensor() = default;
    Tensor(TDtype dtype, TDevice device, void* data, size_t dim);
    Tensor(Tensor&& t);
    Tensor& operator=(Tensor&& t);
    Tensor(const Tensor& t);
    Tensor& operator=(const Tensor& t);

    // -----------------------------------------------------------------------------

    Tensor operator[](size_t idx) const;
    Tensor real() const;
    Tensor imag() const;
    Tensor conj() const;
    Tensor astype(TDtype type) const;

    // -----------------------------------------------------------------------------

    Tensor& operator+=(float other);
    Tensor& operator+=(double other);
    Tensor& operator+=(const std::complex<double>& other);
    Tensor& operator+=(const std::complex<float>& other);
    Tensor& operator+=(const Tensor& other);

    // -----------------------------------------------------------------------------

    Tensor& operator-=(float other);
    Tensor& operator-=(double other);
    Tensor& operator-=(const std::complex<double>& other);
    Tensor& operator-=(const std::complex<float>& other);
    Tensor& operator-=(const Tensor& other);

    // -----------------------------------------------------------------------------

    Tensor& operator*=(float other);
    Tensor& operator*=(double other);
    Tensor& operator*=(const std::complex<double>& other);
    Tensor& operator*=(const std::complex<float>& other);
    Tensor& operator*=(const Tensor& other);

    // -----------------------------------------------------------------------------

    Tensor& operator/=(float other);
    Tensor& operator/=(double other);
    Tensor& operator/=(const std::complex<double>& other);
    Tensor& operator/=(const std::complex<float>& other);
    Tensor& operator/=(const Tensor& other);

    // -----------------------------------------------------------------------------
};

// -----------------------------------------------------------------------------

Tensor operator+(const Tensor& lhs, float rhs);
Tensor operator+(const Tensor& lhs, double rhs);
Tensor operator+(const Tensor& lhs, const std::complex<double>& rhs);
Tensor operator+(const Tensor& lhs, const std::complex<float>& rhs);
Tensor operator+(const Tensor& lhs, const Tensor& rhs);

// -----------------------------------------------------------------------------

Tensor operator-(const Tensor& lhs, float rhs);
Tensor operator-(const Tensor& lhs, double rhs);
Tensor operator-(const Tensor& lhs, const std::complex<double>& rhs);
Tensor operator-(const Tensor& lhs, const std::complex<float>& rhs);
Tensor operator-(float rhs, const Tensor& lhs);
Tensor operator-(double rhs, const Tensor& lhs);
Tensor operator-(const std::complex<double>& rhs, const Tensor& lhs);
Tensor operator-(const std::complex<float>& rhs, const Tensor& lhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);

// -----------------------------------------------------------------------------

Tensor operator*(const Tensor& lhs, float rhs);
Tensor operator*(const Tensor& lhs, double rhs);
Tensor operator*(const Tensor& lhs, const std::complex<double>& rhs);
Tensor operator*(const Tensor& lhs, const std::complex<float>& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);

// -----------------------------------------------------------------------------

Tensor operator/(const Tensor& lhs, float rhs);
Tensor operator/(const Tensor& lhs, double rhs);
Tensor operator/(const Tensor& lhs, const std::complex<double>& rhs);
Tensor operator/(const Tensor& lhs, const std::complex<float>& rhs);
Tensor operator/(float rhs, const Tensor& lhs);
Tensor operator/(double rhs, const Tensor& lhs);
Tensor operator/(const std::complex<double>& rhs, const Tensor& lhs);
Tensor operator/(const std::complex<float>& rhs, const Tensor& lhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
}  // namespace tensor
#endif
