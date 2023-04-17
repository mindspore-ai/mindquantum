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

#include "math/tensor/tensor.hpp"

#include <iostream>

#include "math/tensor/ops.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/traits.hpp"
#define TENSOR_PLUS_EQUAL(dtype)                                                                                       \
    Tensor& Tensor::operator+=(dtype other) {                                                                          \
        ops::inplace_add(this, other);                                                                                 \
        return *this;                                                                                                  \
    }

#define TENSOR_SUB_EQUAL(dtype)                                                                                        \
    Tensor& Tensor::operator-=(dtype other) {                                                                          \
        ops::inplace_sub(this, other);                                                                                 \
        return *this;                                                                                                  \
    }

#define TENSOR_MUL_EQUAL(dtype)                                                                                        \
    Tensor& Tensor::operator*=(dtype other) {                                                                          \
        ops::inplace_mul(this, other);                                                                                 \
        return *this;                                                                                                  \
    }

#define TENSOR_DIV_EQUAL(dtype)                                                                                        \
    Tensor& Tensor::operator/=(dtype other) {                                                                          \
        ops::inplace_div(this, other);                                                                                 \
        return *this;                                                                                                  \
    }

#define TENSOR_ADD(dtype)                                                                                              \
    Tensor operator+(const Tensor& lhs, dtype rhs) {                                                                   \
        return ops::add(lhs, rhs);                                                                                     \
    }

#define TENSOR_SUB(dtype)                                                                                              \
    Tensor operator-(const Tensor& lhs, dtype rhs) {                                                                   \
        return ops::sub(lhs, rhs);                                                                                     \
    }

#define TENSOR_SUB_REV(dtype)                                                                                          \
    Tensor operator-(dtype lhs, const Tensor& rhs) {                                                                   \
        return ops::sub(lhs, rhs);                                                                                     \
    }

#define TENSOR_MUL(dtype)                                                                                              \
    Tensor operator*(const Tensor& lhs, dtype rhs) {                                                                   \
        return ops::mul(lhs, rhs);                                                                                     \
    }

#define TENSOR_DIV(dtype)                                                                                              \
    Tensor operator/(const Tensor& lhs, dtype rhs) {                                                                   \
        return ops::div(lhs, rhs);                                                                                     \
    }

#define TENSOR_DIV_REV(dtype)                                                                                          \
    Tensor operator/(dtype lhs, const Tensor& rhs) {                                                                   \
        return ops::div(lhs, rhs);                                                                                     \
    }

namespace tensor {
TENSOR_PLUS_EQUAL(float)
TENSOR_PLUS_EQUAL(double)
TENSOR_PLUS_EQUAL(const std::complex<float>&)
TENSOR_PLUS_EQUAL(const std::complex<double>&)
TENSOR_PLUS_EQUAL(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_SUB_EQUAL(float)
TENSOR_SUB_EQUAL(double)
TENSOR_SUB_EQUAL(const std::complex<float>&)
TENSOR_SUB_EQUAL(const std::complex<double>&)
TENSOR_SUB_EQUAL(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_MUL_EQUAL(float)
TENSOR_MUL_EQUAL(double)
TENSOR_MUL_EQUAL(const std::complex<float>&)
TENSOR_MUL_EQUAL(const std::complex<double>&)
TENSOR_MUL_EQUAL(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_DIV_EQUAL(float)
TENSOR_DIV_EQUAL(double)
TENSOR_DIV_EQUAL(const std::complex<float>&)
TENSOR_DIV_EQUAL(const std::complex<double>&)
TENSOR_DIV_EQUAL(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_ADD(float)
TENSOR_ADD(double)
TENSOR_ADD(const std::complex<float>&)
TENSOR_ADD(const std::complex<double>&)
TENSOR_ADD(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_SUB(float)
TENSOR_SUB(double)
TENSOR_SUB(const std::complex<float>&)
TENSOR_SUB(const std::complex<double>&)
TENSOR_SUB_REV(float)
TENSOR_SUB_REV(double)
TENSOR_SUB_REV(const std::complex<float>&)
TENSOR_SUB_REV(const std::complex<double>&)
TENSOR_SUB(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_MUL(float)
TENSOR_MUL(double)
TENSOR_MUL(const std::complex<float>&)
TENSOR_MUL(const std::complex<double>&)
TENSOR_MUL(const Tensor&)

// -----------------------------------------------------------------------------

TENSOR_DIV(float)
TENSOR_DIV(double)
TENSOR_DIV(const std::complex<float>&)
TENSOR_DIV(const std::complex<double>&)
TENSOR_DIV_REV(float)
TENSOR_DIV_REV(double)
TENSOR_DIV_REV(const std::complex<float>&)
TENSOR_DIV_REV(const std::complex<double>&)
TENSOR_DIV(const Tensor&)

// -----------------------------------------------------------------------------

Tensor Tensor::operator[](size_t idx) const {
    return tensor::ops::get(*this, idx);
}

Tensor Tensor::real() const {
    return tensor::ops::real(*this);
}

Tensor Tensor::imag() const {
    return tensor::ops::imag(*this);
}

Tensor Tensor::conj() const {
    return tensor::ops::conj(*this);
}

Tensor Tensor::astype(TDtype dtype) const {
    return tensor::ops::cast_to(*this, dtype);
}

std::vector<bool> Tensor::operator==(const Tensor& other) const {
    return tensor::ops::is_equal_to(*this, other);
}
std::vector<bool> Tensor::operator==(float other) const {
    return tensor::ops::is_equal_to(*this, other);
}
std::vector<bool> Tensor::operator==(double other) const {
    return tensor::ops::is_equal_to(*this, other);
}
std::vector<bool> Tensor::operator==(const std::complex<float>& other) const {
    return tensor::ops::is_equal_to(*this, other);
}
std::vector<bool> Tensor::operator==(const std::complex<double>& other) const {
    return tensor::ops::is_equal_to(*this, other);
}
}  // namespace tensor
