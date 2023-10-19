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

#ifndef MATH_TENSOR_OPS_BASIC_MATH_HPP_
#define MATH_TENSOR_OPS_BASIC_MATH_HPP_

#include <complex>

#include "math/tensor/csr_matrix.h"
#include "math/tensor/matrix.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops {
void inplace_add(Tensor* t, float a);
void inplace_add(Tensor* t, double a);
void inplace_add(Tensor* t, const std::complex<float>& a);
void inplace_add(Tensor* t, const std::complex<double>& a);
void inplace_add(Tensor* t, const Tensor& other);
Tensor add(const Tensor& t, float a);
Tensor add(const Tensor& t, double a);
Tensor add(const Tensor& t, const std::complex<float>& a);
Tensor add(const Tensor& t, const std::complex<double>& a);
Tensor add(float a, const Tensor& t);
Tensor add(double a, const Tensor& t);
Tensor add(const std::complex<float>& a, const Tensor& t);
Tensor add(const std::complex<double>& a, const Tensor& t);
Tensor add(const Tensor& t, const Tensor& other);

// -----------------------------------------------------------------------------

void inplace_sub(Tensor* t, float a);
void inplace_sub(Tensor* t, double a);
void inplace_sub(Tensor* t, const std::complex<float>& a);
void inplace_sub(Tensor* t, const std::complex<double>& a);
void inplace_sub(Tensor* t, const Tensor& other);
void inplace_sub(float a, Tensor* t);
void inplace_sub(double a, Tensor* t);
void inplace_sub(const std::complex<float>& a, Tensor* t);
void inplace_sub(const std::complex<double>& a, Tensor* t);
void inplace_sub(const Tensor& other, Tensor* t);
Tensor sub(const Tensor& t, float a);
Tensor sub(const Tensor& t, double a);
Tensor sub(const Tensor& t, const std::complex<float>& a);
Tensor sub(const Tensor& t, const std::complex<double>& a);
Tensor sub(const Tensor& t, const Tensor& other);
Tensor sub(float a, const Tensor& t);
Tensor sub(double a, const Tensor& t);
Tensor sub(const std::complex<float>& a, const Tensor& t);
Tensor sub(const std::complex<double>& a, const Tensor& t);

// -----------------------------------------------------------------------------

void inplace_mul(Tensor* t, float a);
void inplace_mul(Tensor* t, double a);
void inplace_mul(Tensor* t, const std::complex<float>& a);
void inplace_mul(Tensor* t, const std::complex<double>& a);
void inplace_mul(Tensor* t, const Tensor& other);
Tensor mul(const Tensor& t, float a);
Tensor mul(const Tensor& t, double a);
Tensor mul(const Tensor& t, const std::complex<float>& a);
Tensor mul(const Tensor& t, const std::complex<double>& a);
Tensor mul(float a, const Tensor& t);
Tensor mul(double a, const Tensor& t);
Tensor mul(const std::complex<float>& a, const Tensor& t);
Tensor mul(const std::complex<double>& a, const Tensor& t);
Tensor mul(const Tensor& t, const Tensor& other);

// -----------------------------------------------------------------------------

void inplace_div(Tensor* t, float a);
void inplace_div(Tensor* t, double a);
void inplace_div(Tensor* t, const std::complex<float>& a);
void inplace_div(Tensor* t, const std::complex<double>& a);
void inplace_div(Tensor* t, const Tensor& other);
void inplace_div(float a, Tensor* t);
void inplace_div(double a, Tensor* t);
void inplace_div(const std::complex<float>& a, Tensor* t);
void inplace_div(const std::complex<double>& a, Tensor* t);
void inplace_div(const Tensor& other, Tensor* t);
Tensor div(const Tensor& t, float a);
Tensor div(const Tensor& t, double a);
Tensor div(const Tensor& t, const std::complex<float>& a);
Tensor div(const Tensor& t, const std::complex<double>& a);
Tensor div(const Tensor& t, const Tensor& other);
Tensor div(float a, const Tensor& t);
Tensor div(double a, const Tensor& t);
Tensor div(const std::complex<float>& a, const Tensor& t);
Tensor div(const std::complex<double>& a, const Tensor& t);

// -----------------------------------------------------------------------------

Matrix MatMul(const Matrix& m1, const Matrix& m2);
Tensor MatMul(const CsrMatrix& m1, const Tensor& m2);
}  // namespace tensor::ops
#endif
