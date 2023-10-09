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

#ifndef MATH_TENSOR_OPS_MEMORY_OPERATOR_HPP_
#define MATH_TENSOR_OPS_MEMORY_OPERATOR_HPP_
#include <algorithm>
#include <complex>
#include <ostream>
#include <string>
#include <vector>

#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops {
// Memory operator
/**
 * Initialize a empty tensor, the element is undefined.
 * @param  {size_t} len     : length of the tensor.
 * @param  {TDtype} dtype   : type of the tensor.
 * @param  {TDevice} device : target device of the tensor.
 * @return {Tensor}         : a empty tensor with element undefined.
 */
Tensor init(size_t len, TDtype dtype = TDtype::Float64, TDevice device = TDevice::CPU);

/**
 * Cast a tensor to target type.
 * @param  {Tensor} t            : given tensor.
 * @param  {TDtype} target_dtype : target data type.
 * @return {Tensor}              : converted tensor.
 */
Tensor cast_to(const Tensor& t, TDtype target_dtype);

Tensor init_with_value(float a, TDevice device = TDevice::CPU);
Tensor init_with_value(double a, TDevice device = TDevice::CPU);
Tensor init_with_value(const std::complex<float>& a, TDevice device = TDevice::CPU);
Tensor init_with_value(const std::complex<double>& a, TDevice device = TDevice::CPU);
Tensor init_with_vector(const std::vector<float>& a, TDevice device = TDevice::CPU);
Tensor init_with_vector(const std::vector<double>& a, TDevice device = TDevice::CPU);
Tensor init_with_vector(const std::vector<std::complex<float>>& a, TDevice device = TDevice::CPU);
Tensor init_with_vector(const std::vector<std::complex<double>>& a, TDevice device = TDevice::CPU);

/**
 * Destroy a tensor object.
 * @param  {Tensor*} t : tensor pointer.
 */
void destroy(Tensor* t);

/**
 * Copy a tensor object.
 * @param  {Tensor} t : Source tensor.
 * @return {Tensor}   : Copied tensor.
 */
Tensor copy(const Tensor& t);

void set(Tensor* t, float a, size_t idx);
void set(Tensor* t, double a, size_t idx);
void set(Tensor* t, const std::complex<float>& a, size_t idx);
void set(Tensor* t, const std::complex<double>& a, size_t idx);
void set(Tensor* t, const Tensor& source, size_t idx);
Tensor get(const Tensor& t, size_t idx);

std::string to_string(const Tensor& t, bool simplify = false);
}  // namespace tensor::ops

std::ostream& operator<<(std::ostream& os, const tensor::Tensor& t);
#endif
