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

#ifndef MATH_TENSOR_OPS_CONCRETE_TENSOR_HPP_
#define MATH_TENSOR_OPS_CONCRETE_TENSOR_HPP_
#include <complex>
#include <vector>

#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops {
// Concrete  initialize operator
/**
 * Generate a tensor initialize with 0.
 * @param  {size_t} len     : length of the tensor.
 * @param  {TDtype} dtype   : type of the tensor.
 * @param  {TDevice} device : target device of the tensor.
 * @return {Tensor}         : zero tensor.
 */
Tensor zeros(size_t len, TDtype dtype = TDtype::Float64, TDevice device = TDevice::CPU);

/**
 * Generate a tensor initialize with 1.
 * @param  {size_t} len     : length of the tensor.
 * @param  {TDtype} dtype   : type of the tensor.
 * @param  {TDevice} device : target device of the tensor.
 * @return {Tensor}         : one tensor.
 */
Tensor ones(size_t len, TDtype dtype = TDtype::Float64, TDevice device = TDevice::CPU);
}  // namespace tensor::ops
#endif
