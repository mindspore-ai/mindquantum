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

#ifndef MATH_TENSOR_OPS_ADVANCE_MATH_HPP
#define MATH_TENSOR_OPS_ADVANCE_MATH_HPP
#include <type_traits>
#include <vector>

#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

namespace tensor::ops {
/**
 * Get the real part of tensor.
 * @param  {Tensor} t : given tensor.
 * @return {Tensor}   : the real part of tensor.
 */
Tensor real(const Tensor& t);

/**
 * Get the imaginary part of tensor.
 * @param  {Tensor} t : given tensor.
 * @return {Tensor}   : the imaginary part of tensor.
 */
Tensor imag(const Tensor& t);

Tensor conj(const Tensor& t);

Tensor vdot(const Tensor& t1, const Tensor& t2);
Tensor sin(const Tensor& t);
Tensor cos(const Tensor& t);
Tensor exp(const Tensor& t);
Tensor sqrt(const Tensor& t);
Tensor gather(const std::vector<Tensor>& tensors);

bool all_less_than(const Tensor& t);

bool is_all_zero(const Tensor& t);

std::vector<bool> is_equal_to(const Tensor& lhs, const Tensor& rhs);

template <typename T, typename = std::enable_if_t<!std::is_same_v<T, Tensor>>>
std::vector<bool> is_equal_to(const Tensor& lhs, const T& rhs) {
    return is_equal_to(lhs, ops::init_with_value(rhs, lhs.device));
}
template <typename T>
bool all_equal_to(const Tensor& lhs, const T& rhs) {
    auto equal_vec = is_equal_to(lhs, rhs);
    return std::all_of(equal_vec.begin(), equal_vec.end(), [](auto i) { return i; });
}
}  // namespace tensor::ops
#endif /* MATH_TENSOR_OPS_ADVANCE_MATH_HPP */
