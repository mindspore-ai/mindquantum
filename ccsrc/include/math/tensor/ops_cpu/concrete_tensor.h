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

#ifndef MATH_TENSOR_OPS_CPU_CONCRETE_TENSOR_HPP_
#define MATH_TENSOR_OPS_CPU_CONCRETE_TENSOR_HPP_
#include "math/tensor/ops.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
namespace tensor::ops::cpu {
Tensor zeros(size_t len, TDtype dtype = TDtype::Float64);

template <TDtype dtype>
Tensor ones(size_t len) {
    using calc_t = to_device_t<dtype>;
    Tensor out = cpu::init<dtype>(len);
    auto c_data = reinterpret_cast<calc_t*>(out.data);
    for (size_t i = 0; i < len; i++) {
        c_data[i] = 1.0;
    }
    return out;
}

Tensor ones(size_t len, TDtype dtype = TDtype::Float64);
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_CONCRETE_TENSOR_HPP_ */
