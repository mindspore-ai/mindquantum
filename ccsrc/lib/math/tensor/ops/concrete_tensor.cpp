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

#ifndef MATH_TENSOR_OPS_CONCRETE_TENSOR_HPP_
#define MATH_TENSOR_OPS_CONCRETE_TENSOR_HPP_
#include "math/tensor/ops_cpu/concrete_tensor.hpp"

#include <complex>
#include <vector>

#include "math/tensor/ops.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops {
Tensor zeros(size_t len, TDtype dtype, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::zeros(len, dtype);
    } else {
    }
}
Tensor ones(size_t len, TDtype dtype, TDevice device) {
    if (device == TDevice::CPU) {
        return cpu::ones(len, dtype);
    } else {
    }
}
}  // namespace tensor::ops
#endif
