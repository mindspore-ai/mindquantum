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

#ifndef MATH_TENSOR_OPS_CPU_HPP_
#define MATH_TENSOR_OPS_CPU_HPP_
#include <cstdlib>
#include <cstring>

#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace tensor::ops::cpu {
template <TDtype dtype>
Tensor init(size_t len) {
    using calc_t = to_device_t<dtype>;
    auto data = reinterpret_cast<void*>(malloc(sizeof(calc_t) * len));
    return Tensor{dtype, TDevice::CPU, data, len};
}

Tensor init(size_t len, TDtype dtype);

// -----------------------------------------------------------------------------

template <TDtype src, TDtype des>
Tensor cast_to(void* data, size_t len) {
    using d_src = to_device_t<src>;
    using d_des = to_device_t<des>;
    auto c_data = reinterpret_cast<d_src*>(data);
    auto out = init<des>(len);
    auto c_out = reinterpret_cast<d_des*>(out.data);
    for (size_t i = 0; i < len; i++) {
        c_out[i] = c_data[i];
    }
    return out;
}

Tensor cast_to(void* data, TDtype src, TDtype des, size_t len);
}  // namespace tensor::ops::cpu
#endif /* MATH_TENSOR_OPS_CPU_HPP_ */
