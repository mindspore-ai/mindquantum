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

#ifndef PYTHON_TENSOR_HPP_
#define PYTHON_TENSOR_HPP_
#include <algorithm>

#include <pybind11/numpy.h>

#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/tensor.h"

namespace mindquantum::python {
template <typename T>
tensor::Tensor from_numpy(const pybind11::array_t<T> &arr) {
    pybind11::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be one.");
    }
    return tensor::ops::cpu::copy<tensor::to_dtype_v<T>>(buf.ptr, buf.size);
}
}  // namespace mindquantum::python
#endif /* PYTHON_TENSOR_HPP_ */
