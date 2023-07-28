/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef MQ_PYTHON_BASIC_GATE_HPP
#define MQ_PYTHON_BASIC_GATE_HPP

#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ops/basic_gate.h"

namespace mindquantum::python {
template <typename T>
inline VVT<CT<T>> CastArray(const pybind11::object& fun, T theta) {
    pybind11::array_t<CT<T>> a = fun(theta);
    pybind11::buffer_info buf = a.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Gate matrix must be two dimension!");
    }
    if (buf.shape[0] != buf.shape[1]) {
        throw std::runtime_error("Gate matrix need a square matrix!");
    }
    CTP<T> ptr = static_cast<CTP<T>>(buf.ptr);

    VVT<CT<T>> matrix;
    for (size_t i = 0; i < buf.shape[0]; i++) {
        matrix.push_back({});
        for (size_t j = 0; j < buf.shape[1]; j++) {
            matrix[i].push_back(ptr[i * buf.shape[1] + j]);
        }
    }
    return matrix;
}
}  // namespace mindquantum::python

#endif /* MQ_PYTHON_BASIC_GATE_HPP */
