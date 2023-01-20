//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef MQ_PYTHON_BASIC_GATE_HPP
#define MQ_PYTHON_BASIC_GATE_HPP

#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ops/basic_gate.hpp"

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

// template <typename T>
// struct BasicGate : mindquantum::BasicGate<T> {
//     using mindquantum::BasicGate<T>::BasicGate;

//     using base_t = mindquantum::BasicGate<T>;

//     BasicGate(const mindquantum::BasicGate<T>& gate) : base_t(gate) {  // NOLINT
//     }

//     BasicGate(mindquantum::BasicGate<T>&& gate) : base_t(std::move(gate)) {  // NOLINT
//     }

//     BasicGate(const std::string& name, int64_t hermitian_prop, pybind11::object matrix_fun,
//               pybind11::object diff_matrix_fun)
//         : base_t(
//             true, name, hermitian_prop,
//             [matrix_fun](T theta) {
//                 auto matrix = CastArray<T>(matrix_fun, theta);
//                 Dim2Matrix<T> res = Dim2Matrix<T>(matrix);
//                 return res;
//             },
//             [diff_matrix_fun](T theta) {
//                 auto matirx = CastArray<T>(diff_matrix_fun, theta);
//                 Dim2Matrix<T> res = Dim2Matrix<T>(matirx);
//                 return res;
//             }) {
//     }
// };
}  // namespace mindquantum::python

#endif /* MQ_PYTHON_BASIC_GATE_HPP */
