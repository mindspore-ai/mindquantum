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
#ifndef MQ_CORE_NUMBER_WRAPPER_HPP_
#define MQ_CORE_NUMBER_WRAPPER_HPP_

#include <memory>
#include <stdexcept>

#include "core/two_dim_matrix.hpp"
namespace mindquantum {
struct NumbaMatFunWrapper {
    using mat_t = void (*)(double, std::complex<double>*);
    NumbaMatFunWrapper() = default;
    NumbaMatFunWrapper(uint64_t addr, int dim) : dim(dim) {
        if (dim != 2 && dim != 4) {
            throw std::runtime_error("Can only custom one or two qubits matrix.");
        }
        fun = reinterpret_cast<mat_t>(addr);
    }

    auto operator()(double coeff) const {
        mindquantum::Dim2Matrix<double> out;
        auto tmp = std::make_unique<std::complex<double>[]>(dim * dim);
        fun(coeff, tmp.get());
        mindquantum::VVT<std::complex<double>> m(dim, mindquantum::VT<std::complex<double>>(dim, 0.0));
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                m[i][j] = tmp[i * dim + j];
            }
        }

        out = mindquantum::Dim2Matrix<double>(m);
        return out;
    }
    mat_t fun;
    int dim;
};
}  // namespace mindquantum
#endif
