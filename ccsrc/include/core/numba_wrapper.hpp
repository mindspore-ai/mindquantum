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
#include <type_traits>
#include <utility>

#include "config/type_promotion.hpp"

#include "math/tensor/matrix.hpp"
#include "math/tensor/ops.hpp"
#include "math/tensor/ops_cpu/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"
namespace mindquantum {
struct NumbaMatFunWrapper {
    using mat_t = void (*)(double, std::complex<double>*);
    NumbaMatFunWrapper() = default;
    NumbaMatFunWrapper(uint64_t addr, int dim, tensor::TDtype dtype = tensor::TDtype::Complex128)
        : dim(dim), dtype(dtype) {
        fun = reinterpret_cast<mat_t>(addr);
    }

    auto operator()(double coeff) const {
        auto tmp = tensor::ops::init(dim * dim, tensor::TDtype::Complex128, tensor::TDevice::CPU);
        fun(coeff, reinterpret_cast<std::complex<double>*>(tmp.data));
        auto out = tensor::Matrix(std::move(tmp), dim, dim);
        if (this->dtype != tensor::TDtype::Complex128) {
            out = tensor::Matrix(out.astype(this->dtype), dim, dim);
        }
        return out;
    }
    mat_t fun;
    int dim;
    tensor::TDtype dtype = tensor::TDtype::Complex128;
};
}  // namespace mindquantum
#endif
