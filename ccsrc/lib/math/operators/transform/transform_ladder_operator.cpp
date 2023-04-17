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
#include "math/operators/transform.hpp"
#include "math/tensor/ops.hpp"
#include "math/tensor/ops/memory_operator.hpp"

namespace operators::transform {
qubit_op_t transform_ladder_operator(const fermion::TermValue& value, const qlist_t& x1, const qlist_t& y1,
                                     const qlist_t& z1, const qlist_t& x2, const qlist_t& y2, const qlist_t& z2) {
    namespace tn = tensor;
    auto coefficient_1 = tn::ops::init_with_value(0.5);
    auto coefficient_2 = tn::ops::init_with_value(std::complex<double>(0, -0.5));
    if (value == fermion::TermValue::A) {
        coefficient_2 *= -1.0;
    }
    qubit_op_t::terms_t t1;
    t1.reserve(std::size(x1) + std::size(y1) + std::size(z1));
    std::for_each(begin(x1), end(x1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, qubit::TermValue::X); });
    std::for_each(begin(y1), end(y1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, qubit::TermValue::Y); });
    std::for_each(begin(z1), end(z1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, qubit::TermValue::Z); });
    qubit_op_t::terms_t t2;
    t2.reserve(std::size(x2) + std::size(y2) + std::size(z2));
    std::for_each(begin(x2), end(x2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, qubit::TermValue::X); });
    std::for_each(begin(y2), end(y2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, qubit::TermValue::Y); });
    std::for_each(begin(z2), end(z2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, qubit::TermValue::Z); });
    qubit_op_t out = qubit_op_t(t1, parameter::ParameterResolver(coefficient_1));
    out += qubit_op_t(t2, parameter::ParameterResolver(coefficient_2));
    return out;
}
}  // namespace operators::transform
