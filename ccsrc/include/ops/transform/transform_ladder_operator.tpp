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

#include "ops/transform/transform_ladder_operator.hpp"
#include "ops/transform/types.hpp"

namespace mindquantum::ops::transform {
template <typename fermion_op_t>
auto transform_ladder_operator(const TermValue& value, const qlist_t& x1, const qlist_t& y1, const qlist_t& z1,
                               const qlist_t& x2, const qlist_t& y2, const qlist_t& z2)
    -> to_qubit_operator_t<traits::to_cmplx_type_t<fermion_op_t>> {
    using qubit_op_t = to_qubit_operator_t<traits::to_cmplx_type_t<fermion_op_t>>;
    using coefficient_t = typename qubit_op_t::coefficient_t;

    auto coefficient_1 = coefficient_t(0.5);
    auto coefficient_2 = coefficient_t(std::complex<double>(0, -0.5));
    if (value == TermValue::a) {
        coefficient_2 *= -1;
    }
    terms_t t1;
    t1.reserve(std::size(x1) + std::size(y1) + std::size(z1));
    std::for_each(begin(x1), end(x1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, TermValue::X); });
    std::for_each(begin(y1), end(y1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, TermValue::Y); });
    std::for_each(begin(z1), end(z1), [&t1](const auto& qubit_id) { t1.emplace_back(qubit_id, TermValue::Z); });
    terms_t t2;
    t2.reserve(std::size(x2) + std::size(y2) + std::size(z2));
    std::for_each(begin(x2), end(x2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, TermValue::X); });
    std::for_each(begin(y2), end(y2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, TermValue::Y); });
    std::for_each(begin(z2), end(z2), [&t2](const auto& qubit_id) { t2.emplace_back(qubit_id, TermValue::Z); });
    qubit_op_t out = qubit_op_t(t1, coefficient_1);
    out += qubit_op_t(t2, coefficient_2);
    return out;
}

}  // namespace mindquantum::ops::transform
