//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#include "ops/gates/terms_operator.hpp"

namespace mindquantum::ops::transform {
// template <typename fermion_t, typename qubit_t>
qubit_t transform_ladder_operator(const TermValue& value, const qlist_t& x1, const qlist_t& y1, const qlist_t& z1,
                                         const qlist_t& x2, const qlist_t& y2, const qlist_t& z2) {
    auto coefficient_1 = qubit_t::coefficient_t(0.5);
    auto coefficient_2 = qubit_t::coefficient_t(std::complex<double>(0, -0.5));
    if (value == TermValue::a) {
        coefficient_2 *= -1;
    }
    terms_t t1, t2;
    for (auto& i : x1) {
        t1.emplace_back(i, TermValue::X);
    }
    for (auto& i : y1) {
        t1.emplace_back(i, TermValue::Y);
    }
    for (auto& i : z1) {
        t1.emplace_back(i, TermValue::Z);
    }
    for (auto& i : x2) {
        t2.emplace_back(i, TermValue::X);
    }
    for (auto& i : y2) {
        t2.emplace_back(i, TermValue::Y);
    }
    for (auto& i : z2) {
        t2.emplace_back(i, TermValue::Z);
    }
    qubit_t out = qubit_t(t1, coefficient_1);
    out += qubit_t(t2, coefficient_2);
    return out;
}
}  // namespace mindquantum::ops::transform
