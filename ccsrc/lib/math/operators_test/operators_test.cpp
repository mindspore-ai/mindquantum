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

#include <iostream>

#include "math/operators/qubit_operator_view.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/traits.hpp"

int main() {
    // operators::qubit::SinglePauliStr qv(
    //     {
    //         {operators::qubit::TermValue::X, 0},
    //         {operators::qubit::TermValue::Y, 1},
    //         {operators::qubit::TermValue::Z, 1},
    //     },
    //     tensor::ops::ones(1));
    // std::cout << qv.pauli_string.size() << std::endl;
    // std::cout << qv.pauli_string[0] << std::endl;
    // std::cout << qv.coeff << std::endl;

    // std::string str = "X0 Y1 Y1-9";
    // std::istringstream iss(str);

    // for (std::string s; iss >> s;) {
    //     parse_token(s);
    // }
    auto qv = operators::qubit::SinglePauliStr("X0 Y1 Z1", tensor::ops::init_with_value(2.0));
    auto other = operators::qubit::SinglePauliStr("X0 X1 Z40 X40", tensor::ops::init_with_value(2.0));
    std::cout << qv.IsSameString(other) << std::endl;
    return 0;
}
