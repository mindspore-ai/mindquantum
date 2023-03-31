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

#include <omp.h>

#include <iostream>

#include "math/operators/fermion_operator_view.hpp"
#include "math/operators/qubit_operator_view.hpp"
#include "math/pr/parameter_resolver.hpp"
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
    // auto qv = operators::qubit::SinglePauliStr("X0 Y1 Z5 Z2 X8 Z2 Y4 Y9 Y4 Z1 X4",
    // tensor::ops::init_with_value(2.0)); auto other = operators::qubit::SinglePauliStr("Z3 X6 Y1 Z6 Y3 X7 Z8 X1 X3 Y1
    // Z7 X6",
    //                                               tensor::ops::init_with_value(2.0));
    // // std::cout << qv.IsSameString(other) << std::endl;
    // std::cout << qv.GetString() << std::endl;
    // std::cout << other.GetString() << std::endl;
    // auto res = qv.Mul(other);
    // std::cout << res.GetString() << std::endl;

    // auto q = operators::qubit::QubitOperator("X0 Y1");

    // auto p = operators::qubit::SinglePauliStr::init("X0 Y1", tensor::ops::init_with_value(2.3));
    // q.Update(operators::qubit::SinglePauliStr::init("X0", tensor::ops::init_with_value(2.45)));
    // q.Update(operators::qubit::SinglePauliStr::init("Z5", tensor::ops::init_with_value(2.15)));
    // std::cout << q.ToString() << std::endl;
    // q.Update(p);
    // std::cout << q.ToString() << std::endl;
    // q += tensor::ops::init_with_value(std::complex<double>(3.4, 5.6));
    // std::cout << q.ToString() << std::endl;
    // auto x = q + tensor::ops::init_with_value(std::complex<double>(3.4, 5.6));
    // std::cout << x.ToString() << std::endl;
    // auto a = tensor::ops::ones(1);
    // std::cout << a << std::endl;
    // auto p = operators::qubit::QubitOperator("X1")+ operators::qubit::QubitOperator("Y1");
    // auto q = operators::qubit::QubitOperator("Z0 Y1") + operators::qubit::QubitOperator("Z0 X1");
    // q += p;
    // std::cout << q.ToString() << std::endl;
    // std::cout << p.ToString() << std::endl;
    // std::cout << q.ToString() << std::endl;

    // auto x = p * q;
    // std::cout << x.ToString() << std::endl;

    // parameter::ParameterResolver var = tensor::ops::ones(1);
    // std::cout << var << std::endl;
    auto p = operators::qubit::QubitOperator("X24");
    std::cout << p << std::endl;
    auto f = operators::fermion::FermionOperator("0^ 2^ 23^ 45^ 22^");
    auto f2 = operators::fermion::FermionOperator("1 22");
    auto f3 = operators::fermion::FermionOperator("45^ 23^ 22^ 2^ 0^ 22 1");
    std::cout << f << std::endl;
    std::cout << f2 << std::endl;
    std::cout << f * f2 << std::endl;
    std::cout << f * f2 + f3 << std::endl;
    return 0;
}
