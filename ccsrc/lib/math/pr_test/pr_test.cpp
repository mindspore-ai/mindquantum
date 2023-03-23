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

#include "math/pr/parameter_resolver.hpp"

int main() {
    std::map<std::string, double> a;
    a["a"] = 1.2;
    auto pr = parameter::ParameterResolver(static_cast<double>(2.3), a);
    pr.SetItem("a", 1.0);
    // std::cout << pr << std::endl;
    pr -= std::complex<double>(4.5, 4.0);
    // std::cout << pr << std::endl;

    auto lhs = parameter::ParameterResolver(std::complex<float>(1.1, 3.4));
    // std::cout << lhs << std::endl;
    auto rhs = pr + lhs;
    std::cout << pr << std::endl;
    std::cout << lhs << std::endl;
    std::cout << rhs << std::endl;
    pr += lhs;
    pr.AsEncoder();
    pr.NoGrad();
    std::cout << pr << std::endl;
}
