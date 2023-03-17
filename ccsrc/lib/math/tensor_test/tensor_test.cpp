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

#include "math/tensor/tensor.hpp"

#include <iostream>
#include <vector>

#include "math/tensor/ops.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/ops_cpu/utils.hpp"
#include "math/tensor/traits.hpp"

template <typename T>

void TestSet(T a) {
    std::vector<tensor::Tensor> ts;
    ts.push_back(tensor::ops::init(1, tensor::TDtype::Float32));
    ts.push_back(tensor::ops::init(1, tensor::TDtype::Float64));
    ts.push_back(tensor::ops::init(1, tensor::TDtype::Complex64));
    ts.push_back(tensor::ops::init(1, tensor::TDtype::Complex128));
    tensor::ops::set(&ts[0], a, 0);
    tensor::ops::set(&ts[1], a, 0);
    tensor::ops::set(&ts[2], a, 0);
    tensor::ops::set(&ts[3], a, 0);
    for (auto& t : ts) {
        std::cout << t << std::endl;
    }
    std::cout << "===============" << std::endl;
}

void TestSetAll() {
    double a = 1.0;
    float b = 2.0;
    std::complex<float> c = {3.0, 4.0};
    std::complex<double> d = {5.0, 6.0};
    TestSet(a);
    TestSet(b);
    TestSet(c);
    TestSet(d);
}

void TensorSetTensor() {
    float v = 1.2;
    auto t = tensor::ops::init_with_value(v);
    std::cout << t << std::endl;
    std::complex<double> x = {4.5, 6.7};
    auto y = tensor::ops::init_with_value(x);
    std::cout << y << std::endl;
    tensor::ops::set(&t, y, 0);
    std::cout << t << std::endl;
}

void TensorGet() {
    std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
    auto t = tensor::ops::init_with_vector(data);
    std::cout << t << std::endl;
    auto k = tensor::ops::get(t, 0);
    std::cout << k << std::endl;
    auto l = tensor::ops::get(t, 1);
    std::cout << l << std::endl;
}
int main() {
    {
        auto t = tensor::ops::init(3);
        auto a = reinterpret_cast<double*>(t.data);
        a[0] = 1.2;
        a[1] = 2.3;
        a[2] = 3.4;
        std::cout << t << std::endl;
    }
    {
        auto t = tensor::ops::init_with_value(1.2);
        std::cout << t << std::endl;
    }
    {
        std::vector<std::complex<float>> a{{1.0, 2.0}, {2.0, 3.0}};
        auto t = tensor::ops::init_with_vector(a);
        std::cout << t << std::endl;
    }

    // -----------------------------------------------------------------------------

    TestSetAll();

    // -----------------------------------------------------------------------------
    TensorSetTensor();

    // -----------------------------------------------------------------------------

    TensorGet();
}
