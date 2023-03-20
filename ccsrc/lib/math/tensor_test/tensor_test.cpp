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
#include "math/tensor/ops/advance_math.hpp"
#include "math/tensor/ops/basic_math.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/ops_cpu/utils.hpp"
#include "math/tensor/traits.hpp"

static const std::vector<tensor::TDtype> all_type = {
    tensor::TDtype::Float32,
    tensor::TDtype::Float64,
    tensor::TDtype::Complex64,
    tensor::TDtype::Complex128,
};

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

void TensorConcrete() {
    auto t1 = tensor::ops::zeros(2, tensor::TDtype::Complex128);
    auto t2 = tensor::ops::ones(2, tensor::TDtype::Complex128);
    std::cout << t1 << "\n" << t2 << std::endl;
}

void TensorInplace() {
    auto t1 = tensor::ops::ones(2, tensor::TDtype::Complex128);
}

template <typename T>
void TensorInplaceAdd_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        tensor::ops::inplace_add(&t, a);
        std::cout << t << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorInplaceAdd() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorInplaceAdd_all(a);
    TensorInplaceAdd_all(c);
    TensorInplaceAdd_all(d);
    TensorInplaceAdd_all(e);
}

template <typename T>
void TensorInplaceSub_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        tensor::ops::inplace_sub(&t, a);
        std::cout << t << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorInplaceSub() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorInplaceSub_all(a);
    TensorInplaceSub_all(c);
    TensorInplaceSub_all(d);
    TensorInplaceSub_all(e);
}

template <typename T>
void TensorInplaceMul_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        tensor::ops::inplace_mul(&t, a);
        std::cout << t << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorInplaceMul() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorInplaceMul_all(a);
    TensorInplaceMul_all(c);
    TensorInplaceMul_all(d);
    TensorInplaceMul_all(e);
}

template <typename T>
void TensorInplaceDiv_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        tensor::ops::inplace_div(&t, a);
        std::cout << t << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorInplaceDiv() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorInplaceDiv_all(a);
    TensorInplaceDiv_all(c);
    TensorInplaceDiv_all(d);
    TensorInplaceDiv_all(e);
}

template <typename T>
void TensorAdd_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        auto res = tensor::ops::add(t, a);
        std::cout << res << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorAdd() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorAdd_all(a);
    TensorAdd_all(c);
    TensorAdd_all(d);
    TensorAdd_all(e);
}

template <typename T>
void TensorSub_all(T a) {
    for (const auto& type : all_type) {
        auto t = tensor::ops::ones(2, type);
        auto res = tensor::ops::sub(t, a);
        std::cout << res << std::endl;
    }
    std::cout << "==========" << std::endl;
}

void TensorSub() {
    float a = 100;
    double c = 101;
    std::complex<float> d = 103;
    std::complex<double> e = 104;
    TensorSub_all(a);
    TensorSub_all(c);
    TensorSub_all(d);
    TensorSub_all(e);
}

void TensorTensorAdd(int size) {
    for (auto dtype1 : all_type) {
        for (auto dtype2 : all_type) {
            auto t1 = tensor::ops::ones(size, dtype1);
            auto t2 = tensor::ops::ones(size, dtype2);
            auto t3 = tensor::ops::add(t1, t2);
            std::cout << t1 << std::endl;
            std::cout << t2 << std::endl;
            std::cout << t3 << std::endl;
            std::cout << "==============" << std::endl;
        }
    }
}

void TensorTensorSub(int size) {
    for (auto dtype1 : all_type) {
        for (auto dtype2 : all_type) {
            auto t1 = tensor::ops::ones(size, dtype1);
            auto t2 = tensor::ops::ones(size, dtype2);
            auto t3 = tensor::ops::sub(t1, t2);
            std::cout << t1 << std::endl;
            std::cout << t2 << std::endl;
            std::cout << t3 << std::endl;
            std::cout << "==============" << std::endl;
        }
    }
}

void TensorTensorMul(int size) {
    for (auto dtype1 : all_type) {
        for (auto dtype2 : all_type) {
            auto t1 = tensor::ops::ones(size, dtype1);
            auto t2 = tensor::ops::ones(size, dtype2);
            auto t3 = tensor::ops::mul(t1, t2);
            std::cout << t1 << std::endl;
            std::cout << t2 << std::endl;
            std::cout << t3 << std::endl;
            std::cout << "==============" << std::endl;
        }
    }
}

void TensorTensorDiv(int size) {
    for (auto dtype1 : all_type) {
        for (auto dtype2 : all_type) {
            auto t1 = tensor::ops::ones(size, dtype1);
            auto t2 = tensor::ops::ones(size, dtype2);
            auto t3 = tensor::ops::div(t1, t2);
            std::cout << t1 << std::endl;
            std::cout << t2 << std::endl;
            std::cout << t3 << std::endl;
            std::cout << "==============" << std::endl;
        }
    }
}

void Benchmark1(size_t step) {
    for (size_t i = 0; i < step; i++) {
        auto t1 = tensor::ops::ones(2, tensor::TDtype::Complex64);
        double a = 2.0;
        auto t2 = t1 + a;
    }
}

void TensorOps() {
    auto t1 = tensor::ops::ones(2, tensor::TDtype::Complex64);
    double a = 2.0;
    // t1 += a;
    auto t2 = t1 + a;
    // std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    auto t3 = tensor::ops::add(t1, a);
    std::cout << t3 << std::endl;
}

void vdot() {
    auto t1 = tensor::ops::ones(2, tensor::TDtype::Complex64);
    t1 += std::complex<float>(0.0, 3.0);
    auto t2 = tensor::ops::ones(2, tensor::TDtype::Float64);
    auto res = tensor::ops::vdot(t1, t2);
    std::cout << res << std::endl;
    std::cout << res.real() << std::endl;
    std::cout << res.imag() << std::endl;
    std::cout << res.conj() << std::endl;
}
int main() {
    // {
    //     auto t = tensor::ops::init(3);
    //     auto a = reinterpret_cast<double*>(t.data);
    //     a[0] = 1.2;
    //     a[1] = 2.3;
    //     a[2] = 3.4;
    //     std::cout << t << std::endl;
    // }
    // {
    //     auto t = tensor::ops::init_with_value(1.2);
    //     std::cout << t << std::endl;
    // }
    // {
    //     std::vector<std::complex<float>> a{{1.0, 2.0}, {2.0, 3.0}};
    //     auto t = tensor::ops::init_with_vector(a);
    //     std::cout << t << std::endl;
    // }

    // -----------------------------------------------------------------------------

    // TestSetAll();

    // -----------------------------------------------------------------------------
    // TensorSetTensor();

    // -----------------------------------------------------------------------------

    // TensorGet();

    // -----------------------------------------------------------------------------

    // TensorConcrete();

    // -----------------------------------------------------------------------------

    // TensorInplaceAdd();

    // -----------------------------------------------------------------------------

    // TensorInplaceSub();

    // -----------------------------------------------------------------------------

    // TensorInplaceMul();

    // -----------------------------------------------------------------------------
    // TensorInplaceDiv();

    // -----------------------------------------------------------------------------

    // TensorAdd();

    // -----------------------------------------------------------------------------

    // TensorSub();

    // -----------------------------------------------------------------------------

    // TensorTensorAdd(1);

    // -----------------------------------------------------------------------------

    // TensorTensorAdd(2);

    // -----------------------------------------------------------------------------

    // TensorTensorSub(1);

    // -----------------------------------------------------------------------------

    // TensorTensorSub(2);

    // -----------------------------------------------------------------------------

    // TensorTensorMul(1);

    // -----------------------------------------------------------------------------

    // TensorTensorMul(2);

    // -----------------------------------------------------------------------------

    // TensorTensorDiv(1);

    // -----------------------------------------------------------------------------

    // TensorTensorDiv(2);

    // -----------------------------------------------------------------------------

    // TensorOps();

    // -----------------------------------------------------------------------------

    // Benchmark1(10000000);

    // -----------------------------------------------------------------------------
    vdot();
}
