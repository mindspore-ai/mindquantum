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

#include "ops/gates.hpp"

namespace mindquantum {
tensor::Matrix U3Matrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda) {
    auto el = tensor::ops::exp(lambda * std::complex<float>(0, 1));
    auto ep = tensor::ops::exp(phi * std::complex<float>(0, 1));
    auto ct_2 = tensor::ops::cos(theta / 2.0).astype(el.dtype);
    auto st_2 = tensor::ops::sin(theta / 2.0).astype(el.dtype);
    auto elp = el * ep;
    auto out = tensor::ops::gather({ct_2, 0.0 - el * st_2, ep * st_2, elp * ct_2});
    return tensor::Matrix(std::move(out), 2, 2);
}

tensor::Matrix FSimMatrix(tensor::Tensor theta, tensor::Tensor phi) {
    auto b = tensor::ops::sin(theta) * std::complex<float>(0, -1);
    auto a = tensor::ops::cos(theta).astype(b.dtype);
    auto c = tensor::ops::exp(phi * std::complex<float>(0, 1));
    auto one = tensor::ops::ones(1, c.dtype);
    auto zero = tensor::ops::zeros(1, c.dtype);
    return tensor::Matrix(
        tensor::ops::gather({one, zero, zero, zero, zero, a, b, zero, zero, b, a, zero, zero, zero, zero, c}), 4, 4);
}

tensor::Matrix U3DiffThetaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda) {
    auto m = U3Matrix(theta + M_PI, phi, lambda);
    m = tensor::Matrix(tensor::ops::mul(m, 0.5), m.n_row, m.n_col);
    return m;
}

tensor::Matrix FSimDiffThetaMatrix(tensor::Tensor theta) {
    auto b = tensor::ops::cos(theta) * std::complex<float>(0, -1.0);
    auto a = (0.0 - tensor::ops::sin(theta)).astype(b.dtype);
    auto zero = tensor::ops::zeros(1, a.dtype);
    return tensor::Matrix(
        tensor::ops::gather({zero, zero, zero, zero, zero, a, b, zero, zero, b, a, zero, zero, zero, zero, zero}), 4,
        4);
}

tensor::Matrix U3DiffPhiMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda) {
    auto m = U3Matrix(theta, phi + M_PI_2, lambda);
    tensor::ops::set(&m, tensor::ops::zeros(1, m.dtype), 0);
    tensor::ops::set(&m, tensor::ops::zeros(1, m.dtype), 1);
    return m;
}

tensor::Matrix FSimDiffPhiMatrix(tensor::Tensor phi) {
    auto c = tensor::ops::exp((phi + M_PI_2) * std::complex<float>(0, 1));
    auto out = tensor::Matrix(tensor::ops::zeros(16, c.dtype), 4, 4);
    tensor::ops::set(&out, c, 15);
    return out;
}

tensor::Matrix U3DiffLambdaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda) {
    auto m = U3Matrix(theta, phi, lambda + M_PI_2);
    tensor::ops::set(&m, tensor::ops::zeros(1, m.dtype), 0);
    tensor::ops::set(&m, tensor::ops::zeros(1, m.dtype), 2);
    return m;
}

U3::U3(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
       const parameter::ParameterResolver& lambda, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits)
    : theta(theta)
    , phi(phi)
    , lambda(lambda)
    , Parameterizable(GateID::U3, {theta, phi, lambda}, obj_qubits, ctrl_qubits) {
    if (!this->parameterized_) {
        this->base_matrix_ = U3Matrix(theta.const_value, phi.const_value, lambda.const_value);
    }
}

FSim::FSim(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
           const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits)
    : theta(theta), phi(phi), Parameterizable(GateID::FSim, {theta, phi}, obj_qubits, ctrl_qubits) {
    if (!this->parameterized_) {
        this->base_matrix_ = FSimMatrix(theta.const_value, phi.const_value);
    }
}
}  // namespace mindquantum
