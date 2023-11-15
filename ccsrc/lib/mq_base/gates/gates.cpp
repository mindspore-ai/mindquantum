/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/gates.h"

#include <utility>

#include "core/mq_base_types.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/advance_math.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"

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

tensor::Matrix RnMatrix(tensor::Tensor alpha, tensor::Tensor beta, tensor::Tensor gamma) {
    auto cmplx = tensor::ToComplexType(alpha.dtype);
    auto c = tensor::ops::sqrt(alpha * alpha + beta * beta + gamma * gamma);
    auto cx = (alpha / c).astype(cmplx);
    auto cy = (beta / c).astype(cmplx);
    auto cz = (gamma / c).astype(cmplx);
    auto im = tensor::Tensor(std::complex<float>(0, 1), cmplx);
    auto zero = tensor::ops::zeros(1, c.dtype);
    auto cc_2 = tensor::ops::cos(c / 2.0).astype(c.dtype);
    auto sc_2 = tensor::ops::sin(c / 2.0).astype(c.dtype);
    auto v1 = tensor::ops::gather({cc_2, zero, zero, cc_2});
    auto v2 = tensor::ops::gather({cz, cx - im * cy, cx + im * cy, zero - cz});
    auto m = v1 - v2 * im * sc_2;
    return tensor::Matrix(std::move(m), 2, 2);
}
tensor::Matrix RnDiffAlphaMatrix(tensor::Tensor a, tensor::Tensor b, tensor::Tensor c) {
    auto cmplx = tensor::ToComplexType(a.dtype);
    auto f = tensor::ops::sqrt(a * a + b * b + c * c);
    auto fa = a / f;
    auto cc_2 = tensor::ops::cos(f / 2.0).astype(c.dtype);
    auto sc_2 = tensor::ops::sin(f / 2.0).astype(c.dtype);
    auto im = tensor::Tensor(std::complex<float>(0, 1), cmplx);
    float two = 2.0;
    auto sigma_i = tensor::Tensor(VT<CT<double>>({1, 0, 0, 1}), cmplx);
    auto sigma_x = tensor::Tensor(VT<CT<double>>({0, 1, 1, 0}), cmplx);
    auto sigma_y = tensor::Tensor(VT<CT<double>>({0, {0, -1}, {0, 1}, 0}), cmplx);
    auto sigma_z = tensor::Tensor(VT<CT<double>>({1, 0, 0, -1}), cmplx);
    auto h = sigma_x * a + sigma_y * b + sigma_z * c;
    auto m = sigma_i * fa / -two * sc_2 - h * fa * im * (cc_2 / two / f - sc_2 / f / f) - sigma_x * im * sc_2 / f;
    return tensor::Matrix(std::move(m), 2, 2);
}

tensor::Matrix RnDiffBetaMatrix(tensor::Tensor a, tensor::Tensor b, tensor::Tensor c) {
    auto cmplx = tensor::ToComplexType(a.dtype);
    auto f = tensor::ops::sqrt(a * a + b * b + c * c);
    auto fb = b / f;
    auto cc_2 = tensor::ops::cos(f / 2.0).astype(c.dtype);
    auto sc_2 = tensor::ops::sin(f / 2.0).astype(c.dtype);
    auto im = tensor::Tensor(std::complex<float>(0, 1), cmplx);
    float two = 2.0;
    auto sigma_i = tensor::Tensor(VT<CT<double>>({1, 0, 0, 1}), cmplx);
    auto sigma_x = tensor::Tensor(VT<CT<double>>({0, 1, 1, 0}), cmplx);
    auto sigma_y = tensor::Tensor(VT<CT<double>>({0, {0, -1}, {0, 1}, 0}), cmplx);
    auto sigma_z = tensor::Tensor(VT<CT<double>>({1, 0, 0, -1}), cmplx);
    auto h = sigma_x * a + sigma_y * b + sigma_z * c;
    auto m = sigma_i * fb / -two * sc_2 - h * fb * im * (cc_2 / two / f - sc_2 / f / f) - sigma_y * im * sc_2 / f;
    return tensor::Matrix(std::move(m), 2, 2);
}

tensor::Matrix RnDiffGammaMatrix(tensor::Tensor a, tensor::Tensor b, tensor::Tensor c) {
    auto cmplx = tensor::ToComplexType(a.dtype);
    auto f = tensor::ops::sqrt(a * a + b * b + c * c);
    auto fc = c / f;
    auto cc_2 = tensor::ops::cos(f / 2.0).astype(c.dtype);
    auto sc_2 = tensor::ops::sin(f / 2.0).astype(c.dtype);
    auto im = tensor::Tensor(std::complex<float>(0, 1), cmplx);
    float two = 2.0;
    auto sigma_i = tensor::Tensor(VT<CT<double>>({1, 0, 0, 1}), cmplx);
    auto sigma_x = tensor::Tensor(VT<CT<double>>({0, 1, 1, 0}), cmplx);
    auto sigma_y = tensor::Tensor(VT<CT<double>>({0, {0, -1}, {0, 1}, 0}), cmplx);
    auto sigma_z = tensor::Tensor(VT<CT<double>>({1, 0, 0, -1}), cmplx);
    auto h = sigma_x * a + sigma_y * b + sigma_z * c;
    auto m = sigma_i * fc / -two * sc_2 - h * fc * im * (cc_2 / two / f - sc_2 / f / f) - sigma_z * im * sc_2 / f;
    return tensor::Matrix(std::move(m), 2, 2);
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
       const parameter::ParameterResolver& lambda, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits)
    : Parameterizable(GateID::U3, {theta, phi, lambda}, obj_qubits, ctrl_qubits)
    , theta(theta)
    , phi(phi)
    , lambda(lambda) {
    if (!this->parameterized_) {
        this->base_matrix_ = U3Matrix(theta.const_value, phi.const_value, lambda.const_value);
    }
}

Rn::Rn(const parameter::ParameterResolver& alpha, const parameter::ParameterResolver& beta,
       const parameter::ParameterResolver& gamma, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits)
    : Parameterizable(GateID::Rn, {alpha, beta, gamma}, obj_qubits, ctrl_qubits)
    , alpha(alpha)
    , beta(beta)
    , gamma(gamma) {
    if (!this->parameterized_) {
        this->base_matrix_ = RnMatrix(alpha.const_value, beta.const_value, gamma.const_value);
    }
}

FSim::FSim(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
           const qbits_t& obj_qubits, const qbits_t& ctrl_qubits)
    : Parameterizable(GateID::FSim, {theta, phi}, obj_qubits, ctrl_qubits), theta(theta), phi(phi) {
    if (!this->parameterized_) {
        this->base_matrix_ = FSimMatrix(theta.const_value, phi.const_value);
    }
}
}  // namespace mindquantum
