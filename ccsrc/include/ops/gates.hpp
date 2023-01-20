/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDQUANTUM_GATE_GATES_HPP_
#define MINDQUANTUM_GATE_GATES_HPP_

#include <cmath>

#include <functional>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "core/parameter_resolver.hpp"
#include "core/two_dim_matrix.hpp"
#include "core/utils.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"

#ifndef M_SQRT1_2
#    define M_SQRT1_2 0.707106781186547524400844362104849039
#endif  // !M_SQRT1_2

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif  // !M_PI

#ifndef M_PI_2
#    define M_PI_2 1.57079632679489661923132169163975144
#endif  // !M_PI_2

namespace mindquantum {
template <typename T>
Dim2Matrix<T> U3Matrix(T theta, T phi, T lambda) {
    auto ct_2 = std::cos(theta / 2);
    auto st_2 = std::sin(theta / 2);
    auto el = std::exp(std::complex<T>(0, lambda));
    auto ep = std::exp(std::complex<T>(0, phi));
    auto elp = el * ep;
    return Dim2Matrix<T>({{ct_2, -el * st_2}, {ep * st_2, elp * ct_2}});
}

template <typename T>
Dim2Matrix<T> FSimMatrix(T theta, T phi) {
    auto a = std::cos(theta);
    auto b = CT<T>(0, -std::sin(theta));
    auto c = std::exp(std::complex<T>(0, phi));
    return Dim2Matrix<T>({{1, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, c}});
}

template <typename T>
Dim2Matrix<T> U3DiffThetaMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta + static_cast<T>(M_PI), phi, lambda);
    Dim2MatrixBinary<T>(&m, 0.5, std::multiplies<CT<T>>());
    return m;
}
template <typename T>
Dim2Matrix<T> FSimDiffThetaMatrix(T theta) {
    auto a = -std::sin(theta);
    auto b = CT<T>(0, -std::cos(theta));
    return Dim2Matrix<T>({{0, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, 0}});
}

template <typename T>
Dim2Matrix<T> U3DiffPhiMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta, phi + static_cast<T>(M_PI_2), lambda);
    m.matrix_[0][0] = 0;
    m.matrix_[0][1] = 0;
    return m;
}

template <typename T>
Dim2Matrix<T> FSimDiffPhiMatrix(T phi) {
    auto c = std::exp(std::complex<T>(0, phi + static_cast<T>(M_PI_2)));
    return Dim2Matrix<T>({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, c}});
}

template <typename T>
Dim2Matrix<T> U3DiffLambdaMatrix(T theta, T phi, T lambda) {
    auto m = U3Matrix(theta, phi, lambda + static_cast<T>(M_PI_2));
    m.matrix_[0][0] = 0;
    m.matrix_[1][0] = 0;
    return m;
}

template <typename T>
struct U3 : public Parameterizable<T> {
    ParameterResolver<T> theta;
    ParameterResolver<T> phi;
    ParameterResolver<T> lambda;
    std::pair<MST<size_t>, Dim2Matrix<T>> jacobi;
    Dim2Matrix<T> base_matrix_;
    U3(const ParameterResolver<T>& theta, const ParameterResolver<T>& phi, const ParameterResolver<T>& lambda,
       const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits)
        : theta(theta)
        , phi(phi)
        , lambda(lambda)
        , Parameterizable<T>(GateID::U3, {theta, phi, lambda}, obj_qubits, ctrl_qubits) {
        if (!this->parameterized_) {
            this->base_matrix_ = U3Matrix(theta.const_value, phi.const_value, lambda.const_value);
        }
        jacobi = Jacobi(this->prs_);
    }
};

template <typename T>
struct FSim : public Parameterizable<T> {
    ParameterResolver<T> theta;
    ParameterResolver<T> phi;
    std::pair<MST<size_t>, Dim2Matrix<T>> jacobi;
    Dim2Matrix<T> base_matrix_;
    FSim(const ParameterResolver<T>& theta, const ParameterResolver<T>& phi, const VT<Index>& obj_qubits,
         const VT<Index>& ctrl_qubits)
        : theta(theta), phi(phi), Parameterizable<T>(GateID::FSim, {theta, phi}, obj_qubits, ctrl_qubits) {
        if (!this->parameterized_) {
            this->base_matrix_ = FSimMatrix(theta.const_value, phi.const_value);
        }
        jacobi = Jacobi(this->prs_);
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_GATES_HPP_
