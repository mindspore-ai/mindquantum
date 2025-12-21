/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef INCLUDE_SIMULATOR_MPS_COMPILER_MPS_COMPILER_HPP
#define INCLUDE_SIMULATOR_MPS_COMPILER_MPS_COMPILER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "simulator/mps/compiler/mps_program.h"

namespace mindquantum::sim::mps::compiler::detail {
template <typename complex_t>
using Mat2 = std::array<complex_t, 4>;
template <typename complex_t>
using Mat4 = std::array<complex_t, 16>;

template <typename complex_t>
constexpr Mat2<complex_t> Eye2() {
    Mat2<complex_t> out{};
    out[0] = complex_t{1, 0};
    out[1] = complex_t{0, 0};
    out[2] = complex_t{0, 0};
    out[3] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
constexpr Mat4<complex_t> Eye4() {
    Mat4<complex_t> out{};
    for (size_t i = 0; i < 16; ++i) {
        out[i] = complex_t{0, 0};
    }
    out[0] = complex_t{1, 0};
    out[5] = complex_t{1, 0};
    out[10] = complex_t{1, 0};
    out[15] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
inline Mat2<complex_t> MatMul(const Mat2<complex_t>& a, const Mat2<complex_t>& b) {
    Mat2<complex_t> c{};
    c[0] = a[0] * b[0] + a[1] * b[2];
    c[1] = a[0] * b[1] + a[1] * b[3];
    c[2] = a[2] * b[0] + a[3] * b[2];
    c[3] = a[2] * b[1] + a[3] * b[3];
    return c;
}

template <typename complex_t>
inline Mat4<complex_t> MatMul(const Mat4<complex_t>& a, const Mat4<complex_t>& b) {
    Mat4<complex_t> c{};
    for (size_t r = 0; r < 4; ++r) {
        for (size_t col = 0; col < 4; ++col) {
            complex_t sum{0, 0};
            for (size_t k = 0; k < 4; ++k) {
                sum += a[r * 4 + k] * b[k * 4 + col];
            }
            c[r * 4 + col] = sum;
        }
    }
    return c;
}

template <typename complex_t>
inline Mat4<complex_t> KronLeft(const Mat2<complex_t>& u) {
    // (u ⊗ I) in basis |00>,|01>,|10>,|11> (left,right)
    Mat4<complex_t> out{};
    for (size_t i = 0; i < 16; ++i) {
        out[i] = complex_t{0, 0};
    }
    // Acts on indices (00,10) and (01,11) in |00>,|01>,|10>,|11> ordering.
    out[0] = u[0];
    out[2] = u[1];
    out[8] = u[2];
    out[10] = u[3];

    out[5] = u[0];
    out[7] = u[1];
    out[13] = u[2];
    out[15] = u[3];
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> KronRight(const Mat2<complex_t>& u) {
    // (I ⊗ u) in basis |00>,|01>,|10>,|11> (left,right)
    Mat4<complex_t> out{};
    for (size_t i = 0; i < 16; ++i) {
        out[i] = complex_t{0, 0};
    }
    // left=0 block
    out[0] = u[0];
    out[1] = u[1];
    out[4] = u[2];
    out[5] = u[3];
    // left=1 block
    out[10] = u[0];
    out[11] = u[1];
    out[14] = u[2];
    out[15] = u[3];
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> SwapQubits(const Mat4<complex_t>& u_lr) {
    // Convert a 2Q matrix from order (left,right) to (right,left) (i.e. conjugate by SWAP permutation).
    // perm maps new basis index -> old basis index.
    constexpr std::array<size_t, 4> perm = {0, 2, 1, 3};
    Mat4<complex_t> out{};
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            out[r * 4 + c] = u_lr[perm[r] * 4 + perm[c]];
        }
    }
    return out;
}

template <typename complex_t>
constexpr Mat4<complex_t> SwapMatrix() {
    Mat4<complex_t> out{};
    for (size_t i = 0; i < 16; ++i) {
        out[i] = complex_t{0, 0};
    }
    out[0] = complex_t{1, 0};
    out[6] = complex_t{1, 0};
    out[9] = complex_t{1, 0};
    out[15] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
inline Mat2<complex_t> GateMatrix1Q(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                    bool diff) {
    if (diff) {
        throw std::runtime_error("mqmps does not support applying differential gates yet.");
    }
    using calc_t = typename complex_t::value_type;
    constexpr calc_t inv_sqrt2 = static_cast<calc_t>(0.7071067811865475244);
    constexpr complex_t I{0, 1};
    const auto id = gate->GetID();
    switch (id) {
        case GateID::I:
            return Eye2<complex_t>();
        case GateID::X:
            return Mat2<complex_t>{complex_t{0, 0}, complex_t{1, 0}, complex_t{1, 0}, complex_t{0, 0}};
        case GateID::Y:
            return Mat2<complex_t>{complex_t{0, 0}, -I, I, complex_t{0, 0}};
        case GateID::Z:
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{-1, 0}};
        case GateID::H:
            return Mat2<complex_t>{complex_t{inv_sqrt2, 0}, complex_t{inv_sqrt2, 0}, complex_t{inv_sqrt2, 0},
                                   complex_t{-inv_sqrt2, 0}};
        case GateID::S:
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, I};
        case GateID::Sdag:
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, -I};
        case GateID::T: {
            calc_t pi = static_cast<calc_t>(M_PI);
            complex_t phase = std::exp(I * (pi / static_cast<calc_t>(4)));
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, phase};
        }
        case GateID::Tdag: {
            calc_t pi = static_cast<calc_t>(M_PI);
            complex_t phase = std::exp(-I * (pi / static_cast<calc_t>(4)));
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, phase};
        }
        case GateID::SX: {
            // SX = sqrt(X)
            complex_t a{static_cast<calc_t>(0.5), static_cast<calc_t>(0.5)};
            complex_t b{static_cast<calc_t>(0.5), static_cast<calc_t>(-0.5)};
            return Mat2<complex_t>{a, b, b, a};
        }
        case GateID::SXdag: {
            complex_t a{static_cast<calc_t>(0.5), static_cast<calc_t>(-0.5)};
            complex_t b{static_cast<calc_t>(0.5), static_cast<calc_t>(0.5)};
            return Mat2<complex_t>{a, b, b, a};
        }
        case GateID::RX: {
            auto g = static_cast<RXGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            return Mat2<complex_t>{complex_t{c, 0}, -I * s, -I * s, complex_t{c, 0}};
        }
        case GateID::RY: {
            auto g = static_cast<RYGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            return Mat2<complex_t>{complex_t{c, 0}, complex_t{-s, 0}, complex_t{s, 0}, complex_t{c, 0}};
        }
        case GateID::RZ: {
            auto g = static_cast<RZGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            complex_t e0 = std::exp(-I * (val / static_cast<calc_t>(2)));
            complex_t e1 = std::exp(I * (val / static_cast<calc_t>(2)));
            return Mat2<complex_t>{e0, complex_t{0, 0}, complex_t{0, 0}, e1};
        }
        case GateID::PS: {
            auto g = static_cast<PSGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            complex_t e1 = std::exp(I * val);
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, e1};
        }
        case GateID::GP: {
            auto g = static_cast<GPGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            complex_t e = std::exp(-I * val);
            return Mat2<complex_t>{e, complex_t{0, 0}, complex_t{0, 0}, e};
        }
        case GateID::U3: {
            auto g = static_cast<U3*>(gate.get());
            tensor::Matrix m;
            if (!g->Parameterized()) {
                m = g->GetBaseMatrix();
            } else {
                auto theta = g->GetTheta().Combination(pr).const_value;
                auto phi = g->GetPhi().Combination(pr).const_value;
                auto lambda = g->GetLambda().Combination(pr).const_value;
                m = U3Matrix(theta, phi, lambda);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            // tensor::Matrix is row-major in nested vector form: mat[row][col]
            return Mat2<complex_t>{mat[0][0], mat[0][1], mat[1][0], mat[1][1]};
        }
        case GateID::Rn: {
            auto g = static_cast<Rn*>(gate.get());
            tensor::Matrix m;
            if (!g->Parameterized()) {
                m = g->GetBaseMatrix();
            } else {
                auto alpha = g->GetAlpha().Combination(pr).const_value;
                auto beta = g->GetBeta().Combination(pr).const_value;
                auto gamma = g->GetGamma().Combination(pr).const_value;
                m = RnMatrix(alpha, beta, gamma);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            return Mat2<complex_t>{mat[0][0], mat[0][1], mat[1][0], mat[1][1]};
        }
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix m;
            if (!g->Parameterized()) {
                m = g->GetBaseMatrix();
            } else {
                auto val = tensor::ops::cpu::to_vector<double>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
                m = g->GetMatrixWrapper()(val);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            if (mat.size() != 2 || mat[0].size() != 2) {
                throw std::runtime_error("mqmps only supports 1-qubit CUSTOM gate in this path.");
            }
            return Mat2<complex_t>{mat[0][0], mat[0][1], mat[1][0], mat[1][1]};
        }
        default:
            throw std::runtime_error(fmt::format("mqmps: unsupported 1-qubit gate id {}.", id));
    }
}

template <typename complex_t>
inline Mat4<complex_t> GateMatrix2QObjOrder12(const std::shared_ptr<BasicGate>& gate,
                                              const parameter::ParameterResolver& pr, bool diff) {
    // Returns matrix in basis |00>,|01>,|10>,|11> with qubit order (obj[1], obj[0]) for 2-object-qubit gates.
    if (diff) {
        throw std::runtime_error("mqmps does not support applying differential gates yet.");
    }
    using calc_t = typename complex_t::value_type;
    constexpr complex_t I{0, 1};
    const auto id = gate->GetID();
    switch (id) {
        case GateID::SWAP:
            return SwapMatrix<complex_t>();
        case GateID::ISWAP: {
            bool daggered = static_cast<ISWAPGate*>(gate.get())->Daggered();
            complex_t phase = daggered ? -I : I;
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{1, 0};
            out[6] = phase;
            out[9] = phase;
            out[15] = complex_t{1, 0};
            return out;
        }
        case GateID::SWAPalpha: {
            auto g = static_cast<SWAPalphaGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            complex_t e = std::exp(I * (static_cast<calc_t>(M_PI) * val));
            complex_t a = (complex_t{1, 0} + e) / static_cast<calc_t>(2);
            complex_t b = (complex_t{1, 0} - e) / static_cast<calc_t>(2);
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{1, 0};
            out[5] = a;
            out[6] = b;
            out[9] = b;
            out[10] = a;
            out[15] = complex_t{1, 0};
            return out;
        }
        case GateID::Rxx: {
            auto g = static_cast<RxxGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            complex_t is = -I * s;
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{c, 0};
            out[3] = is;
            out[5] = complex_t{c, 0};
            out[6] = is;
            out[9] = is;
            out[10] = complex_t{c, 0};
            out[12] = is;
            out[15] = complex_t{c, 0};
            return out;
        }
        case GateID::Ryy: {
            auto g = static_cast<RyyGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            complex_t is = -I * s;
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            // exp(-i θ/2 Y⊗Y)
            out[0] = complex_t{c, 0};
            out[3] = -is;
            out[5] = complex_t{c, 0};
            out[6] = is;
            out[9] = is;
            out[10] = complex_t{c, 0};
            out[12] = -is;
            out[15] = complex_t{c, 0};
            return out;
        }
        case GateID::Rzz: {
            auto g = static_cast<RzzGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            complex_t e0 = std::exp(-I * (val / static_cast<calc_t>(2)));
            complex_t e1 = std::exp(I * (val / static_cast<calc_t>(2)));
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = e0;
            out[5] = e1;
            out[10] = e1;
            out[15] = e0;
            return out;
        }
        case GateID::Rxy: {
            auto g = static_cast<RxyGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{c, 0};
            out[3] = complex_t{-s, 0};
            out[5] = complex_t{c, 0};
            out[6] = complex_t{-s, 0};
            out[9] = complex_t{s, 0};
            out[10] = complex_t{c, 0};
            out[12] = complex_t{s, 0};
            out[15] = complex_t{c, 0};
            return out;
        }
        case GateID::Rxz: {
            auto g = static_cast<RxzGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{c, 0};
            out[1] = -I * s;
            out[4] = -I * s;
            out[5] = complex_t{c, 0};
            out[10] = complex_t{c, 0};
            out[11] = I * s;
            out[14] = I * s;
            out[15] = complex_t{c, 0};
            return out;
        }
        case GateID::Ryz: {
            auto g = static_cast<RyzGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val / static_cast<calc_t>(2));
            calc_t s = std::sin(val / static_cast<calc_t>(2));
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{c, 0};
            out[1] = complex_t{-s, 0};
            out[4] = complex_t{s, 0};
            out[5] = complex_t{c, 0};
            out[10] = complex_t{c, 0};
            out[11] = complex_t{s, 0};
            out[14] = complex_t{-s, 0};
            out[15] = complex_t{c, 0};
            return out;
        }
        case GateID::Givens: {
            auto g = static_cast<GivensGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<calc_t>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            calc_t c = std::cos(val);
            calc_t s = std::sin(val);
            Mat4<complex_t> out{};
            for (size_t i = 0; i < 16; ++i) {
                out[i] = complex_t{0, 0};
            }
            out[0] = complex_t{1, 0};
            out[5] = complex_t{c, 0};
            out[6] = complex_t{-s, 0};
            out[9] = complex_t{s, 0};
            out[10] = complex_t{c, 0};
            out[15] = complex_t{1, 0};
            return out;
        }
        case GateID::FSim: {
            auto fsim = static_cast<FSim*>(gate.get());
            tensor::Matrix m;
            if (!fsim->Parameterized()) {
                m = fsim->GetBaseMatrix();
            } else {
                auto theta = fsim->GetTheta().Combination(pr).const_value;
                auto phi = fsim->GetPhi().Combination(pr).const_value;
                m = FSimMatrix(theta, phi);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            Mat4<complex_t> out{};
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    out[r * 4 + c] = mat[r][c];
                }
            }
            return out;
        }
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix m;
            if (!g->Parameterized()) {
                m = g->GetBaseMatrix();
            } else {
                auto val = tensor::ops::cpu::to_vector<double>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
                m = g->GetMatrixWrapper()(val);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            if (mat.size() != 4 || mat[0].size() != 4) {
                throw std::runtime_error("mqmps only supports 2-qubit CUSTOM gate in this path.");
            }
            Mat4<complex_t> out{};
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    out[r * 4 + c] = mat[r][c];
                }
            }
            return out;
        }
        case GateID::CUSTOM_TWO_PARAM: {
            auto g = static_cast<CustomTwoParamGate*>(gate.get());
            tensor::Matrix m;
            if (!g->Parameterized()) {
                m = g->GetBaseMatrix();
            } else {
                auto coeffs = g->GetCoeffs();
                auto p1 = tensor::ops::cpu::to_vector<double>(coeffs[0].Combination(pr).const_value)[0];
                auto p2 = tensor::ops::cpu::to_vector<double>(coeffs[1].Combination(pr).const_value)[0];
                m = g->GetMatrixWrapper()(p1, p2);
            }
            auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
            if (mat.size() != 4 || mat[0].size() != 4) {
                throw std::runtime_error("mqmps only supports 2-qubit CUSTOM_TWO_PARAM gate in this path.");
            }
            Mat4<complex_t> out{};
            for (size_t r = 0; r < 4; ++r) {
                for (size_t c = 0; c < 4; ++c) {
                    out[r * 4 + c] = mat[r][c];
                }
            }
            return out;
        }
        default:
            throw std::runtime_error(fmt::format("mqmps: unsupported 2-qubit gate id {}.", id));
    }
}

template <typename complex_t>
inline Mat4<complex_t> ControlledMatrix(const Mat2<complex_t>& u_target) {
    // Matrix in order (control, target), basis |00>,|01>,|10>,|11>.
    Mat4<complex_t> out = Eye4<complex_t>();
    // Replace bottom-right 2x2 with u_target
    out[10] = u_target[0];
    out[11] = u_target[1];
    out[14] = u_target[2];
    out[15] = u_target[3];
    return out;
}

template <typename complex_t>
inline std::vector<Stage<complex_t>> ScheduleUnitary(const std::vector<std::variant<Op1Q<complex_t>, Op2Q<complex_t>>>& ops,
                                                     qbit_t n_qubits) {
    std::vector<size_t> last_layer(static_cast<size_t>(n_qubits), 0);
    std::vector<Stage<complex_t>> stages;
    for (const auto& opv : ops) {
        std::vector<qbit_t> sites;
        size_t layer = 1;
        if (std::holds_alternative<Op1Q<complex_t>>(opv)) {
            const auto& op = std::get<Op1Q<complex_t>>(opv);
            sites = {op.pos};
        } else {
            const auto& op = std::get<Op2Q<complex_t>>(opv);
            sites = {op.bond, static_cast<qbit_t>(op.bond + 1)};
        }
        for (auto s : sites) {
            layer = std::max(layer, last_layer.at(static_cast<size_t>(s)) + 1);
        }
        if (stages.size() < layer) {
            stages.resize(layer);
        }
        if (std::holds_alternative<Op1Q<complex_t>>(opv)) {
            stages[layer - 1].op1.push_back(std::get<Op1Q<complex_t>>(opv));
        } else {
            stages[layer - 1].op2.push_back(std::get<Op2Q<complex_t>>(opv));
        }
        for (auto s : sites) {
            last_layer.at(static_cast<size_t>(s)) = layer;
        }
    }
    return stages;
}
}  // namespace mindquantum::sim::mps::compiler::detail

namespace mindquantum::sim::mps::compiler {
template <typename calc_t>
class MPSCompiler {
 public:
    using complex_t = std::complex<calc_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;

    explicit MPSCompiler(qbit_t n_qubits) : n_qubits_(n_qubits) {
    }

    MPSProgram<complex_t> Compile(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                  const qbits_t& pos_of_logical_in, const qbits_t& logical_at_pos_in,
                                  bool diff = false) const {
        using detail::Eye2;
        using detail::Eye4;
        using detail::GateMatrix1Q;
        using detail::GateMatrix2QObjOrder12;
        using detail::KronLeft;
        using detail::KronRight;
        using detail::Mat2;
        using detail::Mat4;
        using detail::MatMul;
        using detail::ScheduleUnitary;
        using detail::SwapMatrix;
        using detail::SwapQubits;

        if (pos_of_logical_in.size() != static_cast<size_t>(n_qubits_)
            || logical_at_pos_in.size() != static_cast<size_t>(n_qubits_)) {
            throw std::runtime_error("mqmps compiler: invalid mapping size.");
        }

        qbits_t pos_of_logical = pos_of_logical_in;
        qbits_t logical_at_pos = logical_at_pos_in;

        std::vector<Mat2<complex_t>> pending1(static_cast<size_t>(n_qubits_), Eye2<complex_t>());
        std::vector<bool> pending1_active(static_cast<size_t>(n_qubits_), false);
        std::vector<Mat4<complex_t>> pending2(static_cast<size_t>(std::max<qbit_t>(0, n_qubits_ - 1)), Eye4<complex_t>());
        std::vector<bool> pending2_active(static_cast<size_t>(std::max<qbit_t>(0, n_qubits_ - 1)), false);

        using UnitOp = std::variant<Op1Q<complex_t>, Op2Q<complex_t>>;
        std::vector<Stage<complex_t>> program_stages;
        std::vector<UnitOp> unit_ops_segment;
        std::vector<OpMeasure> measure_ops;
        bool seen_measure = false;

        auto flush_pending1_pos = [&](qbit_t pos) {
            if (pos < 0 || pos >= n_qubits_) {
                return;
            }
            auto idx = static_cast<size_t>(pos);
            if (!pending1_active[idx]) {
                return;
            }
            unit_ops_segment.emplace_back(Op1Q<complex_t>{pos, pending1[idx]});
            pending1[idx] = Eye2<complex_t>();
            pending1_active[idx] = false;
        };

        auto flush_pending2_bond = [&](qbit_t bond) {
            if (bond < 0 || bond >= n_qubits_ - 1) {
                return;
            }
            auto idx = static_cast<size_t>(bond);
            if (!pending2_active[idx]) {
                return;
            }
            unit_ops_segment.emplace_back(Op2Q<complex_t>{bond, pending2[idx]});
            pending2[idx] = Eye4<complex_t>();
            pending2_active[idx] = false;
        };

        auto flush_pending_for_sites = [&](qbit_t left, qbit_t right) {
            flush_pending1_pos(left);
            flush_pending1_pos(right);
            flush_pending2_bond(left - 1);
            flush_pending2_bond(left);
            flush_pending2_bond(right);
        };

        auto flush_conflicting_bonds_for_2q = [&](qbit_t bond) {
            flush_pending2_bond(bond - 1);
            flush_pending2_bond(bond + 1);
        };

        auto flush_all_pending = [&]() {
            for (qbit_t b = 0; b < n_qubits_ - 1; ++b) {
                flush_pending2_bond(b);
            }
            for (qbit_t p = 0; p < n_qubits_; ++p) {
                flush_pending1_pos(p);
            }
        };

        auto emit_current_segment_stages = [&]() {
            if (unit_ops_segment.empty()) {
                return;
            }
            auto stages = ScheduleUnitary(unit_ops_segment, n_qubits_);
            program_stages.insert(program_stages.end(), stages.begin(), stages.end());
            unit_ops_segment.clear();
        };

        auto swap_positions = [&](qbit_t bond) {
            // Ensure no pending ops touch this bond.
            flush_pending_for_sites(bond, bond + 1);
            // Emit SWAP as a standalone 2Q op.
            unit_ops_segment.emplace_back(Op2Q<complex_t>{bond, SwapMatrix<complex_t>()});

            // Update mapping.
            auto lq = logical_at_pos.at(static_cast<size_t>(bond));
            auto rq = logical_at_pos.at(static_cast<size_t>(bond + 1));
            std::swap(logical_at_pos.at(static_cast<size_t>(bond)), logical_at_pos.at(static_cast<size_t>(bond + 1)));
            pos_of_logical.at(static_cast<size_t>(lq)) = bond + 1;
            pos_of_logical.at(static_cast<size_t>(rq)) = bond;
        };

        auto route_to_adjacent = [&](qbit_t qa, qbit_t qb) {
            while (std::abs(pos_of_logical.at(static_cast<size_t>(qa)) - pos_of_logical.at(static_cast<size_t>(qb))) > 1) {
                auto pa = pos_of_logical.at(static_cast<size_t>(qa));
                auto pb = pos_of_logical.at(static_cast<size_t>(qb));
                if (pa < pb) {
                    swap_positions(pb - 1);
                } else {
                    swap_positions(pa - 1);
                }
            }
        };

        auto absorb_1q_on_pos = [&](qbit_t pos, const Mat2<complex_t>& u) {
            // If a pending 2Q touches this pos, absorb into that bond.
            if (pos > 0 && pending2_active.at(static_cast<size_t>(pos - 1))) {
                auto b = pos - 1;
                auto factor = KronRight(u);  // pos is right site of bond b
                pending2.at(static_cast<size_t>(b)) = MatMul(factor, pending2.at(static_cast<size_t>(b)));
                pending2_active.at(static_cast<size_t>(b)) = true;
                return;
            }
            if (pos < n_qubits_ - 1 && pending2_active.at(static_cast<size_t>(pos))) {
                auto b = pos;
                auto factor = KronLeft(u);  // pos is left site of bond b
                pending2.at(static_cast<size_t>(b)) = MatMul(factor, pending2.at(static_cast<size_t>(b)));
                pending2_active.at(static_cast<size_t>(b)) = true;
                return;
            }
            // Otherwise accumulate into pending 1Q.
            auto idx = static_cast<size_t>(pos);
            pending1[idx] = MatMul(u, pending1[idx]);
            pending1_active[idx] = true;
        };

        auto apply_adjacent_2q_on_bond = [&](qbit_t bond, Mat4<complex_t> u_lr) {
            // Flush neighbor bonds that would conflict.
            flush_conflicting_bonds_for_2q(bond);

            // Fold in pending 1Q gates on the two sites.
            auto left = bond;
            auto right = bond + 1;
            auto kron_lr = detail::Mat4<complex_t>{};
            // kron(P1_left, P1_right) in basis (left,right)
            // Build explicitly from the two 2x2 matrices.
            // kron = [[a00*b, a01*b],[a10*b,a11*b]] with b=P1_right.
            const auto& a = pending1.at(static_cast<size_t>(left));
            const auto& b = pending1.at(static_cast<size_t>(right));
            for (size_t i = 0; i < 16; ++i) {
                kron_lr[i] = complex_t{0, 0};
            }
            // block (0,0)
            kron_lr[0] = a[0] * b[0];
            kron_lr[1] = a[0] * b[1];
            kron_lr[4] = a[0] * b[2];
            kron_lr[5] = a[0] * b[3];
            // block (0,1)
            kron_lr[2] = a[1] * b[0];
            kron_lr[3] = a[1] * b[1];
            kron_lr[6] = a[1] * b[2];
            kron_lr[7] = a[1] * b[3];
            // block (1,0)
            kron_lr[8] = a[2] * b[0];
            kron_lr[9] = a[2] * b[1];
            kron_lr[12] = a[2] * b[2];
            kron_lr[13] = a[2] * b[3];
            // block (1,1)
            kron_lr[10] = a[3] * b[0];
            kron_lr[11] = a[3] * b[1];
            kron_lr[14] = a[3] * b[2];
            kron_lr[15] = a[3] * b[3];

            pending1.at(static_cast<size_t>(left)) = Eye2<complex_t>();
            pending1.at(static_cast<size_t>(right)) = Eye2<complex_t>();
            pending1_active.at(static_cast<size_t>(left)) = false;
            pending1_active.at(static_cast<size_t>(right)) = false;

            u_lr = MatMul(u_lr, kron_lr);

            // Accumulate into pending 2Q on that bond.
            pending2.at(static_cast<size_t>(bond)) = MatMul(u_lr, pending2.at(static_cast<size_t>(bond)));
            pending2_active.at(static_cast<size_t>(bond)) = true;
        };

        // -------------------------
        // Main compilation loop
        // -------------------------
        for (const auto& g : circ) {
            const auto id = g->GetID();
            if (id == GateID::I) {
                continue;
            }
            if (id == GateID::M) {
                // Measurement is only supported at the end of the circuit.
                if (!seen_measure) {
                    flush_all_pending();
                    emit_current_segment_stages();
                    seen_measure = true;
                }
                auto mg = static_cast<MeasureGate*>(g.get());
                const auto q = g->GetObjQubits().at(0);
                const auto pos = pos_of_logical.at(static_cast<size_t>(q));
                measure_ops.push_back(OpMeasure{mg->Name(), pos, mg->WillReset(), mg->GetRestTo()});
                continue;
            }
            if (seen_measure) {
                throw std::runtime_error("mqmps: mid-circuit measurement is not supported.");
            }

            // Multi-qubit pauli string gate: expand to local 1Q ops (no routing, no entanglement).
            if (id == GateID::PauliString) {
                auto ps = static_cast<PauliString*>(g.get());
                if (ps->GetCtrlQubits().size() != 0) {
                    throw std::runtime_error("mqmps: controlled PauliString is not supported.");
                }
                auto mask = ps->GetPauliMask();
                for (qbit_t q = 0; q < n_qubits_; ++q) {
                    char p = '\0';
                    if ((mask.mask_x >> q) & 1U) {
                        p = 'X';
                    }
                    if ((mask.mask_y >> q) & 1U) {
                        p = 'Y';
                    }
                    if ((mask.mask_z >> q) & 1U) {
                        p = 'Z';
                    }
                    if (p == '\0') {
                        continue;
                    }
                    std::shared_ptr<BasicGate> local;
                    if (p == 'X') {
                        local = std::make_shared<XGate>(qbits_t{q}, qbits_t{});
                    } else if (p == 'Y') {
                        local = std::make_shared<YGate>(qbits_t{q}, qbits_t{});
                    } else {
                        local = std::make_shared<ZGate>(qbits_t{q}, qbits_t{});
                    }
                    auto u = GateMatrix1Q<complex_t>(local, pr, diff);
                    absorb_1q_on_pos(pos_of_logical.at(static_cast<size_t>(q)), u);
                }
                continue;
            }

            const auto& obj = g->GetObjQubits();
            const auto& ctrl = g->GetCtrlQubits();
            const auto total_qubits = obj.size() + ctrl.size();

            if (total_qubits == 1 && ctrl.empty()) {
                auto q = obj.at(0);
                auto pos = pos_of_logical.at(static_cast<size_t>(q));
                auto u = GateMatrix1Q<complex_t>(g, pr, diff);
                absorb_1q_on_pos(pos, u);
                continue;
            }

            if (total_qubits != 2) {
                throw std::runtime_error(fmt::format("mqmps: only 1Q/2Q gates supported, got id {} with {} qubits.", id,
                                                     total_qubits));
            }

            // 2-qubit gate handling.
            qbit_t q_first = -1;
            qbit_t q_second = -1;
            Mat4<complex_t> u_qfirst_qsecond{};

            if (obj.size() == 2 && ctrl.empty()) {
                // MindQuantum convention: matrix is for (obj[1], obj[0]).
                q_first = obj.at(1);
                q_second = obj.at(0);
                u_qfirst_qsecond = GateMatrix2QObjOrder12<complex_t>(g, pr, diff);
            } else if (obj.size() == 1 && ctrl.size() == 1) {
                // Controlled single-qubit gate: build matrix in (control, target) order.
                q_first = ctrl.at(0);
                q_second = obj.at(0);
                Mat2<complex_t> u_target = Eye2<complex_t>();
                std::shared_ptr<BasicGate> bare;
                switch (id) {
                    case GateID::X:
                        bare = std::make_shared<XGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::Y:
                        bare = std::make_shared<YGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::Z:
                        bare = std::make_shared<ZGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::H:
                        bare = std::make_shared<HGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::S:
                        bare = std::make_shared<SGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::Sdag:
                        bare = std::make_shared<SdagGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::T:
                        bare = std::make_shared<TGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::Tdag:
                        bare = std::make_shared<TdagGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::SX:
                        bare = std::make_shared<SXGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::SXdag:
                        bare = std::make_shared<SXdagGate>(qbits_t{q_second}, qbits_t{});
                        break;
                    case GateID::RX: {
                        auto gg = static_cast<RXGate*>(g.get());
                        bare = std::make_shared<RXGate>(gg->GetCoeffs()[0], qbits_t{q_second}, qbits_t{});
                        break;
                    }
                    case GateID::RY: {
                        auto gg = static_cast<RYGate*>(g.get());
                        bare = std::make_shared<RYGate>(gg->GetCoeffs()[0], qbits_t{q_second}, qbits_t{});
                        break;
                    }
                    case GateID::RZ: {
                        auto gg = static_cast<RZGate*>(g.get());
                        bare = std::make_shared<RZGate>(gg->GetCoeffs()[0], qbits_t{q_second}, qbits_t{});
                        break;
                    }
                    case GateID::PS: {
                        auto gg = static_cast<PSGate*>(g.get());
                        bare = std::make_shared<PSGate>(gg->GetCoeffs()[0], qbits_t{q_second}, qbits_t{});
                        break;
                    }
                    case GateID::GP: {
                        auto gg = static_cast<GPGate*>(g.get());
                        bare = std::make_shared<GPGate>(gg->GetCoeffs()[0], qbits_t{q_second}, qbits_t{});
                        break;
                    }
                    default:
                        throw std::runtime_error(fmt::format("mqmps: controlled gate id {} not supported.", id));
                }
                u_target = GateMatrix1Q<complex_t>(bare, pr, diff);
                u_qfirst_qsecond = detail::ControlledMatrix<complex_t>(u_target);
            } else {
                throw std::runtime_error("mqmps: unsupported 2-qubit gate form.");
            }

            // Route the two logical qubits to be adjacent.
            route_to_adjacent(q_first, q_second);

            // Determine the bond and physical ordering.
            const auto p_first = pos_of_logical.at(static_cast<size_t>(q_first));
            const auto p_second = pos_of_logical.at(static_cast<size_t>(q_second));
            const auto bond = std::min(p_first, p_second);
            const auto left_pos = bond;
            const auto right_pos = bond + 1;
            const auto q_left = logical_at_pos.at(static_cast<size_t>(left_pos));
            const auto q_right = logical_at_pos.at(static_cast<size_t>(right_pos));

            Mat4<complex_t> u_lr = u_qfirst_qsecond;
            if (q_left == q_first && q_right == q_second) {
                // ok
            } else if (q_left == q_second && q_right == q_first) {
                u_lr = SwapQubits(u_lr);
            } else {
                throw std::runtime_error("mqmps: internal routing error (qubits not on the expected bond).");
            }

            apply_adjacent_2q_on_bond(bond, u_lr);
        }

        // Final flush.
        flush_all_pending();
        emit_current_segment_stages();
        if (!measure_ops.empty()) {
            Stage<complex_t> st;
            st.meas = std::move(measure_ops);
            program_stages.push_back(std::move(st));
        }

        MPSProgram<complex_t> program;
        program.stages = std::move(program_stages);
        program.pos_of_logical_final = std::move(pos_of_logical);
        program.logical_at_pos_final = std::move(logical_at_pos);
        return program;
    }

 private:
    qbit_t n_qubits_ = 0;
};
}  // namespace mindquantum::sim::mps::compiler

#endif  // INCLUDE_SIMULATOR_MPS_COMPILER_MPS_COMPILER_HPP
