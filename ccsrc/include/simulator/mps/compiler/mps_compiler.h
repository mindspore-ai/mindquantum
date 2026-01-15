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

#include <cmath>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
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
inline Mat4<complex_t> Kron(const Mat2<complex_t>& u_left, const Mat2<complex_t>& u_right) {
    // (u_left ⊗ u_right) in basis |00>,|01>,|10>,|11> (left,right)
    Mat4<complex_t> out{};
    // block (0,0)
    out[0] = u_left[0] * u_right[0];
    out[1] = u_left[0] * u_right[1];
    out[4] = u_left[0] * u_right[2];
    out[5] = u_left[0] * u_right[3];
    // block (0,1)
    out[2] = u_left[1] * u_right[0];
    out[3] = u_left[1] * u_right[1];
    out[6] = u_left[1] * u_right[2];
    out[7] = u_left[1] * u_right[3];
    // block (1,0)
    out[8] = u_left[2] * u_right[0];
    out[9] = u_left[2] * u_right[1];
    out[12] = u_left[2] * u_right[2];
    out[13] = u_left[2] * u_right[3];
    // block (1,1)
    out[10] = u_left[3] * u_right[0];
    out[11] = u_left[3] * u_right[1];
    out[14] = u_left[3] * u_right[2];
    out[15] = u_left[3] * u_right[3];
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
constexpr Mat2<complex_t> XMatrix1Q() {
    return Mat2<complex_t>{complex_t{0, 0}, complex_t{1, 0}, complex_t{1, 0}, complex_t{0, 0}};
}

template <typename complex_t>
constexpr Mat2<complex_t> YMatrix1Q() {
    return Mat2<complex_t>{complex_t{0, 0}, complex_t{0, -1}, complex_t{0, 1}, complex_t{0, 0}};
}

template <typename complex_t>
constexpr Mat2<complex_t> ZMatrix1Q() {
    return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{-1, 0}};
}

template <typename complex_t>
constexpr Mat2<complex_t> HMatrix1Q() {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t inv_sqrt2 = static_cast<calc_t>(M_SQRT1_2);
    return Mat2<complex_t>{complex_t{inv_sqrt2, 0}, complex_t{inv_sqrt2, 0}, complex_t{inv_sqrt2, 0},
                           complex_t{-inv_sqrt2, 0}};
}

template <typename complex_t>
constexpr Mat2<complex_t> SMatrix1Q() {
    return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 1}};
}

template <typename complex_t>
constexpr Mat2<complex_t> SdagMatrix1Q() {
    return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{0, -1}};
}

template <typename complex_t>
constexpr Mat2<complex_t> TMatrix1Q() {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t inv_sqrt2 = static_cast<calc_t>(M_SQRT1_2);
    return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{inv_sqrt2, inv_sqrt2}};
}

template <typename complex_t>
constexpr Mat2<complex_t> TdagMatrix1Q() {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t inv_sqrt2 = static_cast<calc_t>(M_SQRT1_2);
    return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, complex_t{inv_sqrt2, -inv_sqrt2}};
}

template <typename complex_t>
constexpr Mat2<complex_t> SXMatrix1Q() {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t half = static_cast<calc_t>(0.5);
    return Mat2<complex_t>{complex_t{half, half}, complex_t{half, -half}, complex_t{half, -half},
                           complex_t{half, half}};
}

template <typename complex_t>
constexpr Mat2<complex_t> SXdagMatrix1Q() {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t half = static_cast<calc_t>(0.5);
    return Mat2<complex_t>{complex_t{half, -half}, complex_t{half, half}, complex_t{half, half},
                           complex_t{half, -half}};
}

template <typename complex_t>
inline Mat2<complex_t> ToMat2(const tensor::Matrix& m) {
    const auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
    // tensor::Matrix is row-major in nested vector form: mat[row][col]
    return Mat2<complex_t>{mat[0][0], mat[0][1], mat[1][0], mat[1][1]};
}

template <typename complex_t>
inline Mat4<complex_t> ToMat4(const tensor::Matrix& m) {
    const auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
    Mat4<complex_t> out{};
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            out[r * 4 + c] = mat[r][c];
        }
    }
    return out;
}

template <typename calc_t, typename gate_t>
inline calc_t GetSingleCoeffValue(const gate_t* gate, const parameter::ParameterResolver& pr) {
    return tensor::ops::cpu::to_vector<calc_t>(gate->GetCoeffs()[0].Combination(pr).const_value)[0];
}

template <typename complex_t>
inline Mat2<complex_t> GateMatrix1QParam(const std::shared_ptr<BasicGate>& gate,
                                         const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto id = gate->GetID();
    switch (id) {
        case GateID::RX: {
            const auto val = GetSingleCoeffValue<calc_t>(static_cast<RXGate*>(gate.get()), pr);
            const calc_t c = std::cos(val / static_cast<calc_t>(2));
            const calc_t s = std::sin(val / static_cast<calc_t>(2));
            return Mat2<complex_t>{complex_t{c, 0}, -I * s, -I * s, complex_t{c, 0}};
        }
        case GateID::RY: {
            const auto val = GetSingleCoeffValue<calc_t>(static_cast<RYGate*>(gate.get()), pr);
            const calc_t c = std::cos(val / static_cast<calc_t>(2));
            const calc_t s = std::sin(val / static_cast<calc_t>(2));
            return Mat2<complex_t>{complex_t{c, 0}, complex_t{-s, 0}, complex_t{s, 0}, complex_t{c, 0}};
        }
        case GateID::RZ: {
            const auto val = GetSingleCoeffValue<calc_t>(static_cast<RZGate*>(gate.get()), pr);
            const complex_t e0 = std::exp(-I * (val / static_cast<calc_t>(2)));
            const complex_t e1 = std::exp(I * (val / static_cast<calc_t>(2)));
            return Mat2<complex_t>{e0, complex_t{0, 0}, complex_t{0, 0}, e1};
        }
        case GateID::PS: {
            const auto val = GetSingleCoeffValue<calc_t>(static_cast<PSGate*>(gate.get()), pr);
            const complex_t e1 = std::exp(I * val);
            return Mat2<complex_t>{complex_t{1, 0}, complex_t{0, 0}, complex_t{0, 0}, e1};
        }
        case GateID::GP: {
            const auto val = GetSingleCoeffValue<calc_t>(static_cast<GPGate*>(gate.get()), pr);
            const complex_t e = std::exp(-I * val);
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
            return ToMat2<complex_t>(m);
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
            return ToMat2<complex_t>(m);
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
            const auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
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
inline Mat2<complex_t> GateMatrix1Q(const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                    bool diff) {
    if (diff) {
        // Match mqvector semantics: only apply the differential form when the gate is marked as grad-required.
        // This keeps `diff=True` as an internal/advanced feature without affecting normal circuit execution.
        if (!gate->GradRequired()) {
            diff = false;
        } else {
            using calc_t = typename complex_t::value_type;
            const complex_t I{0, 1};
            constexpr calc_t half = static_cast<calc_t>(0.5);
            const auto id = gate->GetID();
            switch (id) {
                case GateID::RX: {
                    const auto val = GetSingleCoeffValue<calc_t>(static_cast<RXGate*>(gate.get()), pr);
                    const calc_t c = std::cos(val * half);
                    const calc_t s = std::sin(val * half);
                    const calc_t dc = -s * half;
                    const calc_t ds = c * half;
                    return Mat2<complex_t>{complex_t{dc, 0}, -I * ds, -I * ds, complex_t{dc, 0}};
                }
                case GateID::RY: {
                    const auto val = GetSingleCoeffValue<calc_t>(static_cast<RYGate*>(gate.get()), pr);
                    const calc_t c = std::cos(val * half);
                    const calc_t s = std::sin(val * half);
                    const calc_t dc = -s * half;
                    const calc_t ds = c * half;
                    return Mat2<complex_t>{complex_t{dc, 0}, complex_t{-ds, 0}, complex_t{ds, 0}, complex_t{dc, 0}};
                }
                case GateID::RZ: {
                    const auto val = GetSingleCoeffValue<calc_t>(static_cast<RZGate*>(gate.get()), pr);
                    const complex_t e0 = std::exp(-I * (val * half));
                    const complex_t e1 = std::exp(I * (val * half));
                    return Mat2<complex_t>{(-I * half) * e0, complex_t{0, 0}, complex_t{0, 0}, (I * half) * e1};
                }
                case GateID::PS: {
                    const auto val = GetSingleCoeffValue<calc_t>(static_cast<PSGate*>(gate.get()), pr);
                    const complex_t e1 = std::exp(I * val);
                    return Mat2<complex_t>{complex_t{0, 0}, complex_t{0, 0}, complex_t{0, 0}, I * e1};
                }
                case GateID::GP: {
                    const auto val = GetSingleCoeffValue<calc_t>(static_cast<GPGate*>(gate.get()), pr);
                    const complex_t e = std::exp(-I * val);
                    return Mat2<complex_t>{-I * e, complex_t{0, 0}, complex_t{0, 0}, -I * e};
                }
                default:
                    throw std::runtime_error(fmt::format("mqmps: differential 1-qubit gate id {} not supported.", id));
            }
        }
    }
    const auto id = gate->GetID();
    switch (id) {
        case GateID::I:
            return Eye2<complex_t>();
        case GateID::X:
            return XMatrix1Q<complex_t>();
        case GateID::Y:
            return YMatrix1Q<complex_t>();
        case GateID::Z:
            return ZMatrix1Q<complex_t>();
        case GateID::H:
            return HMatrix1Q<complex_t>();
        case GateID::S:
            return SMatrix1Q<complex_t>();
        case GateID::Sdag:
            return SdagMatrix1Q<complex_t>();
        case GateID::T:
            return TMatrix1Q<complex_t>();
        case GateID::Tdag:
            return TdagMatrix1Q<complex_t>();
        case GateID::SX:
            return SXMatrix1Q<complex_t>();
        case GateID::SXdag:
            return SXdagMatrix1Q<complex_t>();
        default:
            return GateMatrix1QParam<complex_t>(gate, pr);
    }
}

template <typename complex_t>
inline Mat4<complex_t> ISWAPMatrix2Q(const ISWAPGate* gate) {
    const complex_t I{0, 1};
    const complex_t phase = gate->Daggered() ? -I : I;
    Mat4<complex_t> out{};
    out[0] = complex_t{1, 0};
    out[6] = phase;
    out[9] = phase;
    out[15] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> SWAPalphaMatrix2Q(const SWAPalphaGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const complex_t e = std::exp(I * (static_cast<calc_t>(M_PI) * val));
    const complex_t a = (complex_t{1, 0} + e) / static_cast<calc_t>(2);
    const complex_t b = (complex_t{1, 0} - e) / static_cast<calc_t>(2);
    Mat4<complex_t> out{};
    out[0] = complex_t{1, 0};
    out[5] = a;
    out[6] = b;
    out[9] = b;
    out[10] = a;
    out[15] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> SWAPalphaDiffMatrix2Q(const SWAPalphaGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const complex_t e = std::exp(I * (static_cast<calc_t>(M_PI) * val));
    const complex_t de = (I * static_cast<calc_t>(M_PI)) * e;
    const complex_t da = de / static_cast<calc_t>(2);
    const complex_t db = -de / static_cast<calc_t>(2);
    Mat4<complex_t> out{};
    out[5] = da;
    out[6] = db;
    out[9] = db;
    out[10] = da;
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RxxMatrix2Q(const RxxGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val / static_cast<calc_t>(2));
    const calc_t s = std::sin(val / static_cast<calc_t>(2));
    const complex_t is = -I * s;
    Mat4<complex_t> out{};
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

template <typename complex_t>
inline Mat4<complex_t> RxxDiffMatrix2Q(const RxxGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val * half);
    const calc_t s = std::sin(val * half);
    const calc_t dc = -s * half;
    const calc_t ds = c * half;
    const complex_t ids = -I * ds;
    Mat4<complex_t> out{};
    out[0] = complex_t{dc, 0};
    out[3] = ids;
    out[5] = complex_t{dc, 0};
    out[6] = ids;
    out[9] = ids;
    out[10] = complex_t{dc, 0};
    out[12] = ids;
    out[15] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RyyMatrix2Q(const RyyGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val / static_cast<calc_t>(2));
    const calc_t s = std::sin(val / static_cast<calc_t>(2));
    const complex_t is = -I * s;
    Mat4<complex_t> out{};
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

template <typename complex_t>
inline Mat4<complex_t> RyyDiffMatrix2Q(const RyyGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val * half);
    const calc_t s = std::sin(val * half);
    const calc_t dc = -s * half;
    const calc_t ds = c * half;
    const complex_t ids = -I * ds;
    Mat4<complex_t> out{};
    out[0] = complex_t{dc, 0};
    out[3] = -ids;
    out[5] = complex_t{dc, 0};
    out[6] = ids;
    out[9] = ids;
    out[10] = complex_t{dc, 0};
    out[12] = -ids;
    out[15] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RzzMatrix2Q(const RzzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const complex_t e0 = std::exp(-I * (val / static_cast<calc_t>(2)));
    const complex_t e1 = std::exp(I * (val / static_cast<calc_t>(2)));
    Mat4<complex_t> out{};
    out[0] = e0;
    out[5] = e1;
    out[10] = e1;
    out[15] = e0;
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RzzDiffMatrix2Q(const RzzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const complex_t e0 = std::exp(-I * (val * half));
    const complex_t e1 = std::exp(I * (val * half));
    Mat4<complex_t> out{};
    out[0] = (-I * half) * e0;
    out[5] = (I * half) * e1;
    out[10] = (I * half) * e1;
    out[15] = (-I * half) * e0;
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RxyMatrix2Q(const RxyGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val / static_cast<calc_t>(2));
    const calc_t s = std::sin(val / static_cast<calc_t>(2));
    Mat4<complex_t> out{};
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

template <typename complex_t>
inline Mat4<complex_t> RxyDiffMatrix2Q(const RxyGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val * half);
    const calc_t s = std::sin(val * half);
    const calc_t dc = -s * half;
    const calc_t ds = c * half;
    Mat4<complex_t> out{};
    out[0] = complex_t{dc, 0};
    out[3] = complex_t{-ds, 0};
    out[5] = complex_t{dc, 0};
    out[6] = complex_t{-ds, 0};
    out[9] = complex_t{ds, 0};
    out[10] = complex_t{dc, 0};
    out[12] = complex_t{ds, 0};
    out[15] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RxzMatrix2Q(const RxzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val / static_cast<calc_t>(2));
    const calc_t s = std::sin(val / static_cast<calc_t>(2));
    Mat4<complex_t> out{};
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

template <typename complex_t>
inline Mat4<complex_t> RxzDiffMatrix2Q(const RxzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const complex_t I{0, 1};
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val * half);
    const calc_t s = std::sin(val * half);
    const calc_t dc = -s * half;
    const calc_t ds = c * half;
    const complex_t ids = -I * ds;
    Mat4<complex_t> out{};
    out[0] = complex_t{dc, 0};
    out[1] = ids;
    out[4] = ids;
    out[5] = complex_t{dc, 0};
    out[10] = complex_t{dc, 0};
    out[11] = -ids;
    out[14] = -ids;
    out[15] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> RyzMatrix2Q(const RyzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val / static_cast<calc_t>(2));
    const calc_t s = std::sin(val / static_cast<calc_t>(2));
    Mat4<complex_t> out{};
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

template <typename complex_t>
inline Mat4<complex_t> RyzDiffMatrix2Q(const RyzGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    constexpr calc_t half = static_cast<calc_t>(0.5);
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val * half);
    const calc_t s = std::sin(val * half);
    const calc_t dc = -s * half;
    const calc_t ds = c * half;
    Mat4<complex_t> out{};
    out[0] = complex_t{dc, 0};
    out[1] = complex_t{-ds, 0};
    out[4] = complex_t{ds, 0};
    out[5] = complex_t{dc, 0};
    out[10] = complex_t{dc, 0};
    out[11] = complex_t{ds, 0};
    out[14] = complex_t{-ds, 0};
    out[15] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> GivensMatrix2Q(const GivensGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val);
    const calc_t s = std::sin(val);
    Mat4<complex_t> out{};
    out[0] = complex_t{1, 0};
    out[5] = complex_t{c, 0};
    out[6] = complex_t{-s, 0};
    out[9] = complex_t{s, 0};
    out[10] = complex_t{c, 0};
    out[15] = complex_t{1, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> GivensDiffMatrix2Q(const GivensGate* gate, const parameter::ParameterResolver& pr) {
    using calc_t = typename complex_t::value_type;
    const auto val = GetSingleCoeffValue<calc_t>(gate, pr);
    const calc_t c = std::cos(val);
    const calc_t s = std::sin(val);
    const calc_t dc = -s;
    const calc_t ds = c;
    Mat4<complex_t> out{};
    out[5] = complex_t{dc, 0};
    out[6] = complex_t{-ds, 0};
    out[9] = complex_t{ds, 0};
    out[10] = complex_t{dc, 0};
    return out;
}

template <typename complex_t>
inline Mat4<complex_t> FSimMatrix2Q(const FSim* gate, const parameter::ParameterResolver& pr) {
    tensor::Matrix m;
    if (!gate->Parameterized()) {
        m = gate->GetBaseMatrix();
    } else {
        auto theta = gate->GetTheta().Combination(pr).const_value;
        auto phi = gate->GetPhi().Combination(pr).const_value;
        m = FSimMatrix(theta, phi);
    }
    return ToMat4<complex_t>(m);
}

template <typename complex_t>
inline Mat4<complex_t> CustomMatrix2Q(const CustomGate* gate, const parameter::ParameterResolver& pr) {
    tensor::Matrix m;
    if (!gate->Parameterized()) {
        m = gate->GetBaseMatrix();
    } else {
        auto val = tensor::ops::cpu::to_vector<double>(gate->GetCoeffs()[0].Combination(pr).const_value)[0];
        m = gate->GetMatrixWrapper()(val);
    }
    const auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
    if (mat.size() != 4 || mat[0].size() != 4) {
        throw std::runtime_error("mqmps only supports 2-qubit CUSTOM gate in this path.");
    }
    return ToMat4<complex_t>(m);
}

template <typename complex_t>
inline Mat4<complex_t> CustomTwoParamMatrix2Q(const CustomTwoParamGate* gate, const parameter::ParameterResolver& pr) {
    tensor::Matrix m;
    if (!gate->Parameterized()) {
        m = gate->GetBaseMatrix();
    } else {
        const auto coeffs = gate->GetCoeffs();
        const auto p1 = tensor::ops::cpu::to_vector<double>(coeffs[0].Combination(pr).const_value)[0];
        const auto p2 = tensor::ops::cpu::to_vector<double>(coeffs[1].Combination(pr).const_value)[0];
        m = gate->GetMatrixWrapper()(p1, p2);
    }
    const auto mat = tensor::ops::cpu::to_vector<complex_t>(m);
    if (mat.size() != 4 || mat[0].size() != 4) {
        throw std::runtime_error("mqmps only supports 2-qubit CUSTOM_TWO_PARAM gate in this path.");
    }
    return ToMat4<complex_t>(m);
}

template <typename complex_t>
inline Mat4<complex_t> GateMatrix2QObjOrder12Diff(const std::shared_ptr<BasicGate>& gate,
                                                  const parameter::ParameterResolver& pr) {
    const auto id = gate->GetID();
    switch (id) {
        case GateID::SWAPalpha:
            return SWAPalphaDiffMatrix2Q<complex_t>(static_cast<SWAPalphaGate*>(gate.get()), pr);
        case GateID::Rxx:
            return RxxDiffMatrix2Q<complex_t>(static_cast<RxxGate*>(gate.get()), pr);
        case GateID::Ryy:
            return RyyDiffMatrix2Q<complex_t>(static_cast<RyyGate*>(gate.get()), pr);
        case GateID::Rzz:
            return RzzDiffMatrix2Q<complex_t>(static_cast<RzzGate*>(gate.get()), pr);
        case GateID::Rxy:
            return RxyDiffMatrix2Q<complex_t>(static_cast<RxyGate*>(gate.get()), pr);
        case GateID::Rxz:
            return RxzDiffMatrix2Q<complex_t>(static_cast<RxzGate*>(gate.get()), pr);
        case GateID::Ryz:
            return RyzDiffMatrix2Q<complex_t>(static_cast<RyzGate*>(gate.get()), pr);
        case GateID::Givens:
            return GivensDiffMatrix2Q<complex_t>(static_cast<GivensGate*>(gate.get()), pr);
        default:
            throw std::runtime_error(fmt::format("mqmps: differential 2-qubit gate id {} not supported.", id));
    }
}

template <typename complex_t>
inline Mat4<complex_t> GateMatrix2QObjOrder12NoDiff(const std::shared_ptr<BasicGate>& gate,
                                                    const parameter::ParameterResolver& pr) {
    const auto id = gate->GetID();
    switch (id) {
        case GateID::SWAP:
            return SwapMatrix<complex_t>();
        case GateID::ISWAP:
            return ISWAPMatrix2Q<complex_t>(static_cast<ISWAPGate*>(gate.get()));
        case GateID::SWAPalpha:
            return SWAPalphaMatrix2Q<complex_t>(static_cast<SWAPalphaGate*>(gate.get()), pr);
        case GateID::Rxx:
            return RxxMatrix2Q<complex_t>(static_cast<RxxGate*>(gate.get()), pr);
        case GateID::Ryy:
            return RyyMatrix2Q<complex_t>(static_cast<RyyGate*>(gate.get()), pr);
        case GateID::Rzz:
            return RzzMatrix2Q<complex_t>(static_cast<RzzGate*>(gate.get()), pr);
        case GateID::Rxy:
            return RxyMatrix2Q<complex_t>(static_cast<RxyGate*>(gate.get()), pr);
        case GateID::Rxz:
            return RxzMatrix2Q<complex_t>(static_cast<RxzGate*>(gate.get()), pr);
        case GateID::Ryz:
            return RyzMatrix2Q<complex_t>(static_cast<RyzGate*>(gate.get()), pr);
        case GateID::Givens:
            return GivensMatrix2Q<complex_t>(static_cast<GivensGate*>(gate.get()), pr);
        case GateID::FSim:
            return FSimMatrix2Q<complex_t>(static_cast<FSim*>(gate.get()), pr);
        case GateID::CUSTOM:
            return CustomMatrix2Q<complex_t>(static_cast<CustomGate*>(gate.get()), pr);
        case GateID::CUSTOM_TWO_PARAM:
            return CustomTwoParamMatrix2Q<complex_t>(static_cast<CustomTwoParamGate*>(gate.get()), pr);
        default:
            throw std::runtime_error(fmt::format("mqmps: unsupported 2-qubit gate id {}.", id));
    }
}

template <typename complex_t>
inline Mat4<complex_t> GateMatrix2QObjOrder12(const std::shared_ptr<BasicGate>& gate,
                                              const parameter::ParameterResolver& pr, bool diff) {
    // Returns matrix in basis |00>,|01>,|10>,|11> with qubit order (obj[1], obj[0]) for 2-object-qubit gates.
    // Match mqvector semantics: only apply the differential form when the gate is marked as grad-required.
    // This keeps `diff=true` as an internal/advanced feature without affecting normal circuit execution.
    const bool diff_active = diff && gate->GradRequired();
    if (diff_active) {
        return GateMatrix2QObjOrder12Diff<complex_t>(gate, pr);
    }
    return GateMatrix2QObjOrder12NoDiff<complex_t>(gate, pr);
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
inline Mat4<complex_t> ControlledDiffMatrix(const Mat2<complex_t>& du_target) {
    // Differential of a controlled single-qubit gate w.r.t. the target parameter:
    // d(|0><0|⊗I + |1><1|⊗U)/dθ = |1><1|⊗dU.
    Mat4<complex_t> out{};
    out[10] = du_target[0];
    out[11] = du_target[1];
    out[14] = du_target[2];
    out[15] = du_target[3];
    return out;
}

template <typename complex_t>
inline std::vector<Stage<complex_t>> ScheduleUnitary(
    const std::vector<std::variant<Op1Q<complex_t>, Op2Q<complex_t>>>& ops, qbit_t n_qubits) {
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

template <typename complex_t>
class MPSCompileState {
 public:
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;

    MPSCompileState(qbit_t n_qubits, const parameter::ParameterResolver& pr, bool diff,
                    const qbits_t& pos_of_logical_in, const qbits_t& logical_at_pos_in)
        : n_qubits_(n_qubits)
        , pr_(pr)
        , diff_(diff)
        , pos_of_logical_(pos_of_logical_in)
        , logical_at_pos_(logical_at_pos_in)
        , pending1_(static_cast<size_t>(n_qubits_), Eye2<complex_t>())
        , pending1_active_(static_cast<size_t>(n_qubits_), false)
        , pending2_(static_cast<size_t>(std::max<qbit_t>(0, n_qubits_ - 1)), Eye4<complex_t>())
        , pending2_active_(static_cast<size_t>(std::max<qbit_t>(0, n_qubits_ - 1)), false) {
        if (pos_of_logical_in.size() != static_cast<size_t>(n_qubits_)
            || logical_at_pos_in.size() != static_cast<size_t>(n_qubits_)) {
            throw std::runtime_error("mqmps compiler: invalid mapping size.");
        }
    }

    MPSProgram<complex_t> Compile(const circuit_t& circ) {
        for (const auto& g : circ) {
            ApplyGate(g);
        }
        Finalize();

        MPSProgram<complex_t> program;
        program.stages = std::move(program_stages_);
        program.pos_of_logical_final = std::move(pos_of_logical_);
        program.logical_at_pos_final = std::move(logical_at_pos_);
        return program;
    }

 private:
    using Mat2 = detail::Mat2<complex_t>;
    using Mat4 = detail::Mat4<complex_t>;
    using UnitOp = std::variant<Op1Q<complex_t>, Op2Q<complex_t>>;

    void ApplyGate(const std::shared_ptr<BasicGate>& g) {
        const auto id = g->GetID();
        if (id == GateID::I) {
            return;
        }
        if (TryApplyMeasurement(g, id)) {
            return;
        }
        if (seen_measure_) {
            throw std::runtime_error("mqmps: mid-circuit measurement is not supported.");
        }
        if (TryApplyPauliString(g, id)) {
            return;
        }
        if (TryApplySingleQubitGate(g)) {
            return;
        }
        ApplyTwoQubitGate(g, id);
    }

    bool TryApplyMeasurement(const std::shared_ptr<BasicGate>& g, GateID id) {
        if (id != GateID::M) {
            return false;
        }
        // Measurement is only supported at the end of the circuit.
        if (!seen_measure_) {
            FlushAllPending();
            EmitCurrentSegmentStages();
            seen_measure_ = true;
        }
        auto mg = static_cast<MeasureGate*>(g.get());
        const auto q = g->GetObjQubits().at(0);
        const auto pos = pos_of_logical_.at(static_cast<size_t>(q));
        measure_ops_.push_back(OpMeasure{mg->Name(), pos, mg->WillReset(), mg->GetRestTo()});
        return true;
    }

    bool TryApplyPauliString(const std::shared_ptr<BasicGate>& g, GateID id) {
        if (id != GateID::PauliString) {
            return false;
        }
        auto ps = static_cast<PauliString*>(g.get());
        if (!ps->GetCtrlQubits().empty()) {
            throw std::runtime_error("mqmps: controlled PauliString is not supported.");
        }
        const auto mask = ps->GetPauliMask();
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
            switch (p) {
                case 'X':
                    local = std::make_shared<XGate>(qbits_t{q}, qbits_t{});
                    break;
                case 'Y':
                    local = std::make_shared<YGate>(qbits_t{q}, qbits_t{});
                    break;
                default:
                    local = std::make_shared<ZGate>(qbits_t{q}, qbits_t{});
                    break;
            }
            const auto u = GateMatrix1Q<complex_t>(local, pr_, diff_);
            AbsorbSingleQubitOnPos(pos_of_logical_.at(static_cast<size_t>(q)), u);
        }
        return true;
    }

    bool TryApplySingleQubitGate(const std::shared_ptr<BasicGate>& g) {
        const auto& obj = g->GetObjQubits();
        const auto& ctrl = g->GetCtrlQubits();
        const auto total_qubits = obj.size() + ctrl.size();
        if (total_qubits != 1 || !ctrl.empty()) {
            return false;
        }
        const auto q = obj.at(0);
        const auto pos = pos_of_logical_.at(static_cast<size_t>(q));
        const auto u = GateMatrix1Q<complex_t>(g, pr_, diff_);
        AbsorbSingleQubitOnPos(pos, u);
        return true;
    }

    void ApplyTwoQubitGate(const std::shared_ptr<BasicGate>& g, GateID id) {
        const auto& obj = g->GetObjQubits();
        const auto& ctrl = g->GetCtrlQubits();
        const auto total_qubits = obj.size() + ctrl.size();
        if (total_qubits != 2) {
            throw std::runtime_error(
                fmt::format("mqmps: only 1Q/2Q gates supported, got id {} with {} qubits.", id, total_qubits));
        }

        qbit_t q_first = -1;
        qbit_t q_second = -1;
        Mat4 u_qfirst_qsecond{};

        if (obj.size() == 2 && ctrl.empty()) {
            // MindQuantum convention: matrix is for (obj[1], obj[0]).
            q_first = obj.at(1);
            q_second = obj.at(0);
            u_qfirst_qsecond = GateMatrix2QObjOrder12<complex_t>(g, pr_, diff_);
        } else if (obj.size() == 1 && ctrl.size() == 1) {
            // Controlled single-qubit gate: build matrix in (control, target) order.
            q_first = ctrl.at(0);
            q_second = obj.at(0);
            u_qfirst_qsecond = ControlledSingleQubitGateMatrix(id, g, q_second);
        } else {
            throw std::runtime_error("mqmps: unsupported 2-qubit gate form.");
        }

        // Route the two logical qubits to be adjacent.
        RouteToAdjacent(q_first, q_second);

        // Determine the bond and physical ordering.
        const auto p_first = pos_of_logical_.at(static_cast<size_t>(q_first));
        const auto p_second = pos_of_logical_.at(static_cast<size_t>(q_second));
        const auto bond = std::min(p_first, p_second);
        const auto left_pos = bond;
        const auto right_pos = bond + 1;
        const auto q_left = logical_at_pos_.at(static_cast<size_t>(left_pos));
        const auto q_right = logical_at_pos_.at(static_cast<size_t>(right_pos));

        Mat4 u_lr = u_qfirst_qsecond;
        if (q_left == q_first && q_right == q_second) {
            // ok
        } else if (q_left == q_second && q_right == q_first) {
            u_lr = SwapQubits(u_lr);
        } else {
            throw std::runtime_error("mqmps: internal routing error (qubits not on the expected bond).");
        }

        ApplyAdjacentTwoQubitOnBond(bond, u_lr);
    }

    Mat4 ControlledSingleQubitGateMatrix(GateID id, const std::shared_ptr<BasicGate>& g, qbit_t target) {
        const auto bare = MakeBareSingleQubitGate(id, g, target);
        const bool diff_active = diff_ && bare->GradRequired();
        const auto u_target = GateMatrix1Q<complex_t>(bare, pr_, diff_);
        if (diff_active) {
            return detail::ControlledDiffMatrix<complex_t>(u_target);
        }
        return detail::ControlledMatrix<complex_t>(u_target);
    }

    std::shared_ptr<BasicGate> MakeBareSingleQubitGate(GateID id, const std::shared_ptr<BasicGate>& g, qbit_t target) {
        switch (id) {
            case GateID::X:
                return std::make_shared<XGate>(qbits_t{target}, qbits_t{});
            case GateID::Y:
                return std::make_shared<YGate>(qbits_t{target}, qbits_t{});
            case GateID::Z:
                return std::make_shared<ZGate>(qbits_t{target}, qbits_t{});
            case GateID::H:
                return std::make_shared<HGate>(qbits_t{target}, qbits_t{});
            case GateID::S:
                return std::make_shared<SGate>(qbits_t{target}, qbits_t{});
            case GateID::Sdag:
                return std::make_shared<SdagGate>(qbits_t{target}, qbits_t{});
            case GateID::T:
                return std::make_shared<TGate>(qbits_t{target}, qbits_t{});
            case GateID::Tdag:
                return std::make_shared<TdagGate>(qbits_t{target}, qbits_t{});
            case GateID::SX:
                return std::make_shared<SXGate>(qbits_t{target}, qbits_t{});
            case GateID::SXdag:
                return std::make_shared<SXdagGate>(qbits_t{target}, qbits_t{});
            case GateID::RX: {
                auto gg = static_cast<RXGate*>(g.get());
                return std::make_shared<RXGate>(gg->GetCoeffs()[0], qbits_t{target}, qbits_t{});
            }
            case GateID::RY: {
                auto gg = static_cast<RYGate*>(g.get());
                return std::make_shared<RYGate>(gg->GetCoeffs()[0], qbits_t{target}, qbits_t{});
            }
            case GateID::RZ: {
                auto gg = static_cast<RZGate*>(g.get());
                return std::make_shared<RZGate>(gg->GetCoeffs()[0], qbits_t{target}, qbits_t{});
            }
            case GateID::PS: {
                auto gg = static_cast<PSGate*>(g.get());
                return std::make_shared<PSGate>(gg->GetCoeffs()[0], qbits_t{target}, qbits_t{});
            }
            case GateID::GP: {
                auto gg = static_cast<GPGate*>(g.get());
                return std::make_shared<GPGate>(gg->GetCoeffs()[0], qbits_t{target}, qbits_t{});
            }
            default:
                throw std::runtime_error(fmt::format("mqmps: controlled gate id {} not supported.", id));
        }
    }

    void FlushPending1Pos(qbit_t pos) {
        if (pos < 0 || pos >= n_qubits_) {
            return;
        }
        const auto idx = static_cast<size_t>(pos);
        if (!pending1_active_[idx]) {
            return;
        }
        unit_ops_segment_.emplace_back(Op1Q<complex_t>{pos, pending1_[idx]});
        pending1_[idx] = Eye2<complex_t>();
        pending1_active_[idx] = false;
    }

    void FlushPending2Bond(qbit_t bond) {
        if (bond < 0 || bond >= n_qubits_ - 1) {
            return;
        }
        const auto idx = static_cast<size_t>(bond);
        if (!pending2_active_[idx]) {
            return;
        }
        unit_ops_segment_.emplace_back(Op2Q<complex_t>{bond, pending2_[idx]});
        pending2_[idx] = Eye4<complex_t>();
        pending2_active_[idx] = false;
    }

    void FlushPendingForSites(qbit_t left, qbit_t right) {
        FlushPending1Pos(left);
        FlushPending1Pos(right);
        FlushPending2Bond(left - 1);
        FlushPending2Bond(left);
        FlushPending2Bond(right);
    }

    void FlushConflictingBondsForTwoQ(qbit_t bond) {
        FlushPending2Bond(bond - 1);
        FlushPending2Bond(bond + 1);
    }

    void FlushAllPending() {
        for (qbit_t b = 0; b < n_qubits_ - 1; ++b) {
            FlushPending2Bond(b);
        }
        for (qbit_t p = 0; p < n_qubits_; ++p) {
            FlushPending1Pos(p);
        }
    }

    void EmitCurrentSegmentStages() {
        if (unit_ops_segment_.empty()) {
            return;
        }
        const auto stages = ScheduleUnitary(unit_ops_segment_, n_qubits_);
        program_stages_.insert(program_stages_.end(), stages.begin(), stages.end());
        unit_ops_segment_.clear();
    }

    void SwapPositions(qbit_t bond) {
        // Ensure no pending ops touch this bond.
        FlushPendingForSites(bond, bond + 1);
        // Emit SWAP as a standalone 2Q op.
        unit_ops_segment_.emplace_back(Op2Q<complex_t>{bond, SwapMatrix<complex_t>()});

        // Update mapping.
        const auto lq = logical_at_pos_.at(static_cast<size_t>(bond));
        const auto rq = logical_at_pos_.at(static_cast<size_t>(bond + 1));
        std::swap(logical_at_pos_.at(static_cast<size_t>(bond)), logical_at_pos_.at(static_cast<size_t>(bond + 1)));
        pos_of_logical_.at(static_cast<size_t>(lq)) = bond + 1;
        pos_of_logical_.at(static_cast<size_t>(rq)) = bond;
    }

    void RouteToAdjacent(qbit_t qa, qbit_t qb) {
        while (std::abs(pos_of_logical_.at(static_cast<size_t>(qa)) - pos_of_logical_.at(static_cast<size_t>(qb)))
               > 1) {
            const auto pa = pos_of_logical_.at(static_cast<size_t>(qa));
            const auto pb = pos_of_logical_.at(static_cast<size_t>(qb));
            if (pa < pb) {
                SwapPositions(pb - 1);
            } else {
                SwapPositions(pa - 1);
            }
        }
    }

    void AbsorbSingleQubitOnPos(qbit_t pos, const Mat2& u) {
        // If a pending 2Q touches this pos, absorb into that bond.
        if (pos > 0 && pending2_active_.at(static_cast<size_t>(pos - 1))) {
            const auto b = pos - 1;
            const auto factor = KronRight(u);  // pos is right site of bond b
            pending2_.at(static_cast<size_t>(b)) = MatMul(factor, pending2_.at(static_cast<size_t>(b)));
            pending2_active_.at(static_cast<size_t>(b)) = true;
            return;
        }
        if (pos < n_qubits_ - 1 && pending2_active_.at(static_cast<size_t>(pos))) {
            const auto b = pos;
            const auto factor = KronLeft(u);  // pos is left site of bond b
            pending2_.at(static_cast<size_t>(b)) = MatMul(factor, pending2_.at(static_cast<size_t>(b)));
            pending2_active_.at(static_cast<size_t>(b)) = true;
            return;
        }
        // Otherwise accumulate into pending 1Q.
        const auto idx = static_cast<size_t>(pos);
        pending1_[idx] = MatMul(u, pending1_[idx]);
        pending1_active_[idx] = true;
    }

    void ApplyAdjacentTwoQubitOnBond(qbit_t bond, Mat4 u_lr) {
        // Flush neighbor bonds that would conflict.
        FlushConflictingBondsForTwoQ(bond);

        // Fold in pending 1Q gates on the two sites.
        const auto left = bond;
        const auto right = bond + 1;

        const auto& u_left = pending1_.at(static_cast<size_t>(left));
        const auto& u_right = pending1_.at(static_cast<size_t>(right));
        const auto kron_lr = Kron(u_left, u_right);

        pending1_.at(static_cast<size_t>(left)) = Eye2<complex_t>();
        pending1_.at(static_cast<size_t>(right)) = Eye2<complex_t>();
        pending1_active_.at(static_cast<size_t>(left)) = false;
        pending1_active_.at(static_cast<size_t>(right)) = false;

        u_lr = MatMul(u_lr, kron_lr);

        // Accumulate into pending 2Q on that bond.
        pending2_.at(static_cast<size_t>(bond)) = MatMul(u_lr, pending2_.at(static_cast<size_t>(bond)));
        pending2_active_.at(static_cast<size_t>(bond)) = true;
    }

    void Finalize() {
        FlushAllPending();
        EmitCurrentSegmentStages();
        if (!measure_ops_.empty()) {
            Stage<complex_t> st;
            st.meas = std::move(measure_ops_);
            program_stages_.push_back(std::move(st));
        }
    }

 private:
    qbit_t n_qubits_ = 0;
    const parameter::ParameterResolver& pr_;
    bool diff_ = false;

    qbits_t pos_of_logical_;
    qbits_t logical_at_pos_;

    std::vector<Mat2> pending1_;
    std::vector<uint8_t> pending1_active_;
    std::vector<Mat4> pending2_;
    std::vector<uint8_t> pending2_active_;

    std::vector<UnitOp> unit_ops_segment_;
    std::vector<Stage<complex_t>> program_stages_;
    std::vector<OpMeasure> measure_ops_;
    bool seen_measure_ = false;
};
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
        detail::MPSCompileState<complex_t> state(n_qubits_, pr, diff, pos_of_logical_in, logical_at_pos_in);
        return state.Compile(circ);
    }

 private:
    qbit_t n_qubits_ = 0;
};
}  // namespace mindquantum::sim::mps::compiler

#endif  // INCLUDE_SIMULATOR_MPS_COMPILER_MPS_COMPILER_HPP
