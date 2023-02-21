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

#ifndef MINDQUANTUM_GATE_basic_gate_H_
#define MINDQUANTUM_GATE_basic_gate_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "core/mq_base_types.hpp"
#include "core/numba_wrapper.hpp"
#include "core/parameter_resolver.hpp"
#include "core/two_dim_matrix.hpp"
#include "core/utils.hpp"
#include "ops/gate_id.hpp"

namespace mindquantum {
struct BasicGate {
    GateID id_ = GateID::I;
    VT<Index> obj_qubits_ = {};
    VT<Index> ctrl_qubits_ = {};
    BasicGate() = default;
    BasicGate(GateID id, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits)
        : id_(id), obj_qubits_(obj_qubits), ctrl_qubits_(ctrl_qubits) {
    }
    virtual bool Parameterized() {
        return false;
    }
    virtual bool GradRequired() {
        return false;
    }
};

struct MeasureGate : public BasicGate {
    std::string name_;
    MeasureGate(const std::string& name, const VT<Index>& obj_qubits)
        : name_(name), BasicGate(GateID::M, obj_qubits, {}) {
    }
};

struct IGate : public BasicGate {
    explicit IGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::I, obj_qubits, ctrl_qubits) {
    }
};
struct XGate : public BasicGate {
    explicit XGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::X, obj_qubits, ctrl_qubits) {
    }
};
struct YGate : public BasicGate {
    explicit YGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::Y, obj_qubits, ctrl_qubits) {
    }
};

struct ZGate : public BasicGate {
    explicit ZGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::Z, obj_qubits, ctrl_qubits) {
    }
};

struct HGate : public BasicGate {
    explicit HGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::H, obj_qubits, ctrl_qubits) {
    }
};
struct SWAPGate : public BasicGate {
    explicit SWAPGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::SWAP, obj_qubits, ctrl_qubits) {
    }
};
struct ISWAPGate : public BasicGate {
    bool daggered_;
    ISWAPGate(bool daggered, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : daggered_(daggered), BasicGate(GateID::ISWAP, obj_qubits, ctrl_qubits) {
    }
};
struct SGate : public BasicGate {
    explicit SGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::S, obj_qubits, ctrl_qubits) {
    }
};
struct SdagGate : public BasicGate {
    explicit SdagGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::Sdag, obj_qubits, ctrl_qubits) {
    }
};
struct TGate : public BasicGate {
    explicit TGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::T, obj_qubits, ctrl_qubits) {
    }
};
struct TdagGate : public BasicGate {
    explicit TdagGate(const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::Tdag, obj_qubits, ctrl_qubits) {
    }
};

template <typename T>
struct Parameterizable : public BasicGate {
    int n_pr;
    VT<ParameterResolver<T>> prs_;
    bool parameterized_;
    bool grad_required_;
    std::pair<MST<size_t>, Dim2Matrix<T>> jacobi;
    Parameterizable(GateID id, const VT<ParameterResolver<T>>& prs, const VT<Index>& obj_qubits,
                    const VT<Index>& ctrl_qubits)
        : n_pr(prs.size()), prs_(prs), BasicGate(id, obj_qubits, ctrl_qubits) {
        parameterized_ = !std::all_of(this->prs_.begin(), this->prs_.end(),
                                      [](const auto& pr) { return pr.IsConst(); });
        grad_required_ = std::any_of(this->prs_.begin(), this->prs_.end(),
                                     [](const auto& pr) { return pr.data_.size() != pr.no_grad_parameters_.size(); });
        jacobi = Jacobi(this->prs_);
    }
    bool Parameterized() override {
        return this->parameterized_;
    }
    bool GradRequired() override {
        return this->grad_required_;
    }
};

template <typename T>
struct RXGate : public Parameterizable<T> {
    RXGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::RX, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RYGate : public Parameterizable<T> {
    RYGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::RY, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RZGate : public Parameterizable<T> {
    RZGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::RZ, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RxxGate : public Parameterizable<T> {
    RxxGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Rxx, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RyyGate : public Parameterizable<T> {
    RyyGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Ryy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RzzGate : public Parameterizable<T> {
    RzzGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Rzz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RxyGate : public Parameterizable<T> {
    RxyGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Rxy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RxzGate : public Parameterizable<T> {
    RxzGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Rxz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct RyzGate : public Parameterizable<T> {
    RyzGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::Ryz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
template <typename T>
struct GPGate : public Parameterizable<T> {
    GPGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::GP, {pr}, obj_qubits, ctrl_qubits) {
    }
};

template <typename T>
struct PSGate : public Parameterizable<T> {
    PSGate(const ParameterResolver<T> pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable<T>(GateID::PS, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct PauliChannel : public BasicGate {
    VT<double> cumulative_probs_;
    VT<double> probs_;
    PauliChannel(double px, double py, double pz, const VT<Index>& obj_qubits,
                 const VT<Index>& ctrl_qubits = {})  // for pauli channel
        : probs_{px, py, pz, 1 - px - py - pz}, BasicGate(GateID::PL, obj_qubits, ctrl_qubits) {
        double sum = 0.;
        cumulative_probs_.push_back(sum);
        for (auto it = probs_.begin(); it != probs_.end(); it++) {
            sum += *it;
            cumulative_probs_.push_back(sum);
        }
    }
};

template <typename T>
struct KrausChannel : public BasicGate {
    VT<VVT<CT<T>>> kraus_operator_set_;
    KrausChannel(const VT<VVT<CT<T>>>& kraus_operator_set, const VT<Index>& obj_qubits,
                 const VT<Index>& ctrl_qubits = {})
        : kraus_operator_set_(kraus_operator_set), BasicGate(GateID::KRAUS, obj_qubits, ctrl_qubits) {
    }
};

struct AmplitudeDampingChannel : public BasicGate {
    double damping_coeff_;
    AmplitudeDampingChannel(double damping_coeff, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : damping_coeff_(damping_coeff), BasicGate(GateID::AD, obj_qubits, ctrl_qubits) {
    }
};
struct PhaseDampingChannel : public BasicGate {
    double damping_coeff_;
    PhaseDampingChannel(double damping_coeff, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : damping_coeff_(damping_coeff), BasicGate(GateID::PD, obj_qubits, ctrl_qubits) {
    }
};

template <typename T>
struct CustomGate : public Parameterizable<T> {
    using matrix_t = Dim2Matrix<T>;
    std::string name_;
    NumbaMatFunWrapper<T> numba_param_matrix_;
    NumbaMatFunWrapper<T> numba_param_diff_matrix_;
    Dim2Matrix<T> base_matrix_;
    CustomGate(const std::string& name, uint64_t m_addr, uint64_t dm_addr, int dim, const ParameterResolver<T> pr,
               const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : name_(name)
        , numba_param_matrix_(m_addr, dim)
        , numba_param_diff_matrix_(dm_addr, dim)
        , Parameterizable<T>(GateID::CUSTOM, {pr}, obj_qubits, ctrl_qubits) {
        if (!this->Parameterized()) {
            base_matrix_ = this->numba_param_matrix_(this->prs_[0].const_value);
        }
    }
    CustomGate(const std::string& name, const Dim2Matrix<T>& mat, const VT<Index>& obj_qubits,
               const VT<Index>& ctrl_qubits = {})
        : name_(name)
        , base_matrix_(mat)
        , Parameterizable<T>(GateID::CUSTOM, {ParameterResolver<T>()}, obj_qubits, ctrl_qubits) {
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
