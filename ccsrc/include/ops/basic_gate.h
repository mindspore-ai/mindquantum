/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>

#include "core/mq_base_types.h"
#include "core/numba_wrapper.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/traits.h"
#include "ops/gate_id.h"

namespace mindquantum {
struct BasicGate {
    GateID id_ = GateID::I;
    qbits_t obj_qubits_ = {};
    qbits_t ctrl_qubits_ = {};
    BasicGate() = default;
    BasicGate(GateID id, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits)
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
    MeasureGate(const std::string& name, const qbits_t& obj_qubits)
        : BasicGate(GateID::M, obj_qubits, {}), name_(name) {
    }
};

struct IGate : public BasicGate {
    explicit IGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::I, obj_qubits, ctrl_qubits) {
    }
};
struct XGate : public BasicGate {
    explicit XGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::X, obj_qubits, ctrl_qubits) {
    }
};
struct YGate : public BasicGate {
    explicit YGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Y, obj_qubits, ctrl_qubits) {
    }
};

struct ZGate : public BasicGate {
    explicit ZGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Z, obj_qubits, ctrl_qubits) {
    }
};

struct HGate : public BasicGate {
    explicit HGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::H, obj_qubits, ctrl_qubits) {
    }
};
struct SWAPGate : public BasicGate {
    explicit SWAPGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::SWAP, obj_qubits, ctrl_qubits) {
    }
};
struct ISWAPGate : public BasicGate {
    bool daggered_;
    ISWAPGate(bool daggered, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::ISWAP, obj_qubits, ctrl_qubits), daggered_(daggered) {
    }
};
struct SGate : public BasicGate {
    explicit SGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::S, obj_qubits, ctrl_qubits) {
    }
};
struct SdagGate : public BasicGate {
    explicit SdagGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Sdag, obj_qubits, ctrl_qubits) {
    }
};
struct TGate : public BasicGate {
    explicit TGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::T, obj_qubits, ctrl_qubits) {
    }
};
struct TdagGate : public BasicGate {
    explicit TdagGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Tdag, obj_qubits, ctrl_qubits) {
    }
};

struct Parameterizable : public BasicGate {
    int n_pr;
    VT<parameter::ParameterResolver> prs_;
    bool parameterized_;
    bool grad_required_;
    std::pair<MST<size_t>, tensor::Matrix> jacobi;
    Parameterizable(GateID id, const VT<parameter::ParameterResolver>& prs, const qbits_t& obj_qubits,
                    const qbits_t& ctrl_qubits)
        : BasicGate(id, obj_qubits, ctrl_qubits), n_pr(prs.size()), prs_(prs) {
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

struct RXGate : public Parameterizable {
    RXGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RX, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RYGate : public Parameterizable {
    RYGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RY, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RZGate : public Parameterizable {
    RZGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RZ, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxxGate : public Parameterizable {
    RxxGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxx, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RyyGate : public Parameterizable {
    RyyGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Ryy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RzzGate : public Parameterizable {
    RzzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rzz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxyGate : public Parameterizable {
    RxyGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxzGate : public Parameterizable {
    RxzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RyzGate : public Parameterizable {
    RyzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Ryz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct GPGate : public Parameterizable {
    GPGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::GP, {pr}, obj_qubits, ctrl_qubits) {
    }
};

struct PSGate : public Parameterizable {
    PSGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::PS, {pr}, obj_qubits, ctrl_qubits) {
    }
};

struct SWAPalphaGate : public Parameterizable {
    SWAPalphaGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::SWAPalpha, {pr}, obj_qubits, ctrl_qubits) {
    }
};

struct PauliChannel : public BasicGate {
    VT<double> cumulative_probs_;
    VT<double> probs_;
    PauliChannel(double px, double py, double pz, const qbits_t& obj_qubits,
                 const qbits_t& ctrl_qubits = {})  // for pauli channel
        : BasicGate(GateID::PL, obj_qubits, ctrl_qubits), probs_{px, py, pz, 1 - px - py - pz} {
        double sum = 0.;
        cumulative_probs_.push_back(sum);
        for (auto it = probs_.begin(); it != probs_.end(); it++) {
            sum += *it;
            cumulative_probs_.push_back(sum);
        }
    }
};
struct DepolarizingChannel : public BasicGate {
    double prob_;
    DepolarizingChannel(double p, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::DEP, obj_qubits, ctrl_qubits), prob_(p) {
    }
};

struct KrausChannel : public BasicGate {
    VT<tensor::Matrix> kraus_operator_set_;
    template <typename T>
    KrausChannel(const VT<VVT<CT<T>>>& kraus_operator_set, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::KRAUS, obj_qubits, ctrl_qubits) {
        std::transform(kraus_operator_set.begin(), kraus_operator_set.end(), std::back_inserter(kraus_operator_set_),
                       [](auto& k) { return tensor::Matrix(k); });
    }
};

struct AmplitudeDampingChannel : public BasicGate {
    double damping_coeff_;
    bool daggered_;
    AmplitudeDampingChannel(bool daggered, double damping_coeff, const qbits_t& obj_qubits,
                            const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::AD, obj_qubits, ctrl_qubits), damping_coeff_(damping_coeff), daggered_(daggered) {
    }
};
struct PhaseDampingChannel : public BasicGate {
    double damping_coeff_;
    PhaseDampingChannel(double damping_coeff, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::PD, obj_qubits, ctrl_qubits), damping_coeff_(damping_coeff) {
    }
};

struct CustomGate : public Parameterizable {
    std::string name_;
    NumbaMatFunWrapper numba_param_matrix_;
    NumbaMatFunWrapper numba_param_diff_matrix_;
    tensor::Matrix base_matrix_;
    CustomGate(const std::string& name, uint64_t m_addr, uint64_t dm_addr, int dim,
               const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::CUSTOM, {pr}, obj_qubits, ctrl_qubits)
        , name_(name)
        , numba_param_matrix_(m_addr, dim)
        , numba_param_diff_matrix_(dm_addr, dim) {
        if (!this->Parameterized()) {
            base_matrix_ = this->numba_param_matrix_(tensor::ops::cpu::to_vector<double>(this->prs_[0].const_value)[0]);
        }
    }
    CustomGate(const std::string& name, const tensor::Matrix& mat, const qbits_t& obj_qubits,
               const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::CUSTOM, {parameter::ParameterResolver()}, obj_qubits, ctrl_qubits)
        , name_(name)
        , base_matrix_(mat) {
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
