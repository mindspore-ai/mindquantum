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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>

#include "core/mq_base_types.hpp"
#include "core/numba_wrapper.hpp"
#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/matrix.hpp"
#include "math/tensor/traits.hpp"
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

struct Parameterizable : public BasicGate {
    int n_pr;
    VT<parameter::ParameterResolver> prs_;
    bool parameterized_;
    bool grad_required_;
    std::pair<MST<size_t>, tensor::Matrix> jacobi;
    Parameterizable(GateID id, const VT<parameter::ParameterResolver>& prs, const VT<Index>& obj_qubits,
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

struct RXGate : public Parameterizable {
    RXGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::RX, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RYGate : public Parameterizable {
    RYGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::RY, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RZGate : public Parameterizable {
    RZGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::RZ, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxxGate : public Parameterizable {
    RxxGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Rxx, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RyyGate : public Parameterizable {
    RyyGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Ryy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RzzGate : public Parameterizable {
    RzzGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Rzz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxyGate : public Parameterizable {
    RxyGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Rxy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RxzGate : public Parameterizable {
    RxzGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Rxz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct RyzGate : public Parameterizable {
    RyzGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::Ryz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
struct GPGate : public Parameterizable {
    GPGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::GP, {pr}, obj_qubits, ctrl_qubits) {
    }
};

struct PSGate : public Parameterizable {
    PSGate(const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : Parameterizable(GateID::PS, {pr}, obj_qubits, ctrl_qubits) {
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

struct KrausChannel : public BasicGate {
    VT<tensor::Matrix> kraus_operator_set_;
    template <typename T>
    KrausChannel(const VT<VVT<CT<T>>>& kraus_operator_set, const VT<Index>& obj_qubits,
                 const VT<Index>& ctrl_qubits = {})
        : BasicGate(GateID::KRAUS, obj_qubits, ctrl_qubits) {
        std::transform(kraus_operator_set.begin(), kraus_operator_set.end(), std::back_inserter(kraus_operator_set_),
                       [](auto& k) { return tensor::Matrix(k); });
    }
};

struct AmplitudeDampingChannel : public BasicGate {
    double damping_coeff_;
    bool daggered_;
    AmplitudeDampingChannel(bool daggered, double damping_coeff, const VT<Index>& obj_qubits,
                            const VT<Index>& ctrl_qubits = {})
        : daggered_(daggered), damping_coeff_(damping_coeff), BasicGate(GateID::AD, obj_qubits, ctrl_qubits) {
    }
};
struct PhaseDampingChannel : public BasicGate {
    double damping_coeff_;
    PhaseDampingChannel(double damping_coeff, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : damping_coeff_(damping_coeff), BasicGate(GateID::PD, obj_qubits, ctrl_qubits) {
    }
};

struct CustomGate : public Parameterizable {
    std::string name_;
    NumbaMatFunWrapper numba_param_matrix_;
    NumbaMatFunWrapper numba_param_diff_matrix_;
    tensor::Matrix base_matrix_;
    CustomGate(const std::string& name, uint64_t m_addr, uint64_t dm_addr, int dim,
               const parameter::ParameterResolver pr, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits = {})
        : name_(name)
        , numba_param_matrix_(m_addr, dim)
        , numba_param_diff_matrix_(dm_addr, dim)
        , Parameterizable(GateID::CUSTOM, {pr}, obj_qubits, ctrl_qubits) {
        if (!this->Parameterized()) {
            base_matrix_ = this->numba_param_matrix_(tensor::ops::cpu::to_vector<double>(this->prs_[0].const_value)[0]);
        }
    }
    CustomGate(const std::string& name, const tensor::Matrix& mat, const VT<Index>& obj_qubits,
               const VT<Index>& ctrl_qubits = {})
        : name_(name)
        , base_matrix_(mat)
        , Parameterizable(GateID::CUSTOM, {parameter::ParameterResolver()}, obj_qubits, ctrl_qubits) {
    }
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
