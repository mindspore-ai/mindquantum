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
#include <stdexcept>
#include <string>
#include <utility>

#include "core/mq_base_types.h"
#include "core/numba_wrapper.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/traits.h"
#include "ops/gate_id.h"
#include "ops/hamiltonian.h"

namespace mindquantum {
class BasicGate {
 public:
    BasicGate() = default;
    BasicGate(GateID id, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits)
        : id_(id), obj_qubits_(obj_qubits), ctrl_qubits_(ctrl_qubits) {
    }
    virtual bool Parameterized() const {
        return false;
    }
    virtual bool GradRequired() const {
        return false;
    }
    GateID GetID() const {
        return this->id_;
    }
    qbits_t GetObjQubits() const {
        return obj_qubits_;
    }
    qbits_t GetCtrlQubits() const {
        return ctrl_qubits_;
    }

 private:
    GateID id_ = GateID::I;
    qbits_t obj_qubits_ = {};
    qbits_t ctrl_qubits_ = {};
};

class MeasureGate : public BasicGate {
 public:
    MeasureGate(const std::string& name, const qbits_t& obj_qubits)
        : BasicGate(GateID::M, obj_qubits, {}), name_(name) {
    }
    MeasureGate(const std::string& name, const qbits_t& obj_qubits, index_t reset_to)
        : BasicGate(GateID::M, obj_qubits, {}), name_(name), reset_to_(reset_to), reset_(true) {
    }
    index_t GetRestTo() const {
        return reset_to_;
    }
    bool WillReset() const {
        return reset_;
    }
    std::string Name() const {
        return name_;
    }

 private:
    std::string name_;
    index_t reset_to_;
    bool reset_ = false;
};

class IGate : public BasicGate {
 public:
    explicit IGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::I, obj_qubits, ctrl_qubits) {
    }
};

class PauliString : public BasicGate {
 public:
    PauliString(const std::string& paulis, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::PauliString, obj_qubits, ctrl_qubits) {
        if (paulis.length() != obj_qubits.size()) {
            throw std::runtime_error("pauli string size miss match with qubit size.");
        }
        Index i = 0;
        for (auto pauli : paulis) {
            term.push_back(PauliWord(obj_qubits.at(i), pauli));
            ++i;
        }
        pauli_mask = mindquantum::GetPauliMask(term);
        ctrl_mask = GetControlMask(ctrl_qubits);
    }
    PauliMask GetPauliMask() const {
        return pauli_mask;
    }
    Index GetCtrlMask() const {
        return ctrl_mask;
    }

 private:
    VT<PauliWord> term;
    PauliMask pauli_mask;
    Index ctrl_mask;
};

class XGate : public BasicGate {
 public:
    explicit XGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::X, obj_qubits, ctrl_qubits) {
    }
};
class YGate : public BasicGate {
 public:
    explicit YGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Y, obj_qubits, ctrl_qubits) {
    }
};

class ZGate : public BasicGate {
 public:
    explicit ZGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Z, obj_qubits, ctrl_qubits) {
    }
};

class HGate : public BasicGate {
 public:
    explicit HGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::H, obj_qubits, ctrl_qubits) {
    }
};
class SWAPGate : public BasicGate {
 public:
    explicit SWAPGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::SWAP, obj_qubits, ctrl_qubits) {
    }
};
class ISWAPGate : public BasicGate {
 public:
    ISWAPGate(bool daggered, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::ISWAP, obj_qubits, ctrl_qubits), daggered_(daggered) {
    }
    bool Daggered() {
        return daggered_;
    }

 private:
    bool daggered_;
};
class SXGate : public BasicGate {
 public:
    explicit SXGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::SX, obj_qubits, ctrl_qubits) {
    }
};
class SXdagGate : public BasicGate {
 public:
    explicit SXdagGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::SXdag, obj_qubits, ctrl_qubits) {
    }
};
class SGate : public BasicGate {
 public:
    explicit SGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::S, obj_qubits, ctrl_qubits) {
    }
};
class SdagGate : public BasicGate {
 public:
    explicit SdagGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Sdag, obj_qubits, ctrl_qubits) {
    }
};
class TGate : public BasicGate {
 public:
    explicit TGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::T, obj_qubits, ctrl_qubits) {
    }
};
class TdagGate : public BasicGate {
 public:
    explicit TdagGate(const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::Tdag, obj_qubits, ctrl_qubits) {
    }
};

class Parameterizable : public BasicGate {
 public:
    Parameterizable(GateID id, const VT<parameter::ParameterResolver>& prs, const qbits_t& obj_qubits,
                    const qbits_t& ctrl_qubits)
        : BasicGate(id, obj_qubits, ctrl_qubits), n_pr(prs.size()), prs_(prs) {
        parameterized_ = !std::all_of(this->prs_.begin(), this->prs_.end(),
                                      [](const auto& pr) { return pr.IsConst(); });
        grad_required_ = std::any_of(this->prs_.begin(), this->prs_.end(),
                                     [](const auto& pr) { return pr.data_.size() != pr.no_grad_parameters_.size(); });
        jacobi = Jacobi(this->prs_);
    }
    bool Parameterized() const override {
        return this->parameterized_;
    }
    bool GradRequired() const override {
        return this->grad_required_;
    }
    VT<parameter::ParameterResolver> GetCoeffs() const {
        return prs_;
    }
    void ModifyCoeff(int idx, const parameter::ParameterResolver& pr) {
        prs_[idx] = pr;
    }
    const std::pair<MST<size_t>, tensor::Matrix>& GetJacobi() const {
        return jacobi;
    }

 private:
    int n_pr;
    VT<parameter::ParameterResolver> prs_;
    bool parameterized_;
    bool grad_required_;
    std::pair<MST<size_t>, tensor::Matrix> jacobi;
};

class RXGate : public Parameterizable {
 public:
    RXGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RX, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RYGate : public Parameterizable {
 public:
    RYGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RY, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RZGate : public Parameterizable {
 public:
    RZGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RZ, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RxxGate : public Parameterizable {
 public:
    RxxGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxx, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RyyGate : public Parameterizable {
 public:
    RyyGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Ryy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RzzGate : public Parameterizable {
 public:
    RzzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rzz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RxyGate : public Parameterizable {
 public:
    RxyGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxy, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RxzGate : public Parameterizable {
 public:
    RxzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Rxz, {pr}, obj_qubits, ctrl_qubits) {
    }
};
class RyzGate : public Parameterizable {
 public:
    RyzGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Ryz, {pr}, obj_qubits, ctrl_qubits) {
    }
};

class GivensGate : public Parameterizable {
 public:
    GivensGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::Givens, {pr}, obj_qubits, ctrl_qubits) {
    }
};

class RotPauliString : public Parameterizable {
 public:
    RotPauliString(const std::string& pauli_string, const parameter::ParameterResolver pr, const qbits_t& obj_qubits,
                   const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::RPS, {pr}, obj_qubits, ctrl_qubits)
        , pauli_string(pauli_string, obj_qubits, ctrl_qubits) {
    }
    PauliString GetPauliString() const {
        return pauli_string;
    }

 private:
    PauliString pauli_string;
};

class GPGate : public Parameterizable {
 public:
    GPGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::GP, {pr}, obj_qubits, ctrl_qubits) {
    }
};

class PSGate : public Parameterizable {
 public:
    PSGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::PS, {pr}, obj_qubits, ctrl_qubits) {
    }
};

class SWAPalphaGate : public Parameterizable {
 public:
    SWAPalphaGate(const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::SWAPalpha, {pr}, obj_qubits, ctrl_qubits) {
    }
};

class PauliChannel : public BasicGate {
 public:
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
    VT<double> GetCumulativeProbs() const {
        return cumulative_probs_;
    }
    VT<double> GetProbs() const {
        return probs_;
    }

 private:
    VT<double> cumulative_probs_;
    VT<double> probs_;
};

class GroupedPauliChannel : public BasicGate {
 public:
    GroupedPauliChannel(const VVT<double>& probs, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::GPL, obj_qubits, ctrl_qubits) {
        int idx = 0;
        for (auto const& prob : probs) {
            pauli_channels.push_back(PauliChannel(prob[0], prob[1], prob[2], qbits_t({obj_qubits[idx]}), ctrl_qubits));
            ++idx;
        }
    }
    VT<PauliChannel>::const_iterator begin() const {
        return pauli_channels.begin();
    }
    VT<PauliChannel>::const_iterator end() const {
        return pauli_channels.end();
    }

 private:
    VT<PauliChannel> pauli_channels;
};
class DepolarizingChannel : public BasicGate {
 public:
    DepolarizingChannel(double p, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::DEP, obj_qubits, ctrl_qubits), prob_(p) {
    }
    double GetProb() const {
        return prob_;
    }

 private:
    double prob_;
};

class KrausChannel : public BasicGate {
 public:
    template <typename T>
    KrausChannel(const VT<VVT<CT<T>>>& kraus_operator_set, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::KRAUS, obj_qubits, ctrl_qubits) {
        std::transform(kraus_operator_set.begin(), kraus_operator_set.end(), std::back_inserter(kraus_operator_set_),
                       [](auto& k) { return tensor::Matrix(k); });
    }
    const VT<tensor::Matrix>& GetKrausOperatorSet() const {
        return kraus_operator_set_;
    }

 private:
    VT<tensor::Matrix> kraus_operator_set_;
};

class AmplitudeDampingChannel : public BasicGate {
 public:
    AmplitudeDampingChannel(bool daggered, double damping_coeff, const qbits_t& obj_qubits,
                            const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::AD, obj_qubits, ctrl_qubits), damping_coeff_(damping_coeff), daggered_(daggered) {
    }
    double GetDampingCoeff() const {
        return damping_coeff_;
    }
    bool Daggered() const {
        return daggered_;
    }

 private:
    double damping_coeff_;
    bool daggered_;
};
class PhaseDampingChannel : public BasicGate {
 public:
    PhaseDampingChannel(double damping_coeff, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::PD, obj_qubits, ctrl_qubits), damping_coeff_(damping_coeff) {
    }
    double GetDampingCoeff() const {
        return damping_coeff_;
    }

 private:
    double damping_coeff_;
};
class ThermalRelaxationChannel : public BasicGate {
 public:
    ThermalRelaxationChannel(double t1, double t2, double gate_time, const qbits_t& obj_qubits,
                             const qbits_t& ctrl_qubits = {})
        : BasicGate(GateID::TR, obj_qubits, ctrl_qubits), t1_(t1), t2_(t2), gate_time_(gate_time) {
    }
    double GetT1() const {
        return t1_;
    }
    double GetT2() const {
        return t2_;
    }
    double GetGateTime() const {
        return gate_time_;
    }

 private:
    double t1_;
    double t2_;
    double gate_time_;
};

class CustomGate : public Parameterizable {
 public:
    CustomGate(const std::string& name, uint64_t m_addr, uint64_t dm_addr, int dim,
               const parameter::ParameterResolver pr, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::CUSTOM, {pr}, obj_qubits, ctrl_qubits)
        , name_(name)
        , numba_param_matrix_(m_addr, dim)
        , numba_param_diff_matrix_(dm_addr, dim) {
        if (!this->Parameterized()) {
            base_matrix_ = this->numba_param_matrix_(
                tensor::ops::cpu::to_vector<double>(this->GetCoeffs()[0].const_value)[0]);
        }
    }
    CustomGate(const std::string& name, const tensor::Matrix& mat, const qbits_t& obj_qubits,
               const qbits_t& ctrl_qubits = {})
        : Parameterizable(GateID::CUSTOM, {parameter::ParameterResolver()}, obj_qubits, ctrl_qubits)
        , name_(name)
        , base_matrix_(mat) {
    }
    const tensor::Matrix& GetBaseMatrix() const {
        return base_matrix_;
    }
    const NumbaMatFunWrapper& GetMatrixWrapper() const {
        return numba_param_matrix_;
    }
    const NumbaMatFunWrapper& GetDiffMatrixWrapper() const {
        return numba_param_diff_matrix_;
    }

 private:
    std::string name_;
    NumbaMatFunWrapper numba_param_matrix_;
    NumbaMatFunWrapper numba_param_diff_matrix_;
    tensor::Matrix base_matrix_;
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_basic_gate_H_
