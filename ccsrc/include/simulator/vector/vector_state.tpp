/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef INCLUDE_VECTOR_VECTOR_STATE_TPP
#define INCLUDE_VECTOR_VECTOR_STATE_TPP

#include <cmath>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/basic_math.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/vector/gradient.h"
#include "simulator/vector/vector_state.h"

namespace mindquantum::sim::vector::detail {
template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::SetSeed(unsigned new_seed) {
    this->seed = new_seed;
    this->rnd_eng_ = RndEngine(new_seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
}

template <typename qs_policy_t_>
VectorState<qs_policy_t_>::VectorState(qbit_t n_qubits, unsigned seed)
    : n_qubits(n_qubits), dim(static_cast<uint64_t>(1) << n_qubits), seed(seed), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
VectorState<qs_policy_t_>::VectorState(qbit_t n_qubits, unsigned seed, qs_data_p_t vec)
    : n_qubits(n_qubits), dim(static_cast<uint64_t>(1) << n_qubits), seed(seed), rnd_eng_(seed) {
    qs = qs_policy_t::Copy(vec, dim);
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
VectorState<qs_policy_t_>::VectorState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed)
    : qs(qs), n_qubits(n_qubits), dim(static_cast<uint64_t>(1) << n_qubits), seed(seed), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
VectorState<qs_policy_t_>::VectorState(const VectorState<qs_policy_t>& sim) {
    this->qs = qs_policy_t::Copy(sim.qs, sim.dim);
    this->dim = sim.dim;
    this->n_qubits = sim.n_qubits;
    this->seed = sim.seed;
    this->rnd_eng_ = RndEngine(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::operator=(const VectorState<qs_policy_t>& sim) -> derived_t& {
    qs_policy_t::FreeState(&(this->qs));
    this->qs = qs_policy_t::Copy(sim.qs, sim.dim);
    this->dim = sim.dim;
    this->n_qubits = sim.n_qubits;
    this->seed = sim.seed;
    this->rnd_eng_ = RndEngine(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
    return *this;
};

template <typename qs_policy_t_>
VectorState<qs_policy_t_>::VectorState(VectorState<qs_policy_t>&& sim) {
    this->qs = sim.qs;
    this->dim = sim.dim;
    this->n_qubits = sim.n_qubits;
    this->seed = sim.seed;
    sim.qs = nullptr;
    this->rnd_eng_ = RndEngine(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::operator=(VectorState<qs_policy_t>&& sim) -> derived_t& {
    qs_policy_t::FreeState(&(this->qs));
    this->qs = sim.qs;
    this->dim = sim.dim;
    this->n_qubits = sim.n_qubits;
    this->seed = sim.seed;
    sim.qs = nullptr;
    this->rnd_eng_ = RndEngine(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
    return *this;
}

template <typename qs_policy_t_>
tensor::TDtype VectorState<qs_policy_t_>::DType() {
    return tensor::to_dtype_v<py_qs_data_t>;
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::Reset() {
    qs_policy_t::Reset(&qs);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::Display(qbit_t qubits_limit) const {
    qs_policy_t::Display(qs, n_qubits, qubits_limit);
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetQS() const -> VT<py_qs_data_t> {
    return qs_policy_t::GetQS(qs, dim);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::SetQS(const VT<py_qs_data_t>& qs_out) {
    qs_policy_t::SetQS(&qs, qs_out, dim);
}

template <typename qs_policy_t_>
index_t VectorState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate>& gate,
                                             const parameter::ParameterResolver& pr, bool diff) {
    auto id = gate->GetID();
    switch (id) {
        case GateID::I:
            break;
        case GateID::X:
            qs_policy_t::ApplyX(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::Y:
            qs_policy_t::ApplyY(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::Z:
            qs_policy_t::ApplyZ(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::H:
            qs_policy_t::ApplyH(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::S:
            qs_policy_t::ApplySGate(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::Sdag:
            qs_policy_t::ApplySdag(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::T:
            qs_policy_t::ApplyT(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::Tdag:
            qs_policy_t::ApplyTdag(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::SX:
            qs_policy_t::ApplySX(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::SXdag:
            qs_policy_t::ApplySXdag(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::PauliString: {
            auto g = static_cast<PauliString*>(gate.get());
            qs_policy_t::ApplyPauliString(&qs, g->GetPauliMask(), g->GetCtrlMask(), dim);
            break;
        }
        case GateID::SWAP:
            qs_policy_t::ApplySWAP(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
            break;
        case GateID::ISWAP: {
            bool daggered = static_cast<ISWAPGate*>(gate.get())->Daggered();
            qs_policy_t::ApplyISWAP(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), daggered, dim);
        } break;
        case GateID::SWAPalpha: {
            auto g = static_cast<SWAPalphaGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplySWAPalpha(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::RPS: {
            auto g = static_cast<RotPauliString*>(gate.get());
            const auto& ps = g->GetPauliString();
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRPS(&qs, ps.GetPauliMask(), ps.GetCtrlMask(), val, dim, diff);
        } break;
        case GateID::RX: {
            auto g = static_cast<RXGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRX(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::RY: {
            auto g = static_cast<RYGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRY(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::RZ: {
            auto g = static_cast<RZGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRZ(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Rxx: {
            auto g = static_cast<RxxGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxx(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Ryy: {
            auto g = static_cast<RyyGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRyy(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Rzz: {
            auto g = static_cast<RzzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRzz(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Rxy: {
            auto g = static_cast<RxyGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxy(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Rxz: {
            auto g = static_cast<RxzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxz(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Ryz: {
            auto g = static_cast<RyzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRyz(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::Givens: {
            auto g = static_cast<GivensGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyGivens(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::PS: {
            auto g = static_cast<PSGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyPS(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::GP: {
            auto g = static_cast<GPGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyGP(&qs, gate->GetObjQubits()[0], gate->GetCtrlQubits(), val, dim, diff);
        } break;
        case GateID::U3: {
            if (diff) {
                std::runtime_error("Can not apply differential format of U3 gate on quantum states currently.");
            }
            auto u3 = static_cast<U3*>(gate.get());
            tensor::Matrix m;
            if (!u3->Parameterized()) {
                m = u3->GetBaseMatrix();
            } else {
                auto theta_const = u3->GetTheta().Combination(pr).const_value;
                auto phi_const = u3->GetPhi().Combination(pr).const_value;
                auto lambda_const = u3->GetLambda().Combination(pr).const_value;
                m = U3Matrix(theta_const, phi_const, lambda_const);
            }
            qs_policy_t::ApplySingleQubitMatrix(qs, &qs, gate->GetObjQubits()[0], gate->GetCtrlQubits(),
                                                tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        } break;
        case GateID::Rn: {
            if (diff) {
                std::runtime_error("Can not apply differential format of Rn gate on quantum states currently.");
            }
            auto rn = static_cast<Rn*>(gate.get());
            tensor::Matrix m;
            if (!rn->Parameterized()) {
                m = rn->GetBaseMatrix();
            } else {
                auto alpha = rn->GetAlpha().Combination(pr).const_value;
                auto beta = rn->GetBeta().Combination(pr).const_value;
                auto gamma = rn->GetGamma().Combination(pr).const_value;
                m = RnMatrix(alpha, beta, gamma);
            }
            qs_policy_t::ApplySingleQubitMatrix(qs, &qs, gate->GetObjQubits()[0], gate->GetCtrlQubits(),
                                                tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        } break;
        case GateID::FSim: {
            if (diff) {
                std::runtime_error("Can not apply differential format of FSim gate on quantum states currently.");
            }
            auto fsim = static_cast<FSim*>(gate.get());
            tensor::Matrix m;
            if (!fsim->Parameterized()) {
                m = fsim->GetBaseMatrix();
            } else {
                auto theta = fsim->GetTheta().Combination(pr).const_value;
                auto phi = fsim->GetPhi().Combination(pr).const_value;
                m = FSimMatrix(theta, phi);
            }
            qs_policy_t::ApplyTwoQubitsMatrix(qs, &qs, gate->GetObjQubits(), gate->GetCtrlQubits(),
                                              tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        } break;
        case GateID::M:
            return this->ApplyMeasure(gate);
        case GateID::PL:
            this->ApplyPauliChannel(gate);
            break;
        case GateID::GPL:
            this->ApplyGroupedPauliChannels(gate);
            break;
        case GateID::DEP:
            this->ApplyDepolarizingChannel(gate);
            break;
        case GateID::AD:
        case GateID::PD:
            this->ApplyDampingChannel(gate);
            break;
        case GateID::KRAUS:
            this->ApplyKrausChannel(gate);
            break;
        case GateID::TR:
            this->ApplyThermalRelaxationChannel(gate);
            break;
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix mat;
            if (!g->Parameterized()) {
                mat = g->GetBaseMatrix();
            } else {
                double val = tensor::ops::cpu::to_vector<double>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
                if (!diff) {
                    mat = g->GetMatrixWrapper()(val);
                } else {
                    mat = g->GetDiffMatrixWrapper()(val);
                }
            }
            qs_policy_t::ApplyMatrixGate(qs, &qs, gate->GetObjQubits(), gate->GetCtrlQubits(),
                                         tensor::ops::cpu::to_vector<py_qs_data_t>(mat), dim);
            break;
        }
        default:
            throw std::invalid_argument(fmt::format("Apply of gate {} not implement.", id));
    }
    return 2;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::ApplyMeasure(const std::shared_ptr<BasicGate>& gate) -> index_t {
    auto obj_qubits = gate->GetObjQubits();
    auto ctrl_qubits = gate->GetCtrlQubits();
    auto m_g = static_cast<MeasureGate*>(gate.get());
    index_t one_mask = (static_cast<uint64_t>(1) << obj_qubits[0]);
    auto one_amp = qs_policy_t::OneStateVdot(qs, qs, obj_qubits[0], dim).real();
    bool collapse_to_one = (rng_() < one_amp);
    qs_data_t norm_fact = (collapse_to_one) ? 1 / std::sqrt(one_amp) : 1 / std::sqrt(1 - one_amp);
    if (collapse_to_one) {
        if (m_g->WillReset() && m_g->GetRestTo() == 0) {
            qs_policy_t::ApplyXLike(&qs, obj_qubits, ctrl_qubits, norm_fact, 0, dim);
        } else {
            qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, 0, norm_fact, dim);
        }
    } else {
        if (m_g->WillReset() && m_g->GetRestTo() == 1) {
            qs_policy_t::ApplyXLike(&qs, obj_qubits, ctrl_qubits, 0, norm_fact, dim);
        } else {
            qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, norm_fact, 0, dim);
        }
    }
    return static_cast<index_t>(collapse_to_one);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyChannel(const std::shared_ptr<BasicGate>& gate) {
    auto id = gate->GetID();
    switch (id) {
        case GateID::PL:
            this->ApplyPauliChannel(gate);
            break;
        case GateID::GPL:
            this->ApplyGroupedPauliChannels(gate);
            break;
        case GateID::DEP:
            this->ApplyDepolarizingChannel(gate);
            break;
        case GateID::KRAUS:
            this->ApplyKrausChannel(gate);
            break;
        case GateID::TR:
            this->ApplyThermalRelaxationChannel(gate);
            break;
        case GateID::AD:
        case GateID::PD:
            this->ApplyDampingChannel(gate);
            break;
        default:
            throw std::invalid_argument(fmt::format("{} is not a noise channel.", id));
    }
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyPauliChannel(const std::shared_ptr<BasicGate>& gate) {
    double r = static_cast<double>(rng_());
    auto g = static_cast<PauliChannel*>(gate.get());
    auto cumulative_probs = g->GetCumulativeProbs();
    auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
    size_t gate_index;
    if (it != cumulative_probs.begin()) {
        gate_index = std::distance(cumulative_probs.begin(), it) - 1;
    } else {
        gate_index = 0;
    }
    if (gate_index == 0) {
        qs_policy_t::ApplyX(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
    } else if (gate_index == 1) {
        qs_policy_t::ApplyY(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
    } else if (gate_index == 2) {
        qs_policy_t::ApplyZ(&qs, gate->GetObjQubits(), gate->GetCtrlQubits(), dim);
    }
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyGroupedPauliChannels(const std::shared_ptr<BasicGate>& gate) {
    auto g = static_cast<GroupedPauliChannel*>(gate.get());
    VT<PauliWord> term;
    // for (auto& pauli_channel : g->pauli_channels) {
    for (auto pauli_channel = g->begin(); pauli_channel != g->end(); ++pauli_channel) {
        auto cumulative_probs = pauli_channel->GetCumulativeProbs();
        double r = static_cast<double>(rng_());
        auto it = std::lower_bound(cumulative_probs.begin(), cumulative_probs.end(), r);
        size_t gate_index;
        if (it != cumulative_probs.begin()) {
            gate_index = std::distance(cumulative_probs.begin(), it) - 1;
        } else {
            gate_index = 0;
        }
        if (gate_index == 0) {
            term.push_back(PauliWord(pauli_channel->GetObjQubits().at(0), 'X'));
        } else if (gate_index == 1) {
            term.push_back(PauliWord(pauli_channel->GetObjQubits().at(0), 'Y'));
        } else if (gate_index == 2) {
            term.push_back(PauliWord(pauli_channel->GetObjQubits().at(0), 'Z'));
        }
    }
    PauliMask pauli_mask = GetPauliMask(term);
    qs_policy_t::ApplyPauliString(&qs, pauli_mask, 0, dim);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyDepolarizingChannel(const std::shared_ptr<BasicGate>& gate) {
    struct PauliSampling {
        DepolarizingChannel gate;
        index_t dim;
        std::function<double()> rng_;
        PauliSampling(const DepolarizingChannel& gate_in, index_t dim_in, std::function<double()> rng_in)
            : gate(gate_in), dim(dim_in), rng_(rng_in) {
        }
        void uniform_sampling(qs_data_p_t* qs, int idx) {
            if (idx >= gate.GetObjQubits().size()) {
                return;
            }
            double p_i = static_cast<double>(rng_());
            qbits_t obj{gate.GetObjQubits()[idx]};
            if (p_i < 1.0 / 4.0) {
                qs_policy_t::ApplyX(qs, obj, gate.GetCtrlQubits(), dim);
            } else if (p_i < 2.0 / 4.0) {
                qs_policy_t::ApplyY(qs, obj, gate.GetCtrlQubits(), dim);
            } else if (p_i < 3.0 / 4.0) {
                qs_policy_t::ApplyZ(qs, obj, gate.GetCtrlQubits(), dim);
            }
            PauliSampling::uniform_sampling(qs, idx + 1);
        }
        void non_uniform_sampling(qs_data_p_t* qs, int idx, double p) {
            if (idx >= gate.GetObjQubits().size()) {
                return;
            }
            if (static_cast<double>(rng_()) < p * 3.0 / 4.0) {
                double p_i = static_cast<double>(rng_());
                qbits_t obj{gate.GetObjQubits()[idx]};
                if (p_i < 1.0 / 3.0) {
                    qs_policy_t::ApplyX(qs, obj, gate.GetCtrlQubits(), dim);
                } else if (p_i < 2.0 / 3.0) {
                    qs_policy_t::ApplyY(qs, obj, gate.GetCtrlQubits(), dim);
                } else {
                    qs_policy_t::ApplyZ(qs, obj, gate.GetCtrlQubits(), dim);
                }
                PauliSampling::uniform_sampling(qs, idx + 1);
            } else {
                if (4 > 3 * p) {
                    PauliSampling::non_uniform_sampling(qs, idx + 1, p / (4 - 3 * p));
                }
            }
        }
    };
    auto g = static_cast<DepolarizingChannel*>(gate.get());
    double p = g->GetProb();
    auto sampler = PauliSampling(*g, dim, rng_);
    sampler.non_uniform_sampling(&qs, 0, p);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyKrausChannel(const std::shared_ptr<BasicGate>& gate) {
    calc_type relative_prob = 0;
    calc_type total_prob = 1;
    auto g = static_cast<KrausChannel*>(gate.get());
    auto kraus_operator_set = g->GetKrausOperatorSet();
    for (size_t n_kraus = 0; n_kraus < kraus_operator_set.size(); n_kraus++) {
        auto m = tensor::ops::cpu::to_vector<py_qs_data_t>(kraus_operator_set[n_kraus]);
        // I case
        if ((std::abs(m[0][1]) < 1e-6) && (std::abs(m[1][0]) < 1e-6) && (std::abs(m[0][0] - m[1][1]) < 1e-6)) {
            calc_type prob = std::real(m[0][0] * std::conj(m[0][0]));
            relative_prob = prob / total_prob;
            total_prob = total_prob - prob;
            if (static_cast<calc_type>(rng_()) < relative_prob) {
                if (std::imag(m[0][0]) < 1e-6) {
                    return;
                } else {  // I with a phase
                    qs_policy_t::QSMulValue(qs, &qs, m[0][0] / std::sqrt(prob), dim);
                    return;
                }
            }
        } else {
            auto tmp_qs = qs_policy_t::InitState(dim);
            qs_policy_t::ApplySingleQubitMatrix(qs, &tmp_qs, gate->GetObjQubits()[0], gate->GetCtrlQubits(), m, dim);
            calc_type prob = qs_policy_t::Vdot(tmp_qs, tmp_qs, dim).real();
            relative_prob = prob / total_prob;
            total_prob = total_prob - prob;
            calc_type renormal_factor = 1 / std::sqrt(prob);
            if (static_cast<calc_type>(rng_()) < relative_prob) {
                qs_policy_t::QSMulValue(tmp_qs, &tmp_qs, renormal_factor, dim);
                qs_policy_t::FreeState(&qs);
                qs = tmp_qs;
                tmp_qs = nullptr;
                break;
            }
            qs_policy_t::FreeState(&tmp_qs);
        }
    }
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyDampingChannel(const std::shared_ptr<BasicGate>& gate) {
    auto obj_qubits = gate->GetObjQubits();
    auto ctrl_qubits = gate->GetCtrlQubits();
    calc_type renormal_factor = qs_policy_t::OneStateVdot(qs, qs, obj_qubits[0], dim).real();
    if (renormal_factor < 1e-6) {
        return;
    }
    auto id = gate->GetID();
    double damping_coeff = 0;
    if (id == GateID::AD) {
        damping_coeff = static_cast<AmplitudeDampingChannel*>(gate.get())->GetDampingCoeff();
    } else {
        damping_coeff = static_cast<PhaseDampingChannel*>(gate.get())->GetDampingCoeff();
    }
    calc_type prob = damping_coeff * renormal_factor;
    if (static_cast<calc_type>(rng_()) <= prob) {
        if (id == GateID::AD) {
            qs_policy_t::ApplyXLike(&qs, obj_qubits, ctrl_qubits, 1 / std::sqrt(renormal_factor), 0, dim);
        } else {
            qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, 0, 1 / std::sqrt(renormal_factor), dim);
        }
    } else {
        calc_type coeff_a = 1 / std::sqrt(1 - prob);
        calc_type coeff_b = std::sqrt(1 - damping_coeff) / std::sqrt(1 - prob);
        qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, coeff_a, coeff_b, dim);
    }
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyThermalRelaxationChannel(const std::shared_ptr<BasicGate>& gate) {
    calc_type t1 = static_cast<ThermalRelaxationChannel*>(gate.get())->GetT1();
    calc_type t2 = static_cast<ThermalRelaxationChannel*>(gate.get())->GetT2();
    calc_type gate_time = static_cast<ThermalRelaxationChannel*>(gate.get())->GetGateTime();
    calc_type e1 = std::exp(-gate_time / t1);
    calc_type e2 = std::exp(-gate_time / t2);
    calc_type p_reset = 1 - e1;
    auto obj_qubits = gate->GetObjQubits();
    auto ctrl_qubits = gate->GetCtrlQubits();
    if (t1 >= t2) {
        calc_type pz = e1 * (1 - e2 / e1) / 2;
        calc_type r = static_cast<calc_type>(rng_());
        if (r < 1 - pz - p_reset) {  // I case
            return;
        } else if (r < 1 - p_reset) {  // Z case
            qs_policy_t::ApplyZ(&qs, obj_qubits, ctrl_qubits, dim);
        } else {  // reset case
            calc_type zero_factor = qs_policy_t::ZeroStateVdot(qs, qs, obj_qubits[0], dim).real();
            if (static_cast<calc_type>(rng_()) < zero_factor) {
                qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, 1 / std::sqrt(zero_factor), 0, dim);
            } else {
                qs_policy_t::ApplyXLike(&qs, obj_qubits, ctrl_qubits, 1 / std::sqrt(1 - zero_factor), 0, dim);
            }
        }
    } else if (2 * t1 > t2) {
        calc_type r = static_cast<calc_type>(rng_());
        calc_type eigenvalue0 = (2 - p_reset + std::sqrt(p_reset * p_reset + 4 * e2 * e2)) / 2;
        calc_type eigenvalue1 = (2 - p_reset - std::sqrt(p_reset * p_reset + 4 * e2 * e2)) / 2;
        calc_type eigen_vector0 = (eigenvalue0 - e1) / e2;
        calc_type eigen_vector1 = (eigenvalue1 - e1) / e2;
        calc_type zero_factor = qs_policy_t::ZeroStateVdot(qs, qs, obj_qubits[0], dim).real();
        calc_type one_factor = 1 - zero_factor;
        calc_type c0 = eigen_vector0 * eigen_vector0 * zero_factor + one_factor;
        calc_type p0 = c0 * eigenvalue0 / (eigen_vector0 * eigen_vector0 + 1);
        if (r < p0) {  // Kraus operator from eigenvalue0 case
            qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, eigen_vector0 / std::sqrt(c0), 1 / std::sqrt(c0),
                                    dim);
        } else if (r < 1 - p_reset * one_factor) {  // Kraus operator from eigenvalue1 case
            calc_type c1 = eigen_vector1 * eigen_vector1 * zero_factor + one_factor;
            qs_policy_t::ApplyILike(&qs, obj_qubits, ctrl_qubits, eigen_vector1 / std::sqrt(c1), 1 / std::sqrt(c1),
                                    dim);
        } else {  // Kraus operator from eigenvalue2 (reset) case
            qs_policy_t::ApplyXLike(&qs, obj_qubits, ctrl_qubits, 1 / std::sqrt(one_factor), 0, dim);
        }
    } else {
        std::runtime_error("(T2 >= 2 * T1) is invalid case for thermal relaxation channel.");
    }
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                               const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    py_qs_data_t out;
    auto sub_seed = static_cast<unsigned int>(static_cast<calc_type>(rng_()) * (1 << 20));
    auto ket = derived_t(n_qubits, sub_seed, qs);
    ket.ApplyCircuit(circ, pr);
    if (ham.how_to_ == ORIGIN) {
        out = qs_policy_t::ExpectationOfTerms(ket.qs, ket.qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, ham.ham_sparse_second_, ket.qs, ket.qs, dim);
    } else {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, ket.qs, ket.qs, dim);
    }
    return out;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                               const circuit_t& circ_left, const parameter::ParameterResolver& pr) const
    -> py_qs_data_t {
    py_qs_data_t out;
    auto sub_seed_bra = static_cast<unsigned int>(static_cast<calc_type>(rng_()) * (1 << 20));
    auto sub_seed_ket = static_cast<unsigned int>(static_cast<calc_type>(rng_()) * (1 << 20));
    auto ket = derived_t(n_qubits, sub_seed_ket, qs);
    auto bra = derived_t(n_qubits, sub_seed_bra, qs);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    if (ham.how_to_ == ORIGIN) {
        out = qs_policy_t::ExpectationOfTerms(bra.qs, ket.qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, ham.ham_sparse_second_, bra.qs, ket.qs, dim);
    } else {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, bra.qs, ket.qs, dim);
    }
    return out;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                               const circuit_t& circ_left, const derived_t& simulator_left,
                                               const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    auto sub_seed_bra = static_cast<unsigned int>(static_cast<calc_type>(simulator_left.rng_()) * (1 << 20));
    auto sub_seed_ket = static_cast<unsigned int>(static_cast<calc_type>(rng_()) * (1 << 20));
    auto ket = derived_t(n_qubits, sub_seed_ket, qs);
    auto bra = derived_t(n_qubits, sub_seed_bra, simulator_left.qs);
    ket.ApplyCircuit(circ_right, pr);
    bra.ApplyCircuit(circ_left, pr);
    py_qs_data_t out;
    if (ham.how_to_ == ORIGIN) {
        out = qs_policy_t::ExpectationOfTerms(bra.qs, ket.qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, ham.ham_sparse_second_, bra.qs, ket.qs, dim);
    } else {
        out = qs_policy_t::ExpectationOfCsr(ham.ham_sparse_main_, bra.qs, ket.qs, dim);
    }
    return out;
}

template <typename qs_policy_t_>
template <typename policy_des, template <typename p_src, typename p_des> class cast_policy>
auto VectorState<qs_policy_t_>::astype(unsigned seed) const -> VectorState<policy_des> {
    return VectorState<policy_des>(cast_policy<qs_policy_t, policy_des>::cast(this->qs, this->dim), this->n_qubits,
                                   seed);
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::ExpectDiffGate(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                               const std::shared_ptr<BasicGate>& gate,
                                               const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    auto id = gate->GetID();
    auto g = static_cast<Parameterizable*>(gate.get());
    auto val = tensor::ops::cpu::to_vector<calc_type>(g->GetCoeffs()[0].Combination(pr).const_value)[0];
    VT<py_qs_data_t> grad = {0};
    switch (id) {
        case GateID::RX:
            grad[0] = qs_policy_t::ExpectDiffRX(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::RY:
            grad[0] = qs_policy_t::ExpectDiffRY(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::RZ:
            grad[0] = qs_policy_t::ExpectDiffRZ(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::RPS: {
            auto rps = static_cast<RotPauliString*>(gate.get());
            const auto& pauli = rps->GetPauliString();
            grad[0] = qs_policy_t::ExpectDiffRPS(bra, ket, pauli.GetPauliMask(), pauli.GetCtrlMask(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        }
        case GateID::Rxx:
            grad[0] = qs_policy_t::ExpectDiffRxx(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rzz:
            grad[0] = qs_policy_t::ExpectDiffRzz(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Ryy:
            grad[0] = qs_policy_t::ExpectDiffRyy(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rxy:
            grad[0] = qs_policy_t::ExpectDiffRxy(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rxz:
            grad[0] = qs_policy_t::ExpectDiffRxz(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Ryz:
            grad[0] = qs_policy_t::ExpectDiffRyz(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Givens:
            grad[0] = qs_policy_t::ExpectDiffGivens(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::PS:
            grad[0] = qs_policy_t::ExpectDiffPS(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::GP:
            grad[0] = qs_policy_t::ExpectDiffGP(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::SWAPalpha:
            grad[0] = qs_policy_t::ExpectDiffSWAPalpha(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(), val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix mat = g->GetDiffMatrixWrapper()(val);
            grad[0] = qs_policy_t::ExpectDiffMatrixGate(bra, ket, gate->GetObjQubits(), gate->GetCtrlQubits(),
                                                        tensor::ops::cpu::to_vector<py_qs_data_t>(mat), dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        }
        case GateID::U3:
            return ExpectDiffU3(bra, ket, gate, pr, dim);
        case GateID::Rn:
            return ExpectDiffRn(bra, ket, gate, pr, dim);
        case GateID::FSim:
            return ExpectDiffFSim(bra, ket, gate, pr, dim);
        default:
            throw std::invalid_argument(fmt::format("Expectation of gate {} not implement.", id));
    }
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::ExpectDiffU3(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                             const std::shared_ptr<BasicGate>& gate,
                                             const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    VT<py_qs_data_t> grad = {0, 0, 0};
    auto u3 = static_cast<U3*>(gate.get());
    if (u3->Parameterized()) {
        tensor::Matrix m;
        auto theta = u3->GetTheta();
        auto phi = u3->GetPhi();
        auto lambda = u3->GetLambda();
        auto theta_cosnt = theta.Combination(pr).const_value;
        auto phi_cosnt = phi.Combination(pr).const_value;
        auto lambda_cosnt = lambda.Combination(pr).const_value;
        if (theta.data_.size() != theta.no_grad_parameters_.size()) {
            m = U3DiffThetaMatrix(theta_cosnt, phi_cosnt, lambda_cosnt);
            grad[0] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->GetObjQubits(), u3->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (phi.data_.size() != phi.no_grad_parameters_.size()) {
            m = U3DiffPhiMatrix(theta_cosnt, phi_cosnt, lambda_cosnt);
            grad[1] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->GetObjQubits(), u3->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (lambda.data_.size() != lambda.no_grad_parameters_.size()) {
            m = U3DiffLambdaMatrix(theta_cosnt, phi_cosnt, lambda_cosnt);
            grad[2] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->GetObjQubits(), u3->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
    }
    return tensor::Matrix(VVT<py_qs_data_t>{grad});
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::ExpectDiffRn(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                             const std::shared_ptr<BasicGate>& gate,
                                             const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    VT<py_qs_data_t> grad = {0, 0, 0};
    auto rn = static_cast<Rn*>(gate.get());
    if (rn->Parameterized()) {
        tensor::Matrix m;
        auto alpha = rn->GetAlpha();
        auto beta = rn->GetBeta();
        auto gamma = rn->GetGamma();
        auto alpha_const = alpha.Combination(pr).const_value;
        auto beta_const = beta.Combination(pr).const_value;
        auto gamma_const = gamma.Combination(pr).const_value;
        if (alpha.data_.size() != alpha.no_grad_parameters_.size()) {
            m = RnDiffAlphaMatrix(alpha_const, beta_const, gamma_const);
            grad[0] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, rn->GetObjQubits(), rn->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (beta.data_.size() != beta.no_grad_parameters_.size()) {
            m = RnDiffBetaMatrix(alpha_const, beta_const, gamma_const);
            grad[1] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, rn->GetObjQubits(), rn->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (gamma.data_.size() != gamma.no_grad_parameters_.size()) {
            m = RnDiffGammaMatrix(alpha_const, beta_const, gamma_const);
            grad[2] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, rn->GetObjQubits(), rn->GetCtrlQubits(),
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
    }
    return tensor::Matrix(VVT<py_qs_data_t>{grad});
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::ExpectDiffFSim(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                               const std::shared_ptr<BasicGate>& gate,
                                               const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    VT<py_qs_data_t> grad = {0, 0};
    auto fsim = static_cast<FSim*>(gate.get());
    if (fsim->Parameterized()) {
        tensor::Matrix m;
        auto theta = fsim->GetTheta();
        auto phi = fsim->GetPhi();
        auto theta_const = theta.Combination(pr).const_value;
        auto phi_const = phi.Combination(pr).const_value;
        if (theta.data_.size() != theta.no_grad_parameters_.size()) {
            m = FSimDiffThetaMatrix(theta_const);  // can be optimized.
            grad[0] = qs_policy_t::ExpectDiffTwoQubitsMatrix(bra, ket, fsim->GetObjQubits(), fsim->GetCtrlQubits(),
                                                             tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (phi.data_.size() != phi.no_grad_parameters_.size()) {
            m = FSimDiffPhiMatrix(phi_const);
            grad[1] = qs_policy_t::ExpectDiffTwoQubitsMatrix(bra, ket, fsim->GetObjQubits(), fsim->GetCtrlQubits(),
                                                             tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
    }
    return tensor::Matrix(VVT<py_qs_data_t>{grad});
}

template <typename qs_policy_t_>
std::map<std::string, int> VectorState<qs_policy_t_>::ApplyCircuit(const circuit_t& circ,
                                                                   const parameter::ParameterResolver& pr) {
    std::map<std::string, int> result;
    for (auto& g : circ) {
        if (g->GetID() == GateID::M) {
            result[static_cast<MeasureGate*>(g.get())->Name()] = ApplyMeasure(g);
        } else {
            ApplyGate(g, pr, false);
        }
    }
    return result;
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    qs_data_p_t new_qs;
    if (ham.how_to_ == ORIGIN) {
        new_qs = qs_policy_t::ApplyTerms(&qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        new_qs = qs_policy_t::CsrDotVec(ham.ham_sparse_main_, ham.ham_sparse_second_, qs, dim);
    } else {
        new_qs = qs_policy_t::CsrDotVec(ham.ham_sparse_main_, qs, dim);
    }
    qs_policy_t::FreeState(&qs);
    qs = new_qs;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetCircuitMatrix(const circuit_t& circ, const parameter::ParameterResolver& pr) const
    -> VVT<py_qs_data_t> {
    VVT<CT<calc_type>> out((static_cast<uint64_t>(1) << n_qubits),
                           VT<CT<calc_type>>((static_cast<uint64_t>(1) << n_qubits), 0));
    for (size_t i = 0; i < (static_cast<uint64_t>(1) << n_qubits); i++) {
        auto sim = VectorState<qs_policy_t>(n_qubits, seed);
        for (qbit_t j = 0; j < n_qubits; ++j) {
            if ((i >> j) & 1) {
                qs_policy_t_::ApplyX(&(sim.qs), qbits_t({j}), qbits_t({}), sim.dim);
            }
        }
        sim.ApplyCircuit(circ, pr);
        auto v = sim.GetQS();
        for (size_t j = 0; j < v.size(); ++j) {
            out[j][i] = v[j];
        }
    }
    return out;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationNonHermitianWithGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
    const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    const derived_t& simulator_left, size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
    using gradient_helper = GradientHelper<VectorStateAdjoint<derived_t>>;
    return gradient_helper::NonHermitianAdjointGradient(&simulator_left, this, left_circ, herm_left_circ, right_circ,
                                                        herm_right_circ, hams, herm_hams, enc_data, ans_data, enc_name,
                                                        ans_name, batch_threads, mea_threads);
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::QramExpectationWithGrad(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<py_qs_data_t>& init_states, const VT<calc_type>& ans_data, const VS& ans_name, size_t batch_threads,
    size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
    using gradient_helper = GradientHelper<VectorStateAdjoint<derived_t>>;
    return gradient_helper::QramHermitianAdjointGradient(this, init_states, circ, herm_circ, hams, ans_data, ans_name,
                                                         batch_threads, mea_threads);
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationWithGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
    using gradient_helper = GradientHelper<VectorStateAdjoint<derived_t>>;
    return gradient_helper::HermitianAdjointGradient(this, circ, herm_circ, hams, enc_data, ans_data, enc_name,
                                                     ans_name, batch_threads, mea_threads);
}

#define CONVERT_GATE(g_t, gate) std::static_pointer_cast<BasicGate>(std::make_shared<g_t>(*static_cast<g_t*>(gate)))

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationWithGradParameterShiftOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ_,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) -> VVT<py_qs_data_t> {
    auto circ = circ_;
    auto n_hams = hams.size();
    int max_thread = 15;
    if (n_thread == 0) {
        throw std::runtime_error("n_thread cannot be zero.");
    }
    if (n_thread > max_thread) {
        n_thread = max_thread;
    }
    if (n_thread > static_cast<int>(n_hams)) {
        n_thread = n_hams;
    }
    auto tmp_pr{pr};
    VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));
    VectorState<qs_policy_t> sim = *this;
    sim.SetSeed(static_cast<unsigned>(this->rng_() * 10000));
    sim.ApplyCircuit(circ, pr);
    int n_group = n_hams / n_thread;
    if (n_hams % n_thread) {
        n_group += 1;
    }
    for (int i = 0; i < n_group; i++) {
        int start = i * n_thread;
        int end = (i + 1) * n_thread;
        if (end > static_cast<int>(n_hams)) {
            end = n_hams;
        }
        std::vector<VectorState<qs_policy_t>> sim_rs(end - start);
        auto sim_l = sim;
        sim_l.SetSeed(static_cast<unsigned>(this->rng_() * 10000));
        for (int j = start; j < end; j++) {
            sim_rs[j - start] = sim_l;
            sim_rs[j - start].SetSeed(static_cast<unsigned>(this->rng_() * 10000));
            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
            f_and_g[j][0] = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
        }
        for (Index g_idx = 0; g_idx < circ.size(); ++g_idx) {
            auto gate = circ[g_idx];
            if (gate->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(gate.get());
                std::shared_ptr<BasicGate> tmp_gate;
                bool is_multi_pr = false;
                switch (gate->GetID()) {
                    case (GateID::RX): {
                        tmp_gate = CONVERT_GATE(RXGate, p_gate);
                        break;
                    }
                    case (GateID::RY): {
                        tmp_gate = CONVERT_GATE(RYGate, p_gate);
                        break;
                    }
                    case (GateID::RZ): {
                        tmp_gate = CONVERT_GATE(RZGate, p_gate);
                        break;
                    }
                    case (GateID::Rxx): {
                        tmp_gate = CONVERT_GATE(RxxGate, p_gate);
                        break;
                    }
                    case (GateID::Ryy): {
                        tmp_gate = CONVERT_GATE(RyyGate, p_gate);
                        break;
                    }
                    case (GateID::Rzz): {
                        tmp_gate = CONVERT_GATE(RzzGate, p_gate);
                        break;
                    }
                    case (GateID::Rxy): {
                        tmp_gate = CONVERT_GATE(RxyGate, p_gate);
                        break;
                    }
                    case (GateID::Rxz): {
                        tmp_gate = CONVERT_GATE(RxzGate, p_gate);
                        break;
                    }
                    case (GateID::Ryz): {
                        tmp_gate = CONVERT_GATE(RyzGate, p_gate);
                        break;
                    }
                    case (GateID::Givens): {
                        tmp_gate = CONVERT_GATE(GivensGate, p_gate);
                        break;
                    }
                    case (GateID::SWAPalpha): {
                        tmp_gate = CONVERT_GATE(SWAPalphaGate, p_gate);
                        break;
                    }
                    case (GateID::GP): {
                        tmp_gate = CONVERT_GATE(GPGate, p_gate);
                        break;
                    }
                    case (GateID::PS): {
                        tmp_gate = CONVERT_GATE(PSGate, p_gate);
                        break;
                    }
                    case (GateID::U3): {
                        tmp_gate = CONVERT_GATE(U3, p_gate);
                        is_multi_pr = true;
                        break;
                    }
                    case (GateID::FSim): {
                        tmp_gate = CONVERT_GATE(FSim, p_gate);
                        is_multi_pr = true;
                        break;
                    }
                    case (GateID::CUSTOM): {
                        tmp_gate = CONVERT_GATE(CustomGate, p_gate);
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            fmt::format("gate {} not supported for parameter shift rule.", gate->GetID()));
                }
                calc_type pr_shift;
                calc_type coeff;
                switch (gate->GetID()) {
                    case GateID::RX:
                    case GateID::RY:
                    case GateID::RZ:
                    case GateID::Rxx:
                    case GateID::Ryy:
                    case GateID::Rzz:
                    case GateID::Rxy:
                    case GateID::Rxz:
                    case GateID::Ryz:
                    case GateID::GP:
                    case GateID::PS:
                    case GateID::U3:
                        pr_shift = M_PI_2;
                        coeff = 0.5;
                        break;
                    case GateID::SWAPalpha:
                        pr_shift = 0.5;
                        coeff = M_PI_2;
                        break;
                    default:
                        pr_shift = 0.001;
                        coeff = 0.5 / pr_shift;
                }
                if (tmp_gate->GetCtrlQubits().size() != 0) {
                    pr_shift = 0.001;
                    coeff = 0.5 / pr_shift;
                }
                circ[g_idx] = tmp_gate;
                auto tmp_p_gate = static_cast<Parameterizable*>(tmp_gate.get());
                auto jacobi = tmp_p_gate->GetJacobi();
                const auto prs = tmp_p_gate->GetCoeffs();
                if (const auto& [title, jac] = jacobi; title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        VT<py_qs_data_t> intrin_grad_list(prs.size());
                        for (int k = 0; k < prs.size(); k++) {
                            if (prs[k].IsConst()) {
                                continue;
                            }
                            tmp_p_gate->ModifyCoeff(k, prs[k] - pr_shift);
                            if (is_multi_pr) {
                                std::string key = prs[k].data_.begin()->first;
                                parameter::tn::Tensor tmp = pr.GetItem(key);
                                tmp += -pr_shift / prs[k].data_.begin()->second;
                                tmp_pr.SetItem(key, tmp);
                            }
                            sim_l = *this;
                            sim_l.SetSeed(static_cast<unsigned>(this->rng_() * 10000));
                            sim_l.ApplyCircuit(circ, tmp_pr);
                            sim_rs[j - start] = sim_l;
                            sim_rs[j - start].SetSeed(static_cast<unsigned>(this->rng_() * 10000));
                            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
                            auto expect0 = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
                            tmp_p_gate->ModifyCoeff(k, prs[k] + pr_shift);
                            if (is_multi_pr) {
                                std::string key = prs[k].data_.begin()->first;
                                parameter::tn::Tensor tmp = pr.GetItem(key);
                                tmp += pr_shift / prs[k].data_.begin()->second;
                                tmp_pr.SetItem(key, tmp);
                            }
                            sim_l = *this;
                            sim_l.SetSeed(static_cast<unsigned>(this->rng_() * 10000));
                            sim_l.ApplyCircuit(circ, tmp_pr);
                            sim_rs[j - start] = sim_l;
                            sim_rs[j - start].SetSeed(static_cast<unsigned>(this->rng_() * 10000));
                            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
                            auto expect1 = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
                            tmp_p_gate->ModifyCoeff(k, prs[k]);
                            if (is_multi_pr) {
                                std::string key = prs[k].data_.begin()->first;
                                tmp_pr.SetItem(key, pr.GetItem(key));
                            }
                            intrin_grad_list[k] = {coeff * std::real(expect1 - expect0), 0};
                        }
                        auto intrin_grad = tensor::Matrix(VVT<py_qs_data_t>{intrin_grad_list});
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += p_grad[0][idx];
                        }
                    }
                }
                circ[g_idx] = gate;
            }
        }
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationWithGradParameterShiftMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) -> VT<VVT<py_qs_data_t>> {
    auto n_hams = hams.size();
    auto n_prs = enc_data.size();
    auto n_params = enc_name.size() + ans_name.size();
    VT<VVT<py_qs_data_t>> output;
    for (size_t i = 0; i < n_prs; i++) {
        output.push_back({});
        for (size_t j = 0; j < n_hams; j++) {
            output[i].push_back({});
            for (size_t k = 0; k < n_params + 1; k++) {
                output[i][j].push_back({0, 0});
            }
        }
    }
    MST<size_t> p_map;
    for (size_t i = 0; i < enc_name.size(); i++) {
        p_map[enc_name[i]] = i;
    }
    for (size_t i = 0; i < ans_name.size(); i++) {
        p_map[ans_name[i]] = i + enc_name.size();
    }
    if (n_prs == 1) {
        parameter::ParameterResolver pr = parameter::ParameterResolver();
        pr.SetItems(enc_name, enc_data[0]);
        pr.SetItems(ans_name, ans_data);
        output[0] = GetExpectationWithGradParameterShiftOneMulti(hams, circ, pr, p_map, mea_threads);
    } else {
        if (batch_threads == 0) {
            throw std::runtime_error("batch_threads cannot be zero.");
        }
        std::vector<std::thread> tasks;
        tasks.reserve(batch_threads);
        size_t end = 0;
        size_t offset = n_prs / batch_threads;
        size_t left = n_prs % batch_threads;
        for (size_t i = 0; i < batch_threads; ++i) {
            size_t start = end;
            end = start + offset;
            if (i < left) {
                end += 1;
            }
            auto task = [&, start, end]() {
                for (size_t n = start; n < end; n++) {
                    parameter::ParameterResolver pr = parameter::ParameterResolver();
                    pr.SetItems(enc_name, enc_data[n]);
                    pr.SetItems(ans_name, ans_data);
                    auto f_g = GetExpectationWithGradParameterShiftOneMulti(hams, circ, pr, p_map, mea_threads);
                    output[n] = f_g;
                }
            };
            tasks.emplace_back(task);
        }
        for (auto& t : tasks) {
            t.join();
        }
    }
    return output;
}

template <typename qs_policy_t_>
VT<unsigned> VectorState<qs_policy_t_>::Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                 size_t shots, const MST<size_t>& key_map, unsigned int seed) const {
    auto key_size = key_map.size();
    VT<unsigned> res(shots * key_size);
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
    for (size_t i = 0; i < shots; i++) {
        auto sim = derived_t(n_qubits, static_cast<unsigned>(rng()), qs);
        auto res0 = sim.ApplyCircuit(circ, pr);
        for (auto& [k, v] : res0) {
            res[i * key_size + key_map.at(k)] = v;
        }
    }
    return res;
}

template <typename qs_policy_t_>
VT<unsigned> VectorState<qs_policy_t_>::SamplingMeasurementEndingWithoutNoise(const circuit_t& circ,
                                                                              const parameter::ParameterResolver& pr,
                                                                              size_t shots, const MST<size_t>& key_map,
                                                                              unsigned int seed) const {
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));

    auto sim = derived_t(n_qubits, static_cast<unsigned>(rng()), qs);

    VT<int> already_measured(this->n_qubits, 0);
    circuit_t mea_circ;

    for (auto& g : circ) {
        if (g->GetID() == GateID::M) {
            auto m_qid = g->GetObjQubits()[0];
            if (already_measured[m_qid] != 0) {
                throw std::runtime_error("Quantum circuit is not a measurement ending circuit.");
            }
            already_measured[m_qid] = 1;
            mea_circ.push_back(g);
        } else {
            sim.ApplyGate(g, pr, false);
        }
    }
    return sim.Sampling(mea_circ, pr, shots, key_map, seed);
}
}  // namespace mindquantum::sim::vector::detail

#endif
