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
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/basic_math.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/vector/vector_state.h"

namespace mindquantum::sim::vector::detail {

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
    auto id = gate->id_;
    switch (id) {
        case GateID::I:
            break;
        case GateID::X:
            qs_policy_t::ApplyX(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::Y:
            qs_policy_t::ApplyY(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::Z:
            qs_policy_t::ApplyZ(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::H:
            qs_policy_t::ApplyH(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::S:
            qs_policy_t::ApplySGate(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::Sdag:
            qs_policy_t::ApplySdag(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::T:
            qs_policy_t::ApplyT(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::Tdag:
            qs_policy_t::ApplyTdag(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::SWAP:
            qs_policy_t::ApplySWAP(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            break;
        case GateID::ISWAP: {
            bool daggered = static_cast<ISWAPGate*>(gate.get())->daggered_;
            qs_policy_t::ApplyISWAP(&qs, gate->obj_qubits_, gate->ctrl_qubits_, daggered, dim);
        } break;
        case GateID::SWAPalpha: {
            auto g = static_cast<SWAPalphaGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplySWAPalpha(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::RX: {
            auto g = static_cast<RXGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRX(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::RY: {
            auto g = static_cast<RYGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRY(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::RZ: {
            auto g = static_cast<RZGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRZ(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Rxx: {
            auto g = static_cast<RxxGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxx(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Ryy: {
            auto g = static_cast<RyyGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRyy(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Rzz: {
            auto g = static_cast<RzzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRzz(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Rxy: {
            auto g = static_cast<RxyGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxy(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Rxz: {
            auto g = static_cast<RxzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRxz(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::Ryz: {
            auto g = static_cast<RyzGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyRyz(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::PS: {
            auto g = static_cast<PSGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyPS(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::GP: {
            auto g = static_cast<GPGate*>(gate.get());
            if (!g->GradRequired()) {
                diff = false;
            }
            auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
            qs_policy_t::ApplyGP(&qs, gate->obj_qubits_[0], gate->ctrl_qubits_, val, dim, diff);
        } break;
        case GateID::U3: {
            if (diff) {
                std::runtime_error("Can not apply differential format of U3 gate on quantum states currently.");
            }
            auto u3 = static_cast<U3*>(gate.get());
            tensor::Matrix m;
            if (!u3->Parameterized()) {
                m = u3->base_matrix_;
            } else {
                auto theta = u3->theta.Combination(pr).const_value;
                auto phi = u3->phi.Combination(pr).const_value;
                auto lambda = u3->lambda.Combination(pr).const_value;
                m = U3Matrix(theta, phi, lambda);
            }
            qs_policy_t::ApplySingleQubitMatrix(qs, &qs, gate->obj_qubits_[0], gate->ctrl_qubits_,
                                                tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        } break;
        case GateID::FSim: {
            if (diff) {
                std::runtime_error("Can not apply differential format of FSim gate on quantum states currently.");
            }
            auto fsim = static_cast<FSim*>(gate.get());
            tensor::Matrix m;
            if (!fsim->Parameterized()) {
                m = fsim->base_matrix_;
            } else {
                auto theta = fsim->theta.Combination(pr).const_value;
                auto phi = fsim->phi.Combination(pr).const_value;
                m = FSimMatrix(theta, phi);
            }
            qs_policy_t::ApplyTwoQubitsMatrix(qs, &qs, gate->obj_qubits_, gate->ctrl_qubits_,
                                              tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        } break;
        case GateID::M:
            return this->ApplyMeasure(gate);
        case GateID::PL:
            this->ApplyPauliChannel(gate);
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
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix mat;
            if (!g->Parameterized()) {
                mat = g->base_matrix_;
            } else {
                double val = tensor::ops::cpu::to_vector<double>(g->prs_[0].Combination(pr).const_value)[0];
                if (!diff) {
                    mat = g->numba_param_matrix_(val);
                } else {
                    mat = g->numba_param_diff_matrix_(val);
                }
            }
            qs_policy_t::ApplyMatrixGate(qs, &qs, gate->obj_qubits_, gate->ctrl_qubits_,
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
    index_t one_mask = (static_cast<uint64_t>(1) << gate->obj_qubits_[0]);
    auto one_amp = qs_policy_t::ConditionalCollect(qs, one_mask, one_mask, true, dim).real();
    index_t collapse_mask = (static_cast<index_t>(rng_() < one_amp) << gate->obj_qubits_[0]);
    qs_data_t norm_fact = (collapse_mask == 0) ? 1 / std::sqrt(1 - one_amp) : 1 / std::sqrt(one_amp);
    qs_policy_t::ConditionalMul(qs, &qs, one_mask, collapse_mask, norm_fact, 0.0, dim);
    return static_cast<index_t>(collapse_mask != 0);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyChannel(const std::shared_ptr<BasicGate>& gate) {
    auto id = gate->id_;
    switch (id) {
        case GateID::PL:
            this->ApplyPauliChannel(gate);
            break;
        case GateID::DEP:
            this->ApplyDepolarizingChannel(gate);
            break;
        case GateID::KRAUS:
            this->ApplyKrausChannel(gate);
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
    auto it = std::lower_bound(g->cumulative_probs_.begin(), g->cumulative_probs_.end(), r);
    size_t gate_index;
    if (it != g->cumulative_probs_.begin()) {
        gate_index = std::distance(g->cumulative_probs_.begin(), it) - 1;
    } else {
        gate_index = 0;
    }
    if (gate_index == 0) {
        qs_policy_t::ApplyX(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (gate_index == 1) {
        qs_policy_t::ApplyY(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (gate_index == 2) {
        qs_policy_t::ApplyZ(&qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
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
            if (idx >= gate.obj_qubits_.size()) {
                return;
            }
            double p_i = static_cast<double>(rng_());
            qbits_t obj{gate.obj_qubits_[idx]};
            if (p_i < 1.0 / 4.0) {
                qs_policy_t::ApplyX(qs, obj, gate.ctrl_qubits_, dim);
            } else if (p_i < 2.0 / 4.0) {
                qs_policy_t::ApplyY(qs, obj, gate.ctrl_qubits_, dim);
            } else if (p_i < 3.0 / 4.0) {
                qs_policy_t::ApplyZ(qs, obj, gate.ctrl_qubits_, dim);
            }
            PauliSampling::uniform_sampling(qs, idx + 1);
        }
        void non_uniform_sampling(qs_data_p_t* qs, int idx, double p) {
            if (idx >= gate.obj_qubits_.size()) {
                return;
            }
            if (static_cast<double>(rng_()) < p * 3.0 / 4.0) {
                double p_i = static_cast<double>(rng_());
                qbits_t obj{gate.obj_qubits_[idx]};
                if (p_i < 1.0 / 3.0) {
                    qs_policy_t::ApplyX(qs, obj, gate.ctrl_qubits_, dim);
                } else if (p_i < 2.0 / 3.0) {
                    qs_policy_t::ApplyY(qs, obj, gate.ctrl_qubits_, dim);
                } else {
                    qs_policy_t::ApplyZ(qs, obj, gate.ctrl_qubits_, dim);
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
    double p = g->prob_;
    auto sampler = PauliSampling(*g, dim, rng_);
    sampler.non_uniform_sampling(&qs, 0, p);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyKrausChannel(const std::shared_ptr<BasicGate>& gate) {
    auto tmp_qs = qs_policy_t::InitState(dim);
    calc_type relative_prob = 0;
    calc_type total_prob = 1;
    auto g = static_cast<KrausChannel*>(gate.get());
    for (size_t n_kraus = 0; n_kraus < g->kraus_operator_set_.size(); n_kraus++) {
        qs_policy_t::ApplySingleQubitMatrix(qs, &tmp_qs, gate->obj_qubits_[0], gate->ctrl_qubits_,
                                            tensor::ops::cpu::to_vector<py_qs_data_t>(g->kraus_operator_set_[n_kraus]),
                                            dim);
        calc_type prob = qs_policy_t::Vdot(tmp_qs, tmp_qs, dim).real();
        relative_prob = prob / total_prob;
        total_prob = total_prob - prob;
        calc_type renormal_factor = 1 / std::sqrt(prob);
        if (static_cast<calc_type>(rng_()) <= relative_prob) {
            qs_policy_t::QSMulValue(tmp_qs, &tmp_qs, renormal_factor, dim);
            qs_policy_t::FreeState(&qs);
            qs = tmp_qs;
            tmp_qs = nullptr;
            break;
        }
    }
    qs_policy_t::FreeState(&tmp_qs);
}

template <typename qs_policy_t_>
void VectorState<qs_policy_t_>::ApplyDampingChannel(const std::shared_ptr<BasicGate>& gate) {
    calc_type reduced_factor_b_square = qs_policy_t::OneStateVdot(qs, qs, gate->obj_qubits_[0], dim).real();
    calc_type reduced_factor_b = std::sqrt(reduced_factor_b_square);
    if (reduced_factor_b < 1e-8) {
        return;
    }
    auto id = gate->id_;
    double damping_coeff = 0;
    if (id == GateID::PD) {
        damping_coeff = static_cast<PhaseDampingChannel*>(gate.get())->damping_coeff_;
    } else {
        damping_coeff = static_cast<AmplitudeDampingChannel*>(gate.get())->damping_coeff_;
    }
    calc_type prob = damping_coeff * reduced_factor_b_square;
    if (static_cast<calc_type>(rng_()) <= prob) {
        auto tmp_qs = qs_policy_t::InitState(dim);
        if (id == GateID::AD) {
            VVT<py_qs_data_t> m({{0, 1 / reduced_factor_b}, {0, 0}});
            qs_policy_t::ApplySingleQubitMatrix(qs, &tmp_qs, gate->obj_qubits_[0], gate->ctrl_qubits_, m, dim);
        } else {
            qs_policy_t::ConditionalMul(qs, &tmp_qs, (static_cast<uint64_t>(1) << gate->obj_qubits_[0]),
                                        (static_cast<uint64_t>(1) << gate->obj_qubits_[0]), 1 / reduced_factor_b, 0,
                                        dim);
        }
        qs = tmp_qs;
        tmp_qs = nullptr;
        qs_policy_t::FreeState(&tmp_qs);
    } else {
        calc_type coeff_a = 1 / std::sqrt(1 - prob);
        calc_type coeff_b = std::sqrt(1 - damping_coeff) / std::sqrt(1 - prob);
        qs_policy_t::ConditionalMul(qs, &qs, (static_cast<uint64_t>(1) << gate->obj_qubits_[0]), 0, coeff_a, coeff_b,
                                    dim);
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
    auto id = gate->id_;
    auto g = static_cast<Parameterizable*>(gate.get());
    auto val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
    VT<py_qs_data_t> grad = {0};
    switch (id) {
        case GateID::RX:
            grad[0] = qs_policy_t::ExpectDiffRX(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::RY:
            grad[0] = qs_policy_t::ExpectDiffRY(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::RZ:
            grad[0] = qs_policy_t::ExpectDiffRZ(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rxx:
            grad[0] = qs_policy_t::ExpectDiffRxx(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rzz:
            grad[0] = qs_policy_t::ExpectDiffRzz(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Ryy:
            grad[0] = qs_policy_t::ExpectDiffRyy(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rxy:
            grad[0] = qs_policy_t::ExpectDiffRxy(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Rxz:
            grad[0] = qs_policy_t::ExpectDiffRxz(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::Ryz:
            grad[0] = qs_policy_t::ExpectDiffRyz(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::PS:
            grad[0] = qs_policy_t::ExpectDiffPS(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::GP:
            grad[0] = qs_policy_t::ExpectDiffGP(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::SWAPalpha:
            grad[0] = qs_policy_t::ExpectDiffSWAPalpha(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix mat = g->numba_param_diff_matrix_(val);
            grad[0] = qs_policy_t::ExpectDiffMatrixGate(bra, ket, gate->obj_qubits_, gate->ctrl_qubits_,
                                                        tensor::ops::cpu::to_vector<py_qs_data_t>(mat), dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        }
        case GateID::U3:
            return ExpectDiffU3(bra, ket, gate, pr, dim);
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
    if (u3->parameterized_) {
        tensor::Matrix m;
        auto theta = u3->theta.Combination(pr).const_value;
        auto phi = u3->phi.Combination(pr).const_value;
        auto lambda = u3->lambda.Combination(pr).const_value;
        if (u3->theta.data_.size() != u3->theta.no_grad_parameters_.size()) {
            m = U3DiffThetaMatrix(theta, phi, lambda);
            grad[0] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->obj_qubits_, u3->ctrl_qubits_,
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (u3->phi.data_.size() != u3->phi.no_grad_parameters_.size()) {
            m = U3DiffPhiMatrix(theta, phi, lambda);
            grad[1] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->obj_qubits_, u3->ctrl_qubits_,
                                                               tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (u3->lambda.data_.size() != u3->lambda.no_grad_parameters_.size()) {
            m = U3DiffLambdaMatrix(theta, phi, lambda);
            grad[2] = qs_policy_t::ExpectDiffSingleQubitMatrix(bra, ket, u3->obj_qubits_, u3->ctrl_qubits_,
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
    if (fsim->parameterized_) {
        tensor::Matrix m;
        auto theta = fsim->theta.Combination(pr).const_value;
        auto phi = fsim->phi.Combination(pr).const_value;
        if (fsim->theta.data_.size() != fsim->theta.no_grad_parameters_.size()) {
            m = FSimDiffThetaMatrix(theta);  // can be optimized.
            grad[0] = qs_policy_t::ExpectDiffTwoQubitsMatrix(bra, ket, fsim->obj_qubits_, fsim->ctrl_qubits_,
                                                             tensor::ops::cpu::to_vector<py_qs_data_t>(m), dim);
        }
        if (fsim->phi.data_.size() != fsim->phi.no_grad_parameters_.size()) {
            m = FSimDiffPhiMatrix(phi);
            grad[1] = qs_policy_t::ExpectDiffTwoQubitsMatrix(bra, ket, fsim->obj_qubits_, fsim->ctrl_qubits_,
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
        if (g->id_ == GateID::M) {
            result[static_cast<MeasureGate*>(g.get())->name_] = ApplyMeasure(g);
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
auto VectorState<qs_policy_t_>::GetExpectationWithGradOneOne(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                             const circuit_t& herm_circ,
                                                             const parameter::ParameterResolver& pr,
                                                             const MST<size_t>& p_map) const -> VT<py_qs_data_t> {
    // auto timer = Timer();
    // timer.Start("First part");
    VT<py_qs_data_t> f_and_g(1 + p_map.size(), 0);
    VectorState<qs_policy_t> sim_l = *this;
    sim_l.ApplyCircuit(circ, pr);
    VectorState<qs_policy_t> sim_r = sim_l;
    sim_r.ApplyHamiltonian(ham);
    f_and_g[0] = qs_policy_t::Vdot(sim_l.qs, sim_r.qs, dim);
    // timer.EndAndStartOther("First part", "Second part");
    for (const auto& g : herm_circ) {
        sim_l.ApplyGate(g, pr);
        if (g->GradRequired()) {
            auto p_gate = static_cast<Parameterizable*>(g.get());
            if (const auto& [title, jac] = p_gate->jacobi; title.size() != 0) {
                auto intrin_grad = ExpectDiffGate(sim_l.qs, sim_r.qs, g, pr, dim);
                auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                }
            }
        }
        sim_r.ApplyGate(g, pr);
    }
    // timer.End("Second part");
    // timer.Analyze();
    return f_and_g;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationNonHermitianWithGradOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
    const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread,
    const derived_t& simulator_left) const -> VVT<py_qs_data_t> {
    auto n_hams = hams.size();

    VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));
    derived_t sim_left = simulator_left;
    derived_t sim_right = *this;
    sim_left.ApplyCircuit(left_circ, pr);
    sim_right.ApplyCircuit(right_circ, pr);
    auto f_g1 = LeftSizeGradOneMulti(hams, herm_left_circ, pr, p_map, n_thread, sim_left, sim_right);
    auto f_g2 = LeftSizeGradOneMulti(herm_hams, herm_right_circ, pr, p_map, n_thread, sim_right, sim_left);
    for (size_t i = 0; i < f_g2.size(); i++) {
        for (size_t j = 1; j < f_g2[i].size(); j++) {
            f_g1[i][j] += std::conj(f_g2[i][j]);
        }
    }
    return f_g1;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::LeftSizeGradOneMulti(const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
                                                     const circuit_t& herm_left_circ,
                                                     const parameter::ParameterResolver& pr, const MST<size_t>& p_map,
                                                     int n_thread, const derived_t& simulator_left,
                                                     const derived_t& simulator_right) const -> VVT<py_qs_data_t> {
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

    VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));

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
        auto sim_l = simulator_left;
        for (int j = start; j < end; j++) {
            sim_rs[j - start] = simulator_right;
            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
            f_and_g[j][0] = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
        }
        for (const auto& g : herm_left_circ) {
            sim_l.ApplyGate(g, pr);
            if (g->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(g.get());
                if (const auto& [title, jac] = p_gate->jacobi; title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffGate(sim_l.qs, sim_rs[j - start].qs, g, pr, dim);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += p_grad[0][idx];
                        }
                    }
                }
            }
            for (int j = start; j < end; j++) {
                sim_rs[j - start].ApplyGate(g, pr);
            }
        }
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationWithGradOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) const -> VVT<py_qs_data_t> {
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
    VVT<py_qs_data_t> f_and_g(n_hams, VT<py_qs_data_t>((1 + p_map.size()), 0));
    VectorState<qs_policy_t> sim = *this;
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
        for (int j = start; j < end; j++) {
            sim_rs[j - start] = sim_l;
            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
            f_and_g[j][0] = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
        }
        for (const auto& g : herm_circ) {
            sim_l.ApplyGate(g, pr);
            if (g->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(g.get());
                if (const auto& [title, jac] = p_gate->jacobi; title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffGate(sim_l.qs, sim_rs[j - start].qs, g, pr, dim);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                        }
                    }
                }
            }
            for (int j = start; j < end; j++) {
                sim_rs[j - start].ApplyGate(g, pr);
            }
        }
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto VectorState<qs_policy_t_>::GetExpectationNonHermitianWithGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
    const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    const derived_t& simulator_left, size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
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
        output[0] = GetExpectationNonHermitianWithGradOneMulti(hams, herm_hams, left_circ, herm_left_circ, right_circ,
                                                               herm_right_circ, pr, p_map, mea_threads, simulator_left);
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
                    auto f_g = GetExpectationNonHermitianWithGradOneMulti(hams, herm_hams, left_circ, herm_left_circ,
                                                                          right_circ, herm_right_circ, pr, p_map,
                                                                          mea_threads, simulator_left);
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
auto VectorState<qs_policy_t_>::QramExpectationWithGrad(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<py_qs_data_t>& init_states, const VT<calc_type>& ans_data, const VS& ans_name, size_t batch_threads,
    size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
    auto n_hams = hams.size();
    auto n_prs = init_states.size();
    auto n_params = ans_name.size();
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
    for (size_t i = 0; i < ans_name.size(); i++) {
        p_map[ans_name[i]] = i;
    }
    if (n_prs == 1) {
        parameter::ParameterResolver pr = parameter::ParameterResolver();
        pr.SetItems(ans_name, ans_data);
        auto sim = VectorState<qs_policy_t_>(this->n_qubits, this->seed);
        sim.SetQS(init_states[0]);
        output[0] = sim.GetExpectationWithGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
                auto sim = VectorState<qs_policy_t_>(this->n_qubits, this->seed);
                for (size_t n = start; n < end; n++) {
                    parameter::ParameterResolver pr = parameter::ParameterResolver();
                    pr.SetItems(ans_name, ans_data);
                    sim.SetQS(init_states[n]);
                    auto f_g = sim.GetExpectationWithGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
auto VectorState<qs_policy_t_>::GetExpectationWithGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) const -> VT<VVT<py_qs_data_t>> {
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
        output[0] = GetExpectationWithGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
                    auto f_g = GetExpectationWithGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
auto VectorState<qs_policy_t_>::GetExpectationWithGradParameterShiftOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) -> VVT<py_qs_data_t> {
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
        for (int j = start; j < end; j++) {
            sim_rs[j - start] = sim_l;
            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
            f_and_g[j][0] = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
        }
        for (auto& gate : circ) {
            if (gate->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(gate.get());
                calc_type pr_shift = M_PI_2;
                calc_type coeff = 0.5;
                if (gate->id_ == GateID::CUSTOM) {
                    pr_shift = 0.001;
                    coeff = 0.5 / pr_shift;
                }
                if (gate->id_ == GateID::SWAPalpha) {
                    pr_shift = 0.5;
                    coeff = M_PI_2;
                }
                if (gate->id_ == GateID::FSim) {
                    pr_shift = 0.001;
                    coeff = 0.5 / pr_shift;
                }
                if (const auto& [title, jac] = p_gate->jacobi; title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        VT<py_qs_data_t> intrin_grad_list(p_gate->prs_.size());
                        for (int k = 0; k < p_gate->prs_.size(); k++) {
                            p_gate->prs_[k] += -pr_shift;
                            if (gate->id_ == GateID::U3 || gate->id_ == GateID::FSim) {
                                parameter::tn::Tensor coeff;
                                parameter::tn::Tensor tmp;
                                std::string key;
                                for (auto& [key_, v] : p_gate->prs_[k].data_) {
                                    key = key_;
                                    coeff = v;
                                    tmp = pr.GetItem(key_);
                                }
                                tmp += -pr_shift / coeff;
                                tmp_pr.SetItem(key, tmp);
                            }
                            sim_l = *this;
                            sim_l.ApplyCircuit(circ, tmp_pr);
                            sim_rs[j - start] = sim_l;
                            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
                            auto expect0 = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
                            p_gate->prs_[k] += 2 * pr_shift;
                            if (gate->id_ == GateID::U3 || gate->id_ == GateID::FSim) {
                                parameter::tn::Tensor coeff;
                                parameter::tn::Tensor tmp;
                                std::string key;
                                for (auto& [key_, v] : p_gate->prs_[k].data_) {
                                    key = key_;
                                    coeff = v;
                                    tmp = pr.GetItem(key_);
                                }
                                tmp += pr_shift / coeff;
                                tmp_pr.SetItem(key, tmp);
                            }
                            sim_l = *this;
                            sim_l.ApplyCircuit(circ, tmp_pr);
                            sim_rs[j - start] = sim_l;
                            sim_rs[j - start].ApplyHamiltonian(*hams[j]);
                            auto expect1 = qs_policy_t::Vdot(sim_l.qs, sim_rs[j - start].qs, dim);
                            p_gate->prs_[k] += -pr_shift;
                            if (gate->id_ == GateID::U3 || gate->id_ == GateID::FSim) {
                                parameter::tn::Tensor coeff;
                                parameter::tn::Tensor tmp;
                                std::string key;
                                for (auto& [key_, v] : p_gate->prs_[k].data_) {
                                    key = key_;
                                    coeff = v;
                                    tmp = pr.GetItem(key_);
                                }
                                tmp_pr.SetItem(key, tmp);
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
        VT<unsigned> res1(key_map.size());
        for (const auto& [name, val] : key_map) {
            res1[val] = res0[name];
        }
        for (size_t j = 0; j < key_size; j++) {
            res[i * key_size + j] = res1[j];
        }
    }
    return res;
}
}  // namespace mindquantum::sim::vector::detail

#endif
