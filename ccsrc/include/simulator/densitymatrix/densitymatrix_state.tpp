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

#ifndef INCLUDE_DENSITYMATRIX_DENSITYMATRIXSTATE_TPP
#define INCLUDE_DENSITYMATRIX_DENSITYMATRIXSTATE_TPP

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
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/basic_math.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "ops/basic_gate.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/densitymatrix/densitymatrix_state.h"

namespace mindquantum::sim::densitymatrix::detail {

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qbit_t n_qubits, unsigned seed)
    : n_qubits(n_qubits), dim(static_cast<uint64_t>(1) << n_qubits), seed(seed), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed)
    : qs(qs), n_qubits(n_qubits), dim(static_cast<uint64_t>(1) << n_qubits), seed(seed), rnd_eng_(seed) {
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(const DensityMatrixState<qs_policy_t>& sim) {
    this->qs = qs_policy_t::Copy(sim.qs, sim.dim);
    this->dim = sim.dim;
    this->n_qubits = sim.n_qubits;
    this->seed = sim.seed;
    this->rnd_eng_ = RndEngine(seed);
    std::uniform_real_distribution<double> dist(0., 1.);
    this->rng_ = std::bind(dist, std::ref(this->rnd_eng_));
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::operator=(const DensityMatrixState<qs_policy_t>& sim) -> derived_t& {
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
DensityMatrixState<qs_policy_t_>::DensityMatrixState(DensityMatrixState<qs_policy_t>&& sim) {
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
auto DensityMatrixState<qs_policy_t_>::operator=(DensityMatrixState<qs_policy_t>&& sim) -> derived_t& {
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
tensor::TDtype DensityMatrixState<qs_policy_t_>::DType() {
    return tensor::to_dtype_v<py_qs_data_t>;
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::Reset() {
    qs_policy_t::Reset(&qs);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::Display(qbit_t qubits_limit) const {
    qs_policy_t::Display(qs, n_qubits, qubits_limit);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetQS() const -> matrix_t {
    return qs_policy_t::GetQS(qs, dim);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::SetQS(const py_qs_datas_t& qs_out) {
    qs_policy_t::SetQS(&qs, qs_out, dim);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::SetDM(const matrix_t& qs_out) {
    qs_policy_t::SetDM(&qs, qs_out, dim);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::CopyQS(const qs_data_p_t& qs_src) {
    qs_policy_t::CopyQS(&qs, qs_src, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::Purity() const -> calc_type {
    return qs_policy_t::Purity(qs, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetPartialTrace(const qbits_t& objs) const -> matrix_t {
    return qs_policy_t::GetPartialTrace(qs, objs, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PureStateVector() -> py_qs_datas_t {
    return qs_policy_t::PureStateVector(qs, dim);
}

template <typename qs_policy_t_>
index_t DensityMatrixState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate>& gate,
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
            qs_policy_t::ApplyGP(&qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
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
            qs_policy_t::ApplyPauli(&qs, gate->obj_qubits_, static_cast<PauliChannel*>(gate.get())->probs_, dim);
            break;
        case GateID::DEP:
            qs_policy_t::ApplyDepolarizing(&qs, gate->obj_qubits_, static_cast<DepolarizingChannel*>(gate.get())->prob_,
                                           dim);
            break;
        case GateID::AD:
            qs_policy_t::ApplyAmplitudeDamping(&qs, gate->obj_qubits_,
                                               static_cast<AmplitudeDampingChannel*>(gate.get())->damping_coeff_,
                                               static_cast<AmplitudeDampingChannel*>(gate.get())->daggered_, dim);
            break;
        case GateID::PD:
            qs_policy_t::ApplyPhaseDamping(&qs, gate->obj_qubits_,
                                           static_cast<PhaseDampingChannel*>(gate.get())->damping_coeff_, dim);
            break;
        case GateID::KRAUS: {
            auto& k_set = static_cast<KrausChannel*>(gate.get())->kraus_operator_set_;
            VT<matrix_t> k_mat;
            std::transform(k_set.begin(), k_set.end(), std::back_inserter(k_mat),
                           [&](auto& k) { return tensor::ops::cpu::to_vector<py_qs_data_t>(k); });
            qs_policy_t::ApplyKraus(&qs, gate->obj_qubits_, k_mat, dim);
        } break;
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            tensor::Matrix mat;
            if (!g->Parameterized()) {
                mat = g->base_matrix_;
            } else {
                calc_type val = tensor::ops::cpu::to_vector<calc_type>(g->prs_[0].Combination(pr).const_value)[0];
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
void DensityMatrixState<qs_policy_t_>::ApplyChannel(const std::shared_ptr<BasicGate>& gate) {
    auto id = gate->id_;
    switch (id) {
        case GateID::DEP:
            qs_policy_t::ApplyDepolarizing(&qs, gate->obj_qubits_, static_cast<DepolarizingChannel*>(gate.get())->prob_,
                                           dim);
            break;
        case GateID::AD:
            qs_policy_t::ApplyAmplitudeDamping(&qs, gate->obj_qubits_,
                                               static_cast<AmplitudeDampingChannel*>(gate.get())->damping_coeff_,
                                               static_cast<AmplitudeDampingChannel*>(gate.get())->daggered_, dim);
            break;
        case GateID::PD:
            qs_policy_t::ApplyPhaseDamping(&qs, gate->obj_qubits_,
                                           static_cast<PhaseDampingChannel*>(gate.get())->damping_coeff_, dim);
            break;
        case GateID::PL:
            qs_policy_t::ApplyPauli(&qs, gate->obj_qubits_, static_cast<PauliChannel*>(gate.get())->probs_, dim);
            break;
        case GateID::KRAUS: {
            auto& k_set = static_cast<KrausChannel*>(gate.get())->kraus_operator_set_;
            VT<matrix_t> k_mat;
            std::transform(k_set.begin(), k_set.end(), std::back_inserter(k_mat),
                           [](auto& m) { return tensor::ops::cpu::to_vector<py_qs_data_t>(m); });
            qs_policy_t::ApplyKraus(&qs, gate->obj_qubits_, k_mat, dim);
        } break;
        default:
            throw std::invalid_argument(fmt::format("{} is not a noise channel.", id));
    }
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    if (ham.how_to_ == ORIGIN) {
        qs_policy_t::ApplyTerms(&qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        qs_policy_t::ApplyCsr(&qs, sparse::Csr_Plus_Csr<calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_), dim);
    } else {
        qs_policy_t::ApplyCsr(&qs, ham.ham_sparse_main_, dim);
    }
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyMeasure(const std::shared_ptr<BasicGate>& gate) -> index_t {
    index_t one_mask = (static_cast<uint64_t>(1) << gate->obj_qubits_[0]);
    auto one_amp = qs_policy_t::DiagonalConditionalCollect(qs, one_mask, one_mask, dim);
    index_t collapse_mask = (static_cast<index_t>(rng_() < one_amp) << gate->obj_qubits_[0]);
    qs_data_t norm_fact = (collapse_mask == 0) ? 1 / (1 - one_amp) : 1 / one_amp;
    qs_policy_t::ConditionalMul(qs, &qs, one_mask, collapse_mask, norm_fact, 0.0, dim);
    return static_cast<index_t>(collapse_mask != 0);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffGate(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                                      const std::shared_ptr<BasicGate>& gate,
                                                      const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    auto id = gate->id_;
    VT<py_qs_data_t> grad = {0};
    switch (id) {
        case GateID::RX:
            grad[0] = qs_policy_t::ExpectDiffRX(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::RY:
            grad[0] = qs_policy_t::ExpectDiffRY(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::RZ:
            grad[0] = qs_policy_t::ExpectDiffRZ(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Rxx:
            grad[0] = qs_policy_t::ExpectDiffRxx(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Rzz:
            grad[0] = qs_policy_t::ExpectDiffRzz(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Ryy:
            grad[0] = qs_policy_t::ExpectDiffRyy(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Rxy:
            grad[0] = qs_policy_t::ExpectDiffRxy(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Rxz:
            grad[0] = qs_policy_t::ExpectDiffRxz(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::Ryz:
            grad[0] = qs_policy_t::ExpectDiffRyz(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::PS:
            grad[0] = qs_policy_t::ExpectDiffPS(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::GP:
            grad[0] = qs_policy_t::ExpectDiffGP(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        case GateID::SWAPalpha:
            grad[0] = qs_policy_t::ExpectDiffSWAPalpha(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_,
                                                       dim);
            return tensor::Matrix(VVT<py_qs_data_t>({grad}));
        case GateID::CUSTOM: {
            auto g = static_cast<CustomGate*>(gate.get());
            auto val = tensor::ops::cpu::to_vector<double>(g->prs_[0].Combination(pr).const_value)[0];
            tensor::Matrix gate_m = g->numba_param_matrix_(val);
            tensor::Matrix diff_m = g->numba_param_diff_matrix_(val);
            grad[0] = qs_policy_t::ExpectDiffMatrixGate(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_,
                                                        tensor::ops::cpu::to_vector<py_qs_data_t>(gate_m),
                                                        tensor::ops::cpu::to_vector<py_qs_data_t>(diff_m), dim);
            return tensor::Matrix(VVT<py_qs_data_t>{grad});
        }
        case GateID::U3:
            return ExpectDiffU3(dens_matrix, ham_matrix, gate, pr, dim);
        case GateID::FSim:
            return ExpectDiffFSim(dens_matrix, ham_matrix, gate, pr, dim);
        default:
            throw std::invalid_argument(fmt::format("Expectation of gate {} not implement.", id));
    }
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffU3(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                                    const std::shared_ptr<BasicGate>& gate,
                                                    const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    py_qs_datas_t grad = {0, 0, 0};
    auto u3 = static_cast<U3*>(gate.get());
    if (u3->parameterized_) {
        auto theta = u3->theta.Combination(pr).const_value;
        auto phi = u3->phi.Combination(pr).const_value;
        auto lambda = u3->lambda.Combination(pr).const_value;
        if (u3->theta.data_.size() != u3->theta.no_grad_parameters_.size()) {
            grad[0] = qs_policy_t::ExpectDiffU3Theta(dens_matrix, ham_matrix, u3->obj_qubits_, u3->ctrl_qubits_,
                                                     tensor::ops::cpu::to_vector<calc_type>(phi)[0], dim);
        }
        if (u3->phi.data_.size() != u3->phi.no_grad_parameters_.size()) {
            grad[1] = qs_policy_t::ExpectDiffU3Phi(dens_matrix, ham_matrix, u3->obj_qubits_, u3->ctrl_qubits_, dim);
        }
        if (u3->lambda.data_.size() != u3->lambda.no_grad_parameters_.size()) {
            tensor::Matrix m{U3Matrix(theta, phi, lambda)};
            tensor::Matrix diff_m{U3DiffLambdaMatrix(theta, phi, lambda)};
            grad[2] = qs_policy_t::ExpectDiffSingleQubitMatrix(
                dens_matrix, ham_matrix, u3->obj_qubits_, u3->ctrl_qubits_,
                tensor::ops::cpu::to_vector<py_qs_data_t>(m), tensor::ops::cpu::to_vector<py_qs_data_t>(diff_m), dim);
        }
    }
    return tensor::Matrix(VVT<py_qs_data_t>{grad});
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffFSim(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                                      const std::shared_ptr<BasicGate>& gate,
                                                      const parameter::ParameterResolver& pr, index_t dim) const
    -> tensor::Matrix {
    py_qs_datas_t grad = {0, 0};
    auto fsim = static_cast<FSim*>(gate.get());
    if (fsim->parameterized_) {
        if (fsim->theta.data_.size() != fsim->theta.no_grad_parameters_.size()) {
            grad[0] = qs_policy_t::ExpectDiffFSimTheta(dens_matrix, ham_matrix, fsim->obj_qubits_, fsim->ctrl_qubits_,
                                                       dim);
        }
        if (fsim->phi.data_.size() != fsim->phi.no_grad_parameters_.size()) {
            grad[1] = qs_policy_t::ExpectDiffFSimPhi(dens_matrix, ham_matrix, fsim->obj_qubits_, fsim->ctrl_qubits_,
                                                     dim);
        }
    }
    return tensor::Matrix(VVT<py_qs_data_t>{grad});
}

template <typename qs_policy_t_>
std::map<std::string, int> DensityMatrixState<qs_policy_t_>::ApplyCircuit(const circuit_t& circ,
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
auto DensityMatrixState<qs_policy_t_>::GetStateExpectation(const qs_data_p_t& qs_out, const Hamiltonian<calc_type>& ham,
                                                           index_t dim) const -> py_qs_data_t {
    py_qs_data_t out;
    if (ham.how_to_ == ORIGIN) {
        out = qs_policy_t::ExpectationOfTerms(qs_out, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        out = qs_policy_t::ExpectationOfCsr(
            qs_out, sparse::Csr_Plus_Csr<calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_), dim);
    } else {
        out = qs_policy_t::ExpectationOfCsr(qs_out, ham.ham_sparse_main_, dim);
    }
    return out;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                      const parameter::ParameterResolver& pr) const -> py_qs_data_t {
    py_qs_data_t out;
    auto sub_seed = static_cast<unsigned int>(static_cast<calc_type>(rng_()) * (1 << 20));
    auto tmp_sim = derived_t(n_qubits, sub_seed);
    tmp_sim.CopyQS(qs);
    tmp_sim.ApplyCircuit(circ, pr);
    if (ham.how_to_ == ORIGIN) {
        out = qs_policy_t::ExpectationOfTerms(tmp_sim.qs, ham.ham_, dim);
    } else if (ham.how_to_ == BACKEND) {
        out = qs_policy_t::ExpectationOfCsr(
            tmp_sim.qs, sparse::Csr_Plus_Csr<calc_type>(ham.ham_sparse_main_, ham.ham_sparse_second_), dim);
    } else {
        out = qs_policy_t::ExpectationOfCsr(tmp_sim.qs, ham.ham_sparse_main_, dim);
    }
    return out;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithReversibleGradOneOne(
    const Hamiltonian<calc_type>& ham, const circuit_t& circ, const circuit_t& herm_circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) const -> py_qs_datas_t {
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = GetStateExpectation(sim_qs.qs, ham, dim);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    index_t n = circ.size();
    for (const auto& g : herm_circ) {
        --n;
        if (g->GradRequired()) {
            auto p_gate = static_cast<Parameterizable*>(circ[n].get());
            const auto& [title, jac] = p_gate->jacobi;
            if (title.size() != 0) {
                auto intrin_grad = ExpectDiffGate(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                }
            }
        }
        sim_ham.ApplyGate(g, pr);
        sim_qs.ApplyGate(g, pr);
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithReversibleGradOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) const -> VT<py_qs_datas_t> {
    auto n_hams = hams.size();
    int max_thread = 15;
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    if (n_thread == 0) {
        throw std::runtime_error("n_thread cannot be zero.");
    }
    if (n_thread > max_thread) {
        n_thread = max_thread;
    }
    if (n_thread > static_cast<int>(n_hams)) {
        n_thread = n_hams;
    }
    VT<py_qs_datas_t> f_and_g(n_hams, py_qs_datas_t((1 + p_map.size()), 0));
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
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
        std::vector<derived_t> sim_hams(end - start);
        for (int j = start; j < end; j++) {
            f_and_g[j][0] = GetStateExpectation(sim_qs.qs, *hams[j], dim);
            auto ham_matrix = qs_policy_t::HamiltonianMatrix(*hams[j], dim);
            sim_hams[j - start] = std::move(derived_t{ham_matrix, n_qubits, seed});
        }
        index_t n = circ.size();
        for (const auto& g : herm_circ) {
            --n;
            if (g->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(circ[n].get());
                const auto& [title, jac] = p_gate->jacobi;
                if (title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffGate(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                        }
                    }
                }
            }
            for (int j = start; j < end; j++) {
                sim_hams[j - start].ApplyGate(g, pr);
            }
            sim_qs.ApplyGate(g, pr);
        }
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithReversibleGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) const -> VT<VT<py_qs_datas_t>> {
    auto n_hams = hams.size();
    auto n_prs = enc_data.size();
    auto n_params = enc_name.size() + ans_name.size();
    VT<VT<py_qs_datas_t>> output;
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
        output[0] = GetExpectationWithReversibleGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
    } else {
        if (batch_threads == 0) {
            throw std::runtime_error("batch_thread cannot be zero.");
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
                    auto f_g = GetExpectationWithReversibleGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithNoiseGradOneOne(const Hamiltonian<calc_type>& ham,
                                                                         const circuit_t& circ,
                                                                         const circuit_t& herm_circ,
                                                                         const parameter::ParameterResolver& pr,
                                                                         const MST<size_t>& p_map) const
    -> py_qs_datas_t {
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = GetStateExpectation(sim_qs.qs, ham, dim);
    sim_qs.CopyQS(this->qs);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    index_t n = circ.size();
    for (const auto& g : herm_circ) {
        --n;
        if (g->GradRequired()) {
            auto p_gate = static_cast<Parameterizable*>(circ[n].get());
            const auto& [title, jac] = p_gate->jacobi;
            if (title.size() != 0) {
                for (index_t a = 0; a <= n; a++) {
                    sim_qs.ApplyGate(circ[a], pr);
                }
                auto intrin_grad = ExpectDiffGate(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                }
                sim_qs.CopyQS(this->qs);
            }
        }
        sim_ham.ApplyGate(g, pr);
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithNoiseGradOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread) const -> VT<py_qs_datas_t> {
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
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
    VT<py_qs_datas_t> f_and_g(n_hams, py_qs_datas_t((1 + p_map.size()), 0));
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
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
        std::vector<derived_t> sim_hams(end - start);
        for (int j = start; j < end; j++) {
            f_and_g[j][0] = GetStateExpectation(sim_qs.qs, *hams[j], dim);
            auto ham_matrix = qs_policy_t::HamiltonianMatrix(*hams[j], dim);
            sim_hams[j - start] = std::move(derived_t{ham_matrix, n_qubits, seed});
        }
        sim_qs.CopyQS(this->qs);
        index_t n = circ.size();
        for (const auto& g : herm_circ) {
            --n;
            if (g->GradRequired()) {
                auto p_gate = static_cast<Parameterizable*>(circ[n].get());
                const auto& [title, jac] = p_gate->jacobi;
                if (title.size() != 0) {
                    for (index_t a = 0; a <= n; a++) {
                        sim_qs.ApplyGate(circ[a], pr);
                    }
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffGate(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto p_grad = tensor::ops::cpu::to_vector<py_qs_data_t>(tensor::ops::MatMul(intrin_grad, jac));
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(p_grad[0][idx]);
                        }
                    }
                    sim_qs.CopyQS(this->qs);
                }
            }
            for (int j = start; j < end; j++) {
                sim_hams[j - start].ApplyGate(g, pr);
            }
        }
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithNoiseGradMultiMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
    size_t batch_threads, size_t mea_threads) const -> VT<VT<py_qs_datas_t>> {
    auto n_hams = hams.size();
    auto n_prs = enc_data.size();
    auto n_params = enc_name.size() + ans_name.size();
    VT<VT<py_qs_datas_t>> output;
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
        output[0] = GetExpectationWithNoiseGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
                    auto f_g = GetExpectationWithNoiseGradOneMulti(hams, circ, herm_circ, pr, p_map, mea_threads);
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
VT<unsigned> DensityMatrixState<qs_policy_t_>::Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                        size_t shots, const MST<size_t>& key_map,
                                                        unsigned int seed) const {
    auto key_size = key_map.size();
    VT<unsigned> res(shots * key_size);
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
    for (size_t i = 0; i < shots; i++) {
        derived_t sim{n_qubits, static_cast<unsigned>(rng())};
        qs_policy_t::CopyQS(&(sim.qs), qs, dim);
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

}  // namespace mindquantum::sim::densitymatrix::detail

#endif
