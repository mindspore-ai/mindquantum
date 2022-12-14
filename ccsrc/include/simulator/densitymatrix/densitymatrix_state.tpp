//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

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

#include "core/mq_base_types.hpp"
#include "core/parameter_resolver.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gates.hpp"
#include "ops/hamiltonian.hpp"
#include "simulator/densitymatrix/densitymatrix_state.hpp"
#include "simulator/types.hpp"

namespace mindquantum::sim::densitymatrix::detail {

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qbit_t n_qubits, unsigned seed)
    : n_qubits(n_qubits), dim(1UL << n_qubits), seed(seed), rnd_eng_(seed) {
    qs = qs_policy_t::InitState(dim);
    std::uniform_real_distribution<double> dist(0., 1.);
    rng_ = std::bind(dist, std::ref(rnd_eng_));
}

template <typename qs_policy_t_>
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed)
    : qs(qs), n_qubits(n_qubits), dim(1UL << n_qubits), seed(seed), rnd_eng_(seed) {
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
    qs_policy_t::FreeState(this->qs);
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
    qs_policy_t::FreeState(this->qs);
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
void DensityMatrixState<qs_policy_t_>::Reset() {
    qs_policy_t::Reset(qs, dim);
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
    qs_policy_t::SetQS(qs, qs_out, dim);
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::CopyQS(const qs_data_p_t& qs_out) {
    qs_policy_t::CopyQS(qs, qs_out, dim);
}

template <typename qs_policy_t_>
bool DensityMatrixState<qs_policy_t_>::IsPure() {
    return qs_policy_t::IsPure(qs, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::PureStateVector() -> py_qs_datas_t {
    return qs_policy_t::PureStateVector(qs, dim);
}

template <typename qs_policy_t_>
index_t DensityMatrixState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                    const ParameterResolver<calc_type>& pr, bool diff) {
    auto name = gate->name_;
    if (gate->is_custom_) {
        std::remove_reference_t<decltype(*gate)>::matrix_t mat;
        if (!gate->parameterized_) {
            mat = gate->base_matrix_;
        } else {
            calc_type val = gate->params_.Combination(pr).const_value;
            if (!diff) {
                mat = gate->numba_param_matrix_(val);
            } else {
                mat = gate->numba_param_diff_matrix_(val);
            }
        }
        qs_policy_t::ApplyMatrixGate(qs, qs, gate->obj_qubits_, gate->ctrl_qubits_, mat.matrix_, dim);
    } else if (name == gI) {
    } else if (name == gX) {
        qs_policy_t::ApplyX(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gCNOT) {
        qbits_t obj_qubits = {gate->obj_qubits_[0]};
        qbits_t ctrl_qubits = gate->ctrl_qubits_;
        std::copy(std::begin(gate->obj_qubits_) + 1, std::end(gate->obj_qubits_), std::back_inserter(ctrl_qubits));
        qs_policy_t::ApplyX(qs, obj_qubits, ctrl_qubits, dim);
    } else if (name == gY) {
        qs_policy_t::ApplyY(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gZ) {
        qs_policy_t::ApplyZ(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gH) {
        qs_policy_t::ApplyH(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gS) {
        if (gate->daggered_) {
            qs_policy_t::ApplySdag(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
        } else {
            qs_policy_t::ApplySGate(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
        }
    } else if (name == gT) {
        if (gate->daggered_) {
            qs_policy_t::ApplyTdag(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
        } else {
            qs_policy_t::ApplyT(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
        }
    } else if (name == gSWAP) {
        qs_policy_t::ApplySWAP(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gISWAP) {
        qs_policy_t::ApplyISWAP(qs, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    } else if (name == gRX) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyRX(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gRY) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyRY(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gRZ) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyRZ(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gXX) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyXX(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gZZ) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyZZ(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gYY) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyYY(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gPS) {
        auto val = gate->applied_value_;
        if (!gate->parameterized_) {
            diff = false;
        } else {
            val = gate->params_.Combination(pr).const_value;
        }
        qs_policy_t::ApplyPS(qs, gate->obj_qubits_, gate->ctrl_qubits_, val, dim, diff);
    } else if (name == gU3) {
        if (diff) {
            std::runtime_error("Can not apply differential format of U3 gate on quatum states currently.");
        }
        auto u3 = static_cast<U3<calc_type>*>(gate.get());
        Dim2Matrix<calc_type> m;
        if (!gate->parameterized_) {
            m = gate->base_matrix_;
        } else {
            auto theta = u3->theta.Combination(pr).const_value;
            auto phi = u3->phi.Combination(pr).const_value;
            auto lambda = u3->lambda.Combination(pr).const_value;
            m = U3Matrix<calc_type>(theta, phi, lambda);
        }
        qs_policy_t::ApplySingleQubitMatrix(qs, qs, gate->obj_qubits_[0], gate->ctrl_qubits_, m.matrix_, dim);
    } else if (name == gFSim) {
        if (diff) {
            std::runtime_error("Can not apply differential format of FSim gate on quatum states currently.");
        }
        auto fsim = static_cast<FSim<calc_type>*>(gate.get());
        Dim2Matrix<calc_type> m;
        if (!gate->parameterized_) {
            m = gate->base_matrix_;
        } else {
            auto theta = fsim->theta.Combination(pr).const_value;
            auto phi = fsim->phi.Combination(pr).const_value;
            m = FSimMatrix<calc_type>(theta, phi);
        }
        qs_policy_t::ApplyTwoQubitsMatrix(qs, qs, gate->obj_qubits_, gate->ctrl_qubits_, m.matrix_, dim);
    } else if (gate->is_measure_) {
        return ApplyMeasure(gate);
    } else if (gate->is_channel_) {
        ApplyChannel(gate);
    } else {
        throw std::invalid_argument("gate " + name + " not implement.");
    }
    return 2;  // qubit should be 1 or 0, 2 means nothing.
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyChannel(const std::shared_ptr<BasicGate<calc_type>>& gate) {
    if (gate->name_ == cAD) {
        qs_policy_t::ApplyAmplitudeDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else if (gate->name_ == cPD) {
        qs_policy_t::ApplyPhaseDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else if (gate->name_ == cPL) {
        qs_policy_t::ApplyPauli(qs, gate->obj_qubits_, gate->probs_, dim);
    } else if (gate->kraus_operator_set_.size() != 0) {
        qs_policy_t::ApplyKraus(qs, gate->obj_qubits_, gate->kraus_operator_set_, dim);
    } else if (gate->name_ == cHAD) {
        qs_policy_t::ApplyHermitianAmplitudeDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else {
        throw std::runtime_error("This noise channel not implemented.");
    }
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    qs_policy_t::ApplyTerms(qs, ham.ham_, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyMeasure(const std::shared_ptr<BasicGate<calc_type>>& gate) {
    assert(gate->is_measure_);
    index_t one_mask = (1UL << gate->obj_qubits_[0]);
    auto one_amp = qs_policy_t::DiagonalConditionalCollect(qs, one_mask, one_mask, dim);
    index_t collapse_mask = (static_cast<index_t>(rng_() < one_amp) << gate->obj_qubits_[0]);
    qs_data_t norm_fact = (collapse_mask == 0) ? 1 / (1 - one_amp) : 1 / one_amp;
    qs_policy_t::ConditionalMul(qs, qs, one_mask, collapse_mask, norm_fact, 0.0, dim);
    return static_cast<index_t>(collapse_mask != 0);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffGate(qs_data_p_t dens_matrix, qs_data_p_t ham_matrix,
                                                      const std::shared_ptr<BasicGate<calc_type>>& gate, index_t dim)
    -> py_qs_data_t {
    auto name = gate->name_;

    if (name == gRX) {
        return qs_policy_t::ExpectDiffRX(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gRY) {
        return qs_policy_t::ExpectDiffRY(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gRZ) {
        return qs_policy_t::ExpectDiffRZ(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gXX) {
        return qs_policy_t::ExpectDiffXX(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gZZ) {
        return qs_policy_t::ExpectDiffZZ(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gYY) {
        return qs_policy_t::ExpectDiffYY(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    if (name == gPS) {
        return qs_policy_t::ExpectDiffPS(dens_matrix, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, dim);
    }
    throw std::invalid_argument("gate " + name + " not implement.");
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffU3(qs_data_p_t dens_matrix, qs_data_p_t ham_matrix,
                                                    const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                    const ParameterResolver<calc_type>& pr, index_t dim)
    -> Dim2Matrix<calc_type> {
    py_qs_datas_t grad = {0, 0, 0};
    auto u3 = static_cast<U3<calc_type>*>(gate.get());
    if (u3->parameterized_) {
        auto theta = u3->theta.Combination(pr).const_value;
        auto phi = u3->phi.Combination(pr).const_value;
        auto lambda = u3->lambda.Combination(pr).const_value;
        if (u3->theta.data_.size() != u3->theta.no_grad_parameters_.size()) {
            grad[0] = qs_policy_t::ExpectDiffU3Theta(dens_matrix, ham_matrix, u3->obj_qubits_, u3->ctrl_qubits_, phi,
                                                     dim);
        }
        if (u3->phi.data_.size() != u3->phi.no_grad_parameters_.size()) {
            grad[1] = qs_policy_t::ExpectDiffU3Phi(dens_matrix, ham_matrix, u3->obj_qubits_, u3->ctrl_qubits_, dim);
        }
        if (u3->lambda.data_.size() != u3->lambda.no_grad_parameters_.size()) {
            Dim2Matrix<calc_type> diff_m{U3DiffLambdaMatrix(theta, phi, lambda)};
            Dim2Matrix<calc_type> herm_m{U3Matrix(-theta, -lambda, -phi)};
            grad[2] = qs_policy_t::ExpectDiffSingleQubitMatrix(dens_matrix, ham_matrix, u3->obj_qubits_,
                                                               u3->ctrl_qubits_, diff_m.matrix_, herm_m.matrix_, dim);
        }
    }
    return Dim2Matrix<calc_type>({grad});
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffFSim(qs_data_p_t dens_matrix, qs_data_p_t ham_matrix,
                                                      const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                      const ParameterResolver<calc_type>& pr, index_t dim)
    -> Dim2Matrix<calc_type> {
    py_qs_datas_t grad = {0, 0};
    auto fsim = static_cast<FSim<calc_type>*>(gate.get());
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
    return Dim2Matrix<calc_type>({grad});
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyCircuit(const circuit_t& circ, const ParameterResolver<calc_type>& pr) {
    std::map<std::string, int> result;
    for (auto& g : circ) {
        if (g->is_measure_) {
            result[g->name_] = ApplyMeasure(g);
        } else {
            ApplyGate(g, pr, false);
        }
    }
    return result;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectation(const Hamiltonian<calc_type>& ham) -> py_qs_data_t {
    return qs_policy_t::GetExpectation(qs, ham.ham_, dim);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithReversibleGradOneOne(const Hamiltonian<calc_type>& ham,
                                                                              const circuit_t& circ,
                                                                              const circuit_t& herm_circ,
                                                                              const ParameterResolver<calc_type>& pr,
                                                                              const MST<size_t>& p_map, int n_thread)
    -> py_qs_datas_t {
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = qs_policy_t::GetExpectation(sim_qs.qs, ham.ham_, dim);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham.ham_, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    index_t n = circ.size();
    for (const auto& g : herm_circ) {
        --n;
        if (g->name_ == gU3) {
            auto u3 = static_cast<U3<calc_type>*>(circ[n].get());
            const auto& [title, jac] = u3->jacobi;
            if (title.size() != 0) {
                auto intrin_grad = ExpectDiffU3(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto u3_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(u3_grad.matrix_[0][idx]);
                }
            }
        } else if (g->name_ == gFSim) {
            auto fsim = static_cast<FSim<calc_type>*>(circ[n].get());
            const auto& [title, jac] = fsim->jacobi;
            if (title.size() != 0) {
                auto intrin_grad = ExpectDiffFSim(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto fsim_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(fsim_grad.matrix_[0][idx]);
                }
            }
        } else if (g->params_.data_.size() != g->params_.no_grad_parameters_.size()) {
            if (g->is_custom_) {
                auto val = circ[n]->params_.Combination(pr).const_value;
                auto herm_val = g->params_.Combination(pr).const_value;
                Dim2Matrix<calc_type> diff_m = circ[n]->numba_param_diff_matrix_(val);
                Dim2Matrix<calc_type> herm_m = g->numba_param_matrix_(herm_val);
                auto gi = qs_policy_t::ExpectDiffMatrixGate(sim_qs.qs, sim_ham.qs, g->obj_qubits_, g->ctrl_qubits_,
                                                            diff_m.matrix_, herm_m.matrix_, dim);
                for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                    f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                }
            } else {
                auto gi = ExpectDiffGate(sim_qs.qs, sim_ham.qs, circ[n], dim);
                for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                    f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
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
    const ParameterResolver<calc_type>& pr, const MST<size_t>& p_map, int n_thread) -> VT<py_qs_datas_t> {
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
            f_and_g[j][0] = qs_policy_t::GetExpectation(sim_qs.qs, hams[j]->ham_, dim);
            auto ham_matrix = qs_policy_t::HamiltonianMatrix(hams[j]->ham_, dim);
            sim_hams[j - start] = std::move(derived_t{ham_matrix, n_qubits, seed});
        }
        index_t n = circ.size();
        for (const auto& g : herm_circ) {
            --n;
            if (g->name_ == gU3) {
                auto u3 = static_cast<U3<calc_type>*>(circ[n].get());
                const auto& [title, jac] = u3->jacobi;
                if (title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffU3(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto u3_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(u3_grad.matrix_[0][idx]);
                        }
                    }
                }
            } else if (g->name_ == gFSim) {
                auto fsim = static_cast<FSim<calc_type>*>(circ[n].get());
                const auto& [title, jac] = fsim->jacobi;
                if (title.size() != 0) {
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffFSim(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto fsim_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(fsim_grad.matrix_[0][idx]);
                        }
                    }
                }
            } else if (g->params_.data_.size() != g->params_.no_grad_parameters_.size()) {
                if (g->is_custom_) {
                    auto val = circ[n]->params_.Combination(pr).const_value;
                    auto herm_val = g->params_.Combination(pr).const_value;
                    Dim2Matrix<calc_type> diff_m = circ[n]->numba_param_diff_matrix_(val);
                    Dim2Matrix<calc_type> herm_m = g->numba_param_matrix_(herm_val);
                    for (int j = start; j < end; j++) {
                        auto gi = qs_policy_t::ExpectDiffMatrixGate(sim_qs.qs, sim_hams[j - start].qs, g->obj_qubits_,
                                                                    g->ctrl_qubits_, diff_m.matrix_, herm_m.matrix_,
                                                                    dim);
                        for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                            f_and_g[j][1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                        }
                    }
                } else {
                    for (int j = start; j < end; j++) {
                        auto gi = ExpectDiffGate(sim_qs.qs, sim_hams[j - start].qs, circ[n], dim);
                        for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                            f_and_g[j][1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
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
    size_t batch_threads, size_t mea_threads) -> VT<VT<py_qs_datas_t>> {
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
        ParameterResolver<calc_type> pr = ParameterResolver<calc_type>();
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
                    ParameterResolver<calc_type> pr = ParameterResolver<calc_type>();
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
                                                                         const ParameterResolver<calc_type>& pr,
                                                                         const MST<size_t>& p_map) -> py_qs_datas_t {
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = qs_policy_t::GetExpectation(sim_qs.qs, ham.ham_, dim);
    sim_qs.CopyQS(this->qs);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham.ham_, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    index_t n = circ.size();
    for (const auto& g : herm_circ) {
        --n;
        if (g->name_ == gU3) {
            auto u3 = static_cast<U3<calc_type>*>(circ[n].get());
            const auto& [title, jac] = u3->jacobi;
            if (title.size() != 0) {
                for (index_t a = 0; a <= n; a++) {
                    sim_qs.ApplyGate(circ[a], pr);
                }
                auto intrin_grad = ExpectDiffU3(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto u3_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(u3_grad.matrix_[0][idx]);
                }
                sim_qs.CopyQS(this->qs);
            }
        } else if (g->name_ == gFSim) {
            auto fsim = static_cast<FSim<calc_type>*>(circ[n].get());
            const auto& [title, jac] = fsim->jacobi;
            if (title.size() != 0) {
                for (index_t a = 0; a <= n; a++) {
                    sim_qs.ApplyGate(circ[a], pr);
                }
                auto intrin_grad = ExpectDiffFSim(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
                auto fsim_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                for (const auto& [name, idx] : title) {
                    f_and_g[1 + p_map.at(name)] += 2 * std::real(fsim_grad.matrix_[0][idx]);
                }
                sim_qs.CopyQS(this->qs);
            }
        } else if (g->params_.data_.size() != g->params_.no_grad_parameters_.size()) {
            for (index_t a = 0; a <= n; a++) {
                sim_qs.ApplyGate(circ[a], pr);
            }
            if (g->is_custom_) {
                auto val = circ[n]->params_.Combination(pr).const_value;
                auto herm_val = g->params_.Combination(pr).const_value;
                Dim2Matrix<calc_type> diff_m = circ[n]->numba_param_diff_matrix_(val);
                Dim2Matrix<calc_type> herm_m = g->numba_param_matrix_(herm_val);
                auto gi = qs_policy_t::ExpectDiffMatrixGate(sim_qs.qs, sim_ham.qs, g->obj_qubits_, g->ctrl_qubits_,
                                                            diff_m.matrix_, herm_m.matrix_, dim);
                for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                    f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                }
            } else {
                auto gi = ExpectDiffGate(sim_qs.qs, sim_ham.qs, circ[n], dim);
                for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                    f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                }
            }
            sim_qs.CopyQS(this->qs);
        }
        sim_ham.ApplyGate(g, pr);
    }
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationWithNoiseGradOneMulti(
    const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ, const circuit_t& herm_circ,
    const ParameterResolver<calc_type>& pr, const MST<size_t>& p_map, int n_thread) -> VT<py_qs_datas_t> {
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
            f_and_g[j][0] = qs_policy_t::GetExpectation(sim_qs.qs, hams[j]->ham_, dim);
            auto ham_matrix = qs_policy_t::HamiltonianMatrix(hams[j]->ham_, dim);
            sim_hams[j - start] = std::move(derived_t{ham_matrix, n_qubits, seed});
        }
        sim_qs.CopyQS(this->qs);
        index_t n = circ.size();
        for (const auto& g : herm_circ) {
            --n;
            if (g->name_ == gU3) {
                auto u3 = static_cast<U3<calc_type>*>(circ[n].get());
                const auto& [title, jac] = u3->jacobi;
                if (title.size() != 0) {
                    for (index_t a = 0; a <= n; a++) {
                        sim_qs.ApplyGate(circ[a], pr);
                    }
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffU3(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto u3_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(u3_grad.matrix_[0][idx]);
                        }
                    }
                    sim_qs.CopyQS(this->qs);
                }
            } else if (g->name_ == gFSim) {
                auto fsim = static_cast<FSim<calc_type>*>(circ[n].get());
                const auto& [title, jac] = fsim->jacobi;
                if (title.size() != 0) {
                    for (index_t a = 0; a <= n; a++) {
                        sim_qs.ApplyGate(circ[a], pr);
                    }
                    for (int j = start; j < end; j++) {
                        auto intrin_grad = ExpectDiffFSim(sim_qs.qs, sim_hams[j - start].qs, circ[n], pr, dim);
                        auto fsim_grad = Dim2MatrixMatMul<calc_type>(intrin_grad, jac);
                        for (const auto& [name, idx] : title) {
                            f_and_g[j][1 + p_map.at(name)] += 2 * std::real(fsim_grad.matrix_[0][idx]);
                        }
                    }
                    sim_qs.CopyQS(this->qs);
                }
            } else if (g->params_.data_.size() != g->params_.no_grad_parameters_.size()) {
                for (index_t a = 0; a <= n; a++) {
                    sim_qs.ApplyGate(circ[a], pr);
                }
                if (g->is_custom_) {
                    auto val = circ[n]->params_.Combination(pr).const_value;
                    auto herm_val = g->params_.Combination(pr).const_value;
                    Dim2Matrix<calc_type> diff_m = circ[n]->numba_param_diff_matrix_(val);
                    Dim2Matrix<calc_type> herm_m = g->numba_param_matrix_(herm_val);
                    for (int j = start; j < end; j++) {
                        auto gi = qs_policy_t::ExpectDiffMatrixGate(sim_qs.qs, sim_hams[j - start].qs, g->obj_qubits_,
                                                                    g->ctrl_qubits_, diff_m.matrix_, herm_m.matrix_,
                                                                    dim);
                        for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                            f_and_g[j][1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                        }
                    }
                } else {
                    for (int j = start; j < end; j++) {
                        auto gi = ExpectDiffGate(sim_qs.qs, sim_hams[j - start].qs, circ[n], dim);
                        for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                            f_and_g[j][1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
                        }
                    }
                }
                sim_qs.CopyQS(this->qs);
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
    size_t batch_threads, size_t mea_threads) -> VT<VT<py_qs_datas_t>> {
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
        ParameterResolver<calc_type> pr = ParameterResolver<calc_type>();
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
                    ParameterResolver<calc_type> pr = ParameterResolver<calc_type>();
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
VT<unsigned> DensityMatrixState<qs_policy_t_>::Sampling(const circuit_t& circ, const ParameterResolver<calc_type>& pr,
                                                        size_t shots, const MST<size_t>& key_map, unsigned int seed) {
    auto key_size = key_map.size();
    VT<unsigned> res(shots * key_size);
    RndEngine rnd_eng = RndEngine(seed);
    std::uniform_real_distribution<double> dist(1.0, (1 << 20) * 1.0);
    std::function<double()> rng = std::bind(dist, std::ref(rnd_eng));
    for (size_t i = 0; i < shots; i++) {
        derived_t sim{n_qubits, static_cast<unsigned>(rng())};
        qs_policy_t::CopyQS(sim.qs, qs, dim);
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
