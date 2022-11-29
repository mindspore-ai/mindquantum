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

#ifndef INCLUDE_VECTOR_VECTORSTATE_TPP
#define INCLUDE_VECTOR_VECTORSTATE_TPP

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
#include <vector>

// #include "core/mq_base_types.hpp"
// #include "core/parameter_resolver.hpp"
// #include "ops/basic_gate.hpp"
// #include "ops/gates.hpp"
// #include "ops/hamiltonian.hpp"
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
DensityMatrixState<qs_policy_t_>::DensityMatrixState(qbit_t n_qubits, unsigned seed, qs_data_p_t vec)
    : n_qubits(n_qubits), dim(1UL << n_qubits), seed(seed), rnd_eng_(seed) {
    qs = qs_policy_t::Copy(vec, dim);
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
void DensityMatrixState<qs_policy_t_>::DisplayQS() const {
    qs_policy_t::DisplayQS(qs, n_qubits, dim);
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
index_t DensityMatrixState<qs_policy_t_>::ApplyGate(const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                    const ParameterResolver<calc_type>& pr, bool diff) {
    auto name = gate->name_;
    if (name == gI) {
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
    assert(gate->is_channel_);
    if (gate->name_ == "ADC") {
        qs_policy_t::ApplyAmplitudeDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else if (gate->name_ == "PDC") {
        qs_policy_t::ApplyPhaseDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else if (gate->name_ == "PL") {
        qs_policy_t::ApplyPauli(qs, gate->obj_qubits_, gate->probs_, dim);
    } else if (gate->kraus_operator_set_.size() != 0) {
        qs_policy_t::ApplyKraus(qs, gate->obj_qubits_, gate->kraus_operator_set_, dim);
    } else if (gate->name_ == "hADC") {
        qs_policy_t::ApplyHermitianAmplitudeDamping(qs, gate->obj_qubits_, gate->damping_coeff_, dim);
    } else {
        throw std::runtime_error("This noise channel not implemented.");
    }
}

template <typename qs_policy_t_>
void DensityMatrixState<qs_policy_t_>::ApplyHamiltonian(const Hamiltonian<calc_type>& ham) {
    auto new_qs = qs_policy_t::ApplyTerms(qs, ham.ham_, dim);
    qs_policy_t::FreeState(qs);
    qs = new_qs;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ApplyMeasure(const std::shared_ptr<BasicGate<calc_type>>& gate) {
    assert(gate->is_measure_);
    index_t one_mask = (1UL << gate->obj_qubits_[0]);
    auto one_amp = qs_policy_t::DiagonalConditionalCollect(qs, one_mask, one_mask, true, dim);
    index_t collapse_mask = (static_cast<index_t>(rng_() < one_amp) << gate->obj_qubits_[0]);
    qs_data_t norm_fact = (collapse_mask == 0) ? 1 / (1 - one_amp) : 1 / one_amp;
    qs_policy_t::ConditionalMul(qs, qs, one_mask, collapse_mask, norm_fact, 0.0, dim);
    return static_cast<index_t>(collapse_mask != 0);
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::ExpectDiffGate(qs_data_p_t qs, qs_data_p_t ham_matrix,
                                                      const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                      const ParameterResolver<calc_type>& pr, index_t dim)
    -> py_qs_data_t {
    auto name = gate->name_;
    auto val = gate->params_.Combination(pr).const_value;

    if (name == gRX) {
        return qs_policy_t::ExpectDiffRX(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gRY) {
        return qs_policy_t::ExpectDiffRY(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gRZ) {
        return qs_policy_t::ExpectDiffRZ(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gXX) {
        return qs_policy_t::ExpectDiffXX(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gZZ) {
        return qs_policy_t::ExpectDiffZZ(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gYY) {
        return qs_policy_t::ExpectDiffYY(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    if (name == gPS) {
        return qs_policy_t::ExpectDiffPS(qs, ham_matrix, gate->obj_qubits_, gate->ctrl_qubits_, val, dim);
    }
    throw std::invalid_argument("gate " + name + " not implement.");
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
auto DensityMatrixState<qs_policy_t_>::GetExpectationReversibleWithGrad(const Hamiltonian<calc_type>& ham,
                                                                        const circuit_t& circ,
                                                                        const circuit_t& herm_circ,
                                                                        const ParameterResolver<calc_type>& pr,
                                                                        const MST<size_t>& p_map) -> py_qs_datas_t {
    // auto timer = Timer();
    // timer.Start("First part");
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = qs_policy_t::GetExpectation(sim_qs.qs, ham.ham_, dim);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham.ham_, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    // timer.EndAndStartOther("First part", "Second part");
    for (const auto& g : herm_circ) {
        if (g->params_.data_.size() != g->params_.no_grad_parameters_.size()) {
            // timer.Start("ExpectDiffGate");
            auto gi = ExpectDiffGate(sim_qs.qs, sim_ham.qs, g, pr, dim);
            // timer.End("ExpectDiffGate");
            for (auto& it : g->params_.GetRequiresGradParameters()) {
                f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * g->params_.data_.at(it);
            }
        }
        sim_ham.ApplyGate(g, pr);
        sim_qs.ApplyGate(g, pr);
    }
    // timer.End("Second part");
    // timer.Analyze();
    qs_policy_t::FreeState(ham_matrix);
    return f_and_g;
}

template <typename qs_policy_t_>
auto DensityMatrixState<qs_policy_t_>::GetExpectationNonReversibleWithGrad(const Hamiltonian<calc_type>& ham,
                                                                           const circuit_t& circ,
                                                                           const circuit_t& herm_circ,
                                                                           const ParameterResolver<calc_type>& pr,
                                                                           const MST<size_t>& p_map) -> py_qs_datas_t {
    // auto timer = Timer();
    // timer.Start("First part");
    py_qs_datas_t f_and_g(1 + p_map.size(), 0);
    derived_t sim_qs = *this;
    sim_qs.ApplyCircuit(circ, pr);
    f_and_g[0] = qs_policy_t::GetExpectation(sim_qs.qs, ham.ham_, dim);
    sim_qs.CopyQS(this->qs);
    auto ham_matrix = qs_policy_t::HamiltonianMatrix(ham.ham_, dim);
    derived_t sim_ham{ham_matrix, n_qubits, seed};
    // timer.EndAndStartOther("First part", "Second part");
    if (circ.size() != herm_circ.size()) {
        std::runtime_error("In density matrix mode, circ and herm_circ must be the same size.");
    }
    index_t n = circ.size();
    for (index_t i = 0; i < herm_circ.size(); i++) {
        --n;
        if (circ[n]->params_.data_.size() != circ[n]->params_.no_grad_parameters_.size()) {
            // timer.Start("ExpectDiffGate");
            for (index_t j = 0; j <= n; j++) {
                sim_qs.ApplyGate(circ[j], pr);
            }
            auto gi = ExpectDiffGate(sim_qs.qs, sim_ham.qs, circ[n], pr, dim);
            // timer.End("ExpectDiffGate");
            for (auto& it : circ[n]->params_.GetRequiresGradParameters()) {
                f_and_g[1 + p_map.at(it)] += 2 * std::real(gi) * circ[n]->params_.data_.at(it);
            }
            sim_qs.CopyQS(this->qs);
        }
        sim_ham.ApplyGate(herm_circ[i], pr);
    }

    // timer.End("Second part");
    // timer.Analyze();
    qs_policy_t::FreeState(ham_matrix);
    return f_and_g;
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

}  // namespace mindquantum::sim::densitymatrix::detail

#endif
