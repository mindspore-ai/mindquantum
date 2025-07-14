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
#ifndef INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_TPP
#define INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_TPP

#include <algorithm>
#include <iterator>

#include "math/tensor/traits.h"
#include "simulator/chemistry/ci_state.h"
#include "simulator/vector/gradient.h"

namespace mindquantum::sim::chem {

// Constructors

template <typename policy_t>
CIState<policy_t>::CIState(const CIState<policy_t>& sim)
    : n_qubits_(sim.n_qubits_), n_electrons_(sim.n_electrons_), seed_(sim.seed_) {
    qs_ = policy_t::AllocateState(n_qubits_, n_electrons_, seed_);
    policy_t::CopyState(sim.qs_, qs_, n_qubits_, n_electrons_);
}

template <typename policy_t>
CIState<policy_t>::CIState(CIState<policy_t>&& sim)
    : n_qubits_(sim.n_qubits_), n_electrons_(sim.n_electrons_), seed_(sim.seed_), qs_(sim.qs_) {
    sim.qs_ = nullptr;
}

template <typename policy_t>
CIState<policy_t>::CIState(qbit_t n_qubits, int n_electrons, unsigned seed)
    : n_qubits_(n_qubits), n_electrons_(n_electrons), seed_(seed) {
    qs_ = policy_t::AllocateState(n_qubits_, n_electrons_, seed_);
}

template <typename policy_t>
CIState<policy_t>::CIState(qbit_t n_qubits, int n_electrons, unsigned seed, qs_data_p_t vec)
    : n_qubits_(n_qubits), n_electrons_(n_electrons), seed_(seed) {
    qs_ = policy_t::AllocateState(n_qubits_, n_electrons_, seed_);
    policy_t::CopyState(vec, qs_, n_qubits_, n_electrons_);
}

// Destructor

template <typename policy_t>
CIState<policy_t>::~CIState() {
    if (qs_) {
        policy_t::FreeState(qs_);
    }
}

// Assignment

template <typename policy_t>
CIState<policy_t>& CIState<policy_t>::operator=(const CIState<policy_t>& sim) {
    if (this != &sim) {
        if (qs_)
            policy_t::FreeState(qs_);
        n_qubits_ = sim.n_qubits_;
        n_electrons_ = sim.n_electrons_;
        seed_ = sim.seed_;
        qs_ = policy_t::AllocateState(n_qubits_, n_electrons_, seed_);
        policy_t::CopyState(sim.qs_, qs_, n_qubits_, n_electrons_);
    }
    return *this;
}

template <typename policy_t>
CIState<policy_t>& CIState<policy_t>::operator=(CIState<policy_t>&& sim) {
    if (this != &sim) {
        if (qs_)
            policy_t::FreeState(qs_);
        n_qubits_ = sim.n_qubits_;
        n_electrons_ = sim.n_electrons_;
        seed_ = sim.seed_;
        qs_ = sim.qs_;
        sim.qs_ = nullptr;
    }
    return *this;
}

// Reset

template <typename policy_t>
void CIState<policy_t>::Reset() {
    if (qs_)
        policy_t::FreeState(qs_);
    qs_ = policy_t::AllocateState(n_qubits_, n_electrons_, seed_);
}

// Set seed

template <typename policy_t>
void CIState<policy_t>::SetSeed(unsigned new_seed) {
    seed_ = new_seed;
}

// Get/Set state

template <typename policy_t>
std::vector<std::pair<uint64_t, typename policy_t::py_qs_data_t>> CIState<policy_t>::GetQS() const {
    return policy_t::GetState(qs_, n_qubits_, n_electrons_);
}

template <typename policy_t>
void CIState<policy_t>::SetQS(const std::vector<std::pair<uint64_t, typename policy_t::py_qs_data_t>>& qs_out) {
    policy_t::SetState(qs_, qs_out, n_qubits_, n_electrons_);
}

// Apply a single UCC excitation gate G with parameter from resolver
template <typename policy_t>
void CIState<policy_t>::ApplySingleUCCGate(const std::shared_ptr<detail::CppExcitationOperator<calc_type>>& gate,
                                            const parameter::ParameterResolver& pr) {
    auto gate_pr = gate->coeff;
    auto combo = gate_pr.Combination(pr);
    auto tmp = combo.GetConstValue();
    auto theta = *reinterpret_cast<const double*>(tmp.data);
    policy_t::ApplyUCCGate(qs_, *gate, theta, n_qubits_, n_electrons_);
}

// Compute expectation value ⟨psi|H|psi⟩ for a CI Hamiltonian on the current state
template <typename policy_t>
typename CIState<policy_t>::calc_type CIState<policy_t>::GetExpectationValue(
    const detail::CppCIHamiltonian<calc_type>& ham) const {
    return ham.get_expectation_value(*qs_);
}

// Data type of quantum state values
template <typename policy_t>
tensor::TDtype CIState<policy_t>::DType() const {
    return tensor::to_dtype_v<py_qs_data_t>;
}

template <typename policy_t>
auto CIState<policy_t>::GetExpectationWithGradMultiMulti(const detail::CppCIHamiltonian<calc_type>& ham,
                                                          const circuit_t& circ, const VVT<calc_type>& enc_data,
                                                          const VT<calc_type>& ans_data, const VS& enc_name,
                                                          const VS& ans_name, size_t batch_threads,
                                                          size_t mea_threads) const
    -> VT<VVT<std::complex<calc_type>>> {
    using policy = detail::CIVectorGradPolicy<calc_type>;
    using helper = GradientHelper<policy>;
    std::vector<std::shared_ptr<detail::CppCIHamiltonian<calc_type>>> hams;
    hams.emplace_back(std::make_shared<detail::CppCIHamiltonian<calc_type>>(ham));
    circuit_t herm_circ;
    herm_circ.reserve(circ.size());
    std::transform(circ.begin(), circ.end(), std::back_inserter(herm_circ), [](const auto& g_ptr) {
        return std::make_shared<detail::CppExcitationOperator<calc_type>>(*g_ptr);
    });
    std::reverse(herm_circ.begin(), herm_circ.end());
    std::for_each(herm_circ.begin(), herm_circ.end(), [](auto& g) { g->coeff *= static_cast<calc_type>(-1); });
    return helper::HermitianAdjointGradient(this, circ, herm_circ, hams, enc_data, ans_data, enc_name, ans_name,
                                            batch_threads, mea_threads);
}

// Apply a sequence of UCC excitation operators from resolver
template <typename policy_t>
void CIState<policy_t>::ApplyCircuit(
    const std::vector<std::shared_ptr<detail::CppExcitationOperator<calc_type>>>& circuit,
    const parameter::ParameterResolver& pr) {
    for (const auto& gate : circuit) {
        auto gate_pr = gate->coeff;
        auto combo = gate_pr.Combination(pr);
        auto tmp = combo.GetConstValue();
        auto theta = *reinterpret_cast<const double*>(tmp.data);
        policy_t::ApplyUCCGate(qs_, *gate, theta, n_qubits_, n_electrons_);
    }
}

}  // namespace mindquantum::sim::chem

#endif  // INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_TPP
