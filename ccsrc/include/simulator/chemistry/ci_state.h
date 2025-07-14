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

#ifndef INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_HPP
#define INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_HPP

#include <complex>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "simulator/chemistry/detail/cpp_ci_hamiltonian.h"
#include "simulator/chemistry/detail/cpp_excitation_operator.h"
#include "simulator/chemistry/detail/ci_vector_grad_policy.h"

namespace mindquantum::sim::chem {

template <typename policy_t>
class CIState {
 public:
    using qs_policy_t = policy_t;
    using calc_type = typename policy_t::calc_type;
    using qs_data_p_t = typename policy_t::qs_data_p_t;
    using py_qs_data_t = typename policy_t::py_qs_data_t;
    using circuit_t = std::vector<std::shared_ptr<detail::CppExcitationOperator<calc_type>>>;

    //! ctor
    CIState() = default;
    explicit CIState(qbit_t n_qubits, int n_electrons, unsigned seed = 42);
    CIState(qbit_t n_qubits, int n_electrons, unsigned seed, qs_data_p_t vec);

    CIState(const CIState<policy_t>& sim);
    CIState(CIState<policy_t>&& sim);
    CIState<policy_t>& operator=(const CIState<policy_t>& sim);
    CIState<policy_t>& operator=(CIState<policy_t>&& sim);

    //! dtor
    ~CIState();

    //! Reset the quantum state to zero state
    void Reset();

    //! Set random seed
    void SetSeed(unsigned new_seed);

    //! Get the quantum state vector (sparse representation)
    std::vector<std::pair<uint64_t, py_qs_data_t>> GetQS() const;

    //! Set the quantum state vector (sparse representation)
    void SetQS(const std::vector<std::pair<uint64_t, py_qs_data_t>>& qs_out);

    //! Get the data type of quantum state values
    tensor::TDtype DType() const;

    //! Apply a single UCC excitation gate G with parameters from resolver.
    void ApplySingleUCCGate(const std::shared_ptr<detail::CppExcitationOperator<calc_type>>& gate,
                            const parameter::ParameterResolver& pr);
    //! Compute expectation value ⟨psi|H|psi⟩ for a CI Hamiltonian on the current state
    calc_type GetExpectationValue(const detail::CppCIHamiltonian<calc_type>& ham) const;

    //! Compute expectation and its gradient for multiple parameter sets via Hermitian-adjoint algorithm.
    VT<VVT<std::complex<calc_type>>> GetExpectationWithGradMultiMulti(const detail::CppCIHamiltonian<calc_type>& ham,
                                                                      const circuit_t& circ,
                                                                      const VVT<calc_type>& enc_data,
                                                                      const VT<calc_type>& ans_data, const VS& enc_name,
                                                                      const VS& ans_name, size_t batch_threads,
                                                                      size_t mea_threads) const;
    template <typename>
    friend struct detail::CIVectorGradPolicy;

    //! Apply a sequence of UCC excitation operators from resolver.
    void ApplyCircuit(const std::vector<std::shared_ptr<detail::CppExcitationOperator<calc_type>>>& circuit,
                      const parameter::ParameterResolver& pr);

 protected:
    qs_data_p_t qs_ = nullptr;
    qbit_t n_qubits_ = 0;
    int n_electrons_ = 0;
    unsigned seed_ = 0;
};

}  // namespace mindquantum::sim::chem

#include "simulator/chemistry/ci_state.tpp"  // NOLINT
#endif  // INCLUDE_SIMULATOR_CHEMISTRY_CI_STATE_HPP
