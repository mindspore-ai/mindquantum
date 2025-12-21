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

#ifndef INCLUDE_SIMULATOR_MPS_MPS_STATE_HPP
#define INCLUDE_SIMULATOR_MPS_MPS_STATE_HPP

#include <cstdint>

#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/hamiltonian.h"

namespace mindquantum::sim::mps::detail {
template <typename qs_policy_t_>
class MPSState {
 public:
    using qs_policy_t = qs_policy_t_;
    using calc_type = typename qs_policy_t::calc_type;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using complex_t = py_qs_data_t;
    using derived_t = MPSState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;
    using RndEngine = std::mt19937;

    MPSState() = default;
    explicit MPSState(qbit_t n_qubits, unsigned seed = 42, uint32_t max_bond_dimension = 0);

    MPSState(const MPSState& other);
    derived_t& operator=(const MPSState& other);
    MPSState(MPSState&& other) noexcept;
    derived_t& operator=(MPSState&& other) noexcept;

    virtual ~MPSState() = default;

    virtual tensor::TDtype DType();

    virtual void Reset();
    virtual void SetSeed(unsigned new_seed);
    virtual void Display(qbit_t qubits_limit = 10) const;

    virtual VT<py_qs_data_t> GetQS() const;
    virtual void SetQS(const VT<py_qs_data_t>& qs);

    virtual VVT<py_qs_data_t> GetReducedDensityMatrix(const qbits_t& kept_qubits) const;

    virtual index_t ApplyGate(const std::shared_ptr<BasicGate>& gate,
                              const parameter::ParameterResolver& pr = parameter::ParameterResolver(), bool diff = false);

    virtual std::map<std::string, int> ApplyCircuit(const circuit_t& circ,
                                                    const parameter::ParameterResolver& pr = parameter::ParameterResolver());

    virtual VT<unsigned> Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr, size_t shots,
                                  const MST<size_t>& key_map, unsigned seed) const;

    virtual VT<unsigned> SamplingMeasurementEndingWithoutNoise(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                               size_t shots, const MST<size_t>& key_map,
                                                               unsigned seed) const;

    virtual void ApplyHamiltonian(const Hamiltonian<calc_type>& ham);

    virtual VVT<py_qs_data_t> GetCircuitMatrix(const circuit_t& circ, const parameter::ParameterResolver& pr) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                        const parameter::ParameterResolver& pr) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                        const circuit_t& circ_left, const parameter::ParameterResolver& pr) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                        const circuit_t& circ_left, const derived_t& simulator_left,
                                        const parameter::ParameterResolver& pr) const;

    virtual calc_type Purity() const;
    virtual VT<py_qs_data_t> PureStateVector() const;

    template <typename policy_des>
    MPSState<policy_des> astype(unsigned new_seed) const;

 private:
    struct GammaSite {
        VT<complex_t> g0;
        VT<complex_t> g1;
        size_t dl = 1;
        size_t dr = 1;
    };

    void InitZeroState_();

    void Apply1QOnPhysical_(qbit_t pos, const std::array<complex_t, 4>& u);
    void Apply2QOnBond_(qbit_t bond, const std::array<complex_t, 16>& u);

    // Full-state helpers (slow paths, used for SetQS/measurement).
    VT<complex_t> ContractToStateVectorPhysicalOrder_() const;
    VT<complex_t> PermutePhysicalToLogical_(const VT<complex_t>& phys) const;

    void SetFromStateVectorLogical_(const VT<complex_t>& vec);
    void SetFromStateVectorPhysical_(const VT<complex_t>& vec_phys);

    std::map<std::string, int> ApplyCircuitCompiled_(const circuit_t& circ, const parameter::ParameterResolver& pr,
                                                     bool diff);

    // Expectation value helpers
    py_qs_data_t ExpectationPauliTerms_(const VT<PauliTerm<calc_type>>& terms) const;
    py_qs_data_t ExpectationSparse_(const Hamiltonian<calc_type>& ham) const;

 private:
    qbit_t n_qubits_ = 0;
    unsigned seed_ = 0;
    uint32_t max_bond_dimension_ = 0;

    RndEngine rnd_eng_;
    std::function<double()> rng_;

    // Vidal Γ/Λ form in physical position order [0..n_qubits_-1].
    VT<VT<calc_type>> lambdas_;   // size n_qubits_ + 1
    VT<GammaSite> gammas_;        // size n_qubits_

    // Dynamic mapping between logical qubits (bit positions) and MPS physical sites.
    qbits_t pos_of_logical_;   // size n_qubits_
    qbits_t logical_at_pos_;   // size n_qubits_
};
}  // namespace mindquantum::sim::mps::detail

#include "simulator/mps/mps_state.tpp"

#endif  // INCLUDE_SIMULATOR_MPS_MPS_STATE_HPP

