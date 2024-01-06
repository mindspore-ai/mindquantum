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

#ifndef INCLUDE_VECTOR_VECTOR_STATE_HPP
#define INCLUDE_VECTOR_VECTOR_STATE_HPP
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
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/timer.h"
#include "simulator/utils.h"

namespace mindquantum::sim::vector::detail {
template <typename qs_policy_t_>
struct BLAS;
template <typename sim_t>
struct VectorStateAdjoint;

template <typename qs_policy_t_>
class VectorState {
    friend struct BLAS<qs_policy_t_>;
    friend struct VectorStateAdjoint<VectorState<qs_policy_t_>>;

 public:
    using qs_policy_t = qs_policy_t_;
    using calc_type = typename qs_policy_t::calc_type;
    using derived_t = VectorState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using qs_data_p_t = typename qs_policy_t::qs_data_p_t;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using RndEngine = std::mt19937;

    //! ctor
    VectorState() = default;
    explicit VectorState(qbit_t n_qubits, unsigned seed = 42);
    VectorState(qbit_t n_qubits, unsigned seed, qs_data_p_t vec);
    VectorState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed = 42);

    VectorState(const VectorState<qs_policy_t>& sim);
    derived_t& operator=(const VectorState<qs_policy_t>& sim);
    VectorState(VectorState<qs_policy_t>&& sim);
    derived_t& operator=(VectorState<qs_policy_t>&& sim);

    //! dtor
    virtual ~VectorState() {
        qs_policy_t::FreeState(&qs);
    }

    virtual tensor::TDtype DType();

    //! Reset the quantum state to quantum zero state
    virtual void Reset();

    //! Set the random seed to a new seed
    virtual void SetSeed(unsigned new_seed);

    //! Display basic information of this quantum state
    virtual void Display(qbit_t qubits_limit = 10) const;

    //! Get the quantum state value
    virtual VT<py_qs_data_t> GetQS() const;

    //! Set the quantum state value
    virtual void SetQS(const VT<py_qs_data_t>& qs_out);

    /*!
     * \brief Apply a quantum gate on this quantum state, quantum gate can be
     * normal quantum gate, measurement gate and noise channel
     */
    virtual index_t ApplyGate(const std::shared_ptr<BasicGate>& gate,
                              const parameter::ParameterResolver& pr = parameter::ParameterResolver(),
                              bool diff = false);

    //! Apply a measurement gate on this quantum state, return the collapsed qubit state
    virtual index_t ApplyMeasure(const std::shared_ptr<BasicGate>& gate);

    //! Apply a noise channel on this quantum state
    virtual void ApplyChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a pauli channel on this quantum state
    virtual void ApplyPauliChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a group of pauli channel
    virtual void ApplyGroupedPauliChannels(const std::shared_ptr<BasicGate>& gate);

    //! Apply a depolarizing channel on this quantum state
    virtual void ApplyDepolarizingChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a customized kraus channel
    virtual void ApplyKrausChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a damping channel
    virtual void ApplyDampingChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a thermal relaxation channel on this quantum state
    virtual void ApplyThermalRelaxationChannel(const std::shared_ptr<BasicGate>& gate);

    //! calculate the expectation of differential form of parameterized gate two quantum state. That is
    //! <bra| \partial_\theta{U} |ket>
    virtual tensor::Matrix ExpectDiffGate(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                          const std::shared_ptr<BasicGate>& gate,
                                          const parameter::ParameterResolver& pr, index_t dim) const;

    virtual tensor::Matrix ExpectDiffU3(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                        const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                        index_t dim) const;

    virtual tensor::Matrix ExpectDiffRn(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                        const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                        index_t dim) const;

    virtual tensor::Matrix ExpectDiffFSim(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                          const std::shared_ptr<BasicGate>& gate,
                                          const parameter::ParameterResolver& pr, index_t dim) const;
    //! Apply a quantum circuit on this quantum state
    virtual std::map<std::string, int> ApplyCircuit(const circuit_t& circ, const parameter::ParameterResolver& pr
                                                                           = parameter::ParameterResolver());

    //! Apply a hamiltonian on this quantum state
    virtual void ApplyHamiltonian(const Hamiltonian<calc_type>& ham);

    //! Get the matrix of quantum circuit.
    virtual VVT<py_qs_data_t> GetCircuitMatrix(const circuit_t& circ, const parameter::ParameterResolver& pr) const;

    //! Get expectation of given hamiltonian
    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                        const parameter::ParameterResolver& pr) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                        const circuit_t& circ_left, const parameter::ParameterResolver& pr) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ_right,
                                        const circuit_t& circ_left, const derived_t& simulator_left,
                                        const parameter::ParameterResolver& pr) const;

    //! Get the expectation of hamiltonian
    //! Here multiple hamiltonian and multiple parameters are needed
    virtual VT<VVT<py_qs_data_t>> GetExpectationWithGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name,
        const VS& ans_name, size_t batch_threads, size_t mea_threads) const;

    virtual VT<VVT<py_qs_data_t>> QramExpectationWithGrad(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const VVT<py_qs_data_t>& init_states, const VT<calc_type>& ans_data,
        const VS& ans_name, size_t batch_threads, size_t mea_threads) const;

    virtual VT<VVT<py_qs_data_t>> GetExpectationNonHermitianWithGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
        const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
        const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
        const derived_t& simulator_left, size_t batch_threads, size_t mea_threads) const;

    //! Get the expectation and gradient of hamiltonian by parameter-shift rule
    virtual VVT<py_qs_data_t> GetExpectationWithGradParameterShiftOneMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const parameter::ParameterResolver& pr, const MST<size_t>& p_map, int n_thread);

    virtual VT<VVT<py_qs_data_t>> GetExpectationWithGradParameterShiftMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
        size_t batch_threads, size_t mea_threads);

    virtual VT<unsigned> Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr, size_t shots,
                                  const MST<size_t>& key_map, unsigned seed) const;

    virtual VT<unsigned> SamplingMeasurementEndingWithoutNoise(const circuit_t& circ,
                                                               const parameter::ParameterResolver& pr, size_t shots,
                                                               const MST<size_t>& key_map, unsigned seed) const;

    template <typename policy_des, template <typename p_src, typename p_des> class cast_policy>
    VectorState<policy_des> astype(unsigned seed) const;

 protected:
    qs_data_p_t qs = nullptr;  // nullptr represent zero state.
    qbit_t n_qubits = 0;
    index_t dim = 0;
    unsigned seed = 0;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};

template <typename sim_t_>
struct VectorStateAdjoint {
    using sim_t = sim_t_;
    using calc_type = typename sim_t::calc_type;
    using circuit_t = typename sim_t::circuit_t;
    using gate_t = std::shared_ptr<BasicGate>;
    using ham_t = Hamiltonian<calc_type>;
    using py_qs_data_t = typename sim_t::py_qs_data_t;
    using qs_policy_t = typename sim_t::qs_policy_t;
    // -----------------------------------------------------------------------------

    static std::shared_ptr<sim_t> CopySimToSharedPtr(const sim_t* sim) {
        return std::make_shared<sim_t>(*sim);
    }
    static std::shared_ptr<sim_t> SimilarSim(const sim_t* sim, const VT<py_qs_data_t>& init_state) {
        std::shared_ptr<sim_t> out = std::make_shared<sim_t>(sim->n_qubits, sim->seed);
        out->SetQS(init_state);
        return out;
    }
    static void ApplyCircuit(sim_t* sim, const circuit_t& circ, const parameter::ParameterResolver& pr) {
        sim->ApplyCircuit(circ, pr);
    }
    static void ApplyHamiltonian(sim_t* sim, const std::shared_ptr<ham_t>& ham) {
        sim->ApplyHamiltonian(*ham.get());
    }
    static py_qs_data_t Vdot(sim_t* psi_l, sim_t* psi_r) {
        if (psi_l->dim != psi_r->dim) {
            throw std::runtime_error("Vdot need same dimension quantum state.");
        }
        return qs_policy_t::Vdot(psi_l->qs, psi_r->qs, psi_l->dim);
    }
    static void ApplyGate(sim_t* sim, const gate_t& g, const parameter::ParameterResolver& pr) {
        sim->ApplyGate(g, pr);
    }
    static bool GateRequiresGrad(const gate_t& g) {
        return g->GradRequired();
    }
    static std::pair<MST<size_t>, tensor::Matrix> GetJacobi(const gate_t& g) {
        auto p_gate = static_cast<Parameterizable*>(g.get());
        return p_gate->GetJacobi();
    }
    static tensor::Matrix ExpectDiffGate(sim_t* psi_l, sim_t* psi_r, const gate_t& g,
                                         const parameter::ParameterResolver& pr) {
        if (psi_l->dim != psi_r->dim) {
            throw std::runtime_error("ExpectDiffGate need same dimension quantum state.");
        }
        return psi_l->ExpectDiffGate(psi_l->qs, psi_r->qs, g, pr, psi_l->dim);
    }
};
}  // namespace mindquantum::sim::vector::detail

#include "simulator/vector/vector_state.tpp"  // NOLINT

#endif
