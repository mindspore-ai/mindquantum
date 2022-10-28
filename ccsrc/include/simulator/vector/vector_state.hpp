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

#ifndef INCLUDE_VECTOR_VECTORSTATE_HPP
#define INCLUDE_VECTOR_VECTORSTATE_HPP
#include <cmath>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "core/mq_base_types.hpp"
#include "core/parameter_resolver.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gates.hpp"
#include "ops/hamiltonian.hpp"
#include "simulator/timer.h"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::vector::detail {
template <typename qs_policy_t_>
struct BLAS;

template <typename qs_policy_t_>
class VectorState {
    friend struct BLAS<qs_policy_t_>;

 public:
    using qs_policy_t = qs_policy_t_;
    using derived_t = VectorState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate<calc_type>>>;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using qs_data_p_t = typename qs_policy_t::qs_data_p_t;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using py_qs_datas_t = typename qs_policy_t::py_qs_datas_t;
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
    ~VectorState() {
        qs_policy_t::FreeState(qs);
    }

    //! Reset the quantum state to quantum zero state
    void Reset();

    //! Display basic information of this quantum state
    void Display(qbit_t qubits_limit = 10) const;

    //! Get the quantum state value
    py_qs_datas_t GetQS() const;

    //! Set the quantum state value
    void SetQS(const py_qs_datas_t& qs_out);

    //! Apply a quantum gate on this quantum state, quantum gate can be normal quantum gate, measurement gate and noise
    //! channel
    index_t ApplyGate(const std::shared_ptr<BasicGate<calc_type>>& gate,

                      const ParameterResolver<calc_type>& pr = ParameterResolver<calc_type>(), bool diff = false);

    //! Apply a measurement gate on this quantum state, return the collapsed qubit state
    auto ApplyMeasure(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! Apply a noise channel on this quantum state
    auto ApplyChannel(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! Apply a pauli channel on this quantum state
    auto ApplyPauliChannel(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! Apply a customed kraus channel
    auto ApplyKrausChannel(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! Apply a damping channel
    auto ApplyDampingChannel(const std::shared_ptr<BasicGate<calc_type>>& gate);

    //! calculate the expectaton of differential form of parameterized gate two quantum state. That is
    //! <bra| \partial_\theta{U} |ket>
    static py_qs_data_t ExpectDiffGate(qs_data_p_t bra, qs_data_p_t ket,
                                       const std::shared_ptr<BasicGate<calc_type>>& gate,
                                       const ParameterResolver<calc_type>& pr, index_t dim);

    static Dim2Matrix<calc_type> ExpectDiffU3(qs_data_p_t bra, qs_data_p_t ket,
                                              const std::shared_ptr<BasicGate<calc_type>>& gate,
                                              const ParameterResolver<calc_type>& pr, index_t dim);
    static Dim2Matrix<calc_type> ExpectDiffFSim(qs_data_p_t bra, qs_data_p_t ket,
                                                const std::shared_ptr<BasicGate<calc_type>>& gate,
                                                const ParameterResolver<calc_type>& pr, index_t dim);
    //! Apply a quantum circuit on this quantum state
    auto ApplyCircuit(const circuit_t& circ, const ParameterResolver<calc_type>& pr = ParameterResolver<calc_type>());

    //! Apply a hamiltonian on this quantum state
    void ApplyHamiltonian(const Hamiltonian<calc_type>& ham);

    //! Get the matrix of quantum circuit.
    VT<py_qs_datas_t> GetCircuitMatrix(const circuit_t& circ, const ParameterResolver<calc_type>& pr);

    //! Get expectation of given hamiltonian
    py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham) {
        auto ket = *this;
        ket.ApplyHamiltonian(ham);
        return qs_policy_t::Vdot(this->qs, ket.qs, dim);
    }

    //! Get the expectation of hamiltonian
    //! Here a single hamiltonian and single parameter data are needed
    py_qs_datas_t GetExpectationWithGradOneOne(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                               const circuit_t& herm_circ, const ParameterResolver<calc_type>& pr,
                                               const MST<size_t>& p_map);

    //! Get the expectation of hamiltonian
    //! Here multiple hamiltonians and single parameter data are needed
    VT<py_qs_datas_t> GetExpectationWithGradOneMulti(const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
                                                     const circuit_t& circ, const circuit_t& herm_circ,
                                                     const ParameterResolver<calc_type>& pr, const MST<size_t>& p_map,
                                                     int n_thread);
    //! Get the expectation of hamiltonian
    //! Here multiple hamiltonian and multiple parameters are needed
    VT<VT<py_qs_datas_t>> GetExpectationWithGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name,
        const VS& ans_name, size_t batch_threads, size_t mea_threads);

    VT<py_qs_datas_t> GetExpectationNonHermitianWithGradOneMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
        const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
        const ParameterResolver<calc_type>& pr, const MST<size_t>& p_map, int n_thread,
        const derived_t& simulator_left);

    VT<py_qs_datas_t> LeftSizeGradOneMulti(const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
                                           const circuit_t& herm_left_circ, const ParameterResolver<calc_type>& pr,
                                           const MST<size_t>& p_map, int n_thread, const derived_t& simulator_left,
                                           const derived_t& simulator_right);

    VT<VT<py_qs_datas_t>> GetExpectationNonHermitianWithGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams,
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& herm_hams, const circuit_t& left_circ,
        const circuit_t& herm_left_circ, const circuit_t& right_circ, const circuit_t& herm_right_circ,
        const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name, const VS& ans_name,
        const derived_t& simulator_left, size_t batch_threads, size_t mea_threads);

    VT<unsigned> Sampling(const circuit_t& circ, const ParameterResolver<calc_type>& pr, size_t shots,
                          const MST<size_t>& key_map, unsigned seed);

 private:
    qs_data_p_t qs = nullptr;
    qbit_t n_qubits = 0;
    index_t dim = 0;
    unsigned seed = 0;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};
}  // namespace mindquantum::sim::vector::detail

#include "simulator/vector/vector_state.tpp"  // NOLINT

#endif
