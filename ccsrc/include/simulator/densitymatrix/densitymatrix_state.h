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

#ifndef INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP
#define INCLUDE_DENSITYMATRIX_DENSITYMATRIX_STATE_HPP
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

#include "core/mq_base_types.h"
#include "math/pr/parameter_resolver.h"
#include "ops/basic_gate.h"
#include "ops/gates.h"
#include "ops/hamiltonian.h"
#include "simulator/timer.h"
#include "simulator/utils.h"

namespace mindquantum::sim::densitymatrix::detail {
template <typename qs_policy_t_>
struct BLAS;

template <typename qs_policy_t_>
class DensityMatrixState {
    friend struct BLAS<qs_policy_t_>;

 public:
    using qs_policy_t = qs_policy_t_;
    using calc_type = typename qs_policy_t::calc_type;
    using derived_t = DensityMatrixState<qs_policy_t>;
    using circuit_t = std::vector<std::shared_ptr<BasicGate>>;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using qs_data_p_t = typename qs_policy_t::qs_data_p_t;
    using py_qs_data_t = typename qs_policy_t::py_qs_data_t;
    using py_qs_datas_t = typename qs_policy_t::py_qs_datas_t;
    using matrix_t = typename qs_policy_t::matrix_t;
    using RndEngine = std::mt19937;

    //! ctor
    DensityMatrixState() = default;
    explicit DensityMatrixState(qbit_t n_qubits, unsigned seed = 42);
    DensityMatrixState(qs_data_p_t qs, qbit_t n_qubits, unsigned seed = 42);

    DensityMatrixState(const DensityMatrixState<qs_policy_t>& sim);
    derived_t& operator=(const DensityMatrixState<qs_policy_t>& sim);
    DensityMatrixState(DensityMatrixState<qs_policy_t>&& sim);
    derived_t& operator=(DensityMatrixState<qs_policy_t>&& sim);

    //! dtor
    virtual ~DensityMatrixState() {
        qs_policy_t::FreeState(&qs);
    }

    virtual tensor::TDtype DType();

    //! Reset the quantum state to quantum zero state
    virtual void Reset();

    //! Display basic information of this quantum state
    virtual void Display(qbit_t qubits_limit = 10) const;

    //! Get the quantum state value
    virtual matrix_t GetQS() const;

    //! Set the quantum state value
    virtual void SetQS(const py_qs_datas_t& qs_out);
    virtual void SetDM(const matrix_t& qs_out);
    virtual void CopyQS(const qs_data_p_t& qs_src);

    //! Get the purity of density matrix
    virtual calc_type Purity() const;

    //! Get the partial trace of density matrix
    virtual matrix_t GetPartialTrace(const qbits_t& objs) const;

    //! Transform to vector from a pure density matrix (with an unknown global phase)
    virtual py_qs_datas_t PureStateVector();

    //! Apply a hamiltonian on this quantum state
    void ApplyHamiltonian(const Hamiltonian<calc_type>& ham);

    /*!
     * \brief Apply a quantum gate on this quantum state, quantum gate can be
     * normal quantum gate, measurement gate and noise channel
     */
    virtual index_t ApplyGate(const std::shared_ptr<BasicGate>& gate,
                              const parameter::ParameterResolver& pr = parameter::ParameterResolver(),
                              bool diff = false);

    virtual void ApplyChannel(const std::shared_ptr<BasicGate>& gate);

    //! Apply a quantum circuit on this quantum state
    virtual std::map<std::string, int> ApplyCircuit(const circuit_t& circ, const parameter::ParameterResolver& pr
                                                                           = parameter::ParameterResolver());

    virtual index_t ApplyMeasure(const std::shared_ptr<BasicGate>& gate);

    virtual tensor::Matrix ExpectDiffGate(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                          const std::shared_ptr<BasicGate>& gate,
                                          const parameter::ParameterResolver& pr, index_t dim) const;

    virtual tensor::Matrix ExpectDiffU3(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                        const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                        index_t dim) const;

    virtual tensor::Matrix ExpectDiffRn(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                        const std::shared_ptr<BasicGate>& gate, const parameter::ParameterResolver& pr,
                                        index_t dim) const;

    virtual tensor::Matrix ExpectDiffFSim(const qs_data_p_t& dens_matrix, const qs_data_p_t& ham_matrix,
                                          const std::shared_ptr<BasicGate>& gate,
                                          const parameter::ParameterResolver& pr, index_t dim) const;

    virtual py_qs_data_t GetStateExpectation(const qs_data_p_t& qs_out, const Hamiltonian<calc_type>& ham,
                                             index_t dim) const;

    virtual py_qs_data_t GetExpectation(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                        const parameter::ParameterResolver& pr) const;

    virtual py_qs_datas_t GetExpectationWithReversibleGradOneOne(const Hamiltonian<calc_type>& ham,
                                                                 const circuit_t& circ, const circuit_t& herm_circ,
                                                                 const parameter::ParameterResolver& pr,
                                                                 const MST<size_t>& p_map, int n_thread) const;

    virtual VT<py_qs_datas_t> GetExpectationWithReversibleGradOneMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const parameter::ParameterResolver& pr, const MST<size_t>& p_map,
        int n_thread) const;

    virtual VT<VT<py_qs_datas_t>> GetExpectationWithReversibleGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name,
        const VS& ans_name, size_t batch_threads, size_t mea_threads) const;

    virtual py_qs_datas_t GetExpectationWithNoiseGradOneOne(const Hamiltonian<calc_type>& ham, const circuit_t& circ,
                                                            const circuit_t& herm_circ,
                                                            const parameter::ParameterResolver& pr,
                                                            const MST<size_t>& p_map) const;

    virtual VT<py_qs_datas_t> GetExpectationWithNoiseGradOneMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const parameter::ParameterResolver& pr, const MST<size_t>& p_map,
        int n_thread) const;

    virtual VT<VT<py_qs_datas_t>> GetExpectationWithNoiseGradMultiMulti(
        const std::vector<std::shared_ptr<Hamiltonian<calc_type>>>& hams, const circuit_t& circ,
        const circuit_t& herm_circ, const VVT<calc_type>& enc_data, const VT<calc_type>& ans_data, const VS& enc_name,
        const VS& ans_name, size_t batch_threads, size_t mea_threads) const;

    virtual VT<unsigned> Sampling(const circuit_t& circ, const parameter::ParameterResolver& pr, size_t shots,
                                  const MST<size_t>& key_map, unsigned int seed) const;

    virtual VT<unsigned> SamplingMeasurementEndingWithoutNoise(const circuit_t& circ,
                                                               const parameter::ParameterResolver& pr, size_t shots,
                                                               const MST<size_t>& key_map, unsigned int seed) const;

    template <typename policy_des, template <typename p_src, typename p_des> class cast_policy>
    DensityMatrixState<policy_des> astype(unsigned new_seed) const {
        return DensityMatrixState<policy_des>(cast_policy<qs_policy_t, policy_des>::cast(this->qs, this->dim),
                                              this->n_qubits, new_seed);
    }

 protected:
    qs_data_p_t qs = nullptr;
    qbit_t n_qubits = 0;
    index_t dim = 0;
    unsigned seed = 0;
    RndEngine rnd_eng_;
    std::function<double()> rng_;
};
}  // namespace mindquantum::sim::densitymatrix::detail

#include "simulator/densitymatrix/densitymatrix_state.tpp"  // NOLINT

#endif
